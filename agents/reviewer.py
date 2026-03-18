from __future__ import annotations

import ast
import json
import os
import re
from typing import Any, Dict, List

try:
    from core.state import ResearchState
except ModuleNotFoundError:  # 兼容直接脚本方式
    from state import ResearchState  # type: ignore

from prompts.system_prompts import REVIEWER_SYSTEM_PROMPT, build_reviewer_rule_hint
from prompts.system_prompts import build_reviewer_user_prompt

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None


ROUTE_END = "end"
ROUTE_RESEARCHER = "researcher"
ROUTE_WRITER = "writer"


MIN_CONTEXT_ITEMS = 2
MIN_DRAFT_LENGTH = 600


def _append_error(errors: List[str], message: str) -> List[str]:
    """将错误信息追加到 errors，避免覆盖既有日志。"""

    updated = list(errors)
    updated.append(message)
    return updated


def _to_issue_list(value: Any) -> List[str]:
    """将 LLM 返回的问题字段统一为字符串列表。"""

    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        # 兼容分号、顿号与换行混排输出
        separators = ["\n", "；", ";", "、"]
        items = [text]
        for sep in separators:
            next_items: List[str] = []
            for item in items:
                next_items.extend(item.split(sep))
            items = next_items
        return [item.strip(" -") for item in items if item.strip(" -")]
    return []


def _extract_json_content(raw: str) -> str:
    """提取模型输出中的 JSON 内容（兼容 fenced block）。"""

    content = raw.strip()
    if not content.startswith("```"):
        return content

    lines = content.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    cleaned = "\n".join(lines).strip()
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()
    return cleaned


def _extract_first_json_object(raw: str) -> str:
    """从混杂文本中提取首个 JSON 对象字符串。"""

    text = raw.strip()
    start = text.find("{")
    if start < 0:
        return text

    depth = 0
    in_str = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start: idx + 1]
    return text[start:]


def _sanitize_json_text(text: str) -> str:
    """轻量修复常见 JSON 格式问题（尾逗号、BOM）。"""

    cleaned = text.strip().lstrip("\ufeff")
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    return cleaned


def _coerce_text_review_to_json(raw: str) -> Dict[str, Any] | None:
    """当模型返回非 JSON 文本时，做最小语义兜底，避免整轮降级。"""

    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None

    lowered = text.lower()
    pass_markers = ["[pass]", "评审通过", "通过当前评审", "可结束流程"]
    is_pass = any(marker in lowered for marker in pass_markers)

    def _extract_after(label_patterns: List[str]) -> List[str]:
        items: List[str] = []
        for pattern in label_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            if not match:
                continue
            block = (match.group(1) or "").strip()
            if not block:
                continue
            items.extend(_to_issue_list(block))
        # 去重并保持顺序
        dedup: List[str] = []
        seen: set[str] = set()
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            dedup.append(item)
        return dedup

    fact_issues = _extract_after([
        r"fact[_\s-]*issues\s*[:：]\s*(.+?)(?:\n\n|$)",
        r"事实问题\s*[:：]\s*(.+?)(?:\n\n|$)",
    ])
    logic_issues = _extract_after([
        r"logic[_\s-]*issues\s*[:：]\s*(.+?)(?:\n\n|$)",
        r"逻辑问题\s*[:：]\s*(.+?)(?:\n\n|$)",
    ])
    info_gaps = _extract_after([
        r"info[_\s-]*gaps\s*[:：]\s*(.+?)(?:\n\n|$)",
        r"信息缺口\s*[:：]\s*(.+?)(?:\n\n|$)",
    ])

    needs_more_research = any(k in text for k in ["补充检索", "信息不足", "证据不足", "缺少来源"]) or bool(info_gaps)

    if is_pass:
        return {
            "supporter": {
                "strengths": ["审稿文本判定为通过"],
                "supported_claims": ["可进入交付阶段"],
            },
            "skeptic": {
                "critical_issues": [],
                "missing_evidence": [],
            },
            "judge": {
                "is_satisfactory": True,
                "needs_more_research": False,
                "critique_feedback": "草稿通过当前评审，可结束流程。",
                "confidence": 0.7,
                "fact_issues": fact_issues,
                "logic_issues": logic_issues,
                "info_gaps": info_gaps,
                "controversy_points": [],
                "evidence_verdicts": [],
            },
        }

    feedback = text[:1200]
    return {
        "supporter": {
            "strengths": [],
            "supported_claims": [],
        },
        "skeptic": {
            "critical_issues": fact_issues + logic_issues,
            "missing_evidence": info_gaps,
        },
        "judge": {
            "is_satisfactory": False,
            "needs_more_research": needs_more_research,
            "critique_feedback": feedback,
            "confidence": 0.55,
            "fact_issues": fact_issues,
            "logic_issues": logic_issues,
            "info_gaps": info_gaps,
            "controversy_points": [],
            "evidence_verdicts": [],
        },
    }


def _parse_review_json(raw: str) -> Dict[str, Any]:
    """解析 Reviewer 输出 JSON，兼容常见格式噪声。"""

    candidates = [
        _sanitize_json_text(_extract_json_content(raw)),
        _sanitize_json_text(_extract_first_json_object(raw)),
        _sanitize_json_text(_extract_first_json_object(_extract_json_content(raw))),
    ]

    for cand in candidates:
        if not cand:
            continue
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    # 最后尝试解析 Python 风格字典，兼容模型偶发单引号输出。
    for cand in candidates:
        if not cand:
            continue
        try:
            parsed = ast.literal_eval(cand)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    # 最后兜底：若是非 JSON 的自然语言评审，转为最小结构化对象。
    coerced = _coerce_text_review_to_json(raw)
    if isinstance(coerced, dict):
        return coerced

    raise ValueError("Reviewer 输出无法解析为结构化 JSON")


def _to_dict_list(value: Any) -> List[Dict[str, Any]]:
    """将 LLM 返回的对象列表统一为字典列表。"""

    if not isinstance(value, list):
        return []
    result: List[Dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            result.append({str(k): item[k] for k in item})
        elif isinstance(item, str) and item.strip():
            result.append({"point": item.strip()})
    return result


def _rule_based_review(
    draft: str,
    retrieved_context: List[Dict[str, Any] | str],
) -> Dict[str, Any]:
    """规则化回退评审。"""

    if len(retrieved_context) < MIN_CONTEXT_ITEMS:
        return {
            "is_satisfactory": False,
            "needs_more_research": True,
            "critique_feedback": (
                f"{REVIEWER_SYSTEM_PROMPT} {build_reviewer_rule_hint()} "
                "当前证据不足，请补充更多高质量来源并覆盖不同观点。"
            ),
            "confidence": 0.65,
            "review_mode": "rule",
            "supporter": {
                "strengths": ["主题聚焦明确"],
                "supported_claims": [],
            },
            "skeptic": {
                "critical_issues": ["证据来源数量不足，无法支撑关键结论"],
                "missing_evidence": ["需要补充多来源检索结果"],
            },
            "judge": {
                "decision": "需补充检索",
                "rationale": "当前上下文数量不足，无法完成高置信审查。",
            },
            "controversy_points": ["当前结论是否建立在足够证据之上"],
            "evidence_verdicts": [
                {
                    "claim": "已有证据可支撑完整报告",
                    "status": "unsupported",
                    "evidence": "上下文数量低于最低阈值",
                    "action": "返回 Researcher 补充检索",
                }
            ],
            "fact_issues": [],
            "logic_issues": [],
            "info_gaps": ["证据来源数量不足", "观点覆盖不足"],
        }

    if len(draft.strip()) < MIN_DRAFT_LENGTH:
        return {
            "is_satisfactory": False,
            "needs_more_research": False,
            "critique_feedback": (
                f"{REVIEWER_SYSTEM_PROMPT} {build_reviewer_rule_hint()} "
                "草稿深度不足，请增强以下部分："
                "方法论细节、关键论据展开、结论可执行性与风险边界。"
            ),
            "confidence": 0.72,
            "review_mode": "rule",
            "supporter": {
                "strengths": ["已有基础结构与主题相关性"],
                "supported_claims": ["草稿具备初步结论框架"],
            },
            "skeptic": {
                "critical_issues": ["论证展开深度不足"],
                "missing_evidence": ["关键论据缺乏细节展开"],
            },
            "judge": {
                "decision": "需重写报告",
                "rationale": "信息量不足但不必重新检索，优先改写论证。",
            },
            "controversy_points": ["现有文本能否支撑可执行建议"],
            "evidence_verdicts": [
                {
                    "claim": "结论可执行性充分",
                    "status": "weak",
                    "evidence": "缺少实施步骤与风险边界",
                    "action": "返回 Writer 扩写方法与建议",
                }
            ],
            "fact_issues": [],
            "logic_issues": ["论证展开不足", "结论可执行性不够明确"],
            "info_gaps": [],
        }

    return {
        "is_satisfactory": True,
        "needs_more_research": False,
        "critique_feedback": "草稿通过当前评审，可结束流程。",
        "confidence": 0.8,
        "review_mode": "rule",
        "supporter": {
            "strengths": ["证据与结论匹配度可接受", "结构完整"],
            "supported_claims": ["可进入交付阶段"],
        },
        "skeptic": {
            "critical_issues": [],
            "missing_evidence": [],
        },
        "judge": {
            "decision": "通过",
            "rationale": "关键审查项达到当前阈值。",
        },
        "controversy_points": [],
        "evidence_verdicts": [
            {
                "claim": "报告达到可交付标准",
                "status": "supported",
                "evidence": "规则评审通过，未发现关键缺口",
                "action": "结束流程",
            }
        ],
        "fact_issues": [],
        "logic_issues": [],
        "info_gaps": [],
    }


def _build_revision_directives(review: Dict[str, Any], next_route: str) -> Dict[str, Any]:
    """将评审结果转换为可执行修订包，供 Writer/Researcher 下一轮消费。"""

    fact_issues = _to_issue_list(review.get("fact_issues", []))
    logic_issues = _to_issue_list(review.get("logic_issues", []))
    info_gaps = _to_issue_list(review.get("info_gaps", []))
    controversy_points = _to_issue_list(review.get("controversy_points", []))
    evidence_verdicts = _to_dict_list(review.get("evidence_verdicts", []))

    if next_route == ROUTE_RESEARCHER:
        route_reason = "当前证据不足，优先补齐检索材料后再写作。"
        focus_areas = ["补充高质量来源", "覆盖相反观点", "补齐评测细节"]
        must_fix = info_gaps or ["新增可验证来源并补充关键证据"]
    elif next_route == ROUTE_WRITER:
        route_reason = "证据基本可用，但草稿存在事实或逻辑问题，需定向改写。"
        focus_areas = ["修复事实引用", "补全论证链", "提升结论可执行性"]
        must_fix = fact_issues + logic_issues
        if not must_fix and controversy_points:
            must_fix = controversy_points
        if not must_fix and evidence_verdicts:
            must_fix = [
                str(item.get("action", "修复争议点"))
                for item in evidence_verdicts[:4]
                if str(item.get("action", "")).strip()
            ]
        if not must_fix:
            must_fix = ["逐条响应 critique_feedback 并修订对应段落"]
    else:
        route_reason = "评审通过。"
        focus_areas = []
        must_fix = []

    return {
        "next_route": next_route,
        "route_reason": route_reason,
        "must_fix": must_fix[:8],
        "focus_areas": focus_areas,
        "fact_issues": fact_issues,
        "logic_issues": logic_issues,
        "info_gaps": info_gaps,
        "controversy_points": controversy_points,
        "evidence_verdicts": evidence_verdicts,
    }


def _llm_review(topic: str, draft: str, context_count: int) -> Dict[str, Any]:
    """调用 DeepSeek 输出结构化评审 JSON。"""

    if ChatOpenAI is None:
        raise RuntimeError("langchain_openai 未安装")

    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not deepseek_api_key:
        raise RuntimeError("未检测到 DEEPSEEK_API_KEY")

    deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    llm = ChatOpenAI(
        model=deepseek_model,
        api_key=lambda: deepseek_api_key,
        base_url=deepseek_base_url,
        temperature=0.0,
    )
    response = llm.invoke(
        [
            ("system", REVIEWER_SYSTEM_PROMPT),
            ("system", build_reviewer_rule_hint()),
            ("human", build_reviewer_user_prompt(topic, draft, context_count)),
        ]
    )

    content_obj = getattr(response, "content", "")
    if isinstance(content_obj, list):
        # 兼容部分模型 SDK 返回分块内容结构
        content = "\n".join(
            [
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in content_obj
            ]
        ).strip()
    else:
        content = str(content_obj or "")

    parsed = _parse_review_json(content)
    parsed_dict = parsed if isinstance(parsed, dict) else {}
    supporter = parsed_dict.get("supporter", {})
    skeptic = parsed_dict.get("skeptic", {})
    judge = parsed_dict.get("judge", {})

    controversy_points = _to_issue_list(judge.get("controversy_points", parsed_dict.get("controversy_points", [])))
    evidence_verdicts = _to_dict_list(judge.get("evidence_verdicts", parsed_dict.get("evidence_verdicts", [])))
    fact_issues = _to_issue_list(judge.get("fact_issues", parsed_dict.get("fact_issues", [])))
    logic_issues = _to_issue_list(judge.get("logic_issues", parsed_dict.get("logic_issues", [])))
    info_gaps = _to_issue_list(judge.get("info_gaps", parsed_dict.get("info_gaps", [])))

    # 兼容模型未按三层返回时，回退到顶层字段。
    if not judge and parsed_dict:
        judge = parsed

    return {
        "is_satisfactory": bool(judge.get("is_satisfactory", parsed_dict.get("is_satisfactory", False))),
        "needs_more_research": bool(judge.get("needs_more_research", parsed_dict.get("needs_more_research", False))),
        "critique_feedback": str(judge.get("critique_feedback", parsed_dict.get("critique_feedback", "请给出更具体的修订建议。"))),
        "confidence": float(judge.get("confidence", parsed_dict.get("confidence", 0.5))),
        "review_mode": "llm",
        "supporter": {
            "strengths": _to_issue_list(supporter.get("strengths", []) if isinstance(supporter, dict) else []),
            "supported_claims": _to_issue_list(supporter.get("supported_claims", []) if isinstance(supporter, dict) else []),
        },
        "skeptic": {
            "critical_issues": _to_issue_list(skeptic.get("critical_issues", []) if isinstance(skeptic, dict) else []),
            "missing_evidence": _to_issue_list(skeptic.get("missing_evidence", []) if isinstance(skeptic, dict) else []),
        },
        "judge": {
            "decision": str(judge.get("decision", "")) if isinstance(judge, dict) else "",
            "rationale": str(judge.get("rationale", "")) if isinstance(judge, dict) else "",
        },
        "controversy_points": controversy_points,
        "evidence_verdicts": evidence_verdicts,
        "fact_issues": fact_issues,
        "logic_issues": logic_issues,
        "info_gaps": info_gaps,
    }


def reviewer_node(state: ResearchState) -> Dict[str, Any]:
    """评审代理节点。

    审查维度（规则化占位版本）：
    1) 信息充分性（上下文数量）
    2) 草稿完整性（长度与结构）
    3) 迭代控制（递增 revision_step）
    """

    revision_step = state.get("revision_step", 0) + 1
    retrieved_context = list(state.get("retrieved_context", []))
    draft = state.get("draft", "")

    errors = list(state.get("errors", []))

    if not isinstance(draft, str):
        errors = _append_error(errors, "draft 字段类型异常，Reviewer 已按空文本处理。")
        draft = ""

    try:
        review = _llm_review(
            topic=state.get("topic", ""),
            draft=draft,
            context_count=len(retrieved_context),
        )
    except Exception as exc:
        errors = _append_error(errors, f"Reviewer LLM 评审失败，已回退规则评审: {exc}")
        review = _rule_based_review(draft=draft, retrieved_context=retrieved_context)

    is_satisfactory = bool(review.get("is_satisfactory", False))
    info_gaps = _to_issue_list(review.get("info_gaps", []))
    needs_more_research = bool(review.get("needs_more_research", False) or info_gaps)

    if is_satisfactory:
        next_route = ROUTE_END
    else:
        next_route = ROUTE_RESEARCHER if needs_more_research else ROUTE_WRITER

    revision_directives = _build_revision_directives(review=review, next_route=next_route)

    trace = list(state.get("execution_trace", []))
    trace.append(
        {
            "node": "reviewer",
            "revision_step": revision_step,
            "mode": review.get("review_mode", "rule"),
            "is_satisfactory": is_satisfactory,
            "next_route": next_route,
            "confidence": review.get("confidence", 0.0),
            "fact_issues": len(review.get("fact_issues", [])),
            "logic_issues": len(review.get("logic_issues", [])),
            "info_gaps": len(review.get("info_gaps", [])),
            "controversies": len(review.get("controversy_points", [])),
            "route_reason": revision_directives.get("route_reason", ""),
        }
    )

    history = list(state.get("iteration_history", []))
    feedback_mapping = list(state.get("feedback_paragraph_mapping", []))
    source_quality_summary = dict(state.get("source_quality_summary", {}) or {})
    history.append(
        {
            "round": revision_step,
            "draft": draft,
            "review": review,
            "next_route": next_route,
            "is_satisfactory": is_satisfactory,
            "feedback_paragraph_mapping": feedback_mapping,
            "source_quality_summary": source_quality_summary,
        }
    )

    result: Dict[str, Any] = {
        "revision_step": revision_step,
        "is_satisfactory": is_satisfactory,
        "needs_more_research": needs_more_research,
        "next_route": next_route,
        "critique_feedback": review.get("critique_feedback", ""),
        "review_result": review,
        "revision_directives": revision_directives,
        "errors": errors,
        "execution_trace": trace,
        "iteration_history": history,
    }
    if is_satisfactory:
        result["final_report"] = draft
    return result

