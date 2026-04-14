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

# 四维权重
SCORE_WEIGHTS = {"S1": 0.35, "S2": 0.25, "S3": 0.25, "S4": 0.15}
# 强制回 Researcher 的单维阈值
RESEARCHER_S1_THRESHOLD = 5
RESEARCHER_S3_THRESHOLD = 4


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


def _escape_json_string_literals(text: str) -> str:
    """将 JSON 字符串值内的字面控制字符（换行、制表等）转义，修复最常见的解析失败原因。"""

    result: List[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if escaped:
            result.append(ch)
            escaped = False
        elif ch == "\\":
            result.append(ch)
            escaped = True
        elif ch == '"':
            result.append(ch)
            in_string = not in_string
        elif in_string and ch == "\n":
            result.append("\\n")
        elif in_string and ch == "\r":
            result.append("\\r")
        elif in_string and ch == "\t":
            result.append("\\t")
        else:
            result.append(ch)
    return "".join(result)


def _sanitize_json_text(text: str) -> str:
    """轻量修复常见 JSON 格式问题（尾逗号、BOM、字符串内控制字符）。"""

    cleaned = text.strip().lstrip("\ufeff")
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    cleaned = _escape_json_string_literals(cleaned)
    return cleaned


def _coerce_text_review_to_json(raw: str) -> Dict[str, Any] | None:
    """当模型返回非 JSON 文本时，做最小语义兜底，避免整轮降级。"""

    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None

    lowered = text.lower()
    pass_markers = ["[pass]", "评审通过", "通过当前评审", "can be finalized", "is_satisfactory.*true"]
    is_pass = any(marker in lowered for marker in pass_markers[:4]) or bool(
        re.search(r"is_satisfactory.*true", lowered)
    )

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

    # 兜底时给保守的低分，不让报告轻易通过
    if is_pass:
        scores = {"S1": 7, "S2": 7, "S3": 7, "S4": 7}
        weighted = 7.0
    else:
        scores = {"S1": 4, "S2": 5, "S3": 4, "S4": 5}
        weighted = 0.35 * 4 + 0.25 * 5 + 0.25 * 4 + 0.15 * 5

    return {
        "scores": scores,
        "weighted_score": round(weighted, 2),
        "is_satisfactory": is_pass,
        "needs_more_research": needs_more_research if not is_pass else False,
        "critique_feedback": "草稿通过当前评审，可结束流程。" if is_pass else text[:1200],
        "citation_checks": [],
        "fact_issues": fact_issues,
        "logic_issues": logic_issues,
        "info_gaps": info_gaps,
        "score_rationale": {"S1": "兜底解析", "S2": "兜底解析", "S3": "兜底解析", "S4": "兜底解析"},
        "supporter": {"strengths": ["审稿文本判定为通过"] if is_pass else [], "supported_claims": []},
        "skeptic": {"critical_issues": fact_issues + logic_issues, "missing_evidence": info_gaps},
        "controversy_points": [],
        "evidence_verdicts": [],
        "review_mode": "coerce_fallback",
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


def _compute_weighted_score(scores: Dict[str, Any]) -> float:
    """根据四维分数计算加权总分。"""
    total = 0.0
    for dim, weight in SCORE_WEIGHTS.items():
        total += float(scores.get(dim, 0)) * weight
    return round(total, 4)


def _rule_based_review(
    draft: str,
    retrieved_context: List[Dict[str, Any] | str],
) -> Dict[str, Any]:
    """规则化回退评审，使用与 LLM 模式相同的评分结构。"""

    n_ctx = len(retrieved_context)
    draft_len = len(draft.strip())

    if n_ctx < MIN_CONTEXT_ITEMS:
        scores = {"S1": 2, "S2": 5, "S3": 2, "S4": 4}
        weighted = _compute_weighted_score(scores)
        return {
            "scores": scores,
            "weighted_score": weighted,
            "is_satisfactory": False,
            "needs_more_research": True,
            "critique_feedback": "当前证据不足，请补充更多高质量来源并覆盖不同观点。",
            "citation_checks": [],
            "fact_issues": [],
            "logic_issues": [],
            "info_gaps": ["证据来源数量不足", "观点覆盖不足"],
            "score_rationale": {
                "S1": f"仅有 {n_ctx} 条来源，无法有效核查事实",
                "S2": "结构待评估",
                "S3": f"来源数量 {n_ctx} 低于阈值，覆盖面存疑",
                "S4": "待评估",
            },
            "supporter": {"strengths": ["主题聚焦明确"], "supported_claims": []},
            "skeptic": {
                "critical_issues": ["证据来源数量不足，无法支撑关键结论"],
                "missing_evidence": ["需要补充多来源检索结果"],
            },
            "controversy_points": ["当前结论是否建立在足够证据之上"],
            "evidence_verdicts": [{
                "claim": "已有证据可支撑完整报告",
                "status": "unsupported",
                "evidence": f"上下文数量 {n_ctx} 低于最低阈值 {MIN_CONTEXT_ITEMS}",
                "action": "返回 Researcher 补充检索",
            }],
            "review_mode": "rule",
        }

    if draft_len < MIN_DRAFT_LENGTH:
        scores = {"S1": 5, "S2": 3, "S3": 4, "S4": 2}
        weighted = _compute_weighted_score(scores)
        return {
            "scores": scores,
            "weighted_score": weighted,
            "is_satisfactory": False,
            "needs_more_research": False,
            "critique_feedback": (
                "草稿深度不足，请增强以下部分："
                "方法论细节、关键论据展开、结论可执行性与风险边界。"
            ),
            "citation_checks": [],
            "fact_issues": [],
            "logic_issues": ["论证展开深度不足", "结论可执行性不够明确"],
            "info_gaps": [],
            "score_rationale": {
                "S1": "草稿过短，事实核查覆盖有限",
                "S2": f"草稿仅 {draft_len} 字，结构不完整",
                "S3": "内容过少，覆盖广度无法评估",
                "S4": "建议部分缺失或过于简略",
            },
            "supporter": {
                "strengths": ["已有基础结构与主题相关性"],
                "supported_claims": ["草稿具备初步结论框架"],
            },
            "skeptic": {
                "critical_issues": ["论证展开深度不足"],
                "missing_evidence": ["关键论据缺乏细节展开"],
            },
            "controversy_points": ["现有文本能否支撑可执行建议"],
            "evidence_verdicts": [{
                "claim": "结论可执行性充分",
                "status": "weak",
                "evidence": f"草稿仅 {draft_len} 字，缺少实施步骤与风险边界",
                "action": "返回 Writer 扩写方法与建议",
            }],
            "review_mode": "rule",
        }

    # 规则通过：给出保守的及格分
    scores = {"S1": 7, "S2": 7, "S3": 7, "S4": 7}
    weighted = _compute_weighted_score(scores)
    return {
        "scores": scores,
        "weighted_score": weighted,
        "is_satisfactory": True,
        "needs_more_research": False,
        "critique_feedback": "草稿通过规则评审，可结束流程。",
        "citation_checks": [],
        "fact_issues": [],
        "logic_issues": [],
        "info_gaps": [],
        "score_rationale": {
            "S1": "来源数量及草稿长度满足基础要求",
            "S2": "结构基本完整",
            "S3": "覆盖面达到规则最低标准",
            "S4": "建议部分存在",
        },
        "supporter": {
            "strengths": ["证据与结论匹配度可接受", "结构完整"],
            "supported_claims": ["可进入交付阶段"],
        },
        "skeptic": {"critical_issues": [], "missing_evidence": []},
        "controversy_points": [],
        "evidence_verdicts": [{
            "claim": "报告达到可交付标准",
            "status": "supported",
            "evidence": "规则评审通过，未发现关键缺口",
            "action": "结束流程",
        }],
        "review_mode": "rule",
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


def _llm_review(
    topic: str,
    draft: str,
    retrieved_context: List[Dict[str, Any] | str],
    revision_step: int = 0,
) -> Dict[str, Any]:
    """调用 DeepSeek 输出四维量化评分 JSON，传入 top-10 原始来源供事实核查。"""

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
            ("human", build_reviewer_user_prompt(topic, draft, retrieved_context, revision_step=revision_step - 1)),
        ]
    )

    content_obj = getattr(response, "content", "")
    if isinstance(content_obj, list):
        content = "\n".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content_obj
        ).strip()
    else:
        content = str(content_obj or "")

    parsed = _parse_review_json(content)
    d = parsed if isinstance(parsed, dict) else {}

    # ── 提取四维分数 ────────────────────────────────────────────────
    raw_scores = d.get("scores", {})
    scores: Dict[str, int] = {}
    for dim in ("S1", "S2", "S3", "S4"):
        val = raw_scores.get(dim, 0)
        scores[dim] = max(0, min(10, int(val)))

    # 优先信任模型计算的加权分，同时用本地公式兜底验证
    local_weighted = _compute_weighted_score(scores)
    reported_weighted = d.get("weighted_score")
    if reported_weighted is not None:
        # 允许模型报告值与本地计算值有 ±0.5 误差，否则以本地计算为准
        weighted_score = float(reported_weighted)
        if abs(weighted_score - local_weighted) > 0.5:
            weighted_score = local_weighted
    else:
        weighted_score = local_weighted

    # ── 通过判定（以加权总分为准，忽略模型自报的布尔值） ──────────
    _pass_threshold = float(os.getenv("REVIEWER_PASS_THRESHOLD", "7.5"))
    is_satisfactory = weighted_score >= _pass_threshold

    # ── 路由判定 ────────────────────────────────────────────────────
    if is_satisfactory:
        needs_more_research = False
    else:
        # S1 < 5（事实严重问题）或 S3 < 4（覆盖严重不足）→ 回 Researcher
        needs_more_research = (
            scores.get("S1", 0) < RESEARCHER_S1_THRESHOLD
            or scores.get("S3", 0) < RESEARCHER_S3_THRESHOLD
        )
        # 允许模型也可以触发 needs_more_research（两者取并集）
        model_nmr = bool(d.get("needs_more_research", False))
        needs_more_research = needs_more_research or model_nmr

    # ── 提取其他字段 ────────────────────────────────────────────────
    citation_checks = d.get("citation_checks", [])
    if not isinstance(citation_checks, list):
        citation_checks = []

    fact_issues = _to_issue_list(d.get("fact_issues", []))
    logic_issues = _to_issue_list(d.get("logic_issues", []))
    info_gaps = _to_issue_list(d.get("info_gaps", []))
    controversy_points = _to_issue_list(d.get("controversy_points", []))
    evidence_verdicts = _to_dict_list(d.get("evidence_verdicts", []))
    score_rationale = d.get("score_rationale", {})
    if not isinstance(score_rationale, dict):
        score_rationale = {}

    supporter = d.get("supporter", {})
    skeptic = d.get("skeptic", {})

    return {
        "scores": scores,
        "weighted_score": round(weighted_score, 4),
        "is_satisfactory": is_satisfactory,
        "needs_more_research": needs_more_research,
        "critique_feedback": str(d.get("critique_feedback", "请给出更具体的修订建议。")),
        "citation_checks": citation_checks,
        "fact_issues": fact_issues,
        "logic_issues": logic_issues,
        "info_gaps": info_gaps,
        "score_rationale": score_rationale,
        "supporter": {
            "strengths": _to_issue_list(supporter.get("strengths", []) if isinstance(supporter, dict) else []),
            "supported_claims": _to_issue_list(supporter.get("supported_claims", []) if isinstance(supporter, dict) else []),
        },
        "skeptic": {
            "critical_issues": _to_issue_list(skeptic.get("critical_issues", []) if isinstance(skeptic, dict) else []),
            "missing_evidence": _to_issue_list(skeptic.get("missing_evidence", []) if isinstance(skeptic, dict) else []),
        },
        "controversy_points": controversy_points,
        "evidence_verdicts": evidence_verdicts,
        "review_mode": "llm_scored",
    }


def reviewer_node(state: ResearchState) -> Dict[str, Any]:
    """评审代理节点。

    审查维度：
    1) S1 事实准确性（对照 top-10 原始来源核查，权重 35%）
    2) S2 逻辑完整性（权重 25%）
    3) S3 信息覆盖广度（权重 25%）
    4) S4 结论可执行性（权重 15%）
    加权总分 ≥ PASS_THRESHOLD(默认7.0) 才通过。
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
            retrieved_context=retrieved_context,
            revision_step=revision_step,
        )
    except Exception as exc:
        errors = _append_error(errors, f"Reviewer LLM 评审失败，已回退规则评审: {exc}")
        review = _rule_based_review(draft=draft, retrieved_context=retrieved_context)

    is_satisfactory = bool(review.get("is_satisfactory", False))
    needs_more_research = bool(review.get("needs_more_research", False))

    if is_satisfactory:
        next_route = ROUTE_END
    else:
        next_route = ROUTE_RESEARCHER if needs_more_research else ROUTE_WRITER

    revision_directives = _build_revision_directives(review=review, next_route=next_route)

    scores = review.get("scores", {})
    weighted_score = review.get("weighted_score", 0.0)

    _pass_threshold = float(os.getenv("REVIEWER_PASS_THRESHOLD", "7.5"))
    trace = list(state.get("execution_trace", []))
    trace.append(
        {
            "node": "reviewer",
            "revision_step": revision_step,
            "mode": review.get("review_mode", "rule"),
            "scores": scores,
            "weighted_score": weighted_score,
            "pass_threshold": _pass_threshold,
            "is_satisfactory": is_satisfactory,
            "next_route": next_route,
            "fact_issues": len(review.get("fact_issues", [])),
            "logic_issues": len(review.get("logic_issues", [])),
            "info_gaps": len(review.get("info_gaps", [])),
            "citation_checks": len(review.get("citation_checks", [])),
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

