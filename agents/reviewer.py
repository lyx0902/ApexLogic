from __future__ import annotations

import json
import os
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
        "fact_issues": [],
        "logic_issues": [],
        "info_gaps": [],
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
        api_key=deepseek_api_key,
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

    content = _extract_json_content(getattr(response, "content", "") or "")
    parsed = json.loads(content)
    return {
        "is_satisfactory": bool(parsed.get("is_satisfactory", False)),
        "needs_more_research": bool(parsed.get("needs_more_research", False)),
        "critique_feedback": str(parsed.get("critique_feedback", "请给出更具体的修订建议。")),
        "confidence": float(parsed.get("confidence", 0.5)),
        "review_mode": "llm",
        "fact_issues": _to_issue_list(parsed.get("fact_issues", [])),
        "logic_issues": _to_issue_list(parsed.get("logic_issues", [])),
        "info_gaps": _to_issue_list(parsed.get("info_gaps", [])),
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
    needs_more_research = bool(review.get("needs_more_research", False))

    if is_satisfactory:
        next_route = ROUTE_END
    else:
        next_route = ROUTE_RESEARCHER if needs_more_research else ROUTE_WRITER

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
        }
    )

    result: Dict[str, Any] = {
        "revision_step": revision_step,
        "is_satisfactory": is_satisfactory,
        "needs_more_research": needs_more_research,
        "next_route": next_route,
        "critique_feedback": review.get("critique_feedback", ""),
        "review_result": review,
        "errors": errors,
        "execution_trace": trace,
    }
    if is_satisfactory:
        result["final_report"] = draft
    return result

