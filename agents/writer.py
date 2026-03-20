from __future__ import annotations

import os
from typing import Any, Dict, List

try:
    from core.state import ResearchState
except ModuleNotFoundError:  # 兼容直接脚本方式
    from state import ResearchState  # type: ignore

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None

from prompts.system_prompts import WRITER_SYSTEM_PROMPT, build_writer_user_prompt


def _append_error(errors: List[str], message: str) -> List[str]:
    """将错误信息追加到 errors，避免覆盖既有日志。"""

    updated = list(errors)
    updated.append(message)
    return updated


def _fallback_draft(
    topic: str,
    revision_step: int,
    critique_feedback: str,
    retrieved_context: List[Dict[str, Any] | str],
) -> str:
    """在无法调用 LLM 时生成可用的占位草稿。"""

    return (
        f"# {topic} 深度研究草稿\n\n"
        f"- 迭代轮次: {revision_step}\n"
        f"- 上下文数量: {len(retrieved_context)}\n"
        f"- 修订依据: {critique_feedback or '首轮生成'}\n\n"
        "## 一、问题定义\n"
        "待补充：请在接入 LLM 后生成完整问题背景与边界。\n\n"
        "## 二、关键发现\n"
        "待补充：请基于检索上下文形成有引用的结构化结论。\n\n"
        "## 三、结论与建议\n"
        "待补充：请输出可执行建议与风险提示。"
    )


def _extract_section_blocks(draft: str) -> List[Dict[str, str]]:
    """按 Markdown 二级标题切分，便于构建反馈-段落映射。"""

    lines = draft.splitlines()
    sections: List[Dict[str, str]] = []
    current_title = "引言"
    buffer: List[str] = []

    for line in lines:
        if line.startswith("## "):
            if buffer:
                sections.append({"section": current_title, "content": "\n".join(buffer).strip()})
                buffer = []
            current_title = line[3:].strip()
            continue
        buffer.append(line)

    if buffer:
        sections.append({"section": current_title, "content": "\n".join(buffer).strip()})

    return sections


def _build_feedback_mapping(draft: str, revision_directives: Dict[str, Any]) -> List[Dict[str, Any]]:
    """将 must_fix 项映射到草稿章节，便于 debug 报告展示可追踪修订。"""

    must_fix = revision_directives.get("must_fix", [])
    if not isinstance(must_fix, list) or not must_fix:
        return []

    sections = _extract_section_blocks(draft)
    if not sections:
        sections = [{"section": "正文", "content": draft}]

    mapping: List[Dict[str, Any]] = []
    for idx, issue in enumerate(must_fix, start=1):
        picked = sections[(idx - 1) % len(sections)]
        snippet = picked.get("content", "")[:220].replace("\n", " ")
        mapping.append(
            {
                "issue": str(issue),
                "mapped_section": picked.get("section", "正文"),
                "evidence_snippet": snippet,
            }
        )
    return mapping


def _get_previous_draft(state: Any) -> str:
    """从迭代历史中取最近一轮草稿，供修订时参考。"""
    history = list(state.get("iteration_history", []) or [])
    if not history:
        return ""
    last = history[-1]
    return str(last.get("draft", "") or "")


def _generate_draft_with_llm(
    llm: Any,
    topic: str,
    revision_step: int,
    critique_feedback: str,
    retrieved_context: List[Dict[str, Any] | str],
    revision_directives: Dict[str, Any],
    previous_draft: str = "",
) -> str:
    """调用 LLM 生成整篇草稿。"""

    user_prompt = build_writer_user_prompt(
        topic=topic,
        revision_step=revision_step,
        critique_feedback=critique_feedback,
        retrieved_context=retrieved_context,
        revision_directives=revision_directives,
        previous_draft=previous_draft,
    )

    response = llm.invoke(
        [
            ("system", WRITER_SYSTEM_PROMPT),
            ("human", user_prompt),
        ]
    )
    return (getattr(response, "content", "") or "").strip()


def writer_node(state: ResearchState) -> Dict[str, Any]:
    """主笔代理节点。

    输入: topic / retrieved_context / critique_feedback / revision_step
    输出: draft / errors
    """

    topic = state.get("topic", "")
    revision_step = state.get("revision_step", 0)
    critique_feedback = state.get("critique_feedback", "")
    retrieved_context = list(state.get("retrieved_context", []))
    revision_directives = dict(state.get("revision_directives", {}) or {})

    errors = list(state.get("errors", []))

    if ChatOpenAI is None:
        errors = _append_error(errors, "langchain_openai 未安装，Writer 使用本地占位草稿。")
        draft = _fallback_draft(
            topic,
            revision_step,
            critique_feedback,
            retrieved_context,
        )
        mapping = _build_feedback_mapping(draft, revision_directives)
        trace = list(state.get("execution_trace", []))
        trace.append(
            {
                "node": "writer",
                "revision_step": revision_step,
                "draft_len": len(draft),
                "mode": "fallback_no_langchain_openai",
                "mapped_items": len(mapping),
            }
        )
        return {
            "draft": draft,
            "errors": errors,
            "feedback_paragraph_mapping": mapping,
            "execution_trace": trace,
        }

    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    if not deepseek_api_key:
        errors = _append_error(errors, "未检测到 DEEPSEEK_API_KEY，Writer 使用本地占位草稿。")
        draft = _fallback_draft(
            topic,
            revision_step,
            critique_feedback,
            retrieved_context,
        )
        mapping = _build_feedback_mapping(draft, revision_directives)
        trace = list(state.get("execution_trace", []))
        trace.append(
            {
                "node": "writer",
                "revision_step": revision_step,
                "draft_len": len(draft),
                "mode": "fallback_no_api_key",
                "mapped_items": len(mapping),
            }
        )
        return {
            "draft": draft,
            "errors": errors,
            "feedback_paragraph_mapping": mapping,
            "execution_trace": trace,
        }

    try:
        llm = ChatOpenAI(
            model=deepseek_model,
            api_key=lambda: deepseek_api_key,
            base_url=deepseek_base_url,
            temperature=0.3,
        )

        draft = _generate_draft_with_llm(
            llm=llm,
            topic=topic,
            revision_step=revision_step,
            critique_feedback=critique_feedback,
            retrieved_context=retrieved_context,
            revision_directives=revision_directives,
            previous_draft=_get_previous_draft(state),
        )
        if not draft:
            draft = _fallback_draft(topic, revision_step, critique_feedback, retrieved_context)

        mapping = _build_feedback_mapping(draft, revision_directives)
        trace = list(state.get("execution_trace", []))
        trace.append(
            {
                "node": "writer",
                "revision_step": revision_step,
                "draft_len": len(draft),
                "mode": "llm",
                "mapped_items": len(mapping),
            }
        )
        return {
            "draft": draft,
            "errors": errors,
            "feedback_paragraph_mapping": mapping,
            "execution_trace": trace,
        }
    except Exception as exc:
        errors = _append_error(errors, f"DeepSeek 生成失败，Writer 已降级: {exc}")
        draft = _fallback_draft(
            topic,
            revision_step,
            critique_feedback,
            retrieved_context,
        )
        mapping = _build_feedback_mapping(draft, revision_directives)
        trace = list(state.get("execution_trace", []))
        trace.append(
            {
                "node": "writer",
                "revision_step": revision_step,
                "draft_len": len(draft),
                "mode": "fallback_exception",
                "mapped_items": len(mapping),
            }
        )
        return {
            "draft": draft,
            "errors": errors,
            "feedback_paragraph_mapping": mapping,
            "execution_trace": trace,
        }

