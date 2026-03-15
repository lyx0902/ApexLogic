from __future__ import annotations

import os
import re
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
    report_length: str,
) -> str:
    """在无法调用 LLM 时生成可用的占位草稿。"""

    return (
        f"# {topic} 深度研究草稿\n\n"
        f"- 迭代轮次: {revision_step}\n"
        f"- 上下文数量: {len(retrieved_context)}\n"
        f"- 目标篇幅: {report_length}\n"
        f"- 修订依据: {critique_feedback or '首轮生成'}\n\n"
        "## 一、问题定义\n"
        "待补充：请在接入 LLM 后生成完整问题背景与边界。\n\n"
        "## 二、关键发现\n"
        "待补充：请基于检索上下文形成有引用的结构化结论。\n\n"
        "## 三、结论与建议\n"
        "待补充：请输出可执行建议与风险提示。"
    )


def _strip_markdown(text: str) -> str:
    """粗略移除常见 Markdown 标记，估算净字数。"""

    clean = re.sub(r"```[\s\S]*?```", "", text)
    clean = re.sub(r"`[^`]*`", "", clean)
    clean = re.sub(r"[#>*\-]", "", clean)
    clean = re.sub(r"\[[^]]+]\([^)]+\)", "", clean)
    clean = re.sub(r"\s+", " ", clean)
    return clean.strip()


def _get_length_limits(report_length: str) -> tuple[int, int]:
    """返回 short/medium/long 的净字数上下限。"""

    if report_length == "short":
        return 800, 1200
    if report_length == "long":
        return 3200, 4500
    return 1800, 2600


def _enforce_length(draft: str, report_length: str) -> tuple[str, Dict[str, Any]]:
    """对草稿执行硬上限裁剪，并返回裁剪元数据。"""

    min_len, max_len = _get_length_limits(report_length)
    net_text = _strip_markdown(draft)
    net_len = len(net_text)

    meta = {
        "target": report_length,
        "min_len": min_len,
        "max_len": max_len,
        "raw_len": len(draft),
        "net_len": net_len,
        "trimmed": False,
    }

    if net_len <= max_len:
        return draft, meta

    # 以净文本长度近似裁剪 Markdown 原文，避免超长输出。
    ratio = max_len / max(net_len, 1)
    cut = max(int(len(draft) * ratio), 600)
    trimmed = draft[:cut].rstrip()
    if not trimmed.endswith(("。", "！", "？", ".", "!", "?")):
        trimmed += "\n\n（篇幅控制：已按目标上限裁剪）"

    meta["trimmed"] = True
    meta["trimmed_raw_len"] = len(trimmed)
    meta["trimmed_net_len"] = len(_strip_markdown(trimmed))
    return trimmed, meta


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


def writer_node(state: ResearchState) -> Dict[str, Any]:
    """主笔代理节点。

    输入: topic / retrieved_context / critique_feedback / revision_step
    输出: draft / errors
    """

    topic = state.get("topic", "")
    revision_step = state.get("revision_step", 0)
    critique_feedback = state.get("critique_feedback", "")
    retrieved_context = list(state.get("retrieved_context", []))
    report_length = str(state.get("report_length", "medium"))
    revision_directives = dict(state.get("revision_directives", {}) or {})

    errors = list(state.get("errors", []))

    if ChatOpenAI is None:
        errors = _append_error(errors, "langchain_openai 未安装，Writer 使用本地占位草稿。")
        draft = _fallback_draft(
            topic,
            revision_step,
            critique_feedback,
            retrieved_context,
            report_length,
        )
        draft, length_meta = _enforce_length(draft, report_length)
        mapping = _build_feedback_mapping(draft, revision_directives)
        trace = list(state.get("execution_trace", []))
        trace.append(
            {
                "node": "writer",
                "revision_step": revision_step,
                "draft_len": len(draft),
                "mode": "fallback_no_langchain_openai",
                "trimmed": length_meta.get("trimmed", False),
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
            report_length,
        )
        draft, length_meta = _enforce_length(draft, report_length)
        mapping = _build_feedback_mapping(draft, revision_directives)
        trace = list(state.get("execution_trace", []))
        trace.append(
            {
                "node": "writer",
                "revision_step": revision_step,
                "draft_len": len(draft),
                "mode": "fallback_no_api_key",
                "trimmed": length_meta.get("trimmed", False),
                "mapped_items": len(mapping),
            }
        )
        return {
            "draft": draft,
            "errors": errors,
            "feedback_paragraph_mapping": mapping,
            "execution_trace": trace,
        }

    user_prompt = build_writer_user_prompt(
        topic=topic,
        revision_step=revision_step,
        critique_feedback=critique_feedback,
        retrieved_context=retrieved_context,
        report_length=report_length if report_length in {"short", "medium", "long"} else "medium",
        revision_directives=revision_directives,
    )

    try:
        llm = ChatOpenAI(
            model=deepseek_model,
            api_key=lambda: deepseek_api_key,
            base_url=deepseek_base_url,
            temperature=0.3,
        )
        response = llm.invoke(
            [
                ("system", WRITER_SYSTEM_PROMPT),
                ("human", user_prompt),
            ]
        )
        draft = getattr(response, "content", "") or _fallback_draft(
            topic,
            revision_step,
            critique_feedback,
            retrieved_context,
            report_length,
        )
        draft, length_meta = _enforce_length(draft, report_length)
        mapping = _build_feedback_mapping(draft, revision_directives)
        trace = list(state.get("execution_trace", []))
        trace.append(
            {
                "node": "writer",
                "revision_step": revision_step,
                "draft_len": len(draft),
                "mode": "llm",
                "trimmed": length_meta.get("trimmed", False),
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
            report_length,
        )
        draft, length_meta = _enforce_length(draft, report_length)
        mapping = _build_feedback_mapping(draft, revision_directives)
        trace = list(state.get("execution_trace", []))
        trace.append(
            {
                "node": "writer",
                "revision_step": revision_step,
                "draft_len": len(draft),
                "mode": "fallback_exception",
                "trimmed": length_meta.get("trimmed", False),
                "mapped_items": len(mapping),
            }
        )
        return {
            "draft": draft,
            "errors": errors,
            "feedback_paragraph_mapping": mapping,
            "execution_trace": trace,
        }

