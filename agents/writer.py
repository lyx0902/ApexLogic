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


def writer_node(state: ResearchState) -> Dict[str, Any]:
    """主笔代理节点。

    输入: topic / retrieved_context / critique_feedback / revision_step
    输出: draft / errors
    """

    topic = state.get("topic", "")
    revision_step = state.get("revision_step", 0)
    critique_feedback = state.get("critique_feedback", "")
    retrieved_context = list(state.get("retrieved_context", []))

    errors = list(state.get("errors", []))

    if ChatOpenAI is None:
        errors = _append_error(errors, "langchain_openai 未安装，Writer 使用本地占位草稿。")
        draft = _fallback_draft(
            topic,
            revision_step,
            critique_feedback,
            retrieved_context,
        )
        trace = list(state.get("execution_trace", []))
        trace.append(
            {
                "node": "writer",
                "revision_step": revision_step,
                "draft_len": len(draft),
                "mode": "fallback_no_langchain_openai",
            }
        )
        return {
            "draft": draft,
            "errors": errors,
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
        trace = list(state.get("execution_trace", []))
        trace.append(
            {
                "node": "writer",
                "revision_step": revision_step,
                "draft_len": len(draft),
                "mode": "fallback_no_api_key",
            }
        )
        return {
            "draft": draft,
            "errors": errors,
            "execution_trace": trace,
        }

    user_prompt = build_writer_user_prompt(
        topic=topic,
        revision_step=revision_step,
        critique_feedback=critique_feedback,
        retrieved_context=retrieved_context,
    )

    try:
        llm = ChatOpenAI(
            model=deepseek_model,
            api_key=deepseek_api_key,
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
        )
        trace = list(state.get("execution_trace", []))
        trace.append(
            {
                "node": "writer",
                "revision_step": revision_step,
                "draft_len": len(draft),
                "mode": "llm",
            }
        )
        return {
            "draft": draft,
            "errors": errors,
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
        trace = list(state.get("execution_trace", []))
        trace.append(
            {
                "node": "writer",
                "revision_step": revision_step,
                "draft_len": len(draft),
                "mode": "fallback_exception",
            }
        )
        return {
            "draft": draft,
            "errors": errors,
            "execution_trace": trace,
        }

