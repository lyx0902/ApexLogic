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

from prompts.system_prompts import (
    RESEARCHER_SYSTEM_PROMPT,
    build_researcher_user_prompt,
)
from tools.arxiv_tool import arxiv_search
from tools.search_tool import tavily_search


def _append_error(errors: List[str], message: str) -> List[str]:
    """将错误信息追加到 errors，避免覆盖既有日志。"""

    updated = list(errors)
    updated.append(message)
    return updated


def _build_queries(topic: str, critique_feedback: str) -> List[str]:
    """构建检索词：首轮按主题展开，迭代轮次融合评审反馈。"""

    queries = [
        f"{topic} 最新研究进展",
        f"{topic} 关键技术挑战",
        f"{topic} 产业应用与案例",
    ]

    if critique_feedback:
        queries.append(f"{topic} 针对问题补充: {critique_feedback[:80]}")

    return queries


def _rewrite_queries_with_llm(
    topic: str,
    critique_feedback: str,
    seed_queries: List[str],
) -> List[str]:
    """可选使用 DeepSeek 重写检索词，提升针对性。"""

    if ChatOpenAI is None:
        return seed_queries

    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not deepseek_api_key:
        return seed_queries

    deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    llm = ChatOpenAI(
        model=deepseek_model,
        api_key=deepseek_api_key,
        base_url=deepseek_base_url,
        temperature=0.2,
    )
    user_prompt = build_researcher_user_prompt(topic, critique_feedback)
    response = llm.invoke(
        [
            ("system", RESEARCHER_SYSTEM_PROMPT),
            ("human", user_prompt),
        ]
    )

    content = (getattr(response, "content", "") or "").strip()
    if not content:
        return seed_queries

    rewritten: List[str] = []
    for line in content.splitlines():
        item = line.strip().lstrip("-0123456789. ")
        if item:
            rewritten.append(item)

    merged = rewritten + seed_queries
    unique_queries: List[str] = []
    for query in merged:
        if query not in unique_queries:
            unique_queries.append(query)
    return unique_queries[:6]


def _normalize_context_item(item: Dict[str, Any] | str) -> Dict[str, Any]:
    """统一上下文字段，便于后续去重与引用。"""

    if isinstance(item, dict):
        return {
            "title": str(item.get("title", "")),
            "url": str(item.get("url", "")),
            "source": str(item.get("source", "unknown")),
            "content": str(item.get("content", "")),
        }
    return {
        "title": "text_context",
        "url": "",
        "source": "legacy",
        "content": str(item),
    }


def _dedupe_and_index_contexts(contexts: List[Dict[str, Any] | str]) -> List[Dict[str, Any]]:
    """按 url 或 title+source 去重，并生成连续 citation_id。"""

    unique: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for raw in contexts:
        item = _normalize_context_item(raw)
        url_key = item.get("url", "").strip().lower()
        fallback_key = f"{item.get('source','')}|{item.get('title','').strip().lower()}"
        key = url_key or fallback_key
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)

    for idx, item in enumerate(unique, start=1):
        item["citation_id"] = f"S{idx}"

    return unique


def researcher_node(state: ResearchState) -> Dict[str, Any]:
    """检索代理节点。

    输入: topic / critique_feedback
    输出: search_queries / retrieved_context / errors
    """

    topic = state.get("topic", "")
    critique_feedback = state.get("critique_feedback", "")
    errors = list(state.get("errors", []))

    seed_queries = _build_queries(topic=topic, critique_feedback=critique_feedback)
    queries = list(seed_queries)

    try:
        queries = _rewrite_queries_with_llm(topic, critique_feedback, seed_queries)
    except Exception as exc:
        errors = _append_error(errors, f"DeepSeek 查询重写失败，已使用规则检索词: {exc}")

    contexts: List[Dict[str, Any] | str] = list(state.get("retrieved_context", []))

    for query in queries[:4]:
        try:
            contexts.extend(tavily_search(query, max_results=2))
        except Exception as exc:
            errors = _append_error(errors, f"Tavily 检索失败: {exc}")

    # 使用 ArXiv 作为学术补充来源，降低信息偏差。
    for query in queries[:2]:
        try:
            contexts.extend(arxiv_search(query, max_results=2))
        except Exception as exc:
            errors = _append_error(errors, f"ArXiv 检索失败: {exc}")

    # 确保在无外部依赖时流程仍然有上下文可用
    if not contexts:
        contexts.append(
            {
                "title": f"{topic} 占位检索结果",
                "url": "",
                "source": "fallback",
                "content": "未连接 Tavily 时生成的占位内容，用于打通工作流。",
            }
        )

    normalized_contexts = _dedupe_and_index_contexts(contexts)
    trace = list(state.get("execution_trace", []))
    trace.append(
        {
            "node": "researcher",
            "revision_step": state.get("revision_step", 0),
            "queries": len(queries),
            "contexts": len(normalized_contexts),
            "errors": len(errors),
        }
    )

    return {
        "search_queries": queries,
        "retrieved_context": normalized_contexts,
        "errors": errors,
        "execution_trace": trace,
    }

