from __future__ import annotations

import os
from typing import Any, Dict, List
from urllib.parse import urlparse

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


def _build_queries(
    topic: str,
    critique_feedback: str,
    revision_directives: Dict[str, Any] | None = None,
) -> List[str]:
    """构建检索词：首轮按主题展开，迭代轮次融合评审反馈。"""

    queries = [
        f"{topic} 最新研究进展",
        f"{topic} 关键技术挑战",
        f"{topic} 产业应用与案例",
    ]

    if critique_feedback:
        queries.append(f"{topic} 针对问题补充: {critique_feedback[:80]}")

    directives = revision_directives or {}
    info_gaps = directives.get("info_gaps", [])
    must_fix = directives.get("must_fix", [])

    for gap in info_gaps[:2]:
        queries.append(f"{topic} 补充信息缺口: {str(gap)[:60]}")
    for item in must_fix[:2]:
        queries.append(f"{topic} 证据核查: {str(item)[:60]}")

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
        api_key=lambda: deepseek_api_key,
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


def _score_source_quality(item: Dict[str, Any]) -> Dict[str, Any]:
    """根据 source + domain 粗粒度评估来源质量。"""

    source = str(item.get("source", "")).lower().strip()
    url = str(item.get("url", "")).strip().lower()
    domain = urlparse(url).netloc if url else ""

    score = 0.5
    tier = "C"
    reason = "默认分层"

    # 论文与学术来源优先
    if source == "arxiv" or "arxiv.org" in domain:
        score = 0.95
        tier = "A"
        reason = "学术预印本"
    # 官方文档和标准机构
    elif any(x in domain for x in ["openai.com", "google.com", "deepmind.com", "anthropic.com", "ietf.org", "iso.org", "nist.gov", "learn.microsoft.com"]):
        score = 0.9
        tier = "A"
        reason = "官方或标准机构"
    # 主流技术媒体/行业报告
    elif any(x in domain for x in ["nature.com", "science.org", "ieee.org", "acm.org", "mckinsey.com", "gartner.com", "forrester.com"]):
        score = 0.82
        tier = "B"
        reason = "高可信行业媒体或机构"
    # 聚合搜索结果的基础可信度
    elif source == "tavily":
        score = 0.72
        tier = "B"
        reason = "聚合检索结果，需二次核验"
    # 博客/论坛/未知
    elif any(x in domain for x in ["github.com", "medium.com", "reddit.com", "zhihu.com", "csdn.net", "cnblogs.com"]):
        score = 0.62
        tier = "C"
        reason = "社区内容，观点价值高但事实需交叉验证"

    rated = dict(item)
    rated["quality_score"] = round(score, 2)
    rated["quality_tier"] = tier
    rated["quality_reason"] = reason
    rated["domain"] = domain
    return rated


def _build_quality_summary(contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """生成来源质量统计摘要。"""

    if not contexts:
        return {"avg_score": 0.0, "tier_counts": {"A": 0, "B": 0, "C": 0}}

    tier_counts: Dict[str, int] = {"A": 0, "B": 0, "C": 0}
    score_sum = 0.0
    for item in contexts:
        tier = str(item.get("quality_tier", "C"))
        if tier not in tier_counts:
            tier_counts[tier] = 0
        tier_counts[tier] += 1
        score_sum += float(item.get("quality_score", 0.0))

    avg_score = round(score_sum / len(contexts), 3)
    return {
        "avg_score": avg_score,
        "tier_counts": tier_counts,
        "total": len(contexts),
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
        unique.append(_score_source_quality(item))

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
    revision_directives = dict(state.get("revision_directives", {}) or {})
    errors = list(state.get("errors", []))

    seed_queries = _build_queries(
        topic=topic,
        critique_feedback=critique_feedback,
        revision_directives=revision_directives,
    )
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
    source_quality_summary = _build_quality_summary(normalized_contexts)
    trace = list(state.get("execution_trace", []))
    trace.append(
        {
            "node": "researcher",
            "revision_step": state.get("revision_step", 0),
            "queries": len(queries),
            "contexts": len(normalized_contexts),
            "quality_avg": source_quality_summary.get("avg_score", 0.0),
            "errors": len(errors),
        }
    )

    return {
        "search_queries": queries,
        "retrieved_context": normalized_contexts,
        "source_quality_summary": source_quality_summary,
        "errors": errors,
        "execution_trace": trace,
    }

