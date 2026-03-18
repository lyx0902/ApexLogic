from __future__ import annotations

import os
import re
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
from tools.search_tool import duckduckgo_search, tavily_search


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


def _extract_core_summary(title: str, content: str) -> str:
    """从长内容中提炼核心信息，避免简单首句截断。"""

    text = (content or "").strip()
    # 去除常见 Markdown/HTML 噪声，降低导航栏与页脚干扰。
    text = re.sub(r"!\[[^]]*]\([^)]*\)", " ", text)
    text = re.sub(r"\[[^]]+]\([^)]*\)", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\|", " ", text)
    text = re.sub(r"\s+", " ", text)

    noise_tokens = [
        "首页",
        "退出",
        "充值",
        "收藏夹",
        "机构管理",
        "切换用户",
        "关注官方微信",
        "关注官方微博",
        "javascript:void",
        "logo",
    ]
    for token in noise_tokens:
        text = text.replace(token, " ")
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    # 句子切分后按关键词密度与长度打分，选择更像“核心结论”的句子。
    raw_sentences = [s.strip() for s in re.split(r"(?<=[。！？.!?;；])\s+", text) if s.strip()]
    sentences: List[str] = []
    for sent in raw_sentences:
        if len(sent) > 220:
            clauses = [c.strip() for c in re.split(r"[，,]\s*", sent) if c.strip()]
            sentences.extend(clauses)
        else:
            sentences.append(sent)

    noise_fragments = ["主办", "定价", "官方", "微博", "微信", "logo", "搜索", "收藏夹", "退出", "充值"]
    sentences = [s for s in sentences if not any(token in s for token in noise_fragments)]
    if not sentences:
        return text[:280]

    keywords = [
        "propose",
        "method",
        "result",
        "conclusion",
        "benchmark",
        "outperform",
        "improve",
        "evaluate",
        "find",
        "architecture",
        "指令集",
        "性能",
        "功耗",
        "兼容",
        "差异",
        "结论",
        "实验",
        "对比",
    ]

    def _signal_ratio(s: str) -> float:
        signal_chars = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", s)
        return len(signal_chars) / max(len(s), 1)

    scored: List[tuple[float, str]] = []
    for s in sentences:
        # 过滤版面符号密集、信息密度低的片段。
        if _signal_ratio(s) < 0.55:
            continue
        lower = s.lower()
        hit = sum(1 for k in keywords if k in lower)
        # 过短句通常信息量不足，轻微惩罚。
        length_bonus = min(len(s) / 80.0, 1.2)
        score = hit * 1.6 + length_bonus
        scored.append((score, s))

    if not scored:
        clean_sentences = [s for s in sentences if _signal_ratio(s) >= 0.45]
        fallback = " ".join(clean_sentences[:2]).strip()
        return fallback[:320] if fallback else text[:220]

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [s for _, s in scored[:2]]
    summary = " ".join(top).strip()
    if len(summary) < 90 and len(sentences) > 2:
        summary = (summary + " " + sentences[0]).strip()
    return summary[:420]


def _score_source_quality(item: Dict[str, Any]) -> Dict[str, Any]:
    """根据 source + domain 粗粒度评估来源质量。"""

    source = str(item.get("source", "")).lower().strip()
    url = str(item.get("url", "")).strip().lower()
    domain = urlparse(url).netloc if url else ""

    score = 0.5
    tier = "C"
    reason = "默认分层"

    # 论文与学术来源优先
    if source in {"arxiv", "openalex"} or "arxiv.org" in domain or "openalex.org" in domain:
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
    # 结构化百科
    elif source == "wikipedia" or "wikipedia.org" in domain:
        score = 0.86
        tier = "B"
        reason = "结构化百科来源"
    # 聚合搜索结果的基础可信度
    elif source in {"tavily", "duckduckgo"}:
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


def _topic_keywords(topic: str) -> List[str]:
    """从主题中抽取轻量关键词，用于相关性过滤。"""

    text = (topic or "").strip().lower()
    if not text:
        return []

    # 中英混写主题（如“nike和adidas的品牌侧重分析”）先按脚本类型切块，避免整句被当成一个 token。
    rough_chunks = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text)

    cn_stop_words = {
        "的",
        "和",
        "与",
        "及",
        "对",
        "在",
        "是",
        "比较",
        "分析",
        "研究",
        "侧重",
        "异同",
        "区别",
        "优劣",
    }
    cn_topic_terms = [
        "品牌",
        "性能",
        "架构",
        "市场",
        "策略",
        "技术",
        "营收",
        "利润",
        "供应链",
        "风险",
        "结论",
    ]
    alias_map = {
        "nike": ["耐克"],
        "adidas": ["阿迪达斯"],
        "iphone": ["苹果", "iphone"],
        "oppo": ["欧珀", "oppo"],
        "risc-v": ["riscv", "risc-v", "risc v"],
    }
    short_allowlist = {"c", "v", "ai", "ml", "isa", "cpu", "gpu", "risc"}

    candidates: List[str] = []
    for chunk in rough_chunks:
        if re.search(r"[a-z0-9]", chunk):
            candidates.append(chunk)
            continue

        # 中文块按常见连接词进一步拆分，提取“品牌/性能/架构”等有效词。
        parts = [
            p.strip()
            for p in re.split(r"[的和与及在对是、，。：；（）()\-\s]+", chunk)
            if p.strip()
        ]
        if parts:
            candidates.extend(parts)
        else:
            candidates.append(chunk)

        for term in cn_topic_terms:
            if term in chunk:
                candidates.append(term)

    picked: List[str] = []
    for token in candidates:
        if token in cn_stop_words:
            continue
        if len(token) < 2 and token not in short_allowlist:
            continue
        if token not in picked:
            picked.append(token)

        if token in alias_map:
            for alias in alias_map[token]:
                if alias not in picked:
                    picked.append(alias)

    return picked[:16]


def _text_signal_ratio(text: str) -> float:
    """计算文本信号比（字母/数字/中文占比），过滤版面噪声。"""

    cleaned = (text or "").strip()
    if not cleaned:
        return 0.0
    signal = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", cleaned)
    return round(len(signal) / max(len(cleaned), 1), 3)


def _topic_relevance_score(topic_keywords: List[str], title: str, summary: str, content: str) -> float:
    """基于关键词命中计算主题相关性（0-1）。"""

    if not topic_keywords:
        return 0.5

    title_text = title.lower()
    summary_text = summary.lower()
    content_text = content[:1800].lower()

    title_hits = sum(1 for kw in topic_keywords if kw in title_text)
    summary_hits = sum(1 for kw in topic_keywords if kw in summary_text)
    content_hits = sum(1 for kw in topic_keywords if kw in content_text)

    base = max(min(len(topic_keywords), 6), 1)
    weighted = (title_hits * 0.5 + summary_hits * 0.3 + content_hits * 0.2) / base
    return round(min(weighted, 1.0), 3)


def _filter_contexts_by_thresholds(
    contexts: List[Dict[str, Any]],
    topic: str,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """按阈值过滤来源，降低低质量/低相关噪声进入写作链路。"""

    min_quality = float(os.getenv("MIN_SOURCE_QUALITY", "0.66"))
    min_signal = float(os.getenv("MIN_SOURCE_SIGNAL", "0.52"))
    min_relevance = float(os.getenv("MIN_SOURCE_RELEVANCE", "0.22"))
    min_composite = float(os.getenv("MIN_SOURCE_COMPOSITE", "0.58"))
    min_keep = int(os.getenv("MIN_CONTEXT_KEEP", "6"))
    weight_quality = float(os.getenv("FILTER_WEIGHT_QUALITY", "0.55"))
    weight_signal = float(os.getenv("FILTER_WEIGHT_SIGNAL", "0.15"))
    weight_relevance = float(os.getenv("FILTER_WEIGHT_RELEVANCE", "0.30"))
    required_sources = [
        s.strip().lower()
        for s in os.getenv("REQUIRED_SOURCE_DIVERSITY", "duckduckgo,arxiv,tavily").split(",")
        if s.strip()
    ]

    total_weight = max(weight_quality + weight_signal + weight_relevance, 1e-6)
    weight_quality /= total_weight
    weight_signal /= total_weight
    weight_relevance /= total_weight

    keywords = _topic_keywords(topic)
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []

    for item in contexts:
        title = str(item.get("title", ""))
        core_summary = str(item.get("core_summary", ""))
        content = str(item.get("content", ""))
        quality = float(item.get("quality_score", 0.0))
        signal_ratio = _text_signal_ratio(core_summary or content)
        relevance = _topic_relevance_score(keywords, title, core_summary, content)
        composite = round(
            quality * weight_quality + signal_ratio * weight_signal + relevance * weight_relevance,
            3,
        )

        reasons: List[str] = []
        if quality < min_quality:
            reasons.append("low_quality")
        if signal_ratio < min_signal:
            reasons.append("low_signal")
        if relevance < min_relevance:
            reasons.append("low_relevance")
        if composite < min_composite:
            reasons.append("low_composite")

        enriched = dict(item)
        enriched["filter_scores"] = {
            "quality": round(quality, 3),
            "signal_ratio": signal_ratio,
            "relevance": relevance,
            "composite": composite,
        }
        enriched["filter_passed"] = not reasons
        enriched["filter_reasons"] = reasons

        if reasons:
            dropped.append(enriched)
        else:
            kept.append(enriched)

    # 多样性兜底：三类来源可用时至少保留各 1 条，避免单一来源主导。
    if required_sources and dropped:
        kept_sources = {str(item.get("source", "")).lower() for item in kept}
        for source in required_sources:
            if source in kept_sources:
                continue
            candidates = [
                item
                for item in dropped
                if str(item.get("source", "")).lower() == source
            ]
            if not candidates:
                continue
            best = sorted(
                candidates,
                key=lambda x: float(x.get("filter_scores", {}).get("composite", 0.0)),
                reverse=True,
            )[0]
            best["filter_passed"] = True
            best["filter_reasons"] = ["rescued_for_source_diversity"]
            kept.append(best)
            dropped = [x for x in dropped if id(x) != id(best)]
            kept_sources.add(source)

    # 兜底：若阈值过严导致有效上下文太少，则按综合分回补。
    if len(kept) < min_keep and dropped:
        rescue = sorted(
            dropped,
            key=lambda x: (
                float(x.get("filter_scores", {}).get("quality", 0.0)) * 0.55
                + float(x.get("filter_scores", {}).get("signal_ratio", 0.0)) * 0.15
                + float(x.get("filter_scores", {}).get("relevance", 0.0)) * 0.30
            ),
            reverse=True,
        )
        need = max(min_keep - len(kept), 0)
        rescued = rescue[:need]
        rescue_ids = {id(x) for x in rescued}
        for item in rescued:
            item["filter_passed"] = True
            item["filter_reasons"] = ["rescued_for_min_context"]
            kept.append(item)
        dropped = [x for x in dropped if id(x) not in rescue_ids]

    # 重新编号引用，保证连续。
    for idx, item in enumerate(kept, start=1):
        item["citation_id"] = f"S{idx}"

    reason_counts: Dict[str, int] = {
        "low_quality": 0,
        "low_signal": 0,
        "low_relevance": 0,
        "low_composite": 0,
        "rescued_for_source_diversity": 0,
        "rescued_for_min_context": 0,
    }
    for item in kept:
        for reason in item.get("filter_reasons", []):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    for item in dropped:
        for reason in item.get("filter_reasons", []):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    filter_summary = {
        "thresholds": {
            "min_quality": min_quality,
            "min_signal": min_signal,
            "min_relevance": min_relevance,
            "min_composite": min_composite,
            "min_keep": min_keep,
        },
        "weights": {
            "quality": round(weight_quality, 3),
            "signal": round(weight_signal, 3),
            "relevance": round(weight_relevance, 3),
        },
        "kept": len(kept),
        "dropped": len(dropped),
        "kept_sources": sorted({str(item.get("source", "")) for item in kept}),
        "dropped_sources": sorted({str(item.get("source", "")) for item in dropped}),
        "reason_counts": reason_counts,
        "avg_scores": {
            "quality": round(sum(float(x.get("filter_scores", {}).get("quality", 0.0)) for x in kept) / max(len(kept), 1), 3),
            "signal": round(sum(float(x.get("filter_scores", {}).get("signal_ratio", 0.0)) for x in kept) / max(len(kept), 1), 3),
            "relevance": round(sum(float(x.get("filter_scores", {}).get("relevance", 0.0)) for x in kept) / max(len(kept), 1), 3),
            "composite": round(sum(float(x.get("filter_scores", {}).get("composite", 0.0)) for x in kept) / max(len(kept), 1), 3),
        },
        "dropped_samples": [
            {
                "title": str(item.get("title", ""))[:120],
                "source": item.get("source", ""),
                "reasons": item.get("filter_reasons", []),
            }
            for item in dropped[:5]
        ],
    }
    return kept, filter_summary


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
        rated = _score_source_quality(item)
        rated["core_summary"] = _extract_core_summary(
            title=str(rated.get("title", "")),
            content=str(rated.get("content", "")),
        )
        unique.append(rated)

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

    ddg_top_k = int(os.getenv("DDG_RESULTS_PER_QUERY", "10"))
    arxiv_top_k = int(os.getenv("ARXIV_RESULTS_PER_QUERY", "5"))
    tavily_top_k = int(os.getenv("TAVILY_RESULTS_PER_QUERY", "3"))
    query_budget = int(os.getenv("SEARCH_QUERY_BUDGET", "3"))

    for query in queries[:query_budget]:
        try:
            contexts.extend(duckduckgo_search(query, max_results=ddg_top_k))
        except Exception as exc:
            errors = _append_error(errors, f"DDG 检索失败: {exc}")

        try:
            contexts.extend(arxiv_search(query, max_results=arxiv_top_k))
        except Exception as exc:
            errors = _append_error(errors, f"ArXiv 检索失败: {exc}")

        try:
            contexts.extend(tavily_search(query, max_results=tavily_top_k))
        except Exception as exc:
            errors = _append_error(errors, f"Tavily 检索失败: {exc}")

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
    filtered_contexts, filter_summary = _filter_contexts_by_thresholds(normalized_contexts, topic)
    source_quality_summary = _build_quality_summary(filtered_contexts)
    source_quality_summary["filter_summary"] = filter_summary
    trace = list(state.get("execution_trace", []))
    trace.append(
        {
            "node": "researcher",
            "revision_step": state.get("revision_step", 0),
            "queries": len(queries),
            "contexts": len(filtered_contexts),
            "dropped": filter_summary.get("dropped", 0),
            "quality_avg": source_quality_summary.get("avg_score", 0.0),
            "errors": len(errors),
        }
    )

    return {
        "search_queries": queries,
        "retrieved_context": filtered_contexts,
        "source_quality_summary": source_quality_summary,
        "errors": errors,
        "execution_trace": trace,
    }

