from __future__ import annotations
from optim.search_audit import search_with_audit, describe, annotate_selection
from optim.parallel_search import map_in_order, runtime_limit

from core.run_config import configured_node, setting
from optim.query_text import strip_list_marker
import re
from threading import Semaphore
from typing import Any, Dict, List, Optional

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
from bge.retriever import retrieve_top_k
from bge.reranker import rerank_top_k
from tools.arxiv_tool import arxiv_search
from tools.search_tool import duckduckgo_search, tavily_search
from optim.mab_search import ThompsonSamplingMAB, compute_source_rewards
from optim.graph_expand import expand_queries_from_contexts
from optim.iterative_retrieval import IterativeRetrievalOptimizer
from optim.query_planner import AdaptiveQueryPlanner, _dependency_query


def _append_error(errors: List[str], message: str) -> List[str]:
    """将错误信息追加到 errors，避免覆盖既有日志。"""

    updated = list(errors)
    updated.append(message)
    return updated


def _generate_reasoning_summary(
    reasoning_chains: List[str],
    reasoning_contexts: List[Dict[str, Any]],
) -> str:
    """生成推理链的结构化摘要，供Writer快速理解。"""
    if not reasoning_chains:
        return ""

    summary_parts = [
        "【IRCoT多跳推理链摘要】",
        f"共 {len(reasoning_chains)} 跳推理，补充检索 {len(reasoning_contexts)} 条专属文档。",
        "",
        "推理过程：",
    ]

    for i, chain in enumerate(reasoning_chains, start=1):
        preview = chain[:200].replace("\n", " ")
        summary_parts.append(f"{i}. {preview}...")

    summary_parts.append("")
    summary_parts.append(
        "注：推理链文档已独立保存，不受BGE筛选影响，"
        "代表系统通过多跳推理主动发现的关键信息缺口。"
    )

    return "\n".join(summary_parts)


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
    revision_directives: Optional[Dict[str, Any]] = None,
    revision_step: int = 0,
) -> List[str]:
    """可选使用 DeepSeek 重写检索词，提升针对性。"""

    if ChatOpenAI is None:
        return seed_queries

    deepseek_api_key = setting("DEEPSEEK_API_KEY", "")
    if not deepseek_api_key:
        return seed_queries

    deepseek_base_url = setting("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    deepseek_model = setting("DEEPSEEK_MODEL", "deepseek-chat")

    llm = ChatOpenAI(
        model=deepseek_model,
        api_key=lambda: deepseek_api_key,
        base_url=deepseek_base_url,
        temperature=0.2,
    )
    user_prompt = build_researcher_user_prompt(
        topic, critique_feedback, revision_directives=revision_directives, revision_step=revision_step
    )
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
        item = strip_list_marker(line)
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
            **{key: item[key] for key in ("origin", "memory_id", "memory_version", "memory_source_run_id",
                "memory_observed_at", "memory_valid_until", "memory_score", "memory_required", "cache_hit", "cache_fetched_at") if key in item},
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


def _dedupe_and_index_contexts(
    contexts: List[Dict[str, Any] | str],
    citation_prefix: str = "S",
) -> List[Dict[str, Any]]:
    """按 url 或 title+source 去重，并生成连续 citation_id。"""

    unique: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for raw in contexts:
        item = _normalize_context_item(raw)
        url_key = item.get("url", "").strip().lower()
        fallback_key = f"{item.get('source','')}|{item.get('title','').strip().lower()}"
        key = ("required-memory:" + str(item["memory_id"])) if item.get("memory_required") else (url_key or fallback_key)
        if key in seen:
            continue
        seen.add(key)
        normalized = dict(item)
        normalized["core_summary"] = normalized["content"] if normalized.get("origin") == "memory" else _extract_core_summary(
            title=str(normalized.get("title", "")),
            content=str(normalized.get("content", "")),
        )
        unique.append(normalized)

    for idx, item in enumerate(unique, start=1):
        item["citation_id"] = f"{citation_prefix}{idx}"

    return unique


def _resolve_base_budgets(q_count: int) -> tuple[Dict[str, int], Dict[str, str]]:
    """从环境变量解析三个搜索源的基础预算，返回预算与来源标记。"""

    def _resolve_total(
        total_key: str,
        legacy_key: str,
        default_total: int,
    ) -> tuple[int, str]:
        total_raw = setting(total_key)
        if total_raw is not None and str(total_raw).strip() != "":
            return max(int(total_raw), 0), total_key

        legacy_raw = setting(legacy_key)
        if legacy_raw is not None and str(legacy_raw).strip() != "":
            legacy_per_query = max(int(legacy_raw), 0)
            return legacy_per_query * q_count, legacy_key

        return default_total, "default"

    ddg_total, ddg_src = _resolve_total("DDG_TOTAL_RESULTS", "DDG_RESULTS_PER_QUERY", 35)
    arxiv_total, arxiv_src = _resolve_total("ARXIV_TOTAL_RESULTS", "ARXIV_RESULTS_PER_QUERY", 20)
    tavily_total, tavily_src = _resolve_total("TAVILY_TOTAL_RESULTS", "TAVILY_RESULTS_PER_QUERY", 5)

    budgets = {"duckduckgo": ddg_total, "arxiv": arxiv_total, "tavily": tavily_total}
    sources = {"duckduckgo": ddg_src, "arxiv": arxiv_src, "tavily": tavily_src}
    return budgets, sources


def _collect_broad_contexts(
    queries: List[str],
    errors: List[str],
    override_budgets: Optional[Dict[str, int]] = None,
) -> tuple[List[Dict[str, Any] | str], List[str], Dict[str, Any]]:
    """按总配额执行广搜，避免按 query 乘法膨胀请求量。

    override_budgets: 若由 MAB 提供，则使用该预算覆盖环境变量配置。
    """

    contexts: List[Dict[str, Any] | str] = []
    q_count = max(len(queries), 1)

    base_budgets, quota_sources = _resolve_base_budgets(q_count)

    if override_budgets is not None:
        # 使用 MAB 调整后的预算，来源标记为 mab
        ddg_total = override_budgets.get("duckduckgo", base_budgets["duckduckgo"])
        arxiv_total = override_budgets.get("arxiv", base_budgets["arxiv"])
        tavily_total = override_budgets.get("tavily", base_budgets["tavily"])
        ddg_source = arxiv_source = tavily_source = "mab"
    else:
        ddg_total = base_budgets["duckduckgo"]
        arxiv_total = base_budgets["arxiv"]
        tavily_total = base_budgets["tavily"]
        ddg_source = quota_sources["duckduckgo"]
        arxiv_source = quota_sources["arxiv"]
        tavily_source = quota_sources["tavily"]

    # 兼容异常配置：若三路总配额全为 0，则回退默认值，避免整轮直接占位符。
    if ddg_total + arxiv_total + tavily_total == 0:
        ddg_total, arxiv_total, tavily_total = 35, 20, 5
        errors = _append_error(
            errors,
            "检测到搜索总配额均为0，已自动回退为默认配额 DDG=35, ArXiv=20, Tavily=5。",
        )
        ddg_source = arxiv_source = tavily_source = "auto_fallback_default"

    provider_stats: Dict[str, Dict[str, int]] = {
        "duckduckgo": {"target": ddg_total, "planned": 0, "fetched": 0, "failed": 0},
        "arxiv": {"target": arxiv_total, "planned": 0, "fetched": 0, "failed": 0},
        "tavily": {"target": tavily_total, "planned": 0, "fetched": 0, "failed": 0},
    }

    def _per_query(total: int, idx: int) -> int:
        base = total // q_count
        extra = 1 if idx < (total % q_count) else 0
        return max(base + extra, 0)

    max_workers = runtime_limit("APEXLOGIC_BROAD_MAX_WORKERS", 4)
    provider_limits = {
        "duckduckgo": min(max_workers, runtime_limit("APEXLOGIC_BROAD_DDG_MAX_INFLIGHT", 2)),
        "arxiv": min(max_workers, runtime_limit("APEXLOGIC_BROAD_ARXIV_MAX_INFLIGHT", 1)),
        "tavily": min(max_workers, runtime_limit("APEXLOGIC_BROAD_TAVILY_MAX_INFLIGHT", 1)),
    }
    semaphores = {name: Semaphore(limit) for name, limit in provider_limits.items()}
    providers = (
        ("duckduckgo", ddg_total, duckduckgo_search),
        ("arxiv", arxiv_total, arxiv_search),
        ("tavily", tavily_total, tavily_search),
    )
    tasks = []
    for idx, query in enumerate(queries):
        for provider, total, search in providers:
            limit = _per_query(total, idx)
            if limit > 0:
                provider_stats[provider]["planned"] += limit
                tasks.append((idx, query, provider, limit, search))

    def _search_one(task):
        _, query, provider, limit, search = task
        records = []
        try:
            with semaphores[provider]:
                docs = search_with_audit(records, provider, query, limit, search)
            return docs, records, None
        except Exception as exc:
            return [], records, exc

    results = map_in_order(_search_one, tasks, max_workers)

    search_records = []
    error_labels = {"duckduckgo": "DDG", "arxiv": "ArXiv", "tavily": "Tavily"}
    for task, (docs, records, failure) in zip(tasks, results):
        idx, _, provider, limit, _ = task
        for record in records:
            record["query_index"] = idx
        search_records.extend(records)
        if failure is None:
            contexts.extend(docs)
            provider_stats[provider]["fetched"] += len(docs)
        else:
            errors = _append_error(errors, f"{error_labels[provider]} 检索失败: {failure}")
            provider_stats[provider]["failed"] += limit

    summary = {
        "broad_targets": {
            "duckduckgo": ddg_total,
            "arxiv": arxiv_total,
            "tavily": tavily_total,
        },
        "quota_source": {
            "duckduckgo": ddg_source,
            "arxiv": arxiv_source,
            "tavily": tavily_source,
        },
        "broad_fetched": provider_stats,
        "broad_attempted": (
            provider_stats["duckduckgo"]["planned"]
            + provider_stats["arxiv"]["planned"]
            + provider_stats["tavily"]["planned"]
        ),
        "broad_total": len(contexts),
        "broad_concurrency": {"max_workers": max_workers, "provider_limits": provider_limits},
        "search_records": search_records,
    }
    return contexts, errors, summary


@configured_node
def researcher_node(state: ResearchState) -> Dict[str, Any]:
    """检索代理节点。

    输入: topic / critique_feedback / mab_state
    输出: search_queries / retrieved_context / errors / mab_state
    """

    topic = state.get("topic", "")
    critique_feedback = state.get("critique_feedback", "")
    revision_directives = dict(state.get("revision_directives", {}) or {})
    errors = list(state.get("errors", []))

    # ── MAB：恢复或初始化 ──────────────────────────────────────────────
    mab_raw = state.get("mab_state") or {}
    mab = ThompsonSamplingMAB.from_dict(mab_raw) if mab_raw else ThompsonSamplingMAB()

    seed_queries = _build_queries(
        topic=topic,
        critique_feedback=critique_feedback,
        revision_directives=revision_directives,
    )
    queries = list(seed_queries)

    from memory.first import prepare_memory_first, retain_required_memory
    early = prepare_memory_first(state, AdaptiveQueryPlanner())
    memory_first = early["summary"]
    memory_hits, memory_stats = early["hits"], early["stats"]
    query_budget = int(setting("SEARCH_QUERY_BUDGET", "3"))
    memory_broad_questions = []
    memory_completed = {}
    memory_resolution = {}
    memory_waves = []
    if memory_first["applied"]:
        queries = [q["search_query"] for q in early["questions"]]
        remaining_ids = {q["id"] for q in early["remaining"]}
        by_memory_id = {hit["memory_id"]: hit for hit in memory_hits}
        for q in memory_first["subquestions"]:
            if q["search_skipped"]:
                memory_completed[q["id"]] = {"raw_docs": [by_memory_id[mid] for mid in q["evidence_ids"]
                                                   if mid in by_memory_id]}
        ready = [q for q in early["remaining"] if not any(
            dep in remaining_ids for dep in q.get("depends_on", []))]
        memory_broad_questions = ready[:query_budget]
        if memory_broad_questions:
            memory_waves.append([q["id"] for q in memory_broad_questions])
        effective_queries = []
        for q in memory_broad_questions:
            resolved, evidence, unresolved = _dependency_query(
                q["search_query"], q.get("depends_on", []), memory_completed)
            memory_resolution[q["id"]] = (resolved, evidence, unresolved)
            effective_queries.append(resolved)
    else:
        try:
            queries = _rewrite_queries_with_llm(
                topic, critique_feedback, seed_queries,
                revision_directives=revision_directives,
                revision_step=state.get("revision_step", 0),
            )
        except Exception as exc:
            errors = _append_error(errors, f"DeepSeek 查询重写失败，已使用规则检索词: {exc}")
        effective_queries = queries[:query_budget]

    # ── MAB：从环境变量获取基础预算，Thompson Sampling 分配本轮预算 ───
    q_count = max(len(effective_queries), 1)
    base_budgets, _ = _resolve_base_budgets(q_count)
    mab_budgets = mab.allocate_budgets(base_budgets)

    if memory_first["applied"] and not effective_queries:
        contexts, bge_stage_summary = [], {"broad_total": 0, "broad_attempted": 0}
        mab_budgets = {key: 0 for key in mab_budgets}
    else:
        contexts, errors, bge_stage_summary = _collect_broad_contexts(
            effective_queries, errors, override_budgets=mab_budgets
        )
    search_records = list(bge_stage_summary.get("search_records", []))
    graph_search_records = []
    if memory_first["applied"]:
        for record in search_records:
            record["subquestion_id"] = memory_broad_questions[record["query_index"]]["id"]
        for q in memory_broad_questions:
            memory_completed[q["id"]] = {"raw_docs": [
                {**doc, "content": doc.get("excerpt", "")}
                for record in search_records if record["subquestion_id"] == q["id"]
                for doc in record["retrieved_docs"]]}
        pending = [q for q in early["remaining"] if q not in memory_broad_questions]
        while pending:
            ready = [q for q in pending if all(dep not in remaining_ids or dep in memory_completed
                     for dep in q.get("depends_on", []))]
            if not ready:
                ready = [pending[0]]  # Invalid legacy cycle: keep the run progressing.
            memory_waves.append([q["id"] for q in ready])
            items = []
            for q in ready:
                resolved, evidence, unresolved = _dependency_query(
                    q["search_query"], q.get("depends_on", []), memory_completed)
                memory_resolution[q["id"]] = (resolved, evidence, unresolved)
                items.append((q, resolved))

            def search_memory_gap(item):
                q, resolved = item
                records = []
                try:
                    docs = search_with_audit(records, "duckduckgo", resolved,
                                             int(setting("AQD_RESULTS_PER_SUBQ", "3")), duckduckgo_search)
                    return q["id"], docs, records, None
                except Exception as exc:
                    return q["id"], [], records, type(exc).__name__

            for qid, docs, records, failure in map_in_order(
                search_memory_gap, items, runtime_limit("APEXLOGIC_AQD_MAX_WORKERS", 2)
            ):
                for record in records:
                    record["subquestion_id"] = qid
                search_records.extend(records)
                contexts.extend(docs)
                memory_completed[qid] = {"raw_docs": docs}
                if failure:
                    errors = _append_error(errors, f"记忆未覆盖子问题补搜失败: {failure}")
            ready_ids = {q["id"] for q in ready}
            pending = [q for q in pending if q["id"] not in ready_ids]
    contexts = list(contexts) + memory_hits

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

    # ── 图扩展查询：从已检索文档构建概念共现图，补搜核心概念方向 ──────
    graph_expand_k = int(setting("GRAPH_EXPAND_QUERIES", "2"))
    graph_expand_summary: Dict[str, Any] = {"enabled": False, "extra_queries": [], "extra_contexts": 0}
    if graph_expand_k > 0 and normalized_contexts and not memory_first["applied"]:
        try:
            extra_queries = expand_queries_from_contexts(
                topic=topic,
                contexts=normalized_contexts,
                top_k=graph_expand_k,
                existing_queries=effective_queries,
            )
            if extra_queries:
                # 用 DDG 对扩展查询各搜少量结果（每条 3 条），不走 MAB 分配
                extra_contexts: List[Dict[str, Any] | str] = []
                def search_graph(eq):
                    records = []
                    try:
                        got = search_with_audit(records, "duckduckgo", eq, 3, duckduckgo_search)
                        return got, records, None
                    except Exception as exc:
                        return [], records, type(exc).__name__
                for got, records, failure in map_in_order(
                    search_graph, extra_queries, runtime_limit("APEXLOGIC_GRAPH_MAX_WORKERS", 2)
                ):
                    graph_search_records.extend(records)
                    extra_contexts.extend(got)
                    if failure:
                        errors = _append_error(errors, f"图扩展查询 DDG 失败: {failure}")
                if extra_contexts:
                    merged = _dedupe_and_index_contexts(list(contexts) + list(extra_contexts))
                    normalized_contexts = merged
                graph_expand_summary = {
                    "enabled": True,
                    "extra_queries": extra_queries,
                    "extra_contexts": len(extra_contexts) if extra_contexts else 0,
                    "total_after_merge": len(normalized_contexts),
                }
        except Exception as exc:
            errors = _append_error(errors, f"图扩展查询失败，已跳过: {exc}")

    # ── 自适应查询分解（AQD）: 分解主题→子问题→逐一补搜 ──────────────
    aqd_enabled = setting("AQD_ENABLED", "1").strip() == "1"
    query_plan: Dict[str, Any] = {"enabled": False}
    if memory_first["applied"]:
        graph_expand_summary = {"enabled": False, "reason": "memory_first_gap_plan"}
        sub_results = []
        for q in memory_first["subquestions"]:
            records = [r for r in search_records if r.get("subquestion_id") == q["id"]] if not q["search_skipped"] else []
            docs = [d for r in records for d in r["retrieved_docs"]]
            resolved, evidence, unresolved = memory_resolution.get(q["id"], (q["search_query"], [], []))
            sub_results.append({**q, "skipped": q["search_skipped"], "new_docs": len(docs),
                                "resolved_query": resolved, "dependency_evidence": evidence,
                                "unresolved_dependencies": unresolved,
                                "search_records": records, "retrieved_docs": docs,
                                "memory_docs": [describe(h) for h in memory_hits if h["memory_id"] in q["recalled_ids"]],
                                "search_stage": "memory" if q["search_skipped"] else "broad_or_gap_search"})
        query_plan = {"enabled": True, "mode": "memory_first", "sub_questions_count": len(early["questions"]),
                      "execution_order": [q["id"] for q in early["questions"]],
                      "execution_waves": memory_waves,
                      "total_new_docs": sum(len(r["retrieved_docs"]) for r in search_records),
                      "sub_results": sub_results}

    if aqd_enabled and normalized_contexts and not memory_first["applied"]:
        try:
            max_sub_questions = int(setting("AQD_MAX_SUB_QUESTIONS", "4"))
            results_per_subq = int(setting("AQD_RESULTS_PER_SUBQ", "3"))
            planner = AdaptiveQueryPlanner(
                max_sub_questions=max_sub_questions,
                results_per_subq=results_per_subq,
            )
            contexts_before_aqd = len(normalized_contexts)
            aqd_contexts, query_plan = planner.run(
                topic=topic,
                contexts=normalized_contexts,
                existing_queries=effective_queries,
                prior_search_records=search_records,
                **({"sub_questions": early["questions"]} if early["questions"] else {}),
            )
            if aqd_contexts:
                normalized_contexts = _dedupe_and_index_contexts(
                    list(normalized_contexts) + list(aqd_contexts)
                )
            if query_plan.get("enabled"):
                query_plan["contexts_before"] = contexts_before_aqd
                query_plan["contexts_after"] = len(normalized_contexts)
                for sub in query_plan.get("sub_results", []):
                    if sub.get("skipped"):
                        sq = sub["search_query"].lower()
                        prior = [r for r in search_records if sq in r["query"].lower() or r["query"].lower() in sq]
                        sub["matched_search_records"] = prior
                        sub["retrieved_docs"] = [dict(d) for r in prior for d in r["retrieved_docs"]]
        except Exception as exc:
            errors = _append_error(errors, f"AQD 查询分解失败，已跳过: {exc}")

    # ── 迭代检索优化（IRCoT）: 推理链→识别缺口→补搜 ─────────────────
    iter_enabled = setting("ITERATIVE_RETRIEVAL_ENABLED", "1").strip() == "1"
    iterative_retrieval_summary: Dict[str, Any] = {"enabled": False}
    reasoning_chains: List[str] = list(state.get("reasoning_chains", []))
    reasoning_contexts_raw: List[Dict[str, Any]] = []
    reasoning_summary = ""

    if iter_enabled and normalized_contexts:
        try:
            max_hops = int(setting("MAX_HOPS", "4"))
            gap_queries_per_hop = int(setting("GAP_QUERIES_PER_HOP", "2"))
            results_per_query = int(setting("GAP_RESULTS_PER_QUERY", "3"))

            optimizer = IterativeRetrievalOptimizer(
                max_hops=max_hops,
                gap_queries_per_hop=gap_queries_per_hop,
                results_per_query=results_per_query,
            )

            # 传入历史推理链（跨轮记忆）
            prior_chains = list(reasoning_chains)

            gap_contexts, new_chains, hop_summaries = optimizer.run(
                topic=topic,
                contexts=normalized_contexts,
                existing_queries=effective_queries,
                prior_reasoning_chains=prior_chains,
            )

            contexts_before = len(normalized_contexts)

            # 保存IRCoT文档到独立通道（归一化+R前缀，不受BGE筛选影响）
            if gap_contexts:
                reasoning_contexts_raw = _dedupe_and_index_contexts(
                    list(gap_contexts), citation_prefix="R"
                )

                # 仍然合并到normalized_contexts供BGE筛选（保持现有逻辑）
                normalized_contexts = _dedupe_and_index_contexts(
                    list(normalized_contexts) + list(gap_contexts)
                )

            # 累积推理链（跨轮持久化）
            reasoning_chains = reasoning_chains + new_chains

            # 生成推理链摘要（供Writer理解）
            reasoning_summary = _generate_reasoning_summary(
                reasoning_chains=reasoning_chains,
                reasoning_contexts=reasoning_contexts_raw,
            )

            iterative_retrieval_summary = {
                "enabled": True,
                "max_hops": max_hops,
                "hops_executed": len(hop_summaries),
                "contexts_before": contexts_before,
                "contexts_after": len(normalized_contexts),
                "gap_contexts_added": len(gap_contexts),
                "reasoning_contexts_count": len(reasoning_contexts_raw),
                "hop_summaries": hop_summaries,
            }
        except Exception as exc:
            errors = _append_error(errors, f"IRCoT 迭代检索失败，已跳过: {exc}")

    # Online evidence wins URL deduplication; recalled items retain immutable provenance.
    from memory.service import recall_for_state
    if memory_stats is None:
        memory_hits, memory_stats = recall_for_state(state)
    if memory_hits and not early["hits"]:
        normalized_contexts = _dedupe_and_index_contexts(list(normalized_contexts) + memory_hits)

    retriever_top_k = int(setting("BGE_RETRIEVER_TOP_K", "20"))
    reranker_top_k = int(setting("BGE_RERANKER_TOP_K", "10"))

    bge_config_snapshot = {
        "retriever_enabled": setting("BGE_RETRIEVER_ENABLED", "1"),
        "reranker_enabled": setting("BGE_RERANKER_ENABLED", "1"),
        "retriever_model": setting("BGE_EMBED_MODEL", "BAAI/bge-m3"),
        "retriever_base_url": setting("BGE_EMBED_BASE_URL", "https://api.siliconflow.cn/v1"),
        "retriever_top_k": retriever_top_k,
        "reranker_model": setting("BGE_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
        "reranker_base_url": setting("BGE_RERANK_BASE_URL", "https://api.siliconflow.cn/v1"),
        "reranker_top_k": reranker_top_k,
    }

    retrieved_20 = list(normalized_contexts)
    retriever_summary: Dict[str, Any] = {
        "enabled": False,
        "mode": "not_run",
        "input": len(normalized_contexts),
        "selected": len(normalized_contexts),
    }
    try:
        retrieved_20, retriever_summary = retrieve_top_k(
            topic=topic,
            queries=effective_queries,
            candidates=normalized_contexts,
            top_k=retriever_top_k,
        )
    except Exception as exc:
        errors = _append_error(errors, f"BGE Retriever 失败，回退原候选: {exc}")
        retriever_summary = {
            "enabled": True,
            "mode": "error_fallback",
            "input": len(normalized_contexts),
            "selected": len(retrieved_20),
            "error": str(exc),
        }

    reranked_10 = list(retrieved_20)
    reranker_summary: Dict[str, Any] = {
        "enabled": False,
        "mode": "not_run",
        "input": len(retrieved_20),
        "selected": len(retrieved_20),
    }
    try:
        reranked_10, reranker_summary = rerank_top_k(
            topic=topic,
            candidates=retrieved_20,
            top_k=reranker_top_k,
        )
    except Exception as exc:
        errors = _append_error(errors, f"BGE Reranker 失败，回退 Retriever 结果: {exc}")
        reranker_summary = {
            "enabled": True,
            "mode": "error_fallback",
            "input": len(retrieved_20),
            "selected": len(reranked_10),
            "error": str(exc),
        }

    filtered_contexts = retain_required_memory(list(reranked_10), normalized_contexts, reranker_top_k)
    memory_first["retained_ids"] = [c["memory_id"] for c in filtered_contexts if c.get("memory_required")]
    if memory_first["retained_ids"]:
        reranker_summary["policy_retained_memory_ids"] = memory_first["retained_ids"]
        reranker_summary["effective_selected"] = len(filtered_contexts)
    for idx, item in enumerate(filtered_contexts, start=1):
        item["citation_id"] = f"S{idx}"

    annotate_selection(query_plan, filtered_contexts)

    dropped_count = max(len(normalized_contexts) - len(filtered_contexts), 0)

    # ── MAB：计算各信源奖励并更新 Beta 参数 ──────────────────────────
    mab_rewards = compute_source_rewards([c for c in filtered_contexts if c.get("origin") != "memory"], mab_budgets)
    mab.update(mab_rewards)
    updated_mab_state = mab.to_dict()

    bge_stage_summary.update(
        {
            "config": bge_config_snapshot,
            "dedup_total": len(normalized_contexts),
            "retriever": retriever_summary,
            "reranker": reranker_summary,
            "dropped": dropped_count,
            "final_contexts": len(filtered_contexts),
            # MAB 本轮摘要，便于 debug 报告可视化
            "mab": {
                "base_budgets": base_budgets,
                "allocated_budgets": mab_budgets,
                "rewards": {k: (round(v, 4) if v is not None else None) for k, v in mab_rewards.items()},
                "expected_rewards_after": mab.expected_rewards(),
                "round": mab.round,
            },
            # 图扩展查询摘要
            "graph_expand": graph_expand_summary,
            # AQD 自适应查询分解摘要
            "query_plan": query_plan,
            # IRCoT 迭代检索摘要
            "iterative_retrieval": iterative_retrieval_summary,
        }
    )

    memory_stats["selected_ids"] = [c["memory_id"] for c in filtered_contexts if c.get("memory_id")]
    memory_stats["selected"] = len(memory_stats["selected_ids"])
    source_quality_summary: Dict[str, Any] = {}
    source_quality_summary["bge_summary"] = bge_stage_summary
    trace = list(state.get("execution_trace", []))
    trace.append(
        {
            "node": "researcher",
            "revision_step": state.get("revision_step", 0),
            "queries": len(effective_queries),
            "broad_total": bge_stage_summary.get("broad_total", 0),
            "dedup_total": bge_stage_summary.get("dedup_total", 0),
            "retriever_topk": retriever_summary.get("selected", 0),
            "reranker_topk": reranker_summary.get("selected", 0),
            "contexts": len(filtered_contexts),
            "dropped": dropped_count,
            "errors": len(errors),
            "mab_budgets": mab_budgets,
            "mab_expected_rewards": mab.expected_rewards(),
            "graph_expand": graph_expand_summary,
            "query_plan": {
                "enabled": query_plan.get("enabled", False),
                "sub_questions_count": query_plan.get("sub_questions_count", 0),
                "total_new_docs": query_plan.get("total_new_docs", 0),
            },
            "iterative_retrieval": iterative_retrieval_summary,
        }
    )

    return {
        "planned_search_queries": queries,
        "search_queries": list(dict.fromkeys(
            [r["query"] for r in search_records + graph_search_records]
            + [r["query"] for sub in query_plan.get("sub_results", []) for r in sub.get("search_records", [])]
            + [q for hop in iterative_retrieval_summary.get("hop_summaries", []) for q in hop.get("gap_queries", [])]
        )),
        "memory_stats": memory_stats,
        "memory_first": memory_first,
        "memory_query_ids": list(dict.fromkeys(state.get("memory_query_ids", []) + ([memory_stats["query_id"]] if memory_stats.get("query_id") else []))),
        "retrieved_context": filtered_contexts,
        "source_quality_summary": source_quality_summary,
        "errors": errors,
        "execution_trace": trace,
        "mab_state": updated_mab_state,
        "reasoning_chains": reasoning_chains,
        "reasoning_contexts": reasoning_contexts_raw,
        "reasoning_summary": reasoning_summary,
        "reasoning_enabled": iter_enabled and bool(reasoning_chains),
        "iterative_retrieval_summary": iterative_retrieval_summary,
        "query_plan": query_plan,
    }

