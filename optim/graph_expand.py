"""Graph-based query expansion via concept co-occurrence.

从已检索的文档中抽取关键概念，构建共现图，用 PageRank 找出核心节点，
生成补充查询，帮助 Researcher 在下一轮覆盖初始检索遗漏的重要子方向。

使用方式
--------
from optim.graph_expand import expand_queries_from_contexts

extra_queries = expand_queries_from_contexts(
    topic="LangGraph多智能体系统优化",
    contexts=normalized_contexts,   # List[Dict]
    top_k=3,                        # 扩展查询条数
)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False


# ── 停用词（中英文常见无信息词）────────────────────────────────────
_STOPWORDS = {
    # 中文
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都",
    "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会",
    "着", "没有", "看", "好", "自己", "这", "那", "中", "大", "来",
    "对", "与", "及", "等", "为", "从", "被", "通过", "基于", "用于",
    "可以", "能够", "研究", "分析", "方法", "系统", "技术", "应用",
    "方面", "问题", "结果", "进行", "实现", "提出", "提供", "包括",
    "主要", "相关", "不同", "多个", "以及", "目前", "已经", "更多",
    # 英文
    "the", "a", "an", "of", "in", "to", "and", "is", "are", "for",
    "with", "this", "that", "on", "by", "from", "it", "as", "be",
    "was", "has", "have", "can", "which", "we", "our", "their",
    "show", "propose", "paper", "study", "result", "model", "based",
}

# 最短有效词长（字符数）
_MIN_TOKEN_LEN = 2
# 每篇文档最多抽取的概念数（控制图规模）
_MAX_CONCEPTS_PER_DOC = 12
# PageRank 阻尼系数
_PAGERANK_ALPHA = 0.85


def _extract_concepts(text: str) -> List[str]:
    """从文本中提取候选概念词（中英文混合简单策略）。"""

    # 英文短语：2-4 个单词组成的名词短语（连字符视为单词内部）
    en_phrases = re.findall(r"\b[A-Za-z][A-Za-z\-]{1,}\b(?:\s+[A-Za-z][A-Za-z\-]{1,}\b){0,2}", text)
    # 中文词：2-6 个汉字组成的词
    zh_phrases = re.findall(r"[\u4e00-\u9fff]{2,6}", text)

    candidates: List[str] = []
    for phrase in en_phrases + zh_phrases:
        token = phrase.strip().lower()
        if len(token) < _MIN_TOKEN_LEN:
            continue
        if token in _STOPWORDS:
            continue
        # 过滤纯数字和极短英文停用词
        if re.fullmatch(r"[\d\s\-]+", token):
            continue
        candidates.append(token)

    # 保留出现频次 ≥ 1 的 top-N（此处简单去重后截取）
    seen: set[str] = set()
    unique: List[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique[:_MAX_CONCEPTS_PER_DOC]


def _build_cooccurrence_graph(
    contexts: List[Dict[str, Any]],
) -> "nx.Graph":
    """对每篇文档内的概念两两连边，构建无向加权共现图。"""

    G = nx.Graph()

    for item in contexts:
        text = " ".join([
            str(item.get("title", "")),
            str(item.get("core_summary", "")),
            str(item.get("content", ""))[:300],
        ])
        concepts = _extract_concepts(text)
        for i, c1 in enumerate(concepts):
            for c2 in concepts[i + 1:]:
                if G.has_edge(c1, c2):
                    G[c1][c2]["weight"] += 1
                else:
                    G.add_edge(c1, c2, weight=1)

    return G


def _top_central_concepts(G: "nx.Graph", top_n: int = 8) -> List[Tuple[str, float]]:
    """用 PageRank 计算概念中心度，返回 top-N (concept, score) 列表。"""

    if len(G) == 0:
        return []

    try:
        scores = nx.pagerank(G, alpha=_PAGERANK_ALPHA, weight="weight")
    except Exception:
        # 图不连通等异常时降级为度中心度
        scores = dict(nx.degree_centrality(G))

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


def expand_queries_from_contexts(
    topic: str,
    contexts: List[Dict[str, Any]],
    top_k: int = 3,
    existing_queries: List[str] | None = None,
) -> List[str]:
    """从已检索文档构建概念图，生成 top_k 条扩展查询。

    参数
    ----
    topic           : 研究主题，用于拼接查询
    contexts        : 已去重归一化的检索结果列表
    top_k           : 返回的扩展查询条数
    existing_queries: 已有查询列表，用于去重避免重复方向

    返回
    ----
    新查询列表（长度 ≤ top_k）
    """

    if not _NX_AVAILABLE:
        return []

    if not contexts:
        return []

    G = _build_cooccurrence_graph(contexts)
    if len(G) < 2:
        return []

    central = _top_central_concepts(G, top_n=top_k * 3)
    if not central:
        return []

    existing_set = set(q.lower() for q in (existing_queries or []))

    queries: List[str] = []
    for concept, _ in central:
        if len(queries) >= top_k:
            break
        # 跳过与已有查询高度重叠的概念（简单子串检查）
        if any(concept in eq or eq in concept for eq in existing_set):
            continue
        query = f"{topic} {concept}"
        queries.append(query)

    return queries
