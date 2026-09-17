from __future__ import annotations

import math
from core.run_config import setting
import re
from datetime import datetime
from typing import Any, Dict, List, Tuple

import requests


def _pack_ranked_items(
    scored_items: List[Dict[str, Any]],
    score_key: str,
    top_k: int,
    sample_limit: int = 10,
) -> Dict[str, Any]:
    """打包检索阶段的入选/淘汰样本，便于 debug 可解释输出。"""

    selected = scored_items[: max(top_k, 0)]
    dropped = scored_items[max(top_k, 0) :]

    timestamp = datetime.now().isoformat(timespec="seconds")

    def _to_view(item: Dict[str, Any], rank: int, reason: str) -> Dict[str, Any]:
        return {
            "rank": rank,
            "title": str(item.get("title", ""))[:140],
            "source": str(item.get("source", "")),
            "url": str(item.get("url", "")),
            "score": float(item.get(score_key, 0.0)),
            "reason": reason,
            "timestamp": timestamp,
        }

    selected_view = [
        _to_view(item, idx, "selected_top_k")
        for idx, item in enumerate(selected[:sample_limit], start=1)
    ]
    dropped_view = [
        _to_view(item, idx + len(selected), "rank_below_top_k")
        for idx, item in enumerate(dropped[:sample_limit], start=1)
    ]

    selected_full = [
        _to_view(item, idx, "selected_top_k")
        for idx, item in enumerate(selected, start=1)
    ]
    dropped_full = [
        _to_view(item, idx + len(selected), "rank_below_top_k")
        for idx, item in enumerate(dropped, start=1)
    ]
    return {
        "selected_samples": selected_view,
        "dropped_samples": dropped_view,
        "selected_records": selected_full,
        "dropped_records": dropped_full,
        "dropped_count": len(dropped),
    }


def _build_document_text(item: Dict[str, Any]) -> str:
    """将候选文档拼接为统一检索文本。"""

    title = str(item.get("title", "")).strip()
    summary = str(item.get("core_summary", "")).strip()
    content = str(item.get("content", "")).strip()
    merged = "\n".join([x for x in [title, summary, content[:1200]] if x])
    return merged[:1800]


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """计算余弦相似度。"""

    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    n1 = math.sqrt(sum(a * a for a in vec1))
    n2 = math.sqrt(sum(b * b for b in vec2))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return dot / (n1 * n2)


def _topic_keywords(topic: str) -> List[str]:
    """轻量关键词抽取，作为 API 失败时的回退方案。"""

    chunks = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", (topic or "").lower())
    stop = {"的", "和", "与", "及", "研究", "分析", "比较", "概述", "性能"}
    picked: List[str] = []
    for token in chunks:
        if token in stop:
            continue
        if len(token) < 2:
            continue
        if token not in picked:
            picked.append(token)
    return picked[:12]


def _keyword_score(topic: str, doc_text: str) -> float:
    """关键词命中打分（回退模式）。"""

    kws = _topic_keywords(topic)
    if not kws:
        return 0.0
    text = doc_text.lower()
    hits = sum(1 for k in kws if k in text)
    return round(hits / max(len(kws), 1), 4)


def _embed_texts(
    texts: List[str],
    model: str,
    api_base: str,
    api_key: str,
    timeout_sec: float,
) -> List[List[float]]:
    """调用 OpenAI 兼容 Embedding API。"""

    endpoint = _resolve_embeddings_endpoint(api_base)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "input": texts,
    }
    resp = requests.post(endpoint, json=payload, headers=headers, timeout=timeout_sec)
    resp.raise_for_status()
    body = resp.json()
    data = body.get("data", [])
    if not isinstance(data, list):
        raise RuntimeError("embedding API 返回格式异常: data 不是列表")

    # 兼容 index 无序返回。
    embeddings: List[Tuple[int, List[float]]] = []
    for row in data:
        idx = int(row.get("index", len(embeddings)))
        emb = row.get("embedding", [])
        if not isinstance(emb, list):
            continue
        embeddings.append((idx, [float(x) for x in emb]))

    embeddings.sort(key=lambda x: x[0])
    ordered = [emb for _, emb in embeddings]
    if len(ordered) != len(texts):
        raise RuntimeError("embedding 数量与输入数量不一致")
    return ordered


def _resolve_embeddings_endpoint(api_base: str) -> str:
    """兼容 base_url 为 /v1、/v1/ 或完整 /embeddings 地址的配置。"""

    base = (api_base or "").strip().rstrip("/")
    if base.endswith("/embeddings"):
        return base
    return base + "/embeddings"


def retrieve_top_k(
    topic: str,
    queries: List[str],
    candidates: List[Dict[str, Any]],
    top_k: int = 20,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """BGE Retriever：粗筛候选上下文，保留 top_k。"""

    if not candidates:
        return [], {
            "enabled": True,
            "mode": "empty",
            "input": 0,
            "selected": 0,
        }

    enabled = setting("BGE_RETRIEVER_ENABLED", "1").strip() == "1"
    if not enabled:
        scored_items = [dict(item) for item in candidates]
        selected = scored_items[: max(top_k, 0)]
        rank_pack = _pack_ranked_items(scored_items, "bge_retriever_score", top_k)
        return selected, {
            "enabled": False,
            "mode": "disabled",
            "api_base": "",
            "model": "",
            "top_k": top_k,
            "input": len(candidates),
            "selected": len(selected),
            **rank_pack,
        }

    api_key = setting("BGE_EMBED_API_KEY", "").strip()
    api_base = setting("BGE_EMBED_BASE_URL", "https://api.siliconflow.cn/v1").strip()
    model = setting("BGE_EMBED_MODEL", "BAAI/bge-m3").strip()
    timeout_sec = float(setting("BGE_EMBED_TIMEOUT", "20"))
    batch_size = max(int(setting("BGE_EMBED_BATCH_SIZE", "32")), 1)

    query_text = topic
    if queries:
        query_text = topic + "\n" + "\n".join(queries[:4])

    docs = [_build_document_text(item) for item in candidates]

    if not api_key:
        # 无密钥时回退关键词粗筛。
        scored: List[Dict[str, Any]] = []
        for item, doc in zip(candidates, docs):
            row = dict(item)
            row["bge_retriever_score"] = _keyword_score(topic, doc)
            scored.append(row)
        scored.sort(key=lambda x: float(x.get("bge_retriever_score", 0.0)), reverse=True)
        selected = scored[: max(top_k, 0)]
        rank_pack = _pack_ranked_items(scored, "bge_retriever_score", top_k)
        return selected, {
            "enabled": True,
            "mode": "keyword_fallback",
            "query": query_text,
            "api_base": api_base,
            "model": "keyword",
            "top_k": top_k,
            "batch_size": batch_size,
            "input": len(candidates),
            "selected": len(selected),
            **rank_pack,
        }

    query_vecs = _embed_texts(
        texts=[query_text],
        model=model,
        api_base=api_base,
        api_key=api_key,
        timeout_sec=timeout_sec,
    )
    query_vec = query_vecs[0]

    doc_vecs: List[List[float]] = []
    for start in range(0, len(docs), batch_size):
        chunk = docs[start : start + batch_size]
        doc_vecs.extend(
            _embed_texts(
                texts=chunk,
                model=model,
                api_base=api_base,
                api_key=api_key,
                timeout_sec=timeout_sec,
            )
        )

    scored_items: List[Dict[str, Any]] = []
    for item, vec in zip(candidates, doc_vecs):
        score = _cosine_similarity(query_vec, vec)
        row = dict(item)
        row["bge_retriever_score"] = round(float(score), 6)
        scored_items.append(row)

    scored_items.sort(key=lambda x: float(x.get("bge_retriever_score", 0.0)), reverse=True)
    selected = scored_items[: max(top_k, 0)]
    rank_pack = _pack_ranked_items(scored_items, "bge_retriever_score", top_k)

    return selected, {
        "enabled": True,
        "mode": "embedding",
        "query": query_text,
        "api_base": api_base,
        "model": model,
        "top_k": top_k,
        "batch_size": batch_size,
        "input": len(candidates),
        "selected": len(selected),
        **rank_pack,
    }

