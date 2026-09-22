from __future__ import annotations

from core.run_config import setting
from typing import Any, Dict, List, Tuple

import requests
import math


def _pack_ranked_items(
    ranked_items: List[Dict[str, Any]],
    score_key: str,
    top_k: int,
    sample_limit: int = 10,
) -> Dict[str, Any]:
    """打包重排序阶段的入选/淘汰样本，便于 debug 展示。"""

    selected = ranked_items[: max(top_k, 0)]
    dropped = ranked_items[max(top_k, 0) :]

    def _to_view(item: Dict[str, Any], rank: int, selected: bool) -> Dict[str, Any]:
        score = item.get(score_key)
        return {
            "rank": rank if score is not None else None,
            "title": str(item.get("title", ""))[:140],
            "source": str(item.get("source", "")),
            "url": str(item.get("url", "")),
            "score": float(score) if score is not None else None,
            "reason": ("selected_top_k" if selected else item.get("_audit_reason", "rank_below_top_k")),
            **{k: item[k] for k in ("origin", "memory_id", "memory_source_run_id") if k in item},
        }

    selected_view = [
        _to_view(item, idx, True)
        for idx, item in enumerate(selected[:sample_limit], start=1)
    ]
    dropped_view = [
        _to_view(item, idx + len(selected), False)
        for idx, item in enumerate(dropped[:sample_limit], start=1)
    ]

    selected_full = [
        _to_view(item, idx, True)
        for idx, item in enumerate(selected, start=1)
    ]
    dropped_full = [
        _to_view(item, idx + len(selected), False)
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
    """将候选文档拼接为重排序输入文本。"""

    title = str(item.get("title", "")).strip()
    summary = str(item.get("core_summary", "")).strip()
    content = str(item.get("content", "")).strip()
    merged = "\n".join([x for x in [title, summary, content[:1200]] if x])
    return merged[:1800]


def _parse_rerank_results(body: Dict[str, Any]) -> List[Tuple[int, float]]:
    """兼容不同服务商的 rerank 返回格式。"""

    raw_results = body.get("results", body.get("data", []))
    if not isinstance(raw_results, list):
        return []

    parsed: List[Tuple[int, float]] = []
    for row in raw_results:
        if not isinstance(row, dict):
            continue
        try:
            raw_idx = row.get("index", row.get("document_index", -1))
            idx = int(raw_idx)
            if isinstance(raw_idx, bool) or str(raw_idx) != str(idx):
                continue
            score = float(row.get("relevance_score", row.get("score")))
        except (TypeError, ValueError, OverflowError):
            continue
        if idx < 0:
            continue
        if math.isfinite(score):
            parsed.append((idx, score))

    parsed.sort(key=lambda x: x[1], reverse=True)
    return parsed


def _resolve_rerank_endpoint(api_base: str) -> str:
    """兼容 base_url 为 /v1、/v1/ 或完整 /rerank 地址的配置。"""

    base = (api_base or "").strip().rstrip("/")
    if base.endswith("/rerank"):
        return base
    return base + "/rerank"


def rerank_top_k(
    topic: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 10,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """BGE Reranker：对 Retriever 输出精排，保留 top_k。"""

    if not candidates:
        return [], {
            "enabled": True,
            "mode": "empty",
            "input": 0,
            "selected": 0,
        }

    enabled = setting("BGE_RERANKER_ENABLED", "1").strip() == "1"
    if not enabled:
        ranked_items = [dict(item) for item in candidates]
        selected = ranked_items[: max(top_k, 0)]
        rank_pack = _pack_ranked_items(ranked_items, "bge_reranker_score", top_k)
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

    api_key = setting("BGE_RERANK_API_KEY", setting("BGE_EMBED_API_KEY", "")).strip()
    api_base = setting("BGE_RERANK_BASE_URL", "https://api.siliconflow.cn/v1").strip()
    model = setting("BGE_RERANK_MODEL", "BAAI/bge-reranker-v2-m3").strip()
    timeout_sec = float(setting("BGE_RERANK_TIMEOUT", "25"))

    if not api_key:
        # 无密钥时按 Retriever 分数回退。
        fallback = sorted(
            [dict(item) for item in candidates],
            key=lambda x: float(x.get("bge_retriever_score", 0.0)),
            reverse=True,
        )
        for row in fallback:
            row["bge_reranker_score"] = round(float(row.get("bge_retriever_score", 0.0)), 6)
        selected = fallback[: max(top_k, 0)]
        rank_pack = _pack_ranked_items(fallback, "bge_reranker_score", top_k)
        return selected, {
            "enabled": True,
            "mode": "retriever_fallback",
            "api_base": api_base,
            "model": "retriever_score",
            "top_k": top_k,
            "input": len(candidates),
            "selected": len(selected),
            **rank_pack,
        }

    documents = [_build_document_text(item) for item in candidates]
    endpoint = _resolve_rerank_endpoint(api_base)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "query": topic,
        "documents": documents,
        "top_n": len(documents),
    }

    resp = requests.post(endpoint, json=payload, headers=headers, timeout=timeout_sec)
    resp.raise_for_status()
    body = resp.json()
    ranked = _parse_rerank_results(body)
    if not ranked:
        raise RuntimeError("rerank API 返回为空或格式无法解析")

    ranked_items: List[Dict[str, Any]] = []
    seen = set()
    for idx, score in ranked:
        if idx >= len(candidates) or idx in seen:
            continue
        seen.add(idx)
        row = dict(candidates[idx])
        row["bge_reranker_score"] = round(float(score), 6)
        ranked_items.append(row)

    if not ranked_items:
        raise RuntimeError("rerank 后无有效候选")
    selected = ranked_items[:max(top_k, 0)]
    returned_count = len(ranked_items)
    for idx, candidate in enumerate(candidates):
        if idx not in seen:
            row = dict(candidate)
            row.pop("bge_reranker_score", None)
            row["_audit_reason"] = "not_returned_by_api"
            ranked_items.append(row)
    # Missing API results are audited, never promoted into the selected evidence.
    rank_pack = _pack_ranked_items(ranked_items, "bge_reranker_score", len(selected))
    return selected, {
        "enabled": True,
        "mode": "api",
        "api_base": api_base,
        "model": model,
        "top_k": top_k,
        "input": len(candidates),
        "selected": len(selected),
        "returned_count": returned_count,
        "unscored_count": len(candidates) - returned_count,
        **rank_pack,
    }

