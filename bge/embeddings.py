"""Shared embedding API with strict index, dimension and finite-value validation."""
from typing import List, Tuple
import requests
import numpy as np


def validate_vectors(vectors):
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError("empty or ragged embeddings")
    norms = np.linalg.norm(array, axis=1)
    if not np.isfinite(array).all() or not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError("invalid embedding values")
    return array


def embed_texts(
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

    if sorted(idx for idx, _ in embeddings) != list(range(len(texts))):
        raise ValueError("invalid embedding indices")
    embeddings.sort(key=lambda x: x[0])
    ordered = [emb for _, emb in embeddings]
    if len(ordered) != len(texts):
        raise RuntimeError("embedding 数量与输入数量不一致")
    validate_vectors(ordered)
    return ordered


def _resolve_embeddings_endpoint(api_base: str) -> str:
    """兼容 base_url 为 /v1、/v1/ 或完整 /embeddings 地址的配置。"""

    base = (api_base or "").strip().rstrip("/")
    if base.endswith("/embeddings"):
        return base
    return base + "/embeddings"


