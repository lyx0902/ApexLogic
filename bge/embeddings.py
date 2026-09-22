"""Shared embedding API with strict index, dimension and finite-value validation."""
from typing import List, Tuple
import hashlib
import os
import time
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


def _request_embeddings(
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


def embed_texts(texts: List[str], model: str, api_base: str, api_key: str,
                timeout_sec: float) -> List[List[float]]:
    """Reuse exact text vectors; retain batching for all misses and validate mixed dimensions."""
    from core.cache import get_cache, key_for, ttl_setting, count
    cache = get_cache()
    ttl = ttl_setting("APEXLOGIC_EMBED_CACHE_TTL", 604800)
    endpoint = _resolve_embeddings_endpoint(api_base)
    # Isolate credentials' model deployments without storing the secret itself.
    identity = [model, endpoint, hashlib.sha256(api_key.encode()).hexdigest(),
                os.getenv("APEXLOGIC_EMBED_CACHE_VERSION", "1")]

    def request(batch):
        count("embedding.external_calls")
        count("embedding.external_texts", len(batch))
        start = time.monotonic()
        try:
            return _request_embeddings(batch, model, api_base, api_key, timeout_sec)
        finally:
            count("embedding.external_seconds", round(time.monotonic() - start, 4))

    if cache is None or not texts:
        count("embedding.disabled_bypass", len(texts))
        return request(texts)

    def valid(value):
        try:
            return isinstance(value, list) and validate_vectors([value]).shape[0] == 1
        except (ValueError, TypeError):
            return False

    unique = list(dict.fromkeys(texts))
    keys = {text: key_for("embedding", [identity, text]) for text in unique}
    vectors = {}
    for text in unique:
        doc = cache.read(keys[text], ttl, valid)
        if doc is not None:
            vectors[text] = doc["value"]
    count("embedding.hits", len(vectors))
    missing = [text for text in unique if text not in vectors]
    count("embedding.misses", len(missing))
    fresh = request(missing) if missing else []
    for text, vector in zip(missing, fresh):
        vectors[text] = vector
    try:
        validate_vectors([vectors[text] for text in unique])
    except ValueError:
        # A provider changed dimensions under the same model name: refresh the whole batch.
        count("embedding.dimension_refresh")
        fresh = request(unique)
        missing = unique
        vectors = dict(zip(unique, fresh))
    for text, vector in zip(missing, fresh):
        cache.write(keys[text], vector, ttl)
    return [vectors[text] for text in texts]


def _resolve_embeddings_endpoint(api_base: str) -> str:
    """兼容 base_url 为 /v1、/v1/ 或完整 /embeddings 地址的配置。"""

    base = (api_base or "").strip().rstrip("/")
    if base.endswith("/embeddings"):
        return base
    return base + "/embeddings"


