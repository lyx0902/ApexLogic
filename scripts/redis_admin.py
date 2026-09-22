"""Connectivity and bounded cache smoke test; no search/LLM API calls."""
import json
from uuid import uuid4
from dotenv import load_dotenv


def main():
    load_dotenv()
    from core.cache import get_cache, cached_search, cache_scope
    cache = get_cache()
    if cache is None:
        raise SystemExit("Redis cache is disabled or client is unavailable")
    try:
        cache.client.ping()
    except Exception as exc:
        raise SystemExit("Redis connection failed: " + type(exc).__name__) from None
    calls = []
    @cached_search("admin-smoke:v1")
    def search(query):
        calls.append(1)
        return [{"title": "smoke", "url": "https://example.org", "source": "stub", "content": "smoke"}]
    nonce = uuid4().hex
    from core.cache import key_for
    key = key_for("search", ["admin-smoke:v1", {"query": nonce}])
    try:
        with cache_scope({}) as stats:
            search(nonce)
            search(nonce)
            assert len(calls) == 1, "cache reuse failed (check FORCE_REFRESH)"
            print(json.dumps({"status": "ok", "stats": stats}))
    finally:
        cache.call("delete", key)


if __name__ == "__main__":
    main()
