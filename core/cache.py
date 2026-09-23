"""Optional, disposable Redis cache. PostgreSQL remains the durable source of truth."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache, wraps
from threading import Lock
import hashlib
import inspect
import json
import os
import re
import time
from uuid import uuid4

_SCOPE = ContextVar("cache_scope", default=None)
_RELEASE = "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end"
_PUBLISH = "if redis.call('get', KEYS[1]) == ARGV[1] then redis.call('set', KEYS[2], ARGV[2], 'EX', ARGV[3]); return 1 else return 0 end"


def sensitive(text):
    return bool(re.search(r"现任|当前|最新|目前|今日|今天|价格|股价|汇率|实时|近期|最近|\b(current|latest|today|price|now|recent)\b", str(text), re.I))


def count(name, amount=1):
    scope = _SCOPE.get()
    if scope is not None:
        with scope["lock"]:
            scope["stats"][name] = scope["stats"].get(name, 0) + amount


@contextmanager
def cache_scope(state):
    scope = {"namespace": state.get("run_config", {}).get("memory", {}).get("namespace", os.getenv("MEMORY_NAMESPACE", "workspace/default")),
             "fresh": sensitive(state.get("topic", "")) or sensitive(state.get("critique_feedback", "")),
             "postgres": state.get("run_config", {}).get("storage", {}).get("backend") == "postgres",
             "stats": {}, "lock": Lock()}
    token = _SCOPE.set(scope)
    try:
        yield scope["stats"]
    finally:
        _SCOPE.reset(token)


def key_for(kind, identity):
    scope = _SCOPE.get()
    namespace = scope["namespace"] if scope else os.getenv("MEMORY_NAMESPACE", "workspace/default")
    body = json.dumps([namespace, identity], sort_keys=True, ensure_ascii=False, allow_nan=False)
    return "apexlogic:cache:v1:" + kind + ":" + hashlib.sha256(body.encode()).hexdigest()


class Cache:
    def __init__(self, client):
        self.client = client
        self.retry_at = 0.0

    def call(self, method, *args, **kwargs):
        if time.monotonic() < self.retry_at:
            count("redis.circuit_bypass")
            return None
        try:
            return getattr(self.client, method)(*args, **kwargs)
        except Exception:
            # Never expose exception messages: client errors may contain credentials.
            self.retry_at = time.monotonic() + 10
            count("redis.errors")
            return None

    def read(self, key, ttl, validator):
        raw = self.call("get", key)
        if raw is None:
            return None
        try:
            doc = json.loads(raw)
            age = time.time() - doc["fetched_at"]
            if not 0 <= age < ttl or not validator(doc["value"]):
                raise ValueError("expired or invalid cache")
            return doc
        except (TypeError, ValueError, KeyError):
            count("cache.invalid")
            return None

    def write(self, key, value, ttl):
        doc = json.dumps({"fetched_at": time.time(), "value": value}, allow_nan=False)
        self.call("set", key, doc, ex=ttl)


@lru_cache(maxsize=8)
def _client(host, port, password, db):
    import redis
    from redis.backoff import NoBackoff
    from redis.retry import Retry
    return Cache(redis.Redis(host=host, port=port, password=password or None, db=db,
                             decode_responses=True, socket_timeout=0.4, socket_connect_timeout=0.4,
                             retry=Retry(NoBackoff(), 0), max_connections=16))


def get_cache():
    if os.getenv("APEXLOGIC_CACHE_ENABLED", "0") != "1":
        return None
    try:
        return _client(os.getenv("APEXLOGIC_REDIS_HOST", "127.0.0.1"),
                       int(os.getenv("APEXLOGIC_REDIS_PORT", "6379")),
                       os.getenv("APEXLOGIC_REDIS_PASSWORD", ""),
                       int(os.getenv("APEXLOGIC_REDIS_DB", "0")))
    except Exception:
        count("redis.errors")
        return None


def ttl_setting(name, default):
    try:
        return max(1, min(int(os.getenv(name, str(default))), 2592000))
    except ValueError:
        return default


def valid_search(value):
    return (isinstance(value, list) and bool(value) and
            all(isinstance(x, dict) and all(isinstance(x.get(k), str) for k in ("title", "url", "content", "source")) for x in value))


def cached_search(provider):
    """Exact provider request cache, including bounded single-flight coordination."""
    def decorate(fn):
        signature = inspect.signature(fn)

        @wraps(fn)
        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            scope = _SCOPE.get()
            fresh = (scope and scope["fresh"]) or sensitive(next(iter(bound.arguments.values()), ""))
            fresh = fresh or os.getenv("APEXLOGIC_CACHE_FORCE_REFRESH", "0") == "1"
            cache = None if fresh else get_cache()
            metric = "search."

            def invoke():
                # A cache hit never occupies a provider permit. All PostgreSQL
                # Workers share the same session-lock capacity for each source.
                from contextlib import nullcontext
                from core.service_limits import capacity, global_slot, rate_ticket
                limits = {"duckduckgo": ("DDG", 4), "arxiv": ("ARXIV", 2),
                          "tavily": ("TAVILY", 2)}
                minute_defaults = {"duckduckgo": 60, "arxiv": 20, "tavily": 60}
                provider_name = provider.split(":", 1)[0]
                code, default = limits.get(provider_name, (provider_name.upper(), 2))
                wait = capacity("APEXLOGIC_PROVIDER_WAIT_SECONDS", 30, maximum=300)
                guard = (global_slot("provider:" + provider_name,
                         capacity("APEXLOGIC_GLOBAL_" + code + "_MAX_INFLIGHT", default),
                         wait_seconds=wait)
                         if scope and scope["postgres"] else nullcontext(True))
                with guard:
                    if scope and scope["postgres"]:
                        waited = rate_ticket(
                            provider_name,
                            capacity("APEXLOGIC_GLOBAL_" + code + "_PER_MINUTE",
                                     minute_defaults.get(provider_name, 60), maximum=100000),
                            wait_seconds=wait,
                        )
                        count(metric + "rate_wait_seconds", round(waited, 4))
                    count(metric + "external_calls")
                    start = time.monotonic()
                    try:
                        return fn(*args, **kwargs)
                    finally:
                        count(metric + "external_seconds", round(time.monotonic() - start, 4))

            if cache is None:
                count(metric + ("fresh_bypass" if fresh else "disabled_bypass"))
                return invoke()
            key = key_for("search", [provider, dict(bound.arguments)])
            ttl = ttl_setting("APEXLOGIC_SEARCH_CACHE_TTL", 1800)

            def hit(doc):
                count(metric + "hits")
                return [dict(x, cache_hit=True, cache_fetched_at=doc["fetched_at"]) for x in doc["value"]]

            doc = cache.read(key, ttl, valid_search)
            if doc is not None:
                return hit(doc)
            count(metric + "misses")
            token = uuid4().hex
            lock = key + ":lock"
            owned = cache.call("set", lock, token, nx=True, ex=60)
            if not owned and cache.retry_at <= time.monotonic():
                count(metric + "waits")
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    time.sleep(0.05)
                    doc = cache.read(key, ttl, valid_search)
                    if doc is not None:
                        return hit(doc)
                    if cache.retry_at > time.monotonic():
                        break
                count(metric + "wait_timeouts")
            try:
                # A previous owner may have published between our first read and lock acquisition.
                if owned:
                    doc = cache.read(key, ttl, valid_search)
                    if doc is not None:
                        return hit(doc)
                result = invoke()
                fetched = time.time()
                if owned and valid_search(result):
                    doc = json.dumps({"fetched_at": fetched, "value": result}, allow_nan=False)
                    cache.call("eval", _PUBLISH, 2, lock, key, token, doc, ttl)
                if valid_search(result):
                    return [dict(x, cache_hit=False, cache_fetched_at=fetched) for x in result]
                return result
            finally:
                if owned:
                    cache.call("eval", _RELEASE, 1, lock, token)
        return wrapped
    return decorate
