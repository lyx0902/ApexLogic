"""Serializable run settings; credentials stay in the environment.

ContextVar binding is per node invocation (including LangGraph's worker threads),
never a mutation of process-wide os.environ. Legacy callers retain env defaults.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import hashlib
import json
import math
import os
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import urlsplit

SCHEMA_VERSION = 1
WORKFLOW_VERSION = "research-rwr-v1"
SETTING_KEYS = frozenset("""
MAX_REVISIONS REVIEWER_PASS_THRESHOLD REVIEWER_ALLOW_DEGRADED_PASS
DEEPSEEK_BASE_URL DEEPSEEK_MODEL SEARCH_QUERY_BUDGET
DDG_TOTAL_RESULTS ARXIV_TOTAL_RESULTS TAVILY_TOTAL_RESULTS
DDG_RESULTS_PER_QUERY ARXIV_RESULTS_PER_QUERY TAVILY_RESULTS_PER_QUERY
GRAPH_EXPAND_QUERIES AQD_ENABLED AQD_MAX_SUB_QUESTIONS AQD_RESULTS_PER_SUBQ
ITERATIVE_RETRIEVAL_ENABLED MAX_HOPS GAP_QUERIES_PER_HOP GAP_RESULTS_PER_QUERY
IRCOT_ENABLE_FINAL_SUMMARY BGE_RETRIEVER_TOP_K BGE_RERANKER_TOP_K
BGE_RETRIEVER_ENABLED BGE_RERANKER_ENABLED BGE_EMBED_MODEL BGE_EMBED_BASE_URL
BGE_EMBED_TIMEOUT BGE_EMBED_BATCH_SIZE BGE_RERANK_MODEL BGE_RERANK_BASE_URL
BGE_RERANK_TIMEOUT ENABLE_TAVILY_FALLBACK ARXIV_SYNONYM_FILE
""".split())
_ACTIVE: ContextVar[Mapping[str, str | None] | None] = ContextVar("run_settings", default=None)


def setting(key: str, default: Any = None) -> Any:
    """Resolve non-secret settings from the frozen snapshot; secrets stay live."""
    active = _ACTIVE.get()
    if active is not None and key in SETTING_KEYS:
        value = active.get(key)
        return default if value is None else value
    return os.getenv(key, default)


def validate_config(config: dict) -> dict:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("任务配置版本不兼容")
    env = config.get("settings")
    if not isinstance(env, dict) or set(env) != SETTING_KEYS:
        raise ValueError("任务配置字段缺失或包含未知字段")
    if any(v is not None and not isinstance(v, str) for v in env.values()):
        raise ValueError("任务配置值必须是字符串或 null")
    revisions = int(env["MAX_REVISIONS"])
    threshold = float(env["REVIEWER_PASS_THRESHOLD"])
    if not 1 <= revisions <= 100 or not math.isfinite(threshold) or not 0 <= threshold <= 10:
        raise ValueError("最大轮数或评审阈值超出范围")
    if env["REVIEWER_ALLOW_DEGRADED_PASS"] not in {"0", "1"}:
        raise ValueError("降级放行配置必须为 0 或 1")
    if config.get("output_mode") not in {"user", "debug", "both", "user_only", "eval"}:
        raise ValueError("未知输出模式")
    flags = {key for key in SETTING_KEYS if key.endswith("ENABLED")} | {"ENABLE_TAVILY_FALLBACK", "IRCOT_ENABLE_FINAL_SUMMARY"}
    numeric = {"SEARCH_QUERY_BUDGET", "DDG_TOTAL_RESULTS", "ARXIV_TOTAL_RESULTS", "TAVILY_TOTAL_RESULTS",
               "DDG_RESULTS_PER_QUERY", "ARXIV_RESULTS_PER_QUERY", "TAVILY_RESULTS_PER_QUERY",
               "GRAPH_EXPAND_QUERIES", "AQD_MAX_SUB_QUESTIONS", "AQD_RESULTS_PER_SUBQ", "MAX_HOPS",
               "GAP_QUERIES_PER_HOP", "GAP_RESULTS_PER_QUERY", "BGE_RETRIEVER_TOP_K", "BGE_RERANKER_TOP_K",
               "BGE_EMBED_BATCH_SIZE"}
    for key in flags:
        if env[key] is not None and env[key].strip() not in {"0", "1"}:
            raise ValueError(f"{key} 必须为 0 或 1")
    for key in numeric:
        if env[key] not in (None, "") and int(env[key]) < 0:
            raise ValueError(f"{key} 不能为负数")
    for key in {"BGE_EMBED_TIMEOUT", "BGE_RERANK_TIMEOUT"}:
        if env[key] is not None and (not math.isfinite(float(env[key])) or float(env[key]) <= 0):
            raise ValueError(f"{key} 必须为正数")
    for key, value in env.items():
        if key.endswith("BASE_URL") and value:
            parsed = urlsplit(value)
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError(f"{key} 不能包含凭据、查询参数或片段")
    return config


def make_run_config(*, max_revisions: int | None = None,
                    pass_threshold: float | None = None, output_mode: str = "debug") -> dict:
    env = {key: os.getenv(key) for key in sorted(SETTING_KEYS)}
    env["MAX_REVISIONS"] = str(max_revisions if max_revisions is not None else int(env["MAX_REVISIONS"] or "3"))
    env["REVIEWER_PASS_THRESHOLD"] = str(pass_threshold if pass_threshold is not None else float(env["REVIEWER_PASS_THRESHOLD"] or "7.5"))
    env["REVIEWER_ALLOW_DEGRADED_PASS"] = env["REVIEWER_ALLOW_DEGRADED_PASS"] or "0"
    config = {"schema_version": SCHEMA_VERSION, "settings": env, "output_mode": output_mode}
    return validate_config(config)


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@contextmanager
def bind_config(config: dict | None):
    if config is None:
        yield
        return
    validate_config(config)
    token = _ACTIVE.set(MappingProxyType(dict(config["settings"])))
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def configured_node(fn):
    @wraps(fn)
    def wrapped(state):
        with bind_config(state.get("run_config")):
            return fn(state)
    return wrapped
