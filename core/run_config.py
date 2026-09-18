"""Serializable run settings; credentials stay in the environment.

ContextVar binding is per node invocation (including LangGraph's worker threads),
never a mutation of process-wide os.environ. Legacy callers retain env defaults.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from datetime import datetime
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
_RESEARCH_AS_OF = ContextVar("research_as_of", default=None)


def research_time_hint():
    value = _RESEARCH_AS_OF.get()
    if not value:
        return "研究截至日期未提供；不得凭模型记忆假定当前日期。对时效结论须明确时间不确定性。"
    return f"本任务固定研究截至时间：{value}。所有‘目前/现任’均相对此时间判断；区分来源发布日期、事件生效日与研究截至日，不得假设当前早于已经过去的生效日。"


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
    if "storage" in config:
        storage = config["storage"]
        if (not isinstance(storage, dict) or set(storage) != {"backend", "profile"}
                or storage["backend"] not in {"sqlite", "postgres"} or storage["profile"] != "default"):
            raise ValueError("无效的存储配置；连接凭据不能写入任务快照")
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
    if "research_as_of" in config:
        if not isinstance(config["research_as_of"], str) or datetime.fromisoformat(config["research_as_of"]).tzinfo is None:
            raise ValueError("研究截至时间必须包含时区")
    # Missing block identifies a legacy task: memory stays disabled without mutation.
    if "memory" in config:
        m = config["memory"]
        if not isinstance(m, dict) or set(m) != {"enabled", "namespace", "top_k", "min_score", "ttl_days", "char_budget", "data_dir"}:
            raise ValueError("无效的记忆配置")
        if type(m["enabled"]) is not bool or not isinstance(m["namespace"], str) or not m["namespace"].strip():
            raise ValueError("无效的记忆开关或命名空间")
        if not isinstance(m["data_dir"], str):
            raise ValueError("无效的记忆目录")
        for key, low, high in [("top_k", 1, 20), ("ttl_days", 1, 3650), ("char_budget", 400, 20000)]:
            if type(m[key]) is not int or not low <= m[key] <= high:
                raise ValueError("无效的记忆数值配置")
        if not isinstance(m["min_score"], (int, float)) or not math.isfinite(m["min_score"]) or not 0 <= m["min_score"] <= 1:
            raise ValueError("无效的记忆相关性阈值")
    return config


def make_run_config(*, max_revisions: int | None = None,
                    pass_threshold: float | None = None, output_mode: str = "debug") -> dict:
    env = {key: os.getenv(key) for key in sorted(SETTING_KEYS)}
    env["MAX_REVISIONS"] = str(max_revisions if max_revisions is not None else int(env["MAX_REVISIONS"] or "3"))
    env["REVIEWER_PASS_THRESHOLD"] = str(pass_threshold if pass_threshold is not None else float(env["REVIEWER_PASS_THRESHOLD"] or "7.5"))
    env["REVIEWER_ALLOW_DEGRADED_PASS"] = env["REVIEWER_ALLOW_DEGRADED_PASS"] or "0"
    flag = os.getenv("MEMORY_ENABLED", "1").strip()
    if flag not in {"0", "1"}:
        raise ValueError("MEMORY_ENABLED 必须为 0 或 1")
    memory = {"enabled": flag == "1", "namespace": os.getenv("MEMORY_NAMESPACE", "workspace/default"),
              "top_k": int(os.getenv("MEMORY_TOP_K", "5")), "min_score": float(os.getenv("MEMORY_MIN_SCORE", "0.65")),
              "ttl_days": int(os.getenv("MEMORY_TTL_DAYS", "30")), "char_budget": int(os.getenv("MEMORY_CHAR_BUDGET", "3000")),
              "data_dir": ""}
    config = {"schema_version": SCHEMA_VERSION, "settings": env, "output_mode": output_mode, "memory": memory, "research_as_of": datetime.now().astimezone().isoformat()}
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
    time_token = _RESEARCH_AS_OF.set(config.get("research_as_of"))
    try:
        yield
    finally:
        _ACTIVE.reset(token)
        _RESEARCH_AS_OF.reset(time_token)


def configured_node(fn):
    @wraps(fn)
    def wrapped(state):
        with bind_config(state.get("run_config")):
            return fn(state)
    return wrapped
