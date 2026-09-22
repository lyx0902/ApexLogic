from __future__ import annotations

from core.run_config import setting
from core.cache import cached_search
import re
from typing import Any, Dict, List

try:
    from tavily import TavilyClient
except Exception:
    TavilyClient = None


HTTP_TIMEOUT = 12
DEFAULT_USER_AGENT = "ApexLogicResearchBot/0.2"


def _trim_text(text: str, limit: int = 2000) -> str:
    """清理并截断文本，避免超长噪声内容进入上下文。"""

    normalized = re.sub(r"\s+", " ", (text or "").strip())
    return normalized[:limit]


def _normalize_result(item: Dict[str, Any]) -> Dict[str, Any]:
    """将 Tavily 原始结果标准化为统一结构。"""

    # 优先使用 raw_content，避免仅依赖短摘要导致信息过浅。
    raw = str(item.get("raw_content", "") or "").strip()
    short = str(item.get("content", "") or "").strip()
    content = raw if raw else short

    return {
        "title": str(item.get("title", "")),
        "url": str(item.get("url", "")),
        "source": "tavily",
        "content": _trim_text(content, 4000),
    }


def _normalize_generic(source: str, title: str, url: str, content: str) -> Dict[str, Any]:
    """统一不同 provider 的上下文结构。"""

    return {
        "title": _trim_text(title, 200),
        "url": url.strip(),
        "source": source,
        "content": _trim_text(content, 2200),
    }


@cached_search("duckduckgo:v1")
def duckduckgo_search(query: str, max_results: int = 4) -> List[Dict[str, Any]]:
    """DuckDuckGo 免费网页搜索（无需 API Key）。"""

    ddgs_cls = None
    try:
        ddgs_module = __import__("ddgs", fromlist=["DDGS"])
        ddgs_cls = getattr(ddgs_module, "DDGS", None)
    except Exception as exc:
        raise RuntimeError("未检测到 ddgs 依赖，请执行 `pip install ddgs`。") from exc

    if ddgs_cls is None:
        raise RuntimeError("ddgs 模块可用但未找到 DDGS 类。")

    results: List[Dict[str, Any]] = []
    with ddgs_cls() as ddgs:
        for item in ddgs.text(query, max_results=max_results):
            title = str(item.get("title", ""))
            url = str(item.get("href", ""))
            body = str(item.get("body", ""))
            if not (title or body):
                continue
            results.append(_normalize_generic("duckduckgo", title, url, body))
    return results


@cached_search("tavily:advanced:raw:v1")
def tavily_search(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """执行 Tavily 检索并返回标准化结果。

    若未安装 SDK 或未配置 API Key，会抛出 RuntimeError，
    由上层 Agent 决定降级策略。
    """

    if TavilyClient is None:
        raise RuntimeError("Tavily SDK 未安装，请先安装 tavily-python。")

    api_key = setting("TAVILY_API_KEY", "")
    if not api_key:
        raise RuntimeError("未检测到 TAVILY_API_KEY。")

    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=query,
        max_results=max_results,
        include_raw_content=True,
        search_depth="advanced",
    )

    return [_normalize_result(item) for item in response.get("results", [])]


def _dedupe_results(items: List[Dict[str, Any]], max_results: int) -> List[Dict[str, Any]]:
    """按 url/title 去重并裁剪结果数量。"""

    unique: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        key = (str(item.get("url", "")).strip().lower() or str(item.get("title", "")).strip().lower())
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(item)
        if len(unique) >= max_results:
            break
    return unique


def unified_search(query: str, max_results: int = 8) -> List[Dict[str, Any]]:
    """统一搜索入口：DDG 主搜 + Tavily 可选补充。"""

    provider_errors: List[str] = []
    merged: List[Dict[str, Any]] = []

    providers = [(duckduckgo_search, max_results)]

    for func, provider_max in providers:
        try:
            merged.extend(func(query, max_results=provider_max))
        except Exception as exc:
            provider_errors.append(f"{func.__name__} failed: {exc}")

    use_tavily = setting("ENABLE_TAVILY_FALLBACK", "0").strip() == "1"
    if use_tavily:
        try:
            merged.extend(tavily_search(query, max_results=max(2, max_results // 3)))
        except Exception as exc:
            provider_errors.append(f"tavily_search failed: {exc}")

    deduped = _dedupe_results(merged, max_results=max_results)
    if deduped:
        return deduped

    # 所有 provider 均失败时抛错，让上层进入既有降级逻辑并记录 errors。
    raise RuntimeError("; ".join(provider_errors) or "no search result returned")


