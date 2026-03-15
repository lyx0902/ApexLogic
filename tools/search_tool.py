from __future__ import annotations

import os
from typing import Any, Dict, List

try:
    from tavily import TavilyClient
except Exception:
    TavilyClient = None


def _normalize_result(item: Dict[str, Any]) -> Dict[str, Any]:
    """将 Tavily 原始结果标准化为统一结构。"""

    return {
        "title": item.get("title", ""),
        "url": item.get("url", ""),
        "source": "tavily",
        "content": item.get("content", "")[:1500],
    }


def tavily_search(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """执行 Tavily 检索并返回标准化结果。

    若未安装 SDK 或未配置 API Key，会抛出 RuntimeError，
    由上层 Agent 决定降级策略。
    """

    if TavilyClient is None:
        raise RuntimeError("Tavily SDK 未安装，请先安装 tavily-python。")

    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        raise RuntimeError("未检测到 TAVILY_API_KEY。")

    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=query,
        max_results=max_results,
        include_raw_content=False,
        search_depth="advanced",
    )

    return [_normalize_result(item) for item in response.get("results", [])]

