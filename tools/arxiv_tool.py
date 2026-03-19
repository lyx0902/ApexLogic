from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import ssl
from pathlib import Path
from typing import Any, Dict, List

import certifi


ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

# 通用中英意图词映射，避免绑定到具体主题。
BUILTIN_SYNONYMS: Dict[str, List[str]] = {
    "比较": ["comparison", "versus"],
    "对比": ["comparison", "vs"],
    "趋势": ["trend", "outlook"],
    "挑战": ["challenge", "bottleneck"],
    "风险": ["risk", "limitation"],
    "应用": ["application", "use case"],
    "案例": ["case study", "empirical study"],
    "方法": ["method", "approach"],
    "评估": ["evaluation", "benchmark"],
    "性能": ["performance", "efficiency"],
    "优化": ["optimization", "improvement"],
    "综述": ["survey", "review"],
    "系统": ["system", "framework"],
    "架构": ["architecture", "design"],
}


def _load_custom_synonyms() -> Dict[str, List[str]]:
    """从 JSON 文件加载可选同义词。"""

    file_path = (os.getenv("ARXIV_SYNONYM_FILE", "") or "").strip()
    if not file_path:
        return {}

    path = Path(file_path)
    if not path.exists():
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if not isinstance(raw, dict):
        return {}

    parsed: Dict[str, List[str]] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, str):
            parsed[key] = [value]
            continue
        if isinstance(value, list):
            parsed[key] = [str(x) for x in value if str(x).strip()]
    return parsed


def _collect_terms(query: str) -> List[str]:
    """抽取中英文 token，供回退查询使用。"""

    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-_/]*|[\u4e00-\u9fff]{2,}", (query or ""))
    unique: List[str] = []
    seen: set[str] = set()
    for token in tokens:
        key = token.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(token.strip())
    return unique[:12]


def _build_fallback_queries(query: str) -> List[str]:
    """构建普适的多轮 ArXiv 查询候选。"""

    raw = (query or "").strip()
    if not raw:
        return []

    terms = _collect_terms(raw)
    queries: List[str] = [raw]

    if terms:
        queries.append(" ".join(terms[:8]))

    synonyms = dict(BUILTIN_SYNONYMS)
    synonyms.update(_load_custom_synonyms())

    expanded_terms: List[str] = []
    for term in terms[:8]:
        expanded_terms.append(term)
        for syn in synonyms.get(term, [])[:2]:
            expanded_terms.append(syn)
    if expanded_terms:
        queries.append(" ".join(expanded_terms[:12]))

    # 仅英文 token 的回退查询，适配混合中英输入。
    alpha_terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-_/]*", raw)
    if alpha_terms:
        queries.append(" ".join(alpha_terms[:8]))

    unique: List[str] = []
    seen: set[str] = set()
    for item in queries:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(item.strip())
    return unique[:6]


def _fetch_arxiv_once(search_text: str, limit: int) -> List[Dict[str, Any]]:
    """单次请求 ArXiv API。"""

    params = {
        "search_query": f"all:{search_text}",
        "start": 0,
        "max_results": max(limit, 1),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"

    ssl_context = ssl.create_default_context(cafile=certifi.where())
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ApexLogicResearchBot/0.2 (+https://export.arxiv.org/)"},
    )

    with urllib.request.urlopen(request, timeout=15, context=ssl_context) as response:
        xml_data = response.read()

    root = ET.fromstring(xml_data)
    entries = root.findall("atom:entry", ATOM_NS)

    results: List[Dict[str, Any]] = []
    for entry in entries:
        title = (entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").strip()
        summary = (
            entry.findtext("atom:summary", default="", namespaces=ATOM_NS) or ""
        ).strip()
        link = ""
        for link_node in entry.findall("atom:link", ATOM_NS):
            href = link_node.attrib.get("href", "")
            rel = link_node.attrib.get("rel", "")
            if href and rel in {"alternate", ""}:
                link = href
                break

        results.append(
            {
                "title": title,
                "url": link,
                "source": "arxiv",
                "content": summary[:1500],
            }
        )
    return results


def arxiv_search(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """查询 ArXiv 并返回标准化结果。"""

    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for search_text in _build_fallback_queries(query):
        remaining = max_results - len(merged)
        if remaining <= 0:
            break

        # 防止首个查询抢占全部配额，保留后续回退机会。
        batch = _fetch_arxiv_once(search_text, min(max(remaining, 1), 8))
        for item in batch:
            key = (
                str(item.get("url", "")).strip().lower()
                or str(item.get("title", "")).strip().lower()
            )
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(item)
            if len(merged) >= max_results:
                break

    return merged[:max_results]


