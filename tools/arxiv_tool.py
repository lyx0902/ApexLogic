from __future__ import annotations

import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Dict, List


ARXIV_API_URL = "http://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def arxiv_search(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """查询 ArXiv 并返回标准化结果。"""

    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
    }
    url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"

    with urllib.request.urlopen(url, timeout=12) as response:
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
            if href:
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

