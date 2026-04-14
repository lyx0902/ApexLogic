"""IRCoT 风格迭代检索-推理优化器。

核心思想
--------
在初始广搜之后，迭代执行「推理 → 识别缺口 → 补搜」循环，
填补多跳问答场景中的信息链断裂，提升最终进入 BGE 的候选上下文质量。

每跳流程：
  1. 对当前所有上下文生成推理链摘要（"已知什么？"）
  2. 识别尚未被覆盖的关键信息缺口
  3. 将缺口转化为精准的补充搜索查询（gap_queries）
  4. 执行 DuckDuckGo 补搜，收集新原始上下文
  重复至达到 max_hops 或无新缺口为止。

与现有模块的关系
---------------
- 在 graph_expand（概念图扩展）之后运行，进一步填补语义缺口
- 补搜结果在 researchers.py 中与 normalized_contexts 合并，再送入 BGE 两阶段过滤
- MAB 负责初始广搜的预算分配；IRCoT 的补搜独立于 MAB，按固定小额查询

新增环境变量
-----------
ITERATIVE_RETRIEVAL_ENABLED  (默认 "1")  – 设为 "0" 可完全禁用
MAX_HOPS                     (默认 "2")  – 最大推理跳数
GAP_QUERIES_PER_HOP          (默认 "2")  – 每跳最多发出的补充查询数
GAP_RESULTS_PER_QUERY        (默认 "4")  – 每条补充查询的 DDG 结果数

"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from langchain_openai import ChatOpenAI
    _LANGCHAIN_AVAILABLE = True
except Exception:
    ChatOpenAI = None  # type: ignore
    _LANGCHAIN_AVAILABLE = False

try:
    from tools.search_tool import duckduckgo_search
    _DDG_AVAILABLE = True
except Exception:
    duckduckgo_search = None  # type: ignore
    _DDG_AVAILABLE = False

try:
    from prompts.system_prompts import (
        ITERATIVE_REASONING_SYSTEM_PROMPT,
        build_iterative_reasoning_prompt,
    )
    _PROMPTS_AVAILABLE = True
except Exception:
    ITERATIVE_REASONING_SYSTEM_PROMPT = ""  # type: ignore
    build_iterative_reasoning_prompt = None  # type: ignore
    _PROMPTS_AVAILABLE = False


# ── 解析辅助 ──────────────────────────────────────────────────────────

def _extract_json_block(text: str) -> str:
    """从 LLM 输出中提取 JSON 块（兼容 ```json ... ``` 包裹）。"""
    s = text.strip()
    # 去除 fenced block
    if s.startswith("```"):
        lines = s.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
        if s.lower().startswith("json"):
            s = s[4:].strip()
    # 若包含多余前后文，提取第一个 { ... }
    start = s.find("{")
    if start < 0:
        return s
    depth = 0
    in_str = escaped = False
    for i in range(start, len(s)):
        c = s[i]
        if escaped:
            escaped = False
            continue
        if c == "\\":
            escaped = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start: i + 1]
    return s[start:]


def _sanitize_json(text: str) -> str:
    """轻量修复常见 JSON 格式问题（尾逗号、字符串内控制字符）。"""
    cleaned = text.strip().lstrip("\ufeff")
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    # 转义字符串内的裸换行
    result: List[str] = []
    in_str = escaped = False
    for ch in cleaned:
        if escaped:
            result.append(ch)
            escaped = False
        elif ch == "\\":
            result.append(ch)
            escaped = True
        elif ch == '"':
            result.append(ch)
            in_str = not in_str
        elif in_str and ch == "\n":
            result.append("\\n")
        elif in_str and ch == "\r":
            result.append("\\r")
        elif in_str and ch == "\t":
            result.append("\\t")
        else:
            result.append(ch)
    return "".join(result)


def _parse_llm_response(content: str) -> Tuple[str, List[str]]:
    """解析 LLM 输出，提取 (reasoning, gap_queries)。

    策略（依优先级）：
    1. JSON 解析：{"reasoning": "...", "gap_queries": ["...", ...]}
    2. 正则回退：从文本中抽取行列表
    """
    raw = _extract_json_block(content)
    for attempt in (raw, _sanitize_json(raw)):
        if not attempt:
            continue
        try:
            obj = json.loads(attempt)
            if isinstance(obj, dict):
                reasoning = str(obj.get("reasoning", "")).strip()
                raw_queries = obj.get("gap_queries", [])
                if isinstance(raw_queries, list):
                    queries = [str(q).strip() for q in raw_queries if str(q).strip()]
                    return reasoning, queries
        except Exception:
            pass

    # 正则回退：每行一条查询（去除序号、破折号等前缀）
    lines = [
        l.strip().lstrip("-•*0123456789. ").strip()
        for l in content.splitlines()
        if l.strip() and not l.strip().startswith("{") and not l.strip().startswith("}")
    ]
    queries = [l for l in lines if 4 < len(l) < 200]
    return content[:300], queries


# ── 主类 ──────────────────────────────────────────────────────────────

class IterativeRetrievalOptimizer:
    """IRCoT 风格迭代检索-推理优化器。

    每跳通过 LLM 从当前上下文生成推理链，识别信息缺口，
    针对缺口生成补充搜索查询，执行 DuckDuckGo 补搜。
    结果由调用方（researchers.py）负责去重归一化后送入 BGE 流水线。

    参数
    ----
    max_hops          : 最大跳数（受 MAX_HOPS 环境变量覆盖）
    gap_queries_per_hop: 每跳最多发出的补充查询数（受 GAP_QUERIES_PER_HOP 覆盖）
    results_per_query : 每条补充查询的 DDG 结果数（受 GAP_RESULTS_PER_QUERY 覆盖）
    """

    def __init__(
        self,
        max_hops: int = 2,
        gap_queries_per_hop: int = 2,
        results_per_query: int = 4,
    ) -> None:
        # 环境变量优先覆盖构造参数
        self.max_hops = int(os.getenv("MAX_HOPS", str(max_hops)))
        self.gap_queries_per_hop = int(os.getenv("GAP_QUERIES_PER_HOP", str(gap_queries_per_hop)))
        self.results_per_query = int(os.getenv("GAP_RESULTS_PER_QUERY", str(results_per_query)))

    # ── 内部：LLM 实例化 ──────────────────────────────────────────────

    def _make_llm(self) -> Optional[Any]:
        """从环境变量构建 DeepSeek LLM 实例。失败返回 None。"""
        if not _LANGCHAIN_AVAILABLE or ChatOpenAI is None:
            return None
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            return None
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        try:
            return ChatOpenAI(
                model=model,
                api_key=lambda: api_key,
                base_url=base_url,
                temperature=0.2,
            )
        except Exception:
            return None

    # ── 内部：单跳推理-缺口生成 ───────────────────────────────────────

    def _call_reasoning_llm(
        self,
        llm: Any,
        topic: str,
        contexts: List[Dict[str, Any]],
        hop: int,
        existing_reasoning: str,
    ) -> Tuple[str, List[str]]:
        """调用 LLM 生成推理链和缺口查询列表。

        返回 (reasoning_text, gap_queries_list)
        失败时返回 ("", [])
        """
        if not _PROMPTS_AVAILABLE or build_iterative_reasoning_prompt is None:
            return "", []

        system_prompt = ITERATIVE_REASONING_SYSTEM_PROMPT
        user_prompt = build_iterative_reasoning_prompt(
            topic=topic,
            contexts=contexts,
            hop=hop,
            max_gap_queries=self.gap_queries_per_hop,
            existing_reasoning=existing_reasoning,
        )

        try:
            response = llm.invoke(
                [
                    ("system", system_prompt),
                    ("human", user_prompt),
                ]
            )
            content_obj = getattr(response, "content", "")
            if isinstance(content_obj, list):
                content = "\n".join(
                    str(item.get("text", "")) if isinstance(item, dict) else str(item)
                    for item in content_obj
                ).strip()
            else:
                content = str(content_obj or "").strip()

            if not content:
                return "", []

            return _parse_llm_response(content)

        except Exception:
            return "", []

    # ── 内部：单条 gap 查询的 DDG 搜索 ──────────────────────────────

    def _search_gap(self, query: str) -> List[Dict[str, Any]]:
        """对单条 gap 查询执行 DuckDuckGo 搜索，返回原始结果列表。"""
        if not _DDG_AVAILABLE or duckduckgo_search is None:
            return []
        try:
            results = duckduckgo_search(query, max_results=self.results_per_query)
            return [r for r in (results or []) if isinstance(r, dict)]
        except Exception:
            return []

    # ── 公开接口 ──────────────────────────────────────────────────────

    def run(
        self,
        topic: str,
        contexts: List[Dict[str, Any]],
        existing_queries: Optional[List[str]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
        """执行迭代检索-推理循环。

        参数
        ----
        topic           : 研究主题
        contexts        : 已归一化的初始上下文列表（含 core_summary 字段）
        existing_queries: 已有查询，用于避免缺口查询与已有查询高度重叠

        返回
        ----
        gap_contexts     : 所有跳中新增的原始文档列表（供调用方合并去重）
        reasoning_chains : 每跳的推理链文本列表
        hop_summaries    : 每跳的统计摘要列表（含 gap_queries, new_contexts 等）
        """
        # 检查前置条件
        if not contexts:
            return [], [], []

        llm = self._make_llm()
        if llm is None:
            return [], [], [{"status": "skipped", "reason": "llm_unavailable"}]

        existing_set: set = set(q.lower() for q in (existing_queries or []))
        all_gap_contexts: List[Dict[str, Any]] = []
        reasoning_chains: List[str] = []
        hop_summaries: List[Dict[str, Any]] = []

        # 用于传入 LLM 的工作上下文（每跳追加新文档的轻量副本）
        working_contexts: List[Dict[str, Any]] = list(contexts)
        existing_reasoning: str = ""

        for hop in range(self.max_hops):
            # 1. 生成推理链 + 缺口查询
            reasoning, gap_queries = self._call_reasoning_llm(
                llm=llm,
                topic=topic,
                contexts=working_contexts,
                hop=hop,
                existing_reasoning=existing_reasoning,
            )

            if reasoning:
                reasoning_chains.append(reasoning)
                existing_reasoning = reasoning

            # 去除与已有查询高度重叠的 gap_queries
            filtered_queries: List[str] = []
            for q in gap_queries:
                q_lower = q.lower()
                if any(q_lower in eq or eq in q_lower for eq in existing_set):
                    continue
                filtered_queries.append(q)
                existing_set.add(q_lower)

            filtered_queries = filtered_queries[: self.gap_queries_per_hop]

            if not filtered_queries:
                hop_summaries.append({
                    "hop": hop + 1,
                    "status": "no_new_gaps",
                    "reasoning_preview": (reasoning or "")[:150],
                    "reasoning_full": (reasoning or ""),
                    "gap_queries": [],
                    "new_contexts": 0,
                    "retrieved_docs": [],
                })
                break

            # 2. 执行补搜
            hop_new_contexts: List[Dict[str, Any]] = []
            for q in filtered_queries:
                results = self._search_gap(q)
                hop_new_contexts.extend(results)

            all_gap_contexts.extend(hop_new_contexts)

            hop_summaries.append({
                "hop": hop + 1,
                "status": "searched",
                "reasoning_preview": (reasoning or "")[:150],
                "reasoning_full": (reasoning or ""),
                "gap_queries": filtered_queries,
                "new_contexts": len(hop_new_contexts),
                "retrieved_docs": [
                    {"title": r.get("title", ""), "url": r.get("url", "")}
                    for r in hop_new_contexts
                ],
            })

            # 3. 更新 working_contexts（为下一跳提供更完整的上下文）
            if hop_new_contexts:
                working_contexts = list(working_contexts) + [
                    {
                        "title": r.get("title", ""),
                        "core_summary": (r.get("content", "") or "")[:400],
                        "content": r.get("content", ""),
                        "source": r.get("source", "duckduckgo"),
                        "url": r.get("url", ""),
                    }
                    for r in hop_new_contexts
                ]

        return all_gap_contexts, reasoning_chains, hop_summaries
