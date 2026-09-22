"""自适应查询分解器（Adaptive Query Decomposition, AQD）

核心思想
--------
将复杂研究主题结构化分解为 N 个带依赖关系的子问题，
按拓扑顺序逐一检索，确保多跳推理场景中的系统性覆盖。

执行流程：
  1. LLM 将研究主题（结合初始上下文）分解为 N 个子问题，每个子问题包含：
     - question   : 子问题描述
     - search_query: 可直接用于搜索引擎的查询字符串
     - depends_on  : 依赖的其他子问题 ID 列表（构成 DAG）
  2. 拓扑排序（Kahn 算法），按依赖关系确定执行顺序
  3. 逐子问题执行 DuckDuckGo 补搜，跳过与已有查询高度重叠的子问题
  4. 返回所有新检索文档供调用方合并去重

与现有模块的关系
---------------
- 在 graph_expand（概念图扩展）之后、IRCoT 之前运行
- AQD 是主动式规划（proactive）：提前分解主题，系统性覆盖各子话题
- IRCoT 是被动式缺口填补（reactive）：在现有上下文中发现缺口再补搜
- 两者互补：AQD 确保广度，IRCoT 确保深度

新增环境变量
-----------
AQD_ENABLED           (默认 "1") – 设为 "0" 可完全禁用
AQD_MAX_SUB_QUESTIONS (默认 "4") – 最大子问题数量
AQD_RESULTS_PER_SUBQ  (默认 "3") – 每个子问题的 DDG 检索结果数
"""

from __future__ import annotations
from optim.query_text import strip_list_marker

import json
from core.run_config import setting
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
        AQD_DECOMPOSE_SYSTEM_PROMPT,
        build_aqd_decompose_prompt,
    )
    _PROMPTS_AVAILABLE = True
except Exception:
    AQD_DECOMPOSE_SYSTEM_PROMPT = ""  # type: ignore
    build_aqd_decompose_prompt = None  # type: ignore
    _PROMPTS_AVAILABLE = False


# ── 拓扑排序辅助 ───────────────────────────────────────────────────────

def _topological_sort(sub_questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """对子问题列表按依赖关系做拓扑排序（Kahn 算法）。

    若检测到循环依赖或 ID 不一致，则安全回退到原始顺序。
    """
    if not sub_questions:
        return []

    all_ids = {sq["id"] for sq in sub_questions}
    in_degree: Dict[int, int] = {sq["id"]: 0 for sq in sub_questions}

    # 计算入度（仅统计指向已知 ID 的依赖）
    for sq in sub_questions:
        for dep_id in sq.get("depends_on", []):
            if dep_id in all_ids and dep_id != sq["id"]:
                in_degree[sq["id"]] += 1

    queue = [sq for sq in sub_questions if in_degree[sq["id"]] == 0]
    sorted_result: List[Dict[str, Any]] = []

    while queue:
        node = queue.pop(0)
        sorted_result.append(node)
        for sq in sub_questions:
            if node["id"] in sq.get("depends_on", []):
                in_degree[sq["id"]] -= 1
                if in_degree[sq["id"]] == 0:
                    queue.append(sq)

    # 若存在循环依赖（排序未覆盖全部节点），回退原始顺序
    if len(sorted_result) != len(sub_questions):
        return list(sub_questions)

    return sorted_result


# ── JSON 解析辅助 ──────────────────────────────────────────────────────

def _extract_json_block(text: str) -> str:
    """从 LLM 输出中提取 JSON 块（兼容 ```json ... ``` 包裹）。"""
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
        if s.lower().startswith("json"):
            s = s[4:].strip()
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


def _parse_decompose_response(
    content: str, n_expected: int, *, strict: bool = False
) -> List[Dict[str, Any]]:
    """解析 LLM 分解输出，提取子问题列表。

    优先 JSON 解析；失败时正则回退，将每行视为一个子问题查询。
    """
    raw = _extract_json_block(content)
    for attempt in (raw, _sanitize_json(raw)):
        if not attempt:
            continue
        try:
            obj = json.loads(attempt)
            if isinstance(obj, dict):
                sub_qs = obj.get("sub_questions", [])
                if isinstance(sub_qs, list) and sub_qs:
                    if strict:
                        if len(sub_qs) > n_expected or any(
                            not isinstance(q, dict) or type(q.get("id")) is not int
                            or any(not isinstance(q.get(k), str) or not q[k].strip()
                                   for k in ("question", "search_query"))
                            or not isinstance(q.get("depends_on"), list)
                            or any(type(d) is not int for d in q["depends_on"])
                            for q in sub_qs
                        ):
                            return []
                        ids = {q["id"] for q in sub_qs}
                        if len(ids) != len(sub_qs) or any(
                            d not in ids or d == q["id"] for q in sub_qs for d in q["depends_on"]
                        ):
                            return []
                        return sub_qs
                    validated: List[Dict[str, Any]] = []
                    for i, sq in enumerate(sub_qs, start=1):
                        if not isinstance(sq, dict):
                            continue
                        validated.append({
                            "id": int(sq.get("id", i)),
                            "question": str(sq.get("question", f"子问题{i}")),
                            "search_query": str(
                                sq.get("search_query", sq.get("question", f"子问题{i}"))
                            ),
                            "depends_on": [
                                int(d)
                                for d in sq.get("depends_on", [])
                                if str(d).strip().lstrip("-").isdigit()
                            ],
                        })
                    if validated:
                        return validated[:n_expected]
        except Exception:
            pass

    if strict:
        return []  # A lossy fallback must never justify suppressing searches.
    # 正则回退：每行视为一个子问题查询
    lines = [
        strip_list_marker(line)
        for line in content.splitlines()
        if line.strip()
        and not line.strip().startswith("{")
        and not line.strip().startswith("}")
    ]
    queries = [line for line in lines if 4 < len(line) < 200][:n_expected]
    return [
        {"id": i + 1, "question": q, "search_query": q, "depends_on": []}
        for i, q in enumerate(queries)
    ]


# ── 主类 ──────────────────────────────────────────────────────────────

class AdaptiveQueryPlanner:
    """自适应查询分解器（AQD）。

    将复杂研究主题分解为带依赖关系的子问题 DAG，
    按拓扑顺序逐一检索，确保多跳场景下的系统性主题覆盖。

    参数
    ----
    max_sub_questions : 最大子问题数量（受 AQD_MAX_SUB_QUESTIONS 环境变量覆盖）
    results_per_subq  : 每个子问题的 DDG 检索结果数（受 AQD_RESULTS_PER_SUBQ 覆盖）
    """

    def __init__(
        self,
        max_sub_questions: int = 4,
        results_per_subq: int = 3,
    ) -> None:
        self.max_sub_questions = int(
            setting("AQD_MAX_SUB_QUESTIONS", str(max_sub_questions))
        )
        self.results_per_subq = int(
            setting("AQD_RESULTS_PER_SUBQ", str(results_per_subq))
        )

    # ── 内部：LLM 实例化 ──────────────────────────────────────────────

    def _make_llm(self) -> Optional[Any]:
        """从环境变量构建 DeepSeek LLM 实例。失败返回 None。"""
        if not _LANGCHAIN_AVAILABLE or ChatOpenAI is None:
            return None
        api_key = setting("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            return None
        base_url = setting("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        model = setting("DEEPSEEK_MODEL", "deepseek-chat")
        try:
            return ChatOpenAI(
                model=model,
                api_key=lambda: api_key,
                base_url=base_url,
                temperature=0.3,
            )
        except Exception:
            return None

    # ── 内部：LLM 分解调用 ────────────────────────────────────────────

    def _call_decompose_llm(
        self,
        llm: Any,
        topic: str,
        contexts: List[Dict[str, Any]],
        existing_queries: List[str],
        pre_search: bool = False,
    ) -> List[Dict[str, Any]]:
        """调用 LLM 将研究主题分解为子问题列表。

        返回 List[{id, question, search_query, depends_on}]，失败时返回 []。
        """
        if not _PROMPTS_AVAILABLE or build_aqd_decompose_prompt is None:
            return []

        user_prompt = build_aqd_decompose_prompt(
            topic=topic,
            contexts=contexts,
            n_sub_questions=self.max_sub_questions,
            existing_queries=existing_queries,
        )
        if pre_search:
            user_prompt += ("\n这是联网前规划：子问题合起来必须覆盖用户请求的全部要求，尤其不能遗漏示例、"
                            "比较维度或限制条件。不要为凑数量引入用户未要求的研究方向。"
                            "每个问题写明实体和范围；未知实体不得猜测，应保留真实的事实依赖。"
                            "用户预设不一定正确，可以把核实或纠正预设列为子问题。")

        try:
            response = llm.invoke(
                [
                    ("system", AQD_DECOMPOSE_SYSTEM_PROMPT),
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
                return []

            return _parse_decompose_response(content, self.max_sub_questions, strict=pre_search)

        except Exception:
            return []

    # ── 内部：单子问题 DDG 检索 ───────────────────────────────────────

    def _search_subq(self, search_query: str) -> List[Dict[str, Any]]:
        """对单个子问题执行 DuckDuckGo 搜索，返回原始结果列表。"""
        if not _DDG_AVAILABLE or duckduckgo_search is None:
            return []
        try:
            results = duckduckgo_search(search_query, max_results=self.results_per_subq)
            return [r for r in (results or []) if isinstance(r, dict)]
        except Exception:
            return []

    # ── 公开接口 ──────────────────────────────────────────────────────

    def plan_before_search(self, topic: str) -> List[Dict[str, Any]]:
        """Plan self-contained subquestions without requiring an initial web search."""
        llm = self._make_llm()
        if llm is None:
            return []
        return _topological_sort(self._call_decompose_llm(llm, topic, [], [], pre_search=True))

    def run(
        self,
        topic: str,
        contexts: List[Dict[str, Any]],
        existing_queries: Optional[List[str]] = None,
        sub_questions: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """执行自适应查询分解与检索。

        参数
        ----
        topic           : 研究主题
        contexts        : 已归一化的初始上下文列表（含 core_summary，供 LLM 参考）
        existing_queries: 已有查询（避免生成重复子问题）

        返回
        ----
        gap_contexts : 所有子问题检索到的新原始文档列表（供调用方合并去重）
        plan_summary : 执行摘要，含子问题列表、执行顺序、每题新增文档数
                       plan_summary["enabled"] == False 时表示 AQD 未执行
        """
        if not contexts:
            return [], {"enabled": False, "reason": "no_initial_contexts"}

        existing_set = set(q.lower() for q in (existing_queries or []))

        # 1. 分解主题为子问题
        reused_plan = sub_questions is not None
        if sub_questions is None:
            llm = self._make_llm()
            if llm is None:
                return [], {"enabled": False, "reason": "llm_unavailable"}
            sub_questions = self._call_decompose_llm(
                llm=llm, topic=topic, contexts=contexts,
                existing_queries=existing_queries or [],
            )

        if not sub_questions:
            return [], {"enabled": False, "reason": "decompose_failed"}

        # 2. 拓扑排序：按依赖关系确定执行顺序
        ordered = _topological_sort(sub_questions)

        # 3. 按序逐子问题检索
        all_gap_contexts: List[Dict[str, Any]] = []
        sub_results: List[Dict[str, Any]] = []

        from optim.search_audit import search_with_audit
        for sq in ordered:
            sq_id = sq["id"]
            question = sq["question"]
            search_query = sq["search_query"]
            depends_on = sq.get("depends_on", [])

            # 跳过与已有查询高度重叠的搜索
            sq_lower = search_query.lower()
            is_dup = any(sq_lower in eq or eq in sq_lower for eq in existing_set)

            new_docs: List[Dict[str, Any]] = []
            records = []
            if not is_dup:
                # The legacy helper swallows failures; audit the provider directly here.
                if _DDG_AVAILABLE and duckduckgo_search is not None:
                    try:
                        new_docs = search_with_audit(records, "duckduckgo", search_query,
                                                     self.results_per_subq, duckduckgo_search)
                    except Exception:
                        new_docs = []
                all_gap_contexts.extend(new_docs)
                existing_set.add(sq_lower)

            sub_results.append({
                "id": sq_id,
                "question": question,
                "search_query": search_query,
                "depends_on": depends_on,
                "new_docs": len(new_docs),
                "skipped": is_dup,
                "search_unavailable": not is_dup and (not _DDG_AVAILABLE or duckduckgo_search is None),
                "search_records": records,
                "retrieved_docs": [d for r in records for d in r["retrieved_docs"]],
            })

        plan_summary: Dict[str, Any] = {
            "enabled": True,
            "reused_pre_search_plan": reused_plan,
            "sub_questions_count": len(sub_questions),
            "execution_order": [sq["id"] for sq in ordered],
            "total_new_docs": len(all_gap_contexts),
            "sub_results": sub_results,
        }

        return all_gap_contexts, plan_summary
