from __future__ import annotations

from typing import Any, Dict, List, Literal, TypedDict, cast


class ResearchStateCore(TypedDict):
    """研究流程的必填核心状态。"""

    topic: str
    search_queries: List[str]
    planned_search_queries: List[str]
    retrieved_context: List[Dict[str, Any] | str]
    draft: str
    critique_feedback: str
    revision_step: int
    is_satisfactory: bool


class ResearchState(ResearchStateCore, total=False):
    """可扩展的全局状态。

    说明：
    - 核心字段继承自 ResearchStateCore，必须存在。
    - 额外字段用于支持图路由、调试信息与最终输出，不影响核心流程。
    """

    answer_status: str
    run_id: str
    thread_id: str
    schema_version: int
    workflow_version: str
    run_config: Dict[str, Any]
    memory_query_ids: List[str]
    memory_used_ids: List[str]
    memory_stats: Dict[str, Any]
    memory_first: Dict[str, Any]
    cache_stats: Dict[str, Any]
    memory_publication: Dict[str, Any]
    memory_write_ids: List[str]

    # Reviewer 给出的下一跳建议：end / researcher / writer
    next_route: str
    # 是否判定为"信息不足"，用于区分回到 Researcher 还是 Writer
    needs_more_research: bool
    # 最终报告（通常与 draft 一致，或为后处理结果）
    final_report: str
    # 运行期错误与告警信息
    errors: List[str]
    # Reviewer 的结构化评审结果
    review_result: Dict[str, Any]
    # Reviewer 给 Writer 的结构化修订指令
    revision_directives: Dict[str, Any]
    # 节点执行轨迹，便于观察每轮决策
    execution_trace: List[Dict[str, Any]]
    # 每轮写作+评审快照，便于导出完整调试报告
    iteration_history: List[Dict[str, Any]]
    # Writer 对评审项的修订映射（must_fix -> 段落/章节）
    feedback_paragraph_mapping: List[Dict[str, Any]]
    # Researcher 生成的检索阶段摘要（当前用于 BGE pipeline 统计）
    source_quality_summary: Dict[str, Any]
    # 报告输出模式：user(面向读者) / debug(含系统细节) / eval(极简答题备忘录)
    output_mode: Literal["user", "debug", "both", "user_only", "eval"]
    # MAB（多臂赌博机）参数状态，跨迭代持久化，驱动自适应检索预算分配
    mab_state: Dict[str, Any]
    # IRCoT 迭代检索：每跳生成的推理链文本列表
    reasoning_chains: List[str]
    # IRCoT 迭代检索：统计摘要（跳数、新增上下文数、gap_queries 等）
    iterative_retrieval_summary: Dict[str, Any]
    # AQD 自适应查询分解：执行计划摘要（子问题列表、执行顺序、每题新增文档数）
    query_plan: Dict[str, Any]
    # IRCoT 推理链专属文档池（不经BGE筛选，完整保留推理链补搜结果）
    reasoning_contexts: List[Dict[str, Any]]
    # IRCoT 推理链结构化摘要（供Writer快速理解推理过程）
    reasoning_summary: str
    # 标记本轮是否启用了IRCoT推理链
    reasoning_enabled: bool
    # Reviewer 评审模式统计（llm_scored / coerce_fallback / rule / degraded_rounds）
    review_stats: Dict[str, Any]
    # 发生降级评审的轮次列表（revision_step），用于报告标注与调试
    review_degraded_rounds: List[int]


def create_initial_state(
    topic: str,
    output_mode: Literal["user", "debug", "both", "user_only", "eval"] = "debug",
) -> ResearchState:
    """创建满足核心约束的初始状态。"""

    return {
        "topic": topic,
        "search_queries": [],
        "retrieved_context": [],
        "draft": "",
        "critique_feedback": "",
        "revision_step": 0,
        "is_satisfactory": False,
        "next_route": "researcher",
        "needs_more_research": True,
        "errors": [],
        "review_result": {},
        "revision_directives": {},
        "execution_trace": [],
        "iteration_history": [],
        "feedback_paragraph_mapping": [],
        "source_quality_summary": {},
        "output_mode": output_mode,
        "mab_state": {},
        "reasoning_chains": [],
        "iterative_retrieval_summary": {},
        "query_plan": {},
        "reasoning_contexts": [],
        "reasoning_summary": "",
        "reasoning_enabled": False,
        "review_stats": {},
        "review_degraded_rounds": [],
    }


def merge_state(base: ResearchState, patch: Dict[str, Any]) -> ResearchState:
    """合并状态更新片段，返回新的状态字典。

    该函数便于在节点实现中保持"输入不可变、输出新状态"风格。
    """

    merged: ResearchState = cast(ResearchState, dict(base))
    merged.update(patch)
    return merged
