from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class ResearchStateCore(TypedDict):
    """研究流程的必填核心状态。"""

    topic: str
    search_queries: List[str]
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

    # Reviewer 给出的下一跳建议：end / researcher / writer
    next_route: str
    # 是否判定为“信息不足”，用于区分回到 Researcher 还是 Writer
    needs_more_research: bool
    # 最终报告（通常与 draft 一致，或为后处理结果）
    final_report: str
    # 运行期错误与告警信息
    errors: List[str]
    # Reviewer 的结构化评审结果
    review_result: Dict[str, Any]
    # 节点执行轨迹，便于观察每轮决策
    execution_trace: List[Dict[str, Any]]


def create_initial_state(topic: str) -> ResearchState:
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
        "execution_trace": [],
    }


def merge_state(base: ResearchState, patch: Dict[str, Any]) -> ResearchState:
    """合并状态更新片段，返回新的状态字典。

    该函数便于在节点实现中保持“输入不可变、输出新状态”风格。
    """

    merged: ResearchState = dict(base)
    merged.update(patch)
    return merged

