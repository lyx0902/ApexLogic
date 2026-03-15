from __future__ import annotations

from typing import Any, Callable, Dict

try:
    from core.state import ResearchState, create_initial_state, merge_state
except ModuleNotFoundError:  # 兼容直接运行 `python core/graph.py`
    from state import ResearchState, create_initial_state, merge_state

try:
    from langgraph.graph import END, START, StateGraph

    _LANGGRAPH_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - 仅用于缺依赖时降级提示
    END = "__END__"
    START = "__START__"
    StateGraph = None
    _LANGGRAPH_IMPORT_ERROR = exc


MAX_REVISIONS = 3

NODE_RESEARCHER = "researcher"
NODE_WRITER = "writer"
NODE_REVIEWER = "reviewer"

ROUTE_END = "end"
ROUTE_RESEARCHER = NODE_RESEARCHER
ROUTE_WRITER = NODE_WRITER


def _load_agent_node(module_path: str, attr_name: str, fallback: Callable[[ResearchState], Dict[str, Any]]) -> Callable[[ResearchState], Dict[str, Any]]:
    """优先加载 agents 目录中的真实节点，失败时回退到本地占位实现。"""

    try:
        module = __import__(module_path, fromlist=[attr_name])
        candidate = getattr(module, attr_name, None)
        if callable(candidate):
            return candidate
    except Exception:
        pass
    return fallback


def _researcher_fallback(state: ResearchState) -> Dict[str, Any]:
    """占位检索节点：生成查询词并补充上下文样例。"""

    topic = state["topic"]
    feedback = state.get("critique_feedback", "")
    queries = list(state.get("search_queries", []))
    context = list(state.get("retrieved_context", []))

    if not queries:
        queries.extend([f"{topic} 最新进展", f"{topic} 关键挑战", f"{topic} 行业案例"])

    if feedback:
        queries.append(f"补充检索: {feedback[:40]}")

    # 用结构化片段模拟检索结果，便于后续 Writer/Reviewer 消费
    context.append(
        {
            "source": "dummy_search",
            "title": f"关于 {topic} 的参考资料 {len(context) + 1}",
            "content": "该内容为占位检索结果，用于连通第一阶段工作流。",
        }
    )

    return {
        "search_queries": queries,
        "retrieved_context": context,
    }


def _writer_fallback(state: ResearchState) -> Dict[str, Any]:
    """占位写作节点：基于上下文生成或修订草稿。"""

    topic = state["topic"]
    revision_step = state.get("revision_step", 0)
    context = state.get("retrieved_context", [])
    feedback = state.get("critique_feedback", "")

    draft = (
        f"# {topic} 研究草稿\n\n"
        f"- 当前迭代轮次: {revision_step}\n"
        f"- 参考上下文数量: {len(context)}\n"
        f"- 修订依据: {feedback or '首轮写作，无反馈'}\n\n"
        "本段为第一阶段占位文本，后续可替换为真实 LLM 生成逻辑，"
        "并结合工具引用输出完整研究报告。"
    )

    return {
        "draft": draft,
    }


def _reviewer_fallback(state: ResearchState) -> Dict[str, Any]:
    """占位评审节点：给出满意度与下一跳建议。"""

    context_len = len(state.get("retrieved_context", []))
    draft_len = len(state.get("draft", ""))
    revision_step = state.get("revision_step", 0) + 1

    if context_len < 2:
        return {
            "revision_step": revision_step,
            "is_satisfactory": False,
            "needs_more_research": True,
            "next_route": ROUTE_RESEARCHER,
            "critique_feedback": "信息来源不足，请补充至少两个高质量参考来源。",
        }

    if draft_len < 120:
        return {
            "revision_step": revision_step,
            "is_satisfactory": False,
            "needs_more_research": False,
            "next_route": ROUTE_WRITER,
            "critique_feedback": "草稿过短，论证深度不足，请扩展方法与结论部分。",
        }

    return {
        "revision_step": revision_step,
        "is_satisfactory": True,
        "needs_more_research": False,
        "next_route": ROUTE_END,
        "critique_feedback": "当前版本通过基础评审。",
        "final_report": state.get("draft", ""),
    }


def reviewer_route(state: ResearchState, max_revisions: int = MAX_REVISIONS) -> str:
    """Reviewer 条件路由函数。"""

    if state.get("is_satisfactory", False):
        return ROUTE_END

    if state.get("revision_step", 0) >= max_revisions:
        return ROUTE_END

    next_route = state.get("next_route", "")
    if next_route in {ROUTE_RESEARCHER, ROUTE_WRITER}:
        return next_route

    if state.get("needs_more_research", False):
        return ROUTE_RESEARCHER
    return ROUTE_WRITER


def compile_graph(max_revisions: int = MAX_REVISIONS):
    """构建并编译循环状态图。"""

    if StateGraph is None:
        raise RuntimeError(
            "未检测到 langgraph，请先安装依赖: pip install langgraph。"
        ) from _LANGGRAPH_IMPORT_ERROR

    researcher_node = _load_agent_node(
        "agents.researchers", "researcher_node", _researcher_fallback
    )
    writer_node = _load_agent_node("agents.writer", "writer_node", _writer_fallback)
    reviewer_node = _load_agent_node(
        "agents.reviewer", "reviewer_node", _reviewer_fallback
    )

    graph = StateGraph(ResearchState)
    graph.add_node(NODE_RESEARCHER, researcher_node)
    graph.add_node(NODE_WRITER, writer_node)
    graph.add_node(NODE_REVIEWER, reviewer_node)

    graph.add_edge(START, NODE_RESEARCHER)
    graph.add_edge(NODE_RESEARCHER, NODE_WRITER)
    graph.add_edge(NODE_WRITER, NODE_REVIEWER)

    graph.add_conditional_edges(
        NODE_REVIEWER,
        lambda state: reviewer_route(state, max_revisions=max_revisions),
        {
            ROUTE_END: END,
            ROUTE_RESEARCHER: NODE_RESEARCHER,
            ROUTE_WRITER: NODE_WRITER,
        },
    )

    return graph.compile()


def run_smoke_test(topic: str = "多智能体系统中的反思机制") -> ResearchState:
    """运行一次最小闭环测试，验证图可执行。"""

    app = compile_graph()
    initial_state = create_initial_state(topic)
    final_state = app.invoke(initial_state)
    return merge_state(initial_state, final_state)


if __name__ == "__main__":
    try:
        result = run_smoke_test()
        print("[SMOKE] finished")
        print(f"topic={result.get('topic')}")
        print(f"revision_step={result.get('revision_step')}")
        print(f"is_satisfactory={result.get('is_satisfactory')}")
        print(f"final_report_len={len(result.get('final_report', result.get('draft', '')))}")
    except Exception as exc:
        print(f"[SMOKE] failed: {exc}")
