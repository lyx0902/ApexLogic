"""Offline graph and subprocess entrypoint for crash/restart acceptance tests."""
from pathlib import Path
import json
import os
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langgraph.graph import END, START, StateGraph
from core.graph import reviewer_route
from core.run_config import configured_node, setting
from core.runner import ResearchRunner
from core.state import ResearchState


def factory(max_revisions=2, *, checkpointer=None, strict=False):
    def entered(node, state):
        log = os.getenv("RECOVERY_TEST_LOG")
        if log:
            with open(log, "a", encoding="utf-8") as f:
                f.write(json.dumps({"node": node, "revision": state["revision_step"]}) + "\n")
                f.flush()
                os.fsync(f.fileno())
        if (os.getenv("RECOVERY_TEST_CRASH_NODE") == node and
                int(os.getenv("RECOVERY_TEST_CRASH_REVISION", "0")) == state["revision_step"]):
            os._exit(23)
        if os.getenv("RECOVERY_TEST_EXCEPTION_NODE") == node:
            raise RuntimeError("模拟节点失败")
        return list(state.get("execution_trace", [])) + [{"node": node, "revision_step": state["revision_step"]}]

    @configured_node
    def researcher(state):
        trace = entered("researcher", state)
        return {"retrieved_context": [{"title": "source", "content": "fact", "citation_id": "S1", "url": "https://example.org"}],
                "mab_state": {"alpha": {"duckduckgo": 2}}, "execution_trace": trace,
                "reasoning_contexts": [{"title": "reason", "content": "evidence", "citation_id": "R1"}],
                "query_plan": {"sub_questions": ["q1"]}}

    @configured_node
    def writer(state):
        trace = entered("writer", state)
        return {"draft": f"draft-{state['revision_step']}-threshold-{setting('REVIEWER_PASS_THRESHOLD')}",
                "execution_trace": trace}

    @configured_node
    def reviewer(state):
        trace = entered("reviewer", state)
        step = state["revision_step"] + 1
        passed = step >= 2 and float(setting("REVIEWER_PASS_THRESHOLD")) <= 8
        result = {"revision_step": step, "is_satisfactory": passed,
                  "next_route": "end" if passed else "writer", "needs_more_research": False,
                  "review_stats": {"rounds_total": step}, "execution_trace": trace,
                  "revision_directives": {"must_fix": []},
                  "iteration_history": list(state.get("iteration_history", [])) + [{"round": step, "draft": state["draft"]}]}
        if passed:
            result["final_report"] = state["draft"]
        return result

    graph = StateGraph(ResearchState)
    graph.add_node("researcher", researcher)
    graph.add_node("writer", writer)
    graph.add_node("reviewer", reviewer)
    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "reviewer")
    graph.add_conditional_edges("reviewer", lambda s: reviewer_route(s, max_revisions),
                               {"end": END, "writer": "writer", "researcher": "researcher"})
    return graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    from core.persistence import run_lock
    folder, run_id = Path(sys.argv[1]), sys.argv[2]
    if len(sys.argv) > 3 and sys.argv[3] == "lock":
        with run_lock(folder, run_id):
            print("locked", flush=True)
            time.sleep(30)
    else:
        ResearchRunner(folder, graph_factory=factory).run(run_id)
