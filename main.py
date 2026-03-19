from __future__ import annotations

import argparse
import os
from typing import Any

from dotenv import load_dotenv

from core.graph import compile_graph
from core.state import create_initial_state


def _format_trace_item(item: dict[str, Any]) -> str:
    """格式化单条执行轨迹，提升终端可读性。"""

    node = item.get("node", "unknown")
    revision = item.get("revision_step", "-")
    mode = item.get("mode", "-")
    if node == "researcher":
        return (
            f"node={node} rev={revision} queries={item.get('queries', 0)} "
            f"broad={item.get('broad_total', 0)} dedup={item.get('dedup_total', 0)} "
            f"ret20={item.get('retriever_topk', 0)} rerank10={item.get('reranker_topk', 0)} "
            f"contexts={item.get('contexts', 0)} "
            f"dropped={item.get('dropped', 0)} "
            f"errors={item.get('errors', 0)}"
        )
    if node == "writer":
        return (
            f"node={node} rev={revision} mode={mode} "
            f"draft_len={item.get('draft_len', 0)} "
            f"mapped={item.get('mapped_items', 0)}"
        )
    if node == "reviewer":
        return (
            f"node={node} rev={revision} mode={mode} "
            f"next={item.get('next_route', '')} "
            f"ok={item.get('is_satisfactory', False)} "
            f"conf={item.get('confidence', 0.0)} "
            f"controversies={item.get('controversies', 0)}"
        )
    return str(item)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Deep Research Multi-Agent Runner")
    parser.add_argument(
        "--topic",
        default="多智能体系统中的反思机制与自我优化",
        help="研究主题",
    )
    parser.add_argument(
        "--output-mode",
        choices=["user", "debug"],
        default="debug",
        help="输出模式：user 仅面向读者，debug 含系统细节",
    )
    return parser.parse_args()


def run() -> dict[str, Any]:
    """加载配置并执行研究图。"""

    env_loaded = load_dotenv()
    if not env_loaded:
        load_dotenv(".env.example")
    args = parse_args()

    max_revisions = int(os.getenv("MAX_REVISIONS", "3"))
    app = compile_graph(max_revisions=max_revisions)
    initial_state = create_initial_state(
        topic=args.topic,
        output_mode=args.output_mode,
    )
    final_state = app.invoke(initial_state)

    return final_state


if __name__ == "__main__":
    state = run()
    print("[RUN] finished")
    print(f"topic={state.get('topic', '')}")
    print(f"revision_step={state.get('revision_step', 0)}")
    print(f"is_satisfactory={state.get('is_satisfactory', False)}")
    print(f"draft_len={len(state.get('draft', '') or '')}")
    print(f"errors={len(state.get('errors', []))}")
    review = state.get("review_result", {})
    print(f"review_mode={review.get('review_mode', 'unknown')}")
    print(f"next_route={state.get('next_route', '')}")
    print(f"fact_issues={review.get('fact_issues', [])}")
    print(f"logic_issues={review.get('logic_issues', [])}")
    print(f"info_gaps={review.get('info_gaps', [])}")

    trace = state.get("execution_trace", [])
    print("[TRACE]")
    for idx, item in enumerate(trace, start=1):
        print(f"{idx}. {_format_trace_item(item)}")

