from __future__ import annotations

import argparse
import json
from typing import Any

from dotenv import load_dotenv

from core.runner import ResearchRunner


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
        default=None,
        help="研究主题",
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--resume", metavar="RUN_ID", help="恢复任务（使用原配置）")
    actions.add_argument("--status", metavar="RUN_ID", help="查看已保存状态")
    actions.add_argument("--list-runs", action="store_true", help="列出研究任务")
    parser.add_argument("--data-dir", default=None, help="持久化目录，默认项目 data/")
    parser.add_argument("--storage-backend", choices=["sqlite", "postgres"], default=None,
                        help="任务存储后端；旧任务使用 sqlite")
    parser.add_argument("--max-revisions", type=int, default=None)
    parser.add_argument("--pass-threshold", type=float, default=None)
    parser.add_argument(
        "--output-mode",
        choices=["user", "debug"],
        default=None,
        help="输出模式：user 仅面向读者，debug 含系统细节",
    )
    args = parser.parse_args()
    if (args.resume or args.status or args.list_runs) and any(
        value is not None for value in (args.topic, args.max_revisions, args.pass_threshold, args.output_mode)
    ):
        parser.error("恢复/查询任务不能覆盖原任务参数；如需修改参数，请创建新任务。")
    return args


def run() -> dict[str, Any] | None:
    """加载配置并执行研究图。"""

    env_loaded = load_dotenv()
    if not env_loaded:
        load_dotenv(".env.example")
    args = parse_args()

    runner = ResearchRunner(args.data_dir, backend=args.storage_backend)
    if args.list_runs:
        for item in runner.repository.list():
            print(f"{item['run_id']}  {item['status']}  {item['topic']}")
        return None
    if args.status:
        info = runner.inspect(args.status)
        print(json.dumps({k: v for k, v in info.items() if k != "state"}, ensure_ascii=False, indent=2))
        return None
    if args.resume:
        run_id = args.resume
    else:
        record = runner.create(args.topic if args.topic is not None else "多智能体系统中的反思机制与自我优化",
            max_revisions=args.max_revisions, pass_threshold=args.pass_threshold,
            output_mode=args.output_mode or "debug")
        run_id = record["run_id"]
    print(f"[RUN] run_id={run_id}", flush=True)
    print(f'[RECOVERY] python main.py --resume {run_id} --storage-backend {runner.storage["backend"]} --data-dir "{runner.data_dir}"', flush=True)
    return runner.run(run_id)


if __name__ == "__main__":
    state = run()
    if state is None:
        raise SystemExit(0)
    print("[RUN] finished")
    print("quality=" + ("passed" if state.get("is_satisfactory") else ("limited" if state.get("answer_status") == "limited" else "not_passed")))
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

