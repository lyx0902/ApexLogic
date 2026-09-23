"""Presentation snapshots derived from durable runs and checkpoints."""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import re
from uuid import uuid4

from core.persistence import validate_run_id


def history_directory() -> Path:
    project_root = Path(__file__).resolve().parent.parent
    configured = os.getenv("APEXLOGIC_HISTORY_DIR", "").strip()
    if not configured:
        return project_root / "appstats"
    path = Path(configured)
    return path if path.is_absolute() else project_root / path


def history_path(run_id: str) -> Path:
    return history_directory() / f"run_{validate_run_id(run_id)}.json"


def snapshot_for_event(event) -> dict | None:
    """Project a committed node state into the existing history view schema."""
    state = event.state
    node = event.node
    if node == "researcher":
        return {
            "node": node, "iteration": state.get("revision_step", 0) + 1,
            "search_queries": state.get("search_queries", []),
            "mab_state": state.get("mab_state", {}),
            "source_quality_summary": state.get("source_quality_summary", {}),
            "query_plan": state.get("query_plan", {}),
            "iterative_retrieval_summary": state.get("iterative_retrieval_summary", {}),
            "reasoning_chains": state.get("reasoning_chains", []),
            "retrieved_context_count": len(state.get("retrieved_context", [])),
        }
    if node == "writer":
        return {"node": node, "iteration": state.get("revision_step", 0) + 1,
                "draft": state.get("draft", "")}
    if node == "reviewer":
        return {
            "node": node, "iteration": state.get("revision_step", 0),
            "review_result": state.get("review_result", {}),
            "critique_feedback": state.get("critique_feedback", ""),
            "revision_directives": state.get("revision_directives", {}),
            "next_route": state.get("next_route", ""),
            "is_satisfactory": bool(state.get("is_satisfactory", False)),
            "answer_status": state.get("answer_status", "unknown"),
        }
    return None


def build_history_record(topic, max_revisions, pass_threshold, final_state, elapsed_seconds,
                         iteration_snapshots=None, completed_at=None, timestamp=None) -> dict:
    review_result = final_state.get("review_result", {}) or {}
    contexts = final_state.get("retrieved_context", [])[:10]
    reasoning_contexts = final_state.get("reasoning_contexts", [])[:15]
    return {
        "run_id": final_state.get("run_id"),
        "timestamp": timestamp or completed_at or datetime.now().isoformat(),
        "completed_at": completed_at,
        "topic": topic,
        "max_revisions": max_revisions,
        "pass_threshold": pass_threshold,
        "iterations_done": final_state.get("revision_step", 0),
        "is_satisfactory": bool(final_state.get("is_satisfactory", False)),
        "answer_status": final_state.get("answer_status", "unknown"),
        "research_as_of": final_state.get("run_config", {}).get("research_as_of"),
        "weighted_score": float(review_result.get("weighted_score", 0.0)),
        "elapsed_seconds": round(elapsed_seconds, 1),
        "final_report": final_state.get("final_report") or final_state.get("draft", ""),
        "references": [
            {
                "citation_id": ctx.get("citation_id", "") if isinstance(ctx, dict) else "",
                "title": ctx.get("title", "") if isinstance(ctx, dict) else str(ctx),
                "url": str(ctx.get("url", "") or "").strip() if isinstance(ctx, dict) else "",
                "score": ctx.get("bge_reranker_score") if isinstance(ctx, dict) else None,
                "summary": ctx.get("core_summary", "")[:200] if isinstance(ctx, dict) else "",
            }
            for ctx in contexts
        ],
        "reasoning_references": [
            {
                "citation_id": ctx.get("citation_id", "") if isinstance(ctx, dict) else "",
                "title": ctx.get("title", "") if isinstance(ctx, dict) else str(ctx),
                "url": str(ctx.get("url", "") or "").strip() if isinstance(ctx, dict) else "",
                "summary": (ctx.get("core_summary", "") or ctx.get("content", ""))[:200]
                if isinstance(ctx, dict) else "",
            }
            for ctx in reasoning_contexts
        ],
        "errors_count": len(final_state.get("errors", [])),
        "errors": final_state.get("errors", [])[:10],
        "run_metadata": {
            "iteration_snapshots": iteration_snapshots or [],
            "memory_stats": final_state.get("memory_stats", {}),
            "cache_stats": final_state.get("cache_stats", {}),
            "memory_first": final_state.get("memory_first", {}),
            "planned_search_queries": final_state.get("planned_search_queries", []),
            "search_queries": final_state.get("search_queries", []),
            "memory_used_ids": final_state.get("memory_used_ids", []),
            "memory_publication": final_state.get("memory_publication", {}),
            "memory_publication_attempts": final_state.get("memory_publication_attempts", []),
            "memory_publication_log_error": final_state.get("memory_publication_log_error"),
            "reasoning_enabled": final_state.get("reasoning_enabled", False),
            "reasoning_chains": final_state.get("reasoning_chains", []),
            "reasoning_contexts": [
                {
                    "title": ctx.get("title", "") if isinstance(ctx, dict) else str(ctx),
                    "url": str(ctx.get("url", "") or "").strip() if isinstance(ctx, dict) else "",
                    "summary": (ctx.get("core_summary", "") or ctx.get("content", ""))[:200]
                    if isinstance(ctx, dict) else "",
                }
                for ctx in reasoning_contexts
            ],
        },
    }


def write_history_record(record: dict, *, directory=None) -> Path:
    directory = Path(directory) if directory is not None else history_directory()
    directory.mkdir(parents=True, exist_ok=True)
    run_id = record.get("run_id")
    if run_id:
        filename = directory / f"run_{validate_run_id(run_id)}.json"
    else:
        safe_topic = re.sub(r"[^\w\u4e00-\u9fff]", "_", record["topic"])[:20]
        filename = directory / f"run_{datetime.now():%Y%m%d_%H%M%S}_{safe_topic}.json"
    if filename.exists():
        try:
            previous = json.loads(filename.read_text(encoding="utf-8"))
            record["timestamp"] = previous.get("timestamp") or record["timestamp"]
        except (OSError, ValueError):
            pass
    temp_path = filename.with_suffix(f".{uuid4().hex}.tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
        temp_path.replace(filename)
    finally:
        temp_path.unlink(missing_ok=True)
    return filename


def save_run_to_history(topic, max_revisions, pass_threshold, final_state, elapsed_seconds,
                        iteration_snapshots=None, completed_at=None, *, directory=None) -> dict:
    record = build_history_record(topic, max_revisions, pass_threshold, final_state,
                                  elapsed_seconds, iteration_snapshots, completed_at)
    write_history_record(record, directory=directory)
    return record


def history_from_run(runner, run_id: str, *, final_state=None, write=False) -> dict:
    """Rebuild a completed run's view without invoking graph nodes or publishing memory."""
    info = runner.peek(run_id)
    record = info["record"]
    if record["status"] != "completed":
        raise ValueError("研究尚未完成，不能生成历史记录")
    state = final_state or info["state"]
    if not state.get("draft") and not state.get("final_report"):
        raise ValueError("已完成任务缺少报告内容")
    events = runner.history(run_id)
    snapshots = [item for event in events if (item := snapshot_for_event(event)) is not None]
    elapsed = sum(float(attempt.get("elapsed_seconds") or 0) for attempt in info["attempts"])
    settings = record["run_config"]["settings"]
    view = build_history_record(record["topic"], int(settings["MAX_REVISIONS"]),
                                float(settings["REVIEWER_PASS_THRESHOLD"]), state, elapsed,
                                snapshots, record.get("completed_at"))
    if write:
        write_history_record(view)
    return view


def load_or_rebuild_run(runner, run_id: str) -> dict:
    """Read a presentation file, rebuilding a missing or broken one from PostgreSQL."""
    path = history_path(run_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if (isinstance(data, dict) and data.get("run_id") == run_id
                and data.get("final_report") and isinstance(data.get("run_metadata"), dict)):
            return data
    except (OSError, ValueError):
        pass
    return history_from_run(runner, run_id, write=True)


def load_history_list(*, directory=None, limit=20) -> list[dict]:
    directory = Path(directory) if directory is not None else history_directory()
    if not directory.exists():
        return []
    records = []
    for path in directory.glob("run_*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("topic"):
                records.append({"file": str(path), "data": data})
        except (OSError, ValueError):
            continue
    return sorted(records, key=lambda entry: history_sort_key(entry["data"]), reverse=True)[:limit]


def history_sort_key(data: dict) -> float:
    value = data.get("completed_at") or data.get("timestamp") or ""
    try:
        return datetime.fromisoformat(value).timestamp()
    except (ValueError, TypeError, OverflowError):
        return float("-inf")


def merge_completed_runs(entries: list[dict], runs: list[dict], *, limit=20) -> list[dict]:
    """Show completed PostgreSQL runs even before their derived file is written."""
    result = list(entries)
    seen = {entry["data"].get("run_id") for entry in entries}
    for run in runs:
        if run["status"] != "completed" or run["run_id"] in seen:
            continue
        result.append({"file": None, "data": {
            "run_id": run["run_id"], "topic": run["topic"],
            "timestamp": run.get("completed_at") or run["created_at"],
            "completed_at": run.get("completed_at"),
            "weighted_score": None,
            "is_satisfactory": run.get("termination_reason") == "passed",
            "answer_status": "limited" if run.get("termination_reason") == "limited" else "unknown",
        }})
        seen.add(run["run_id"])
    return sorted(result, key=lambda entry: history_sort_key(entry["data"]), reverse=True)[:limit]
