"""One entry point for durable CLI/UI execution and read-only report export."""
from contextlib import closing
from copy import deepcopy
from dataclasses import dataclass
import time

from core.graph import compile_graph
from core.persistence import RunBusyError, data_directory, open_checkpointer, run_lock
from core.run_config import SCHEMA_VERSION, WORKFLOW_VERSION, config_hash, make_run_config, validate_config
from core.run_repository import RunRepository
from core import storage as storage_factory
from core.state import create_initial_state


class RecoveryError(RuntimeError):
    pass


@dataclass
class RunEvent:
    kind: str
    node: str | None
    state: dict


class ResearchRunner:
    def __init__(self, data_dir=None, *, graph_factory=compile_graph, backend=None):
        self.data_dir = data_directory(data_dir)
        self.storage = storage_factory.storage_settings(backend)
        self.repository = storage_factory.run_repository(self.data_dir, self.storage)
        self.graph_factory = graph_factory

    def _lock(self, run_id):
        return storage_factory.execution_lock(self.data_dir, run_id, self.storage)

    def _checkpointer(self):
        return storage_factory.checkpointer(self.data_dir, self.storage)

    def create(self, topic, *, max_revisions=None, pass_threshold=None, output_mode="debug"):
        config = make_run_config(max_revisions=max_revisions, pass_threshold=pass_threshold, output_mode=output_mode)
        config["memory"]["data_dir"] = str(self.data_dir)
        config["storage"] = dict(self.storage)
        if self.storage["backend"] == "postgres":
            config["memory"]["data_dir"] = ""
        return self.repository.create(topic, config)

    def _record(self, run_id):
        record = self.repository.get(run_id)
        if storage_factory.config_storage(record["run_config"]) != self.storage:
            raise RecoveryError("任务属于另一存储后端；请切换后端后恢复，不能原地修改历史配置。")
        if record["schema_version"] != SCHEMA_VERSION or record["workflow_version"] != WORKFLOW_VERSION:
            raise RecoveryError("状态结构或工作流版本不兼容；请保留原数据并使用兼容版本恢复。")
        validate_config(record["run_config"])
        if config_hash(record["run_config"]) != record["config_hash"]:
            raise RecoveryError("任务配置快照校验失败")
        return record

    def _graph(self, record, saver):
        return self.graph_factory(
            max_revisions=int(record["run_config"]["settings"]["MAX_REVISIONS"]),
            checkpointer=saver, strict=True,
        )

    @staticmethod
    def _invocation_config(record):
        # The loop has at most three nodes per review round, plus input/end.
        return {"configurable": {"thread_id": record["thread_id"]},
                "recursion_limit": int(record["run_config"]["settings"]["MAX_REVISIONS"]) * 3 + 10}

    def _snapshot(self, record, graph, saver):
        config = self._invocation_config(record)
        if saver.get_tuple(config) is None:
            if (record["checkpoint_seen"] or record["status"] == "completed" or
                    self.repository.attempts(record["run_id"])):
                raise RecoveryError("任务已有执行记录，但 checkpoint 已缺失；不会自动从头重跑，请检查数据库或新建任务。")
            return None
        snapshot = graph.get_state(config)
        state = snapshot.values
        if state and (state.get("run_id") != record["run_id"] or
                      state.get("thread_id") != record["thread_id"] or
                      state.get("schema_version") != SCHEMA_VERSION or
                      state.get("workflow_version") != WORKFLOW_VERSION or
                      config_hash(state.get("run_config", {})) != record["config_hash"]):
            raise RecoveryError("checkpoint 的任务身份、版本或配置与任务记录不一致")
        return snapshot

    def _complete(self, run_id, state, completed_at):
        reason = "passed" if state.get("is_satisfactory") else ("limited" if state.get("answer_status") == "limited" else "max_revisions")
        self.repository.update(run_id, status="completed", termination_reason=reason,
                               checkpoint_seen=1, last_error=None, completed_at=completed_at)

    def _reconcile(self, record, snapshot):
        self.repository.reconcile_attempts(record["run_id"])
        if snapshot is not None and not snapshot.next:
            self._complete(record["run_id"], snapshot.values, snapshot.created_at)
        elif snapshot is not None:
            self.repository.update(record["run_id"], status="interrupted", checkpoint_seen=1,
                                   termination_reason=None)
        elif record["status"] == "running":
            self.repository.update(record["run_id"], status="interrupted")

    def _memory_result(self, record, snapshot, *, publish=False, trigger="direct"):
        state = dict(snapshot.values)
        options = record["run_config"].get("memory", {})
        if not options.get("enabled") or snapshot.next:
            return state
        try:
            from memory.service import MemoryService
            service = MemoryService(record["run_config"])
            checkpoint_id = snapshot.config["configurable"]["checkpoint_id"]
            receipt = (service.publish(state, checkpoint_id, snapshot.created_at, trigger=trigger) if publish else
                       service.repo.publication(record["run_id"], checkpoint_id, options["namespace"]))
            if receipt:
                state["memory_publication"] = receipt
                state["memory_write_ids"] = receipt["item_ids"]
            try:
                state["memory_publication_attempts"] = service.repo.publication_attempts(
                    record["run_id"], checkpoint_id, options["namespace"])
            except Exception as log_exc:
                state["memory_publication_log_error"] = type(log_exc).__name__
        except Exception as exc:
            state["memory_publication"] = {"status": "failed", "error": type(exc).__name__}
        return state

    def inspect(self, run_id):
        """Reconcile stale metadata only when no executor owns the run lock."""
        record = self._record(run_id)
        try:
            with self._lock(run_id):
                with self._checkpointer() as saver:
                    snapshot = self._snapshot(record, self._graph(record, saver), saver)
                    self._reconcile(record, snapshot)
                running = False
        except RunBusyError:
            with self._checkpointer() as saver:
                snapshot = self._snapshot(record, self._graph(record, saver), saver)
            running = True
        return {"record": self.repository.get(run_id), "running": running,
                "state": self._memory_result(record, snapshot) if snapshot else {},
                "next": list(snapshot.next) if snapshot else [],
                "saved_at": snapshot.created_at if snapshot else None,
                "attempts": self.repository.attempts(run_id)}

    def stream(self, run_id):
        """Consume fully or use contextlib.closing to release lock on UI exit."""
        with self._lock(run_id):
            record = self._record(run_id)
            with self._checkpointer() as saver:
                graph = self._graph(record, saver)
                snapshot = self._snapshot(record, graph, saver)
                self._reconcile(record, snapshot)
                if snapshot is not None and not snapshot.next:
                    yield RunEvent("complete", None, self._memory_result(record, snapshot, publish=True, trigger="reopen_completed"))
                    return
                state = dict(snapshot.values) if snapshot else create_initial_state(
                    record["topic"], output_mode=record["run_config"]["output_mode"])
                if snapshot is None:
                    state.update(run_id=run_id, thread_id=record["thread_id"],
                                 schema_version=SCHEMA_VERSION, workflow_version=WORKFLOW_VERSION,
                                 run_config=deepcopy(record["run_config"]))
                attempt_id = self.repository.start_attempt(run_id)
                self.repository.update(run_id, status="running", last_error=None, termination_reason=None)
                started = time.monotonic()
                try:
                    with closing(graph.stream(None if snapshot is not None else state,
                            config=self._invocation_config(record), stream_mode="updates", durability="sync")) as events:
                        for update in events:
                            for node, patch in update.items():
                                if node.startswith("__") or not isinstance(patch, dict):
                                    continue
                                state.update(patch)  # existing nodes return accumulated lists
                                self.repository.update(run_id, checkpoint_seen=1)
                                yield RunEvent("node", node, deepcopy(state))
                    saved = graph.get_state(self._invocation_config(record))
                    if saved.next:
                        raise RecoveryError("工作流仍有待执行节点，已保留进度")
                    self._complete(run_id, saved.values, saved.created_at)
                    self.repository.finish_attempt(attempt_id, "completed", time.monotonic() - started)
                except BaseException as exc:
                    # Never persist arbitrary provider exception text: it may contain credentials.
                    error = f"{type(exc).__name__}: 执行中断；已提交节点可以恢复。"
                    self.repository.update(run_id, status="interrupted", last_error=error)
                    self.repository.finish_attempt(attempt_id, "interrupted", time.monotonic() - started, error)
                    raise
                yield RunEvent("complete", None, self._memory_result(record, saved, publish=True, trigger="research_completed"))

    def run(self, run_id):
        final = None
        with closing(self.stream(run_id)) as events:
            for event in events:
                final = event.state
        return final

    def history(self, run_id):
        record = self._record(run_id)
        with self._checkpointer() as saver:
            graph = self._graph(record, saver)
            self._snapshot(record, graph, saver)
            snapshots = list(graph.get_state_history(self._invocation_config(record)))
        # Node outputs already carry a monotonically growing execution trace.
        result, seen = [], set()
        for snapshot in reversed(snapshots):
            state = dict(snapshot.values)
            trace = state.get("execution_trace", [])
            if not trace or len(trace) in seen:
                continue
            seen.add(len(trace))
            result.append(RunEvent("node", trace[-1]["node"], state))
        return result

    def completed_state(self, run_id):
        info = self.inspect(run_id)
        if info["running"] or info["record"]["status"] != "completed":
            raise RecoveryError("任务尚未结束；请先继续执行后再导出最终报告。")
        return info["state"]
