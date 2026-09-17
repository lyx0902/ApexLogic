from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Barrier

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from recovery_worker import factory
from core.persistence import RunBusyError, open_checkpointer, run_lock
from core.run_config import SETTING_KEYS, bind_config, make_run_config, setting
from core.runner import RecoveryError, ResearchRunner


@pytest.fixture
def runner(tmp_path, monkeypatch):
    # Do not inherit keys, tracing or application settings in offline tests.
    for key in SETTING_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    return ResearchRunner(tmp_path, graph_factory=factory)


@pytest.mark.parametrize("node,revision,expected_next", [
    ("researcher", 0, "researcher"), ("writer", 0, "writer"),
    ("reviewer", 0, "reviewer"), ("writer", 1, "writer"),
])
def test_hard_crash_and_new_process_resume(runner, node, revision, expected_next):
    record = runner.create("crash acceptance", max_revisions=2)
    run_id = record["run_id"]
    log = runner.data_dir / "calls.jsonl"
    env = {**os.environ, "RECOVERY_TEST_CRASH_NODE": node,
           "RECOVERY_TEST_CRASH_REVISION": str(revision), "RECOVERY_TEST_LOG": str(log)}
    command = [sys.executable, str(ROOT / "tests/recovery_worker.py"), str(runner.data_dir), run_id]
    failed = subprocess.run(command, env=env, capture_output=True, timeout=45)
    assert failed.returncode == 23, failed.stderr.decode(errors="replace")
    info = runner.inspect(run_id)
    assert info["next"] == [expected_next]
    assert info["record"]["status"] == "interrupted"
    assert info["attempts"][0]["elapsed_seconds"] is None
    if node != "researcher":
        assert info["state"]["mab_state"]["alpha"]["duckduckgo"] == 2
        assert info["state"]["reasoning_contexts"][0]["citation_id"] == "R1"
    if revision == 1:
        assert info["state"]["review_stats"] == {"rounds_total": 1}

    env.pop("RECOVERY_TEST_CRASH_NODE")
    # Changes after the crash must not alter the saved execution policy.
    env["REVIEWER_PASS_THRESHOLD"] = "9.9"
    env["MAX_REVISIONS"] = "1"
    recovered = subprocess.run(command, env=env, capture_output=True, timeout=45)
    assert recovered.returncode == 0, recovered.stderr.decode(errors="replace")
    info = runner.inspect(run_id)
    assert info["record"]["termination_reason"] == "passed"
    assert info["state"]["review_stats"] == {"rounds_total": 2}
    assert len(info["state"]["iteration_history"]) == 2
    assert len(info["state"]["execution_trace"]) == 5
    assert "threshold-7.5" in info["state"]["draft"]
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert sum(c["node"] == "researcher" for c in calls) == (2 if node == "researcher" else 1)
    assert len(calls) == 6  # five committed nodes + exactly the interrupted node
    assert len(info["attempts"]) == 2


def test_duplicate_execution_cross_process_lock(runner):
    record = runner.create("lock")
    run_id = record["run_id"]
    proc = subprocess.Popen([sys.executable, str(ROOT / "tests/recovery_worker.py"),
        str(runner.data_dir), run_id, "lock"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "locked"
        assert runner.inspect(run_id)["running"]
        with pytest.raises(RunBusyError):
            runner.run(run_id)
    finally:
        proc.terminate()
        proc.communicate(timeout=10)
    assert not runner.inspect(run_id)["running"]
    assert runner.run(run_id)["is_satisfactory"]


def test_completed_task_does_not_run_again_and_reconciles_index(runner):
    run_id = runner.create("complete", max_revisions=2)["run_id"]
    state = runner.run(run_id)
    runner.repository.update(run_id, status="running")
    assert runner.inspect(run_id)["record"]["status"] == "completed"
    assert runner.run(run_id) == state
    assert len(runner.repository.attempts(run_id)) == 1
    assert [e.node for e in runner.history(run_id)] == ["researcher", "writer", "reviewer", "writer", "reviewer"]


def test_round_limit_is_not_quality_pass(runner):
    run_id = runner.create("limit", max_revisions=1)["run_id"]
    state = runner.run(run_id)
    assert not state["is_satisfactory"]
    assert runner.inspect(run_id)["record"]["termination_reason"] == "max_revisions"
    assert runner.completed_state(run_id)["draft"]


def test_missing_checkpoint_refuses_restart(runner):
    run_id = runner.create("missing")["run_id"]
    runner.run(run_id)
    with open_checkpointer(runner.data_dir) as saver:
        saver.delete_thread(run_id)
    with pytest.raises(RecoveryError, match="缺失"):
        runner.run(run_id)


def test_lost_initial_checkpoint_is_not_treated_as_new_task(runner, monkeypatch):
    run_id = runner.create("initial state loss")["run_id"]
    monkeypatch.setenv("RECOVERY_TEST_EXCEPTION_NODE", "researcher")
    with pytest.raises(RuntimeError):
        runner.run(run_id)
    # Researcher did not emit an output, so checkpoint_seen is still false.
    assert not runner.repository.get(run_id)["checkpoint_seen"]
    with open_checkpointer(runner.data_dir) as saver:
        saver.delete_thread(run_id)
    monkeypatch.delenv("RECOVERY_TEST_EXCEPTION_NODE")
    with pytest.raises(RecoveryError, match="缺失"):
        runner.run(run_id)


def test_workflow_version_and_config_integrity(runner):
    run_id = runner.create("versions")["run_id"]
    with runner.repository.connect() as db:
        db.execute("UPDATE runs SET workflow_version='future' WHERE run_id=?", (run_id,))
    with pytest.raises(RecoveryError, match="不兼容"):
        runner.run(run_id)
    with runner.repository.connect() as db:
        from core.run_config import WORKFLOW_VERSION
        db.execute("UPDATE runs SET workflow_version=?, config_hash='bad' WHERE run_id=?", (WORKFLOW_VERSION, run_id))
    with pytest.raises(RecoveryError, match="校验失败"):
        runner.run(run_id)


def test_generator_close_releases_execution_lock(runner):
    run_id = runner.create("close")["run_id"]
    with closing(runner.stream(run_id)) as stream:
        assert next(stream).node == "researcher"
    with run_lock(runner.data_dir, run_id):
        pass
    assert runner.run(run_id)["is_satisfactory"]


def test_config_thread_isolation_and_secret_exclusion(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "private-test-key")
    configs = [make_run_config(pass_threshold=x) for x in (6, 9)]
    assert "private-test-key" not in json.dumps(configs)
    barrier = Barrier(2)
    def read(config):
        with bind_config(config):
            barrier.wait(timeout=5)
            return setting("REVIEWER_PASS_THRESHOLD"), setting("DEEPSEEK_API_KEY")
    with ThreadPoolExecutor(2) as pool:
        assert list(pool.map(read, configs)) == [("6", "private-test-key"), ("9", "private-test-key")]
    monkeypatch.setenv("AQD_ENABLED", "0")
    with bind_config(configs[0]):
        assert setting("AQD_ENABLED", "1") == "1"


def test_real_reviewer_and_planner_use_snapshot(runner, monkeypatch):
    import agents.reviewer as reviewer
    from core.state import create_initial_state
    from optim.query_planner import AdaptiveQueryPlanner
    config = make_run_config(pass_threshold=9)
    monkeypatch.setenv("REVIEWER_PASS_THRESHOLD", "1")
    monkeypatch.setenv("REVIEWER_ALLOW_DEGRADED_PASS", "1")
    monkeypatch.setenv("AQD_MAX_SUB_QUESTIONS", "99")
    state = create_initial_state("snapshot")
    state["run_config"] = config
    state["draft"] = "正文" * 400
    state["retrieved_context"] = [{"content": "a"}, {"content": "b"}]
    monkeypatch.setattr(reviewer, "_llm_review", lambda **kw: reviewer._rule_based_review(kw["draft"], kw["retrieved_context"]))
    result = reviewer.reviewer_node(state)
    assert result["review_result"]["pass_threshold"] == 9
    assert not result["is_satisfactory"]
    with bind_config(config):
        assert AdaptiveQueryPlanner(max_sub_questions=4).max_sub_questions == 4


def test_export_failure_retries_without_research(runner, monkeypatch):
    import export_report
    run_id = runner.create("export")["run_id"]
    state = runner.run(run_id)
    original = export_report.export_state
    monkeypatch.setattr(export_report, "export_state", lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        export_report.export_saved_run(run_id, data_dir=runner.data_dir)
    assert runner.repository.get(run_id)["export_status"] == "failed"
    monkeypatch.setattr(export_report, "export_state", original)
    output = runner.data_dir / "out.md"
    paths = export_report.export_saved_run(run_id, output=str(output), output_mode="user", data_dir=runner.data_dir)
    assert Path(paths[0]).is_file()
    assert runner.completed_state(run_id) == state
    assert len(runner.repository.attempts(run_id)) == 1


def test_strict_graph_loading_never_uses_placeholder():
    from core.graph import _load_agent_node
    with pytest.raises(RuntimeError, match="无法加载"):
        _load_agent_node("agents.nonexistent", "node", lambda s: {}, strict=True)


def test_empty_topic_and_invalid_id(runner):
    with pytest.raises(ValueError):
        runner.create(" ")
    with pytest.raises(ValueError):
        runner.inspect("../outside")


def test_all_nonsecret_runtime_settings_are_snapshotted():
    import ast
    for folder in ("agents", "optim", "bge", "tools"):
        for path in (ROOT / folder).glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and
                        node.func.id == "setting" and node.args and isinstance(node.args[0], ast.Constant)):
                    key = node.args[0].value
                    assert key in SETTING_KEYS or key.endswith("API_KEY"), (path, key)


def test_cli_status_and_export_existing_run(runner):
    run_id = runner.create("CLI integration")["run_id"]
    runner.run(run_id)
    status = subprocess.run([sys.executable, str(ROOT / "main.py"), "--status", run_id,
        "--data-dir", str(runner.data_dir)], capture_output=True, encoding="utf-8", timeout=45,
        env={**os.environ, "PYTHONUTF8": "1"})
    assert status.returncode == 0, status.stderr
    assert json.loads(status.stdout)["record"]["status"] == "completed"
    exported = subprocess.run([sys.executable, str(ROOT / "export_report.py"), "--run-id", run_id,
        "--data-dir", str(runner.data_dir), "--output", str(runner.data_dir / "nested/report.md")],
        capture_output=True, encoding="utf-8", timeout=45, env={**os.environ, "PYTHONUTF8": "1"})
    assert exported.returncode == 0, exported.stderr
    assert (runner.data_dir / "nested/report-user.md").is_file()
    assert (runner.data_dir / "nested/report-debug.md").is_file()
    assert len(runner.repository.attempts(run_id)) == 1


def test_streamlit_create_view_and_reopen_completed(runner, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("APEXLOGIC_DATA_DIR", str(runner.data_dir))
    monkeypatch.setenv("APEXLOGIC_HISTORY_DIR", str(runner.data_dir / "history"))
    monkeypatch.setattr(ResearchRunner, "_graph", lambda self, record, saver: factory(
        max_revisions=int(record["run_config"]["settings"]["MAX_REVISIONS"]), checkpointer=saver))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    app.text_input[0].set_value("UI recovery").run()
    app.button[0].click().run()
    assert not app.exception, [x.message for x in app.exception]
    assert not any("运行出错" in x.value or "无法" in x.value for x in app.error), [x.value for x in app.error]
    assert any("研究完成" in x.value for x in app.success)
    record = runner.repository.list()[0]
    assert record["status"] == "completed"
    assert len(runner.repository.attempts(record["run_id"])) == 1
    history_path = runner.data_dir / "history" / f"run_{record['run_id']}.json"
    original_history = json.loads(history_path.read_text(encoding="utf-8"))
    app.run()
    next(s for s in app.selectbox if s.label == "选择研究任务").select(record["run_id"]).run()
    next(b for b in app.button if b.label == "查看任务状态 / 已有结果").click().run()
    assert not app.exception
    next(b for b in app.button if b.label == "继续研究 / 打开完成结果").click().run()
    assert not app.exception, [x.message for x in app.exception]
    assert not any("运行出错" in x.value or "无法" in x.value for x in app.error), [x.value for x in app.error]
    assert len(runner.repository.attempts(record["run_id"])) == 1
    assert len(list((runner.data_dir / "history").glob("*.json"))) == 1
    after = runner.repository.get(record["run_id"])
    assert after["completed_at"] == record["completed_at"]
    assert after["updated_at"] == record["updated_at"]
    reopened = json.loads(history_path.read_text(encoding="utf-8"))
    assert reopened["timestamp"] == original_history["timestamp"]
    assert reopened["completed_at"] == record["completed_at"]


def test_parallel_tasks_keep_distinct_policy_and_state(runner):
    ids = [runner.create("parallel", max_revisions=2, pass_threshold=threshold)["run_id"] for threshold in (6, 9)]
    with ThreadPoolExecutor(2) as pool:
        states = list(pool.map(runner.run, ids))
    assert [s["run_id"] for s in states] == ids
    assert [s["is_satisfactory"] for s in states] == [True, False]
    assert "threshold-6" in states[0]["draft"]
    assert "threshold-9" in states[1]["draft"]


def test_database_corruption_is_reported(runner):
    import sqlite3
    run_id = runner.create("corrupt")["run_id"]
    (runner.data_dir / "checkpoints.sqlite").write_bytes(b"not a sqlite file")
    with pytest.raises(sqlite3.DatabaseError):
        runner.run(run_id)
    assert not runner.repository.attempts(run_id)


def test_resume_rejects_config_override_on_cli(runner):
    run_id = runner.create("CLI policy")["run_id"]
    result = subprocess.run([sys.executable, str(ROOT / "main.py"), "--resume", run_id,
        "--pass-threshold", "1", "--data-dir", str(runner.data_dir)], capture_output=True, timeout=30)
    assert result.returncode == 2
    assert not runner.repository.attempts(run_id)


def test_real_graph_nodes_preserve_full_state(runner, monkeypatch):
    import agents.researchers as researcher
    import agents.writer as writer
    import agents.reviewer as reviewer
    for key in ("GRAPH_EXPAND_QUERIES", "AQD_ENABLED", "ITERATIVE_RETRIEVAL_ENABLED",
                "BGE_RETRIEVER_ENABLED", "BGE_RERANKER_ENABLED"):
        monkeypatch.setenv(key, "0")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-test-key")
    monkeypatch.setattr(researcher, "_rewrite_queries_with_llm", lambda topic, feedback, seeds, **kw: seeds)
    monkeypatch.setattr(researcher, "_collect_broad_contexts", lambda *args, **kw: ([
        {"title": "A", "url": "https://example.org/a", "content": "evidence A", "source": "duckduckgo"},
        {"title": "B", "url": "https://example.org/b", "content": "evidence B", "source": "duckduckgo"},
    ], [], {}))
    monkeypatch.setattr(writer, "ChatOpenAI", lambda **kw: object())
    monkeypatch.setattr(writer, "_generate_draft_with_llm", lambda **kw: "研究结论 [S1]。" * 100)
    monkeypatch.setattr(reviewer, "_llm_review", lambda **kw: reviewer._finalize_review(
        {"scores": {"S1": 9, "S2": 9, "S3": 9, "S4": 9}, "evidence_verdicts": [{"claim": "test", "status": "supported", "citation_ids": ["S1"], "source_quote": "evidence A"}]}, "llm_scored"))
    real = ResearchRunner(runner.data_dir)
    run_id = real.create("real graph", max_revisions=1)["run_id"]
    result = real.run(run_id)
    saved = ResearchRunner(runner.data_dir).completed_state(run_id)
    assert saved == result
    assert result["is_satisfactory"]
    assert len(result["retrieved_context"]) == 2
    assert result["mab_state"]["round"] == 1
    assert result["review_stats"]["rounds_total"] == 1
    assert [x["node"] for x in result["execution_trace"]] == ["researcher", "writer", "reviewer"]


def test_streamlit_resumes_failed_run(runner, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("APEXLOGIC_DATA_DIR", str(runner.data_dir))
    monkeypatch.setenv("APEXLOGIC_HISTORY_DIR", str(runner.data_dir / "history"))
    monkeypatch.setenv("RECOVERY_TEST_EXCEPTION_NODE", "writer")
    log = runner.data_dir / "calls.jsonl"
    monkeypatch.setenv("RECOVERY_TEST_LOG", str(log))
    run_id = runner.create("resume in UI", max_revisions=2)["run_id"]
    with pytest.raises(RuntimeError):
        runner.run(run_id)
    monkeypatch.delenv("RECOVERY_TEST_EXCEPTION_NODE")
    monkeypatch.setattr(ResearchRunner, "_graph", lambda self, record, saver: factory(
        max_revisions=int(record["run_config"]["settings"]["MAX_REVISIONS"]), checkpointer=saver))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    next(s for s in app.selectbox if s.label == "选择研究任务").select(run_id).run()
    next(b for b in app.button if b.label == "继续研究 / 打开完成结果").click().run()
    assert not app.exception, [x.message for x in app.exception]
    assert not app.error, [x.value for x in app.error]
    assert any("研究完成" in x.value for x in app.success)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert sum(c["node"] == "researcher" for c in calls) == 1
    assert runner.completed_state(run_id)["review_stats"]["rounds_total"] == 2


def test_completion_time_comes_from_checkpoint_and_is_immutable(runner):
    run_id = runner.create("stable completion", max_revisions=2)["run_id"]
    runner.run(run_id)
    before = runner.repository.get(run_id)
    assert before["completed_at"] == runner.inspect(run_id)["saved_at"]
    runner.run(run_id)
    assert runner.repository.get(run_id) == before
    runner.repository.update(run_id, export_status="completed", completed_at="invalid replacement")
    assert runner.repository.get(run_id)["completed_at"] == before["completed_at"]


def test_legacy_completion_migration_preserves_update_time(runner):
    from core.run_repository import RunRepository
    run_id = runner.create("legacy", max_revisions=2)["run_id"]
    runner.run(run_id)
    before = runner.repository.get(run_id)
    ended = runner.repository.attempts(run_id)[-1]["ended_at"]
    with runner.repository.connect() as db:
        db.execute("ALTER TABLE runs DROP COLUMN completed_at")
    migrated = RunRepository(runner.data_dir)
    assert migrated.get(run_id)["completed_at"] == ended
    assert migrated.get(run_id)["updated_at"] == before["updated_at"]
    assert RunRepository(runner.data_dir).get(run_id) == migrated.get(run_id)
