"""Task index and attempts. Checkpoints remain authoritative for execution."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

from core.persistence import validate_run_id
from core.run_config import SCHEMA_VERSION, WORKFLOW_VERSION, config_hash


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class RunRepository:
    def __init__(self, data_dir: Path):
        data_dir.mkdir(parents=True, exist_ok=True)
        self.path = data_dir / "runs.sqlite"
        with self.connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY, thread_id TEXT NOT NULL UNIQUE, topic TEXT NOT NULL,
                status TEXT NOT NULL, termination_reason TEXT, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, config_json TEXT NOT NULL, config_hash TEXT NOT NULL,
                schema_version INTEGER NOT NULL, workflow_version TEXT NOT NULL,
                checkpoint_seen INTEGER NOT NULL DEFAULT 0,
                last_error TEXT, export_status TEXT NOT NULL DEFAULT 'pending'
            );
            CREATE TABLE IF NOT EXISTS attempts (
                attempt_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
                started_at TEXT NOT NULL, ended_at TEXT, status TEXT NOT NULL,
                elapsed_seconds REAL, error TEXT
            );
            """)

            db.execute("BEGIN IMMEDIATE")
            columns = {row["name"] for row in db.execute("PRAGMA table_info(runs)")}
            if "completed_at" not in columns:
                db.execute("ALTER TABLE runs ADD COLUMN completed_at TEXT")
            # Recover known legacy completion times without inventing a timestamp.
            db.execute("""UPDATE runs SET completed_at=(SELECT MAX(ended_at) FROM attempts
                WHERE attempts.run_id=runs.run_id AND attempts.status='completed')
                WHERE status='completed' AND completed_at IS NULL""")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(str(self.path), timeout=30)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys=ON")
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def decode(row):
        record = dict(row)
        record["run_config"] = json.loads(record.pop("config_json"))
        return record

    def create(self, topic: str, config: dict):
        if not topic.strip():
            raise ValueError("研究主题不能为空")
        run_id = uuid4().hex
        now = utc_now()
        with self.connect() as db:
            db.execute("""INSERT INTO runs
                (run_id,thread_id,topic,status,created_at,updated_at,config_json,config_hash,schema_version,workflow_version)
                VALUES (?,?,?,'created',?,?,?,?,?,?)""",
                (run_id, run_id, topic.strip(), now, now, json.dumps(config, ensure_ascii=False),
                 config_hash(config), SCHEMA_VERSION, WORKFLOW_VERSION))
        return self.get(run_id)

    def get(self, run_id):
        validate_run_id(run_id)
        with self.connect() as db:
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise ValueError("未找到研究任务")
        return self.decode(row)

    def list(self, limit=100):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self.decode(row) for row in rows]

    def update(self, run_id, **fields):
        allowed = {"status", "termination_reason", "checkpoint_seen", "last_error", "export_status", "completed_at"}
        if not fields or not set(fields) <= allowed:
            raise ValueError("不可更新的任务字段")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise ValueError("未找到研究任务")
            if row["completed_at"]:
                fields.pop("completed_at", None)
            fields = {k: v for k, v in fields.items() if row[k] != v}
            if not fields:
                return
            fields["updated_at"] = utc_now()
            db.execute(f"UPDATE runs SET {','.join(k+'=?' for k in fields)} WHERE run_id=?",
                       (*fields.values(), run_id))

    def reconcile_attempts(self, run_id):
        # Called only while holding the OS execution lock. A crashed process did
        # not report its duration; leave it unknown instead of counting downtime.
        with self.connect() as db:
            db.execute("UPDATE attempts SET status='interrupted', ended_at=? WHERE run_id=? AND status='running'",
                       (utc_now(), run_id))

    def start_attempt(self, run_id):
        attempt_id = uuid4().hex
        with self.connect() as db:
            db.execute("INSERT INTO attempts(attempt_id,run_id,started_at,status) VALUES (?,?,?,'running')",
                       (attempt_id, run_id, utc_now()))
        return attempt_id

    def finish_attempt(self, attempt_id, status, elapsed, error=None):
        with self.connect() as db:
            db.execute("UPDATE attempts SET status=?,ended_at=?,elapsed_seconds=?,error=? WHERE attempt_id=?",
                       (status, utc_now(), elapsed, error, attempt_id))

    def attempts(self, run_id):
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM attempts WHERE run_id=? ORDER BY started_at", (run_id,))]
