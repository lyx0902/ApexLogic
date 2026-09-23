"""PostgreSQL task repository; preserves the SQLite repository contract."""
import json
from uuid import uuid4
from core.persistence import validate_run_id
from core.run_config import SCHEMA_VERSION, WORKFLOW_VERSION, config_hash
from core.run_repository import utc_now
from core.postgres import connect, require_schema
from core.service_limits import reserve_queue_room


class PostgresRunRepository:
    def __init__(self):
        require_schema()

    connect = staticmethod(connect)

    @staticmethod
    def decode(row):
        record = dict(row)
        record["run_config"] = json.loads(record.pop("config_json"))
        return record

    def create(self, topic: str, config: dict, *, enqueue=False):
        if not topic.strip():
            raise ValueError("研究主题不能为空")
        run_id = uuid4().hex
        now = utc_now()
        with self.connect() as db:
            if enqueue:
                reserve_queue_room(db)
            db.execute("""INSERT INTO runs
                (run_id,thread_id,topic,status,created_at,updated_at,config_json,config_hash,schema_version,workflow_version)
                VALUES (%s,%s,%s,'created',%s,%s,%s,%s,%s,%s)""",
                (run_id, run_id, topic.strip(), now, now, json.dumps(config, ensure_ascii=False),
                 config_hash(config), SCHEMA_VERSION, WORKFLOW_VERSION))
            if enqueue:
                db.execute("INSERT INTO research_jobs(run_id,status,updated_at) VALUES (%s,'queued',now())",
                           (run_id,))
                db.execute("INSERT INTO research_outbox(run_id,created_at) VALUES (%s,now())", (run_id,))
        return self.get(run_id)

    def get(self, run_id):
        validate_run_id(run_id)
        with self.connect() as db:
            row = db.execute("SELECT * FROM runs WHERE run_id=%s", (run_id,)).fetchone()
        if row is None:
            raise ValueError("未找到研究任务")
        return self.decode(row)

    def list(self, limit=100):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT %s", (limit,)).fetchall()
        return [self.decode(row) for row in rows]

    def update(self, run_id, **fields):
        allowed = {"status", "termination_reason", "checkpoint_seen", "last_error", "export_status", "completed_at"}
        if not fields or not set(fields) <= allowed:
            raise ValueError("不可更新的任务字段")
        with self.connect() as db:
            row = db.execute("SELECT * FROM runs WHERE run_id=%s FOR UPDATE", (run_id,)).fetchone()
            if row is None:
                raise ValueError("未找到研究任务")
            if row["completed_at"]:
                fields.pop("completed_at", None)
            fields = {k: v for k, v in fields.items() if row[k] != v}
            if not fields:
                return
            fields["updated_at"] = utc_now()
            db.execute(f"UPDATE runs SET {','.join(k+'=%s' for k in fields)} WHERE run_id=%s",
                       (*fields.values(), run_id))

    def reconcile_attempts(self, run_id):
        # Called only while holding the OS execution lock. A crashed process did
        # not report its duration; leave it unknown instead of counting downtime.
        with self.connect() as db:
            db.execute("UPDATE attempts SET status='interrupted', ended_at=%s WHERE run_id=%s AND status='running'",
                       (utc_now(), run_id))

    def start_attempt(self, run_id):
        attempt_id = uuid4().hex
        with self.connect() as db:
            db.execute("INSERT INTO attempts(attempt_id,run_id,started_at,status) VALUES (%s,%s,%s,'running')",
                       (attempt_id, run_id, utc_now()))
        return attempt_id

    def finish_attempt(self, attempt_id, status, elapsed, error=None):
        with self.connect() as db:
            db.execute("UPDATE attempts SET status=%s,ended_at=%s,elapsed_seconds=%s,error=%s WHERE attempt_id=%s",
                       (status, utc_now(), elapsed, error, attempt_id))

    def attempts(self, run_id):
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM attempts WHERE run_id=%s ORDER BY started_at", (run_id,))]
