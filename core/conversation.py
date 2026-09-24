"""Durable PostgreSQL research operations, separate from research checkpoints."""
from __future__ import annotations

import json
from uuid import uuid4

from core.persistence import validate_run_id
from core.postgres import connect, lock_key
from core.service_limits import QueueFullError, capacity


class ConversationRepository:
    def __init__(self, runner):
        if runner.storage["backend"] != "postgres":
            raise ValueError("后台报告操作仅支持 PostgreSQL 任务")
        with connect() as db:
            if not db.execute("SELECT 1 FROM schema_migrations WHERE version=6").fetchone():
                raise RuntimeError("请先运行 python -m scripts.postgres_admin init 应用研究操作迁移")
            if not db.execute("SELECT 1 FROM schema_migrations WHERE version=10").fetchone():
                raise RuntimeError("请先运行 python -m scripts.postgres_admin init 应用意图审计迁移")
        self.runner = runner

    @staticmethod
    def _row(row):
        return dict(row) if row is not None else None

    def list(self, run_id, *, limit=50):
        validate_run_id(run_id)
        with connect() as db:
            rows = db.execute("""SELECT * FROM conversation_turns WHERE run_id=%s
                ORDER BY turn_seq DESC LIMIT %s""", (run_id, limit)).fetchall()
        return [self._row(row) for row in reversed(rows)]

    def list_versions(self, run_id):
        validate_run_id(run_id)
        with connect() as db:
            rows = db.execute("""SELECT v.* FROM report_versions v
                JOIN conversation_turns t ON t.turn_id=v.operation_turn_id
                WHERE v.run_id=%s ORDER BY t.turn_seq""", (run_id,)).fetchall()
        return [self._row(row) for row in rows]

    def list_intent_events(self, run_id, *, limit=50):
        validate_run_id(run_id)
        with connect() as db:
            rows = db.execute("""SELECT * FROM intent_decision_events WHERE run_id=%s
                ORDER BY created_at DESC,event_id DESC LIMIT %s""", (run_id, limit)).fetchall()
        return [self._row(row) for row in rows]

    def record_intent_event(self, run_id, trace, *, final_intent):
        """Persist routing metadata without the question, model response, or API key."""
        validate_run_id(run_id)
        event_id = uuid4().hex
        probabilities = trace.get("jev_probabilities") or {}
        with connect() as db:
            db.execute("""INSERT INTO intent_decision_events
                (event_id,run_id,selection_mode,decision_route,final_intent,
                 jev_provider,jev_model,jev_status,jev_call_succeeded,jev_accepted,
                 jev_choice,jev_confidence,follow_up_probability,update_probability,
                 rewrite_probability,verify_probability,new_research_probability,clarify_probability,
                 jev_probabilities,jev_error_type,
                 jev_latency_ms,llm_status,llm_latency_ms,jev_http_status,jev_provider_error_type)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s,%s)""",
                (event_id, run_id, trace["selection_mode"], trace["decision_route"], final_intent,
                 trace.get("jev_provider"), trace.get("jev_model"),
                 trace.get("jev_status", "not_attempted"),
                 trace.get("jev_status") == "success", bool(trace.get("jev_accepted")),
                 trace.get("jev_choice"), trace.get("jev_confidence"),
                 probabilities.get("follow_up"), probabilities.get("update"),
                 probabilities.get("rewrite"), probabilities.get("verify"),
                 probabilities.get("new_research"), probabilities.get("clarify"),
                 json.dumps(probabilities), trace.get("jev_error_type"),
                 trace.get("jev_latency_ms"), trace.get("llm_status", "not_called"),
                 trace.get("llm_latency_ms"), trace.get("jev_http_status"),
                 trace.get("jev_provider_error_type")))
        return event_id

    @staticmethod
    def _link_intent_event(db, event_id, run_id, turn_id):
        if event_id:
            linked = db.execute("""UPDATE intent_decision_events SET turn_id=%s
                WHERE event_id=%s AND run_id=%s AND turn_id IS NULL""",
                (turn_id, event_id, run_id))
            if linked.rowcount != 1:
                raise ValueError("意图审计记录无法关联提交任务")

    def submit(self, run_id, question, *, idempotency_key, request=None,
               intent_event_id=None):
        validate_run_id(run_id)
        question = question.strip()
        if not question or len(question) > 4000:
            raise ValueError("操作内容长度须在 1 至 4000 字之间")
        if not idempotency_key or len(idempotency_key) > 128:
            raise ValueError("无效的提交幂等键")
        request = request or {}
        if not isinstance(request, dict) or len(json.dumps(request, ensure_ascii=False)) > 2000:
            raise ValueError("无效的操作约束")
        kind = request.get("intent", "follow_up")
        if kind not in {"follow_up", "update", "rewrite", "verify"}:
            raise ValueError("无效的研究操作类型")
        info = self.runner.peek(run_id)
        if (info["record"]["status"] != "completed" or info["next"] or
                not (info["state"].get("final_report") or info["state"].get("draft"))):
            raise ValueError("只能操作已完成且有报告的任务")
        with connect() as db:
            db.execute("SELECT pg_advisory_xact_lock(%s)",
                       (lock_key("apexlogic:conversation:submit"),))
            existing = db.execute("SELECT * FROM conversation_turns WHERE idempotency_key=%s",
                                  (idempotency_key,)).fetchone()
            if existing:
                if (existing["run_id"] != run_id or existing["question"] != question or
                        (existing["request_json"] or {}).get("intent", "follow_up") != kind):
                    raise ValueError("幂等键已用于其他追问")
                self._link_intent_event(db, intent_event_id, run_id, existing["turn_id"])
                return self._row(existing)
            pending_same = db.execute("""SELECT * FROM conversation_turns
                WHERE run_id=%s AND question=%s AND coalesce(request_json->>'intent','follow_up')=%s
                    AND status IN ('queued','running')
                ORDER BY turn_seq LIMIT 1""", (run_id, question, kind)).fetchone()
            if pending_same:
                self._link_intent_event(db, intent_event_id, run_id, pending_same["turn_id"])
                return self._row(pending_same)
            waiting = db.execute("""SELECT count(*) AS n FROM conversation_turns
                WHERE status IN ('queued','running')""").fetchone()["n"]
            if waiting >= capacity("APEXLOGIC_FOLLOWUP_MAX_PENDING", 100, maximum=100000):
                raise QueueFullError("报告操作队列已满")
            turn_id = uuid4().hex
            row = db.execute("""INSERT INTO conversation_turns
                (turn_id,run_id,idempotency_key,question,request_json,status)
                VALUES (%s,%s,%s,%s,%s::jsonb,'queued') RETURNING *""",
                (turn_id, run_id, idempotency_key, question,
                 json.dumps(request, ensure_ascii=False))).fetchone()
            self._link_intent_event(db, intent_event_id, run_id, turn_id)
        return self._row(row)

    def claim(self, worker_id, *, lease_seconds):
        with connect() as db:
            row = db.execute("""SELECT current_turn.turn_id FROM conversation_turns current_turn WHERE
                ((current_turn.status='queued' AND (current_turn.last_error IS NULL OR
                    current_turn.updated_at<now()-interval '10 seconds')) OR
                 (current_turn.status='running' AND current_turn.lease_until<now()))
                AND NOT EXISTS (SELECT 1 FROM conversation_turns earlier
                    WHERE earlier.run_id=current_turn.run_id
                    AND earlier.status IN ('queued','running')
                    AND earlier.turn_seq<current_turn.turn_seq)
                ORDER BY current_turn.turn_seq
                LIMIT 1 FOR UPDATE OF current_turn SKIP LOCKED""").fetchone()
            if row is None:
                return None
            return self._row(db.execute("""UPDATE conversation_turns SET status='running',
                lease_owner=%s,lease_until=now()+(%s * interval '1 second'),
                attempts=attempts+1,updated_at=now(),last_error=NULL
                WHERE turn_id=%s RETURNING *""",
                (worker_id, lease_seconds, row["turn_id"])).fetchone())

    def heartbeat(self, turn_id, worker_id, *, lease_seconds):
        with connect() as db:
            result = db.execute("""UPDATE conversation_turns SET
                lease_until=now()+(%s * interval '1 second'),updated_at=now()
                WHERE turn_id=%s AND status='running' AND lease_owner=%s""",
                (lease_seconds, turn_id, worker_id))
        return result.rowcount == 1

    def complete(self, turn_id, worker_id, *, answer, citation_ids, checkpoint_id,
                 result=None, report_version=None):
        payload = result or {}
        with connect() as db:
            completed = db.execute("""UPDATE conversation_turns SET status='completed',
                answer=%s,citation_ids=%s::jsonb,source_checkpoint_id=%s,
                result_json=%s::jsonb,
                lease_owner=NULL,lease_until=NULL,updated_at=now()
                WHERE turn_id=%s AND status='running' AND lease_owner=%s
                    AND lease_until>now() RETURNING run_id""",
                (answer, json.dumps(citation_ids), checkpoint_id,
                 json.dumps(payload, ensure_ascii=False), turn_id, worker_id)).fetchone()
            if completed and report_version:
                db.execute("""INSERT INTO report_versions
                    (version_id,run_id,operation_turn_id,parent_version_id,kind,
                     report_markdown,source_manifest,source_checkpoint_id)
                    VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,%s)""",
                    (turn_id, completed["run_id"], turn_id,
                     report_version.get("parent_version_id"), report_version["kind"],
                     report_version["report_markdown"],
                     json.dumps(report_version.get("source_manifest", []), ensure_ascii=False),
                     checkpoint_id))
        return completed is not None

    def fail(self, turn_id, worker_id, error_type):
        with connect() as db:
            db.execute("""UPDATE conversation_turns SET
                status=CASE WHEN attempts>=3 THEN 'failed' ELSE 'queued' END,
                lease_owner=NULL,lease_until=NULL,last_error=%s,updated_at=now()
                WHERE turn_id=%s AND status='running' AND lease_owner=%s""",
                (error_type[:100], turn_id, worker_id))
