"""PostgreSQL is authoritative; Redis Streams only wakes background workers."""
from __future__ import annotations

import os
import logging
import socket
import threading
import time
from uuid import uuid4

from core.persistence import RunBusyError, validate_run_id
from core.postgres import connect
from core.runner import ResearchRunner
from core.conversation import ConversationRepository
from core.followup import answer_followup
from core.report_operations import execute_report_operation
from core.service_limits import capacity, global_slot, reserve_queue_room


GROUP = "apexlogic-workers"
STREAM = "apexlogic:research:v1"
logger = logging.getLogger(__name__)


def queue_client():
    """A separate, persistent Redis instance; never reuse the cache connection."""
    url = os.getenv("APEXLOGIC_QUEUE_REDIS_URL", "").strip()
    if not url:
        raise RuntimeError("Set APEXLOGIC_QUEUE_REDIS_URL to the dedicated queue Redis instance")
    import redis
    client = redis.Redis.from_url(url, socket_connect_timeout=3, socket_timeout=5,
                                  decode_responses=True)
    client.ping()
    return client


class BackgroundScheduler:
    def __init__(self, runner: ResearchRunner):
        if runner.storage["backend"] != "postgres":
            raise ValueError("后台调度仅支持 PostgreSQL；旧 SQLite 任务仍按原方式执行")
        with connect() as db:
            if not db.execute("SELECT 1 FROM schema_migrations WHERE version=4").fetchone():
                raise RuntimeError("后台调度表尚未初始化；请运行 python -m scripts.postgres_admin init")
        self.runner = runner

    def submit(self, topic, *, max_revisions=None, pass_threshold=None, output_mode="user"):
        # Repository inserts run, job and outbox within one database transaction.
        config = self.runner._new_config(max_revisions=max_revisions,
                                         pass_threshold=pass_threshold, output_mode=output_mode)
        return self.runner.repository.create(topic, config, enqueue=True)

    def enqueue(self, run_id):
        """Idempotent resume of a previously created or interrupted run."""
        record = self.runner._record(run_id)
        validate_run_id(run_id)
        with connect() as db:
            row = db.execute("SELECT status,cancel_requested FROM research_jobs WHERE run_id=%s FOR UPDATE",
                             (run_id,)).fetchone()
            if record["status"] == "completed":
                return "completed"
            if row and row["status"] == "running":
                return "cancel_requested" if row["cancel_requested"] else "running"
            if row and row["status"] == "queued" and not row["cancel_requested"]:
                return "queued"
            reserve_queue_room(db)
            db.execute("""INSERT INTO research_jobs(run_id,status,updated_at)
                VALUES (%s,'queued',now()) ON CONFLICT(run_id) DO UPDATE
                SET status='queued',cancel_requested=false,lease_owner=NULL,lease_until=NULL,
                    last_error=NULL,updated_at=now()""", (run_id,))
            db.execute("INSERT INTO research_outbox(run_id,created_at) VALUES (%s,now())", (run_id,))
        return "queued"

    def status(self, run_id):
        validate_run_id(run_id)
        with connect() as db:
            row = db.execute("SELECT * FROM research_jobs WHERE run_id=%s", (run_id,)).fetchone()
            if not row:
                return None
            result = dict(row)
            if row["status"] == "queued" and not row["cancel_requested"]:
                position = db.execute("""SELECT count(*) AS n FROM research_jobs j
                    JOIN runs r ON r.run_id=j.run_id
                    JOIN runs target ON target.run_id=%s
                    WHERE j.status='queued' AND j.cancel_requested=false AND
                    (r.created_at<target.created_at OR
                     (r.created_at=target.created_at AND j.run_id<=%s))""",
                    (run_id, run_id)).fetchone()
                result["queue_position"] = position["n"]
        return result

    def cancel(self, run_id):
        validate_run_id(run_id)
        with connect() as db:
            row = db.execute("SELECT status FROM research_jobs WHERE run_id=%s FOR UPDATE",
                             (run_id,)).fetchone()
            if not row:
                raise ValueError("任务未排入后台队列")
            if row["status"] in {"completed", "cancelled"}:
                return row["status"]
            if row["status"] == "queued":
                db.execute("""UPDATE research_jobs SET status='cancelled',cancel_requested=true,
                    updated_at=now() WHERE run_id=%s""", (run_id,))
                return "cancelled"
            db.execute("UPDATE research_jobs SET cancel_requested=true,updated_at=now() WHERE run_id=%s",
                       (run_id,))
        return "cancel_requested"


class BackgroundWorker:
    def __init__(self, *, runner=None, client=None, worker_id=None, lease_seconds=30):
        self.runner = runner or ResearchRunner(backend="postgres")
        self.scheduler = BackgroundScheduler(self.runner)
        self.conversation = ConversationRepository(self.runner)
        self.client = client or queue_client()
        self.worker_id = worker_id or os.getenv("APEXLOGIC_WORKER_ID") or f"{socket.gethostname()}-{os.getpid()}-{uuid4().hex[:8]}"
        self.manager_id = os.getenv("APEXLOGIC_WORKER_MANAGER_ID") or None
        self._drain_event = threading.Event()
        self.lease_seconds = lease_seconds
        try:
            self.client.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
        except Exception as exc:
            from redis.exceptions import ResponseError
            if not isinstance(exc, ResponseError) or "BUSYGROUP" not in str(exc):
                raise

    def dispatch(self, limit=20):
        """Crash after XADD may duplicate a message; claiming remains idempotent."""
        sent = 0
        for _ in range(limit):
            with connect() as db:
                row = db.execute("""SELECT id,run_id FROM research_outbox
                    WHERE delivered_at IS NULL ORDER BY id LIMIT 1 FOR UPDATE SKIP LOCKED""").fetchone()
                if not row:
                    break
                self.client.xadd(STREAM, {"run_id": row["run_id"]})
                db.execute("UPDATE research_outbox SET delivered_at=now() WHERE id=%s", (row["id"],))
                sent += 1
        return sent

    def _claim(self, run_id):
        with connect() as db:
            # Serialize claims and admit the oldest eligible job first, even
            # when Redis messages arrive out of order on different Workers.
            from core.postgres import lock_key
            db.execute("SELECT pg_advisory_xact_lock(%s)",
                       (lock_key("apexlogic:queue:claim"),))
            oldest = db.execute("""SELECT j.run_id FROM research_jobs j
                JOIN runs r ON r.run_id=j.run_id
                WHERE j.cancel_requested=false AND
                ((j.status='queued' AND (j.last_error IS NULL OR
                  j.updated_at<now()-interval '30 seconds')) OR
                 (j.status='running' AND j.lease_until<now()))
                ORDER BY r.created_at,j.run_id LIMIT 1""").fetchone()
            if not oldest or oldest["run_id"] != run_id:
                return False
            row = db.execute("""UPDATE research_jobs SET status='running',lease_owner=%s,
                lease_until=now()+(%s * interval '1 second'),updated_at=now(),
                last_error=NULL WHERE run_id=%s AND cancel_requested=false AND
                (status='queued' OR (status='running' AND lease_until<now()))
                RETURNING run_id""", (self.worker_id, self.lease_seconds, run_id)).fetchone()
        return bool(row)

    def _heartbeat(self, run_id, stop, lost_lease):
        while not stop.wait(max(1, self.lease_seconds // 3)):
            try:
                with connect() as db:
                    updated = db.execute("""UPDATE research_jobs
                        SET lease_until=now()+(%s * interval '1 second'),updated_at=now()
                        WHERE run_id=%s AND status='running' AND lease_owner=%s""",
                        (self.lease_seconds, run_id, self.worker_id))
                    if updated.rowcount == 0:
                        lost_lease.set()
                        return
            except Exception as exc:
                # Retry transient DB failures; the node-boundary ownership check
                # stops this executor if another worker claims the expired lease.
                logger.warning("Worker heartbeat failed for %s: %s", run_id, type(exc).__name__)

    def _finish(self, run_id, status, error=None):
        with connect() as db:
            db.execute("""UPDATE research_jobs SET status=CASE
                WHEN cancel_requested AND %s='queued' THEN 'cancelled' ELSE %s END,
                lease_owner=NULL,lease_until=NULL,
                last_error=%s,updated_at=now() WHERE run_id=%s AND lease_owner=%s""",
                (status, status, error, run_id, self.worker_id))

    def process(self, run_id):
        if self._drain_event.is_set():
            return False
        validate_run_id(run_id)
        with global_slot("research", capacity("APEXLOGIC_GLOBAL_RESEARCH_MAX_INFLIGHT", 2)) as admitted:
            if not admitted:
                return False
            return self._process_with_slot(run_id)

    def _process_with_slot(self, run_id):
        if not self._claim(run_id):
            return False
        stop = threading.Event()
        lost_lease = threading.Event()
        heartbeat = threading.Thread(target=self._heartbeat,
                                     args=(run_id, stop, lost_lease), daemon=True)
        heartbeat.start()
        try:
            # ResearchRunner owns config validation, checkpoint recovery and the execution lock.
            from contextlib import closing
            final_state = None
            with closing(self.runner.stream(run_id)) as events:
                for event in events:
                    if event.kind == "complete":
                        final_state = event.state
                    if event.kind == "node":
                        with connect() as db:
                            row = db.execute("""UPDATE research_jobs SET last_node=%s,
                                updated_at=now() WHERE run_id=%s AND lease_owner=%s
                                RETURNING cancel_requested""",
                                (event.node, run_id, self.worker_id)).fetchone()
                        if row is None or lost_lease.is_set():
                            raise RunBusyError("Worker 已失去任务租约")
                        if row and row["cancel_requested"]:
                            self._finish(run_id, "cancelled")
                            return True
            if lost_lease.is_set():
                raise RunBusyError("Worker 已失去任务租约")
            try:
                from core.history import history_from_run
                history_from_run(self.runner, run_id, final_state=final_state, write=True)
            except Exception as exc:
                # A presentation export failure must never rerun completed research.
                self._finish(run_id, "completed", f"HistoryExport:{type(exc).__name__}")
            else:
                self._finish(run_id, "completed")
        except RunBusyError:
            # A worker with an expired lease may still hold the authoritative run lock.
            self._finish(run_id, "queued", "RunBusyError")
        except Exception as exc:
            self._finish(run_id, "queued", type(exc).__name__)
            time.sleep(1)
        finally:
            stop.set()
            heartbeat.join(timeout=2)
        return True

    def _sweep(self):
        if self._drain_event.is_set():
            return
        # PostgreSQL reconciliation also covers missing Redis messages or a flushed stream.
        self.process_followup_once()
        with connect() as db:
            cancelled = db.execute("""SELECT run_id FROM research_jobs WHERE status='running'
                AND cancel_requested=true AND lease_until<now() LIMIT 20""").fetchall()
            rows = db.execute("""SELECT run_id FROM research_jobs WHERE cancel_requested=false AND
                ((status='queued' AND (last_error IS NULL OR
                  updated_at<now()-interval '30 seconds')) OR
                 (status='running' AND lease_until<now()))
                ORDER BY (SELECT created_at FROM runs WHERE runs.run_id=research_jobs.run_id),run_id
                LIMIT 20""").fetchall()
        for row in cancelled:
            try:
                # Only settle a lost worker after the authoritative run lock is free.
                with self.runner._lock(row["run_id"]):
                    with connect() as db:
                        db.execute("""UPDATE research_jobs SET status='cancelled',lease_owner=NULL,
                            lease_until=NULL,updated_at=now() WHERE run_id=%s AND status='running'
                            AND cancel_requested=true AND lease_until<now()""", (row["run_id"],))
            except RunBusyError:
                pass
        for row in rows:
            if self._drain_event.is_set():
                break
            self.process(row["run_id"])
            # A large recovery backlog must not defer all follow-up turns.
            self.process_followup_once()

    def process_followup_once(self):
        """PostgreSQL polling is the durable dispatch path for short follow-up turns."""
        if getattr(self, "_drain_event", None) is not None and self._drain_event.is_set():
            return False
        with global_slot("followup", capacity("APEXLOGIC_GLOBAL_FOLLOWUP_MAX_INFLIGHT", 2)) as admitted:
            if not admitted:
                return False
            turn = self.conversation.claim(self.worker_id, lease_seconds=self.lease_seconds)
            if turn is None:
                return False
            stop = threading.Event()
            lost_lease = threading.Event()

            def heartbeat():
                while not stop.wait(max(1, self.lease_seconds // 3)):
                    try:
                        if not self.conversation.heartbeat(turn["turn_id"], self.worker_id,
                                                           lease_seconds=self.lease_seconds):
                            lost_lease.set()
                            return
                    except Exception as exc:
                        logger.warning("Follow-up heartbeat failed for %s: %s",
                                       turn["turn_id"], type(exc).__name__)

            thread = threading.Thread(target=heartbeat, daemon=True)
            thread.start()
            try:
                kind = (turn.get("request_json") or {}).get("intent", "follow_up")
                if kind == "follow_up":
                    prior = self.conversation.list(turn["run_id"])
                    answer, citations, checkpoint_id = answer_followup(
                        self.runner, turn, prior_turns=prior)
                    result, report_version = {}, None
                else:
                    versions = self.conversation.list_versions(turn["run_id"])
                    answer, citations, checkpoint_id, result, report_version = (
                        execute_report_operation(self.runner, turn, prior_versions=versions))
                if not lost_lease.is_set():
                    self.conversation.complete(turn["turn_id"], self.worker_id,
                                               answer=answer, citation_ids=citations,
                                               checkpoint_id=checkpoint_id,
                                               result=result, report_version=report_version)
            except Exception as exc:
                self.conversation.fail(turn["turn_id"], self.worker_id, type(exc).__name__)
                logger.warning("Follow-up attempt failed for %s: %s",
                               turn["turn_id"], type(exc).__name__)
            finally:
                stop.set()
                thread.join(timeout=2)
            return True

    def run_forever(self):
        from core.worker_control import WorkerPresence
        presence = WorkerPresence(self.worker_id, self.manager_id)
        presence.drain_requested = self._drain_event
        presence.start()
        try:
            while not self._drain_event.is_set():
                try:
                    self.dispatch()
                    # Recover abandoned consumer-group messages before taking new ones.
                    claimed = self.client.xautoclaim(STREAM, GROUP, self.worker_id, 30_000, "0-0", count=20)
                    messages = claimed[1] if claimed else []
                    if not messages:
                        batches = self.client.xreadgroup(GROUP, self.worker_id, {STREAM: ">"}, count=1,
                                                         block=1000)
                        messages = batches[0][1] if batches else []
                    for message_id, data in messages:
                        if self._drain_event.is_set():
                            break
                        try:
                            self.process(data["run_id"])
                        finally:
                            if self.client.xack(STREAM, GROUP, message_id):
                                self.client.xdel(STREAM, message_id)
                    self._sweep()
                except KeyboardInterrupt:
                    raise
                except Exception:
                    # Redis failure does not discard PostgreSQL work; retry and sweep.
                    try:
                        self._sweep()
                    except Exception:
                        pass
                    time.sleep(2)
        finally:
            presence.stop()
