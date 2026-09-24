"""PostgreSQL-backed presence and local Streamlit control of independent Workers."""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from uuid import uuid4

from core.persistence import data_directory
from core.postgres import connect, lock_key
from core.service_limits import capacity


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEARTBEAT_SECONDS = 5
LIVE_SECONDS = 20


def ui_manager_id() -> str:
    identity = f"{socket.gethostname()}:{PROJECT_ROOT}".encode("utf-8")
    return "streamlit:" + hashlib.sha256(identity).hexdigest()[:24]


def worker_limit() -> int:
    """Keep the UI process count within the existing global job admission caps."""
    return min(8, capacity("APEXLOGIC_GLOBAL_RESEARCH_MAX_INFLIGHT", 2),
               capacity("APEXLOGIC_GLOBAL_FOLLOWUP_MAX_INFLIGHT", 2))


class WorkerPresence:
    """Keep an idle or busy Worker visible; a drain request ends it between jobs."""

    def __init__(self, worker_id: str, manager_id: str | None = None):
        self.worker_id = worker_id
        self.manager_id = manager_id
        self.drain_requested = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _beat(self) -> None:
        with connect() as db:
            row = db.execute("""INSERT INTO worker_instances
                (worker_id,manager_id,hostname,pid,phase,heartbeat_at)
                VALUES (%s,%s,%s,%s,'online',now())
                ON CONFLICT(worker_id) DO UPDATE SET
                    manager_id=EXCLUDED.manager_id,hostname=EXCLUDED.hostname,
                    pid=EXCLUDED.pid,phase='online',heartbeat_at=now()
                RETURNING drain_requested""",
                (self.worker_id, self.manager_id, socket.gethostname(), os.getpid())).fetchone()
        if row["drain_requested"]:
            self.drain_requested.set()

    def _loop(self) -> None:
        while not self._stop.wait(HEARTBEAT_SECONDS):
            try:
                self._beat()
            except Exception as exc:
                logger.warning("Worker presence heartbeat failed: %s", type(exc).__name__)

    def start(self) -> None:
        self._beat()
        if self.manager_id:
            with connect() as db:
                db.execute("""UPDATE worker_targets SET retry_after=NULL,last_error=NULL
                    WHERE manager_id=%s""", (self.manager_id,))
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        try:
            with connect() as db:
                db.execute("DELETE FROM worker_instances WHERE worker_id=%s", (self.worker_id,))
        except Exception as exc:
            logger.warning("Worker presence cleanup failed: %s", type(exc).__name__)


class StreamlitWorkerManager:
    """Reconcile one local UI's target without touching manually started Workers."""

    def __init__(self, manager_id: str | None = None):
        self.manager_id = manager_id or ui_manager_id()
        with connect() as db:
            if not db.execute("SELECT 1 FROM schema_migrations WHERE version=7").fetchone():
                raise RuntimeError("请先运行 python -m scripts.postgres_admin init 应用 Worker 状态迁移")

    def snapshot(self) -> dict:
        with connect() as db:
            rows = db.execute("""SELECT worker_id,manager_id,phase,drain_requested
                FROM worker_instances WHERE heartbeat_at > now() - interval '20 seconds'""").fetchall()
            target = db.execute("SELECT desired_count,last_error FROM worker_targets WHERE manager_id=%s",
                                (self.manager_id,)).fetchone()
            busy = db.execute("""SELECT lease_owner FROM research_jobs
                    WHERE status='running' AND lease_until>now() AND lease_owner IS NOT NULL
                UNION SELECT lease_owner FROM conversation_turns
                    WHERE status='running' AND lease_until>now() AND lease_owner IS NOT NULL""").fetchall()
        online = [row for row in rows if row["phase"] == "online"]
        busy_ids = {row["lease_owner"] for row in busy}
        return {"online": len(online), "busy": sum(row["worker_id"] in busy_ids for row in online),
                "starting": sum(row["phase"] == "starting" for row in rows),
                "draining": sum(row["drain_requested"] for row in online),
                "external": sum(row["manager_id"] != self.manager_id for row in online),
                "desired": target["desired_count"] if target else None,
                "last_error": target["last_error"] if target else None}

    def set_target(self, target: int) -> None:
        if not 0 <= target <= worker_limit():
            raise ValueError("Worker 数量超出当前全局并发上限")
        with connect() as db:
            db.execute("SELECT pg_advisory_xact_lock(%s)",
                       (lock_key("apexlogic:worker-manager:" + self.manager_id),))
            db.execute("""INSERT INTO worker_targets(manager_id,desired_count)
                VALUES (%s,%s) ON CONFLICT(manager_id) DO UPDATE
                SET desired_count=EXCLUDED.desired_count,retry_after=NULL,
                    last_error=NULL,updated_at=now()""",
                (self.manager_id, target))

    def reconcile(self, default_count: int) -> dict:
        """Reserve starts under a PG lock, then spawn outside the transaction."""
        if not 0 <= default_count <= worker_limit():
            raise ValueError("无效的初始 Worker 数量")
        starts: list[str] = []
        with connect() as db:
            db.execute("SELECT pg_advisory_xact_lock(%s)",
                       (lock_key("apexlogic:worker-manager:" + self.manager_id),))
            stale_starts = db.execute("""DELETE FROM worker_instances
                WHERE heartbeat_at <= now() - interval '20 seconds' AND phase='starting'
                RETURNING manager_id""").fetchall()
            if any(row["manager_id"] == self.manager_id for row in stale_starts):
                db.execute("""UPDATE worker_targets SET last_error='WorkerStartupTimeout',
                    retry_after=now()+interval '30 seconds' WHERE manager_id=%s""",
                    (self.manager_id,))
            db.execute("""DELETE FROM worker_instances
                WHERE heartbeat_at <= now() - interval '20 seconds'""")
            db.execute("""INSERT INTO worker_targets(manager_id,desired_count)
                VALUES (%s,%s) ON CONFLICT(manager_id) DO NOTHING""",
                (self.manager_id, default_count))
            target = db.execute("""SELECT desired_count,
                COALESCE(retry_after>now(),false) AS cooling_down
                FROM worker_targets WHERE manager_id=%s""", (self.manager_id,)).fetchone()
            desired = min(target["desired_count"], worker_limit())
            if desired != target["desired_count"]:
                db.execute("""UPDATE worker_targets SET desired_count=%s,updated_at=now()
                    WHERE manager_id=%s""", (desired, self.manager_id))
            external = db.execute("""SELECT count(*) AS n FROM worker_instances
                WHERE manager_id IS DISTINCT FROM %s""", (self.manager_id,)).fetchone()["n"]
            managed = db.execute("""SELECT worker_id,phase,drain_requested,
                EXISTS(SELECT 1 FROM research_jobs WHERE lease_owner=worker_id
                    AND status='running' AND lease_until>now()) OR
                EXISTS(SELECT 1 FROM conversation_turns WHERE lease_owner=worker_id
                    AND status='running' AND lease_until>now()) AS busy
                FROM worker_instances WHERE manager_id=%s
                ORDER BY started_at DESC""", (self.manager_id,)).fetchall()
            active = [row for row in managed if not row["drain_requested"]]
            needed = max(0, desired - external)
            if len(active) > needed:
                for row in sorted(active, key=lambda item: item["busy"])[:len(active) - needed]:
                    db.execute("UPDATE worker_instances SET drain_requested=true WHERE worker_id=%s",
                               (row["worker_id"],))
            elif len(active) < needed and not target["cooling_down"]:
                for _ in range(needed - len(active)):
                    worker_id = uuid4().hex
                    db.execute("""INSERT INTO worker_instances
                        (worker_id,manager_id,hostname,pid,phase)
                        VALUES (%s,%s,%s,0,'starting')""",
                        (worker_id, self.manager_id, socket.gethostname()))
                    starts.append(worker_id)
        for worker_id in starts:
            try:
                self._spawn(worker_id)
            except Exception as exc:
                with connect() as db:
                    db.execute("""DELETE FROM worker_instances
                        WHERE worker_id=%s AND phase='starting'""", (worker_id,))
                    db.execute("""UPDATE worker_targets SET last_error=%s,
                        retry_after=now()+interval '30 seconds' WHERE manager_id=%s""",
                        (type(exc).__name__[:100], self.manager_id))
                raise
        return self.snapshot()

    def _spawn(self, worker_id: str) -> None:
        from core.background import queue_client
        queue_client().close()  # Report a missing queue before hiding a child process.
        env = os.environ.copy()
        env["APEXLOGIC_WORKER_ID"] = worker_id
        env["APEXLOGIC_WORKER_MANAGER_ID"] = self.manager_id
        folder = data_directory() / "worker_logs"
        folder.mkdir(parents=True, exist_ok=True)
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        with (folder / f"{worker_id}.log").open("ab") as log:
            child = subprocess.Popen([sys.executable, str(PROJECT_ROOT / "worker.py")],
                                     cwd=PROJECT_ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=flags, close_fds=True)
        time.sleep(0.2)
        if child.poll() is not None:
            raise RuntimeError("Worker 启动失败；请查看 data/worker_logs")
        with connect() as db:
            db.execute("""UPDATE worker_instances SET pid=%s
                WHERE worker_id=%s AND phase='starting'""", (child.pid, worker_id))
