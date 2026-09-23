"""Cross-process in-flight caps backed by PostgreSQL session advisory locks.

Each permit owns a dedicated connection. PostgreSQL releases it when a Worker
dies, so no timeout-based semaphore cleanup or Redis state is required.
"""
from contextlib import contextmanager
import os
import time

from core.postgres import connect, dependencies, dsn, lock_key


class ServiceCapacityError(RuntimeError):
    """No global permit became available within the configured wait."""


class QueueFullError(RuntimeError):
    """The configured number of waiting research jobs is already queued."""


def capacity(name: str, default: int, *, maximum: int = 64) -> int:
    try:
        return min(max(int(os.getenv(name, str(default))), 1), maximum)
    except ValueError:
        return default


def reserve_queue_room(db):
    """Serialize enqueue checks with other submitters, within their transaction."""
    db.execute("SELECT pg_advisory_xact_lock(%s)", (lock_key("apexlogic:queue:enqueue"),))
    waiting = db.execute("SELECT count(*) AS n FROM research_jobs WHERE status='queued'").fetchone()["n"]
    if waiting >= capacity("APEXLOGIC_QUEUE_MAX_PENDING", 100, maximum=100000):
        raise QueueFullError("后台等待队列已满，请稍后再提交")


def rate_ticket(provider: str, per_minute: int, *, wait_seconds: float) -> float:
    """Reserve one external request in a database-wide rolling minute.

    Tickets are recorded before an actual HTTP call. A crash may waste a
    ticket, which is safer than allowing a burst above the configured quota.
    """
    started = time.monotonic()
    deadline = started + max(0.0, wait_seconds)
    while True:
        with connect() as db:
            db.execute("SELECT pg_advisory_xact_lock(%s)",
                       (lock_key("apexlogic:rate:" + provider),))
            db.execute("""DELETE FROM provider_rate_events WHERE provider=%s
                AND requested_at<=now()-interval '1 minute'""", (provider,))
            row = db.execute("""SELECT count(*) AS n,
                COALESCE(EXTRACT(EPOCH FROM (min(requested_at)+interval '1 minute'-now())),0)
                    AS wait_seconds
                FROM provider_rate_events WHERE provider=%s""", (provider,)).fetchone()
            if row["n"] < per_minute:
                db.execute("INSERT INTO provider_rate_events(provider) VALUES (%s)", (provider,))
                return time.monotonic() - started
            delay = max(0.05, float(row["wait_seconds"]))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ServiceCapacityError(f"{provider} minute quota exhausted")
        time.sleep(min(delay, remaining, 0.5))


@contextmanager
def global_slot(kind: str, slots: int, *, wait_seconds: float = 0):
    """Yield True while holding one of `slots` database-wide permits.

    A non-blocking call yields False when full. A bounded wait raises, which
    lets retrieval audit record capacity pressure as a tool failure.
    """
    psycopg, dict_row, _ = dependencies()
    deadline = time.monotonic() + max(0.0, wait_seconds)
    keys = [lock_key(f"apexlogic:slot:{kind}:{index}") for index in range(slots)]
    with psycopg.connect(dsn(), autocommit=True, row_factory=dict_row, connect_timeout=5) as db:
        while True:
            for key in keys:
                acquired = db.execute("SELECT pg_try_advisory_lock(%s) AS acquired", (key,)).fetchone()
                if acquired["acquired"]:
                    try:
                        yield True
                    finally:
                        # Closing this dedicated session also releases its lock.
                        # Preserve the original tool/Worker exception if the
                        # database connection died while the permit was held.
                        try:
                            db.execute("SELECT pg_advisory_unlock(%s)", (key,))
                        except Exception:
                            pass
                    return
            if wait_seconds <= 0:
                yield False
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ServiceCapacityError(f"No available {kind} permit")
            time.sleep(min(0.1, remaining))
