"""Lazy PostgreSQL connections, migrations and session-scoped execution locks."""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import atexit
import hashlib
import json
import os
from pathlib import Path
from threading import Lock

from core.persistence import RunBusyError, validate_run_id

_pools = {}
_pool_lock = Lock()
_execution_connection = ContextVar("postgres_execution_connection", default=None)


def dsn():
    value = os.getenv("APEXLOGIC_POSTGRES_DSN", "").strip()
    if not value:
        raise RuntimeError("Set APEXLOGIC_POSTGRES_DSN; see docs/postgres-migration.md")
    return value


def dependencies():
    try:
        import psycopg
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool
    except ImportError:
        raise RuntimeError("Install requirements-postgres.txt first") from None
    return psycopg, dict_row, ConnectionPool


def business_row(cursor):
    """Keep repository API compatible with immutable SQLite snapshots."""
    from psycopg.rows import dict_row
    make = dict_row(cursor)
    def convert(values):
        row = make(values)
        for key, value in row.items():
            if isinstance(value, datetime):
                row[key] = value.astimezone(timezone.utc).isoformat()
            elif key in {"config_json", "item_ids", "details"} and isinstance(value, (dict, list)):
                row[key] = json.dumps(value, ensure_ascii=False)
        return row
    return convert


def pool():
    _, _, Pool = dependencies()
    address = dsn()
    with _pool_lock:
        if address not in _pools:
            _pools[address] = Pool(address, min_size=0, max_size=10, timeout=10, open=True,
                kwargs={"autocommit": True, "row_factory": business_row,
                        "connect_timeout": 5, "options": "-c search_path=apexlogic,public"})
        return _pools[address]


def close_pools():
    with _pool_lock:
        for item in _pools.values():
            item.close()
        _pools.clear()


atexit.register(close_pools)


@contextmanager
def connect():
    # Transaction ends before a pooled connection is returned.
    with pool().connection() as conn:
        with conn.transaction():
            yield conn


def require_schema():
    with connect() as conn:
        row = conn.execute("SELECT to_regclass('apexlogic.schema_migrations') AS name").fetchone()
        if row["name"] is None:
            raise RuntimeError("PostgreSQL is not initialized: python -m scripts.postgres_admin init")
        if not conn.execute("SELECT 1 FROM schema_migrations WHERE version=1").fetchone():
            raise RuntimeError("PostgreSQL schema migration 1 is missing")


def lock_key(value):
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big", signed=True)


@contextmanager
def execution_lock(run_id):
    validate_run_id(run_id)
    psycopg, dict_row, _ = dependencies()
    # Dedicated, not pooled: session lock lifetime equals execution lifetime.
    with psycopg.connect(dsn(), autocommit=True, row_factory=dict_row, connect_timeout=5,
                         options="-c search_path=apexlogic_checkpoints,public") as conn:
        key = lock_key("apexlogic:run:" + run_id)
        if not conn.execute("SELECT pg_try_advisory_lock(%s) AS acquired", (key,)).fetchone()["acquired"]:
            raise RunBusyError("该任务正在其他进程中执行")
        token = _execution_connection.set(conn)
        try:
            yield
        finally:
            _execution_connection.reset(token)
            # Closing this dedicated session releases its locks, including on error.


@contextmanager
def open_checkpointer():
    psycopg, dict_row, _ = dependencies()
    from langgraph.checkpoint.postgres import PostgresSaver
    conn = _execution_connection.get()
    if conn is not None:
        # Losing the execution-lock session also prevents checkpoint writes.
        yield PostgresSaver(conn)
    else:
        with psycopg.connect(dsn(), autocommit=True, row_factory=dict_row, connect_timeout=5,
                             options="-c search_path=apexlogic_checkpoints,public") as conn:
            yield PostgresSaver(conn)


def initialize():
    psycopg, dict_row, _ = dependencies()
    from langgraph.checkpoint.postgres import PostgresSaver
    sql = (Path(__file__).parent.parent / "migrations/postgres/001_initial.sql").read_text(encoding="utf-8")
    checksum = hashlib.sha256(sql.encode()).hexdigest()
    with psycopg.connect(dsn(), autocommit=True, row_factory=dict_row, connect_timeout=5) as conn:
        conn.execute("SELECT pg_advisory_lock(%s)", (lock_key("apexlogic:schema"),))
        with conn.transaction():
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute("CREATE SCHEMA IF NOT EXISTS apexlogic")
            conn.execute("CREATE SCHEMA IF NOT EXISTS apexlogic_checkpoints")
            conn.execute("SET search_path TO apexlogic,public")
            conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations(version integer PRIMARY KEY, checksum text NOT NULL)")
            row = conn.execute("SELECT checksum FROM schema_migrations WHERE version=1").fetchone()
            if row and row["checksum"] != checksum:
                raise RuntimeError("Applied migration checksum differs; do not edit applied migrations")
            if row is None:
                conn.execute(sql)
                conn.execute("INSERT INTO schema_migrations VALUES (1,%s)", (checksum,))
        conn.execute("SET search_path TO apexlogic_checkpoints,public")
        PostgresSaver(conn).setup()
