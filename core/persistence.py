"""Local durable execution storage. Each execution owns its connections/lock."""
from contextlib import contextmanager
from pathlib import Path
import os
import re
import sqlite3

import portalocker

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class RunBusyError(RuntimeError):
    pass


def data_directory(path=None) -> Path:
    return Path(path or os.getenv("APEXLOGIC_DATA_DIR") or DEFAULT_DATA_DIR).resolve()


def validate_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise ValueError("无效的 run_id")
    return run_id


@contextmanager
def run_lock(data_dir: Path, run_id: str):
    validate_run_id(run_id)
    folder = data_dir / "locks"
    folder.mkdir(parents=True, exist_ok=True)
    lock = portalocker.Lock(str(folder / f"{run_id}.lock"), mode="a", timeout=0)
    try:
        lock.acquire()
    except portalocker.exceptions.LockException as exc:
        raise RunBusyError("该任务正在其他窗口或进程中执行，请稍后刷新。") from exc
    try:
        yield
    finally:
        lock.release()


@contextmanager
def open_checkpointer(data_dir: Path):
    from langgraph.checkpoint.sqlite import SqliteSaver
    data_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(data_dir / "checkpoints.sqlite"), timeout=30, check_same_thread=False)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        saver = SqliteSaver(conn)
        saver.setup()
        yield saver
    finally:
        conn.close()
