"""Backend selection. Snapshots contain a profile name, never a database URL."""
import os


def storage_settings(backend=None):
    backend = backend or os.getenv("APEXLOGIC_STORAGE_BACKEND", "sqlite")
    if backend not in {"sqlite", "postgres"}:
        raise ValueError("APEXLOGIC_STORAGE_BACKEND must be sqlite or postgres")
    return {"backend": backend, "profile": "default"}


def config_storage(config):
    # Absence is a historical SQLite snapshot, independent of current env.
    return config.get("storage", {"backend": "sqlite", "profile": "default"})


def run_repository(data_dir, storage):
    if storage["backend"] == "postgres":
        from core.postgres_repository import PostgresRunRepository
        return PostgresRunRepository()
    from core.run_repository import RunRepository
    return RunRepository(data_dir)


def memory_repository(config):
    if config_storage(config)["backend"] == "postgres":
        from memory.postgres_repository import PostgresMemoryRepository
        return PostgresMemoryRepository()
    from memory.repository import MemoryRepository
    return MemoryRepository(config["memory"]["data_dir"])


def execution_lock(data_dir, run_id, storage):
    if storage["backend"] == "postgres":
        from core.postgres import execution_lock as pg_lock
        return pg_lock(run_id)
    from core.persistence import run_lock
    return run_lock(data_dir, run_id)


def checkpointer(data_dir, storage):
    if storage["backend"] == "postgres":
        from core.postgres import open_checkpointer
        return open_checkpointer()
    from core.persistence import open_checkpointer
    return open_checkpointer(data_dir)
