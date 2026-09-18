"""Explicit PostgreSQL setup and offline SQLite evidence import; no LLM calls."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

TABLES = ("memory_items", "memory_embeddings", "memory_publications", "memory_relations", "memory_access_log")
KEYS = {"memory_items": ("id",), "memory_embeddings": ("item_id", "model"),
        "memory_publications": ("run_id", "checkpoint_id", "namespace"),
        "memory_relations": ("left_id", "right_id", "relation"), "memory_access_log": ("query_id",)}


def read_memory(path):
    """One read transaction, no repository constructors/migrations on the source."""
    result = {}
    with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")
        for table in TABLES:
            result[table] = [dict(r) for r in db.execute(
                f"SELECT * FROM {table} ORDER BY {','.join(KEYS[table])}")]
    dimensions = {}
    for row in result["memory_embeddings"]:
        vector = np.frombuffer(row["vector"], dtype="<f4")
        if (len(vector) != row["dimension"] or not len(vector)
                or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0):
            raise ValueError("Invalid source vector; import aborted")
        if dimensions.setdefault(row["model"], len(vector)) != len(vector):
            raise ValueError("Inconsistent embedding dimensions for the same model")
        row["vector"] = vector.tolist()
    for row in result["memory_publications"]:
        row.setdefault("skip_reason", None)
    return result


def import_memory(path, *, apply=False):
    data = read_memory(path)
    counts = {table: len(rows) for table, rows in data.items()}
    fingerprint = hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if not apply:
        return {"status": "dry_run", "counts": counts, "source_digest": fingerprint}
    from psycopg import sql
    from core.postgres import connect, require_schema, lock_key
    require_schema()
    with connect() as db:
        db.execute("SELECT pg_advisory_xact_lock(%s)", (lock_key("apexlogic:memory-import"),))
        # Explicit migration command: serialize writes to all destination memory tables.
        db.execute("LOCK TABLE memory_items,memory_embeddings,memory_publications,memory_relations,memory_access_log IN EXCLUSIVE MODE")
        if db.execute("SELECT 1 FROM memory_imports WHERE source_digest=%s", (fingerprint,)).fetchone():
            return {"status": "already_imported", "counts": counts}
        for table in TABLES:
            if db.execute(sql.SQL("SELECT 1 FROM {} LIMIT 1").format(sql.Identifier(table))).fetchone():
                raise ValueError("Import requires empty target memory tables; existing data is never overwritten")
        for table, rows in data.items():
            for row in rows:
                values = []
                placeholders = []
                for column, value in row.items():
                    if table == "memory_embeddings" and column == "vector":
                        value = json.dumps(value)
                        placeholders.append(sql.SQL("%s::vector"))
                    else:
                        placeholders.append(sql.Placeholder())
                    values.append(value)
                statement = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                    sql.Identifier(table), sql.SQL(",").join(map(sql.Identifier, row)),
                    sql.SQL(",").join(placeholders))
                db.execute(statement, values)
        # A committed receipt makes exact replays idempotent.
        db.execute("INSERT INTO memory_imports(source_digest,counts) VALUES (%s,%s)",
                   (fingerprint, json.dumps(counts)))
    return {"status": "imported", "counts": counts}


def main():
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    sub.add_parser("check")
    migrate = sub.add_parser("import-memory")
    migrate.add_argument("--source", default="data/memory.sqlite")
    migrate.add_argument("--apply", action="store_true", help="Commit import; default only reads source")
    args = parser.parse_args()
    try:
        if args.command == "init":
            from core.postgres import initialize
            initialize()
            result = {"status": "initialized"}
        elif args.command == "check":
            from core.postgres import connect, require_schema
            require_schema()
            with connect() as db:
                extension = db.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()
                checkpoints = db.execute("SELECT to_regclass('apexlogic_checkpoints.checkpoints') AS name").fetchone()
                if not extension or not checkpoints["name"]:
                    raise RuntimeError("Run init to complete checkpoint/pgvector setup")
                result = {"status": "ok", "pgvector": extension["extversion"]}
        else:
            result = import_memory(args.source, apply=args.apply)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        # Drivers may embed connection arguments in exceptions. Never print a DSN.
        parser.exit(1, f"PostgreSQL operation failed ({type(exc).__name__}). Check configuration/service and docs/postgres-migration.md.\n")


if __name__ == "__main__":
    main()
