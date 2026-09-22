"""Immutable evidence versions and retryable publication receipts."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import hashlib
import re
from pathlib import Path
import sqlite3
from memory.publication_log import PublicationLog, SCHEMA


def now():
    return datetime.now(timezone.utc).isoformat()


def conflict_shape(text):
    """Conservative candidate detection, not a truth or contradiction verdict."""
    normalized = re.sub(r"\s+", " ", text).strip().casefold()
    numbers = tuple(re.findall(r"\d+(?:\.\d+)?", normalized))
    negated = bool(re.search(r"\b(not|never|no)\b|不|未|没有", normalized))
    shape = re.sub(r"\d+(?:\.\d+)?", "#", normalized)
    shape = re.sub(r"\b(not|never|no)\b|没有|不|未", "", shape)
    return "".join(shape.split()), numbers, negated


class MemoryRepository(PublicationLog):
    def __init__(self, data_dir):
        self.path = Path(data_dir) / "memory.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript(SCHEMA)
            db.executescript("""
            CREATE TABLE IF NOT EXISTS memory_items (
                id TEXT PRIMARY KEY, namespace TEXT NOT NULL, url TEXT NOT NULL,
                title TEXT NOT NULL, content TEXT NOT NULL, claim TEXT NOT NULL,
                topic TEXT NOT NULL, content_hash TEXT NOT NULL, source_run_id TEXT NOT NULL,
                observed_at TEXT NOT NULL, valid_until TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active', version INTEGER NOT NULL DEFAULT 1
            );
            CREATE INDEX IF NOT EXISTS memory_scope ON memory_items(namespace,status,valid_until);
            CREATE TABLE IF NOT EXISTS memory_embeddings (
                item_id TEXT NOT NULL REFERENCES memory_items(id), model TEXT NOT NULL,
                dimension INTEGER NOT NULL, vector BLOB NOT NULL,
                PRIMARY KEY(item_id,model)
            );
            CREATE TABLE IF NOT EXISTS memory_publications (
                run_id TEXT NOT NULL, checkpoint_id TEXT NOT NULL, namespace TEXT NOT NULL,
                status TEXT NOT NULL, item_ids TEXT NOT NULL, error TEXT,
                updated_at TEXT NOT NULL, PRIMARY KEY(run_id,checkpoint_id,namespace)
            );
            CREATE TABLE IF NOT EXISTS memory_relations (
                left_id TEXT NOT NULL REFERENCES memory_items(id),
                right_id TEXT NOT NULL REFERENCES memory_items(id),
                relation TEXT NOT NULL, reason TEXT NOT NULL, created_at TEXT NOT NULL,
                PRIMARY KEY(left_id,right_id,relation)
            );
            CREATE TABLE IF NOT EXISTS memory_access_log (
                query_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, namespace TEXT NOT NULL,
                details TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            """)
            db.execute("BEGIN IMMEDIATE")
            columns = {r["name"] for r in db.execute("PRAGMA table_info(memory_publications)")}
            if "skip_reason" not in columns:
                db.execute("ALTER TABLE memory_publications ADD COLUMN skip_reason TEXT")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys=ON")
            with db:
                yield db
        finally:
            db.close()

    def publication(self, run_id, checkpoint_id, namespace):
        with self.connect() as db:
            row = db.execute("SELECT * FROM memory_publications WHERE run_id=? AND checkpoint_id=? AND namespace=?",
                             (run_id, checkpoint_id, namespace)).fetchone()
        if row is None:
            return None
        result = {**dict(row), "item_ids": json.loads(row["item_ids"])}
        if result["status"] == "completed" and not result["item_ids"]:
            result.update(status="skipped", skip_reason="legacy_empty_publication")
        return result

    def prepare(self, run_id, checkpoint_id, namespace, items):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for item in items:
                existing = db.execute("""SELECT * FROM memory_items
                    WHERE namespace=? AND url=? AND content_hash=? ORDER BY observed_at DESC,id DESC LIMIT 1""",
                    (namespace, item["url"], item["content_hash"])).fetchone()
                item["version"] = 1
                if existing:
                    # Only a genuinely new online observation can renew an expired item.
                    # Conflict resolution is explicit; re-observation cannot silently clear it.
                    if existing["status"] == "active" and existing["valid_until"] <= item["observed_at"]:
                        item["version"] = existing["version"] + 1
                        item["id"] = hashlib.sha256((existing["id"] + item["observed_at"]).encode()).hexdigest()
                    else:
                        item["id"] = existing["id"]
                inserted = db.execute("""INSERT OR IGNORE INTO memory_items
                    (id,namespace,url,title,content,claim,topic,content_hash,source_run_id,observed_at,valid_until,version)
                    VALUES (:id,:namespace,:url,:title,:content,:claim,:topic,:content_hash,:source_run_id,:observed_at,:valid_until,:version)""", item)
                if inserted.rowcount:
                    if existing and item["version"] > 1:
                        db.execute("INSERT OR IGNORE INTO memory_relations VALUES (?,?,?,?,?)",
                                   (item["id"], existing["id"], "supersedes", "New online observation after expiry", now()))
                        db.execute("UPDATE memory_items SET status='superseded' WHERE id=?", (existing["id"],))
                    shape, numbers, negated = conflict_shape(item["content"])
                    for old in db.execute("SELECT id,content FROM memory_items WHERE namespace=? AND id<>? AND status IN ('active','conflicted') AND valid_until>?",
                                          (namespace, item["id"], now())).fetchall():
                        old_shape, old_numbers, old_negated = conflict_shape(old["content"])
                        if len(shape) >= 16 and shape == old_shape and (numbers, negated) != (old_numbers, old_negated):
                            db.execute("INSERT OR IGNORE INTO memory_relations VALUES (?,?,?,?,?)",
                                       (item["id"], old["id"], "possible_conflict", "Matching sentence except numbers or negation; requires review", now()))
                            db.execute("UPDATE memory_items SET status='conflicted' WHERE id IN (?,?)", (item["id"], old["id"]))
            db.execute("""INSERT OR IGNORE INTO memory_publications (run_id,checkpoint_id,namespace,status,item_ids,error,updated_at) VALUES (?,?,?,?,?,?,?)""",
                       (run_id, checkpoint_id, namespace, "pending", json.dumps([x["id"] for x in items]), None, now()))

    def finish(self, run_id, checkpoint_id, namespace, status, error=None, skip_reason=None):
        with self.connect() as db:
            db.execute("UPDATE memory_publications SET status=?,error=?,updated_at=?,skip_reason=? WHERE run_id=? AND checkpoint_id=? AND namespace=?",
                       (status, error, now(), skip_reason, run_id, checkpoint_id, namespace))

    def get(self, item_id, namespace):
        with self.connect() as db:
            row = db.execute("SELECT * FROM memory_items WHERE id=? AND namespace=?", (item_id, namespace)).fetchone()
        return dict(row) if row else None

    def has_embedding(self, item_id, model):
        with self.connect() as db:
            return db.execute("SELECT 1 FROM memory_embeddings WHERE item_id=? AND model=?", (item_id, model)).fetchone() is not None

    def put_embedding(self, item_id, model, vector):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT dimension FROM memory_embeddings WHERE model=? LIMIT 1", (model,)).fetchone()
            if row and row["dimension"] != len(vector):
                raise ValueError("embedding dimension changed; use a new model identity")
            db.execute("INSERT OR IGNORE INTO memory_embeddings VALUES (?,?,?,?)",
                       (item_id, model, len(vector), vector.astype("<f4").tobytes()))

    def candidates(self, namespace, model, at, limit=10000):
        with self.connect() as db:
            return [dict(r) for r in db.execute("""SELECT i.*,e.dimension,e.vector FROM memory_items i
                JOIN memory_embeddings e ON e.item_id=i.id
                WHERE i.namespace=? AND i.status='active' AND i.valid_until>? AND e.model=?
                ORDER BY i.observed_at DESC,i.id LIMIT ?""", (namespace, at, model, limit))]

    def log_access(self, query_id, run_id, namespace, details):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO memory_access_log VALUES (?,?,?,?,?)",
                       (query_id, run_id, namespace, json.dumps(details, ensure_ascii=False), now()))

    def mark_used(self, query_id, run_id, namespace, selected_ids, used_ids):
        with self.connect() as db:
            row = db.execute("SELECT details FROM memory_access_log WHERE query_id=? AND run_id=? AND namespace=?",
                             (query_id, run_id, namespace)).fetchone()
            if row:
                details = json.loads(row["details"])
                details.update(selected=len(selected_ids), selected_ids=selected_ids, used_ids=used_ids)
                db.execute("UPDATE memory_access_log SET details=?,updated_at=? WHERE query_id=?",
                           (json.dumps(details, ensure_ascii=False), now(), query_id))

    def relate(self, left, right, namespace, relation, reason):
        if relation not in {"supports", "contradicts", "duplicates", "supersedes"} or left == right or not reason.strip():
            raise ValueError("invalid memory relation")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("SELECT id FROM memory_items WHERE namespace=? AND id IN (?,?)", (namespace, left, right)).fetchall()
            if len(rows) != 2:
                raise ValueError("memory relation crosses namespace or missing item")
            db.execute("INSERT OR REPLACE INTO memory_relations VALUES (?,?,?,?,?)", (left, right, relation, reason, now()))
            if relation == "contradicts":
                db.execute("UPDATE memory_items SET status='conflicted' WHERE id IN (?,?)", (left, right))
            elif relation == "supersedes":
                db.execute("UPDATE memory_items SET status='superseded' WHERE id=?", (right,))
                db.execute("UPDATE memory_items SET status='active' WHERE id=?", (left,))
