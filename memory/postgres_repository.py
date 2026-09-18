"""Persistent evidence metadata and exact pgvector cosine ranking."""
import json
import hashlib
from core.postgres import connect, require_schema, lock_key
from memory.repository import now, conflict_shape


class PostgresMemoryRepository:
    def __init__(self):
        require_schema()

    connect = staticmethod(connect)

    def rank_candidates(self, candidates, vector, model):
        by_id = {row["id"]: row for row in candidates}
        if any(row["dimension"] != len(vector) for row in candidates):
            raise ValueError("stored embedding dimension mismatch")
        with self.connect() as db:
            rows = db.execute("""SELECT item_id, 1 - (vector <=> %s::vector) AS score
                FROM memory_embeddings WHERE item_id=ANY(%s) AND model=%s
                ORDER BY score DESC, item_id""", (json.dumps(vector.tolist()), list(by_id), model)).fetchall()
        return [(by_id[row["item_id"]], float(row["score"])) for row in rows]

    def publication(self, run_id, checkpoint_id, namespace):
        with self.connect() as db:
            row = db.execute("SELECT * FROM memory_publications WHERE run_id=%s AND checkpoint_id=%s AND namespace=%s",
                             (run_id, checkpoint_id, namespace)).fetchone()
        if row is None:
            return None
        result = {**dict(row), "item_ids": json.loads(row["item_ids"])}
        if result["status"] == "completed" and not result["item_ids"]:
            result.update(status="skipped", skip_reason="legacy_empty_publication")
        return result

    def prepare(self, run_id, checkpoint_id, namespace, items):
        with self.connect() as db:
            db.execute("SELECT pg_advisory_xact_lock(%s)", (lock_key("apexlogic:memory:" + namespace),))
            if db.execute("SELECT 1 FROM memory_publications WHERE run_id=%s AND checkpoint_id=%s AND namespace=%s", (run_id, checkpoint_id, namespace)).fetchone():
                return
            for item in items:
                existing = db.execute("""SELECT * FROM memory_items
                    WHERE namespace=%s AND url=%s AND content_hash=%s ORDER BY observed_at DESC,id DESC LIMIT 1""",
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
                inserted = db.execute("""INSERT INTO memory_items
                    (id,namespace,url,title,content,claim,topic,content_hash,source_run_id,observed_at,valid_until,version)
                    VALUES (%(id)s,%(namespace)s,%(url)s,%(title)s,%(content)s,%(claim)s,%(topic)s,%(content_hash)s,%(source_run_id)s,%(observed_at)s,%(valid_until)s,%(version)s) ON CONFLICT DO NOTHING""", item)
                if inserted.rowcount:
                    if existing and item["version"] > 1:
                        db.execute("INSERT INTO memory_relations VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                                   (item["id"], existing["id"], "supersedes", "New online observation after expiry", now()))
                        db.execute("UPDATE memory_items SET status='superseded' WHERE id=%s", (existing["id"],))
                    shape, numbers, negated = conflict_shape(item["content"])
                    for old in db.execute("SELECT id,content FROM memory_items WHERE namespace=%s AND id<>%s AND status IN ('active','conflicted') AND valid_until>%s",
                                          (namespace, item["id"], now())).fetchall():
                        old_shape, old_numbers, old_negated = conflict_shape(old["content"])
                        if len(shape) >= 16 and shape == old_shape and (numbers, negated) != (old_numbers, old_negated):
                            db.execute("INSERT INTO memory_relations VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                                       (item["id"], old["id"], "possible_conflict", "Matching sentence except numbers or negation; requires review", now()))
                            db.execute("UPDATE memory_items SET status='conflicted' WHERE id IN (%s,%s)", (item["id"], old["id"]))
            db.execute("""INSERT INTO memory_publications (run_id,checkpoint_id,namespace,status,item_ids,error,updated_at) VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                       (run_id, checkpoint_id, namespace, "pending", json.dumps([x["id"] for x in items]), None, now()))

    def finish(self, run_id, checkpoint_id, namespace, status, error=None, skip_reason=None):
        with self.connect() as db:
            db.execute("UPDATE memory_publications SET status=%s,error=%s,updated_at=%s,skip_reason=%s WHERE run_id=%s AND checkpoint_id=%s AND namespace=%s",
                       (status, error, now(), skip_reason, run_id, checkpoint_id, namespace))

    def get(self, item_id, namespace):
        with self.connect() as db:
            row = db.execute("SELECT * FROM memory_items WHERE id=%s AND namespace=%s", (item_id, namespace)).fetchone()
        return dict(row) if row else None

    def has_embedding(self, item_id, model):
        with self.connect() as db:
            return db.execute("SELECT 1 FROM memory_embeddings WHERE item_id=%s AND model=%s", (item_id, model)).fetchone() is not None

    def put_embedding(self, item_id, model, vector):
        with self.connect() as db:
            db.execute("SELECT pg_advisory_xact_lock(%s)", (lock_key("apexlogic:embedding:" + model),))
            row = db.execute("SELECT dimension FROM memory_embeddings WHERE model=%s LIMIT 1", (model,)).fetchone()
            if row and row["dimension"] != len(vector):
                raise ValueError("embedding dimension changed; use a new model identity")
            db.execute("INSERT INTO memory_embeddings VALUES (%s,%s,%s,%s::vector) ON CONFLICT DO NOTHING",
                       (item_id, model, len(vector), json.dumps(vector.tolist())))

    def candidates(self, namespace, model, at, limit=10000):
        with self.connect() as db:
            return [dict(r) for r in db.execute("""SELECT i.*,e.dimension FROM memory_items i
                JOIN memory_embeddings e ON e.item_id=i.id
                WHERE i.namespace=%s AND i.status='active' AND i.valid_until>%s AND e.model=%s
                ORDER BY i.observed_at DESC,i.id LIMIT %s""", (namespace, at, model, limit))]

    def log_access(self, query_id, run_id, namespace, details):
        with self.connect() as db:
            db.execute("INSERT INTO memory_access_log VALUES (%s,%s,%s,%s,%s) ON CONFLICT(query_id) DO UPDATE SET details=EXCLUDED.details,updated_at=EXCLUDED.updated_at",
                       (query_id, run_id, namespace, json.dumps(details, ensure_ascii=False), now()))

    def mark_used(self, query_id, run_id, namespace, selected_ids, used_ids):
        with self.connect() as db:
            row = db.execute("SELECT details FROM memory_access_log WHERE query_id=%s AND run_id=%s AND namespace=%s",
                             (query_id, run_id, namespace)).fetchone()
            if row:
                details = json.loads(row["details"])
                details.update(selected_ids=selected_ids, used_ids=used_ids)
                db.execute("UPDATE memory_access_log SET details=%s,updated_at=%s WHERE query_id=%s",
                           (json.dumps(details, ensure_ascii=False), now(), query_id))

    def relate(self, left, right, namespace, relation, reason):
        if relation not in {"supports", "contradicts", "duplicates", "supersedes"} or left == right or not reason.strip():
            raise ValueError("invalid memory relation")
        with self.connect() as db:
            db.execute("SELECT pg_advisory_xact_lock(%s)", (lock_key("apexlogic:memory:" + namespace),))
            rows = db.execute("SELECT id FROM memory_items WHERE namespace=%s AND id IN (%s,%s)", (namespace, left, right)).fetchall()
            if len(rows) != 2:
                raise ValueError("memory relation crosses namespace or missing item")
            db.execute("INSERT INTO memory_relations VALUES (%s,%s,%s,%s,%s) ON CONFLICT(left_id,right_id,relation) DO UPDATE SET reason=EXCLUDED.reason,created_at=EXCLUDED.created_at", (left, right, relation, reason, now()))
            if relation == "contradicts":
                db.execute("UPDATE memory_items SET status='conflicted' WHERE id IN (%s,%s)", (left, right))
            elif relation == "supersedes":
                db.execute("UPDATE memory_items SET status='superseded' WHERE id=%s", (right,))
                db.execute("UPDATE memory_items SET status='active' WHERE id=%s", (left,))
