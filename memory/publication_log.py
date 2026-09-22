"""Publication diagnostics with allowlisted error metadata, never exception prose."""
import json
from uuid import uuid4
from datetime import datetime, timezone


def utc_now():
    return datetime.now(timezone.utc).isoformat()

SCHEMA = """CREATE TABLE IF NOT EXISTS memory_publication_attempts (
    attempt_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, checkpoint_id TEXT NOT NULL,
    namespace TEXT NOT NULL, started_at TEXT NOT NULL, details TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS memory_publication_attempts_run
ON memory_publication_attempts(run_id, namespace, started_at);"""


def safe_exception_chain(exc):
    """Requests/urllib3 often nest exceptions in reason/args instead of __cause__."""
    queue, seen, result = [exc], set(), []
    while queue and len(result) < 8:
        item = queue.pop(0)
        if not isinstance(item, BaseException) or id(item) in seen:
            continue
        seen.add(id(item))
        row = {'type': type(item).__name__}
        for attr in ('errno', 'winerror'):
            val = getattr(item, attr, None)
            if type(val) is int:
                row[attr] = val
        status = getattr(getattr(item, 'response', None), 'status_code', None)
        if type(status) is int:
            row['http_status'] = status
        result.append(row)
        queue.extend([item.__cause__, item.__context__, getattr(item, 'reason', None),
                      getattr(item, 'original_error', None)])
        queue.extend(a for a in item.args if isinstance(a, BaseException))
    return result


class PublicationLog:
    log_placeholder = '?'

    def save_publication_attempt(self, attempt):
        p = self.log_placeholder
        with self.connect() as db:
            db.execute(f'''INSERT INTO memory_publication_attempts VALUES ({','.join([p]*6)})
                ON CONFLICT(attempt_id) DO UPDATE SET details=excluded.details''',
                (attempt['attempt_id'], attempt['run_id'], attempt['checkpoint_id'],
                 attempt['namespace'], attempt['started_at'], json.dumps(attempt, ensure_ascii=False)))

    def publication_attempts(self, run_id, checkpoint_id, namespace):
        p = self.log_placeholder
        with self.connect() as db:
            rows = db.execute(f'''SELECT details FROM memory_publication_attempts
                WHERE run_id={p} AND checkpoint_id={p} AND namespace={p}
                ORDER BY started_at DESC,attempt_id DESC LIMIT 50''', (run_id, checkpoint_id, namespace)).fetchall()
        return [json.loads(r['details']) for r in rows]


def new_attempt(run_id, checkpoint_id, namespace, trigger):
    return dict(attempt_id=uuid4().hex, run_id=run_id, checkpoint_id=checkpoint_id,
                namespace=namespace, started_at=utc_now(), ended_at=None, trigger=trigger,
                status='running', stage='prepare_evidence', total=None, completed_before=0,
                completed_this_attempt=0, failed_item_id=None)
