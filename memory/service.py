"""Bounded semantic recall and source-only publication; never trust report prose."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
import re
import time
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from core.evidence import evidence_entries
from bge.embeddings import embed_texts, validate_vectors
from memory.repository import MemoryRepository, now


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize(text):
    return "".join(str(text).split()).casefold()


def time_sensitive(text):
    # Conservative: these tasks must re-check the current web, never answer from memory.
    return bool(re.search(r"现任|当前|最新|目前|今日|今天|价格|股价|汇率|实时|\b(current|latest|today|price|now)\b", text, re.I))


def source_url(value):
    try:
        parsed = urlsplit(str(value).strip())
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        return None
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path, parsed.query, ""))


class MemoryService:
    def __init__(self, config, *, embed=None, repository=None):
        self.options = config["memory"]
        self.namespace = self.options["namespace"]
        self.repo = repository or MemoryRepository(self.options["data_dir"])
        settings = config["settings"]
        self.model = settings.get("BGE_EMBED_MODEL") or "BAAI/bge-m3"
        self.base = settings.get("BGE_EMBED_BASE_URL") or "https://api.siliconflow.cn/v1"
        self.timeout = float(settings.get("BGE_EMBED_TIMEOUT") or "20")
        self.identity = self.model + "@" + digest(self.base.rstrip("/"))[:16]
        self.embed = embed or self._embed

    def _embed(self, texts):
        key = os.getenv("BGE_EMBED_API_KEY", "").strip()
        if not key:
            raise RuntimeError("embedding credentials unavailable")
        return embed_texts(texts, model=self.model, api_base=self.base, api_key=key, timeout_sec=self.timeout)

    def _vectors(self, texts):
        result = validate_vectors(self.embed(texts))
        if len(result) != len(texts):
            raise ValueError("embedding count mismatch")
        return result

    def recall(self, state):
        started = time.monotonic()
        query = state["topic"]
        query_id = digest(f"{state['run_id']}:{state.get('revision_step', 0)}:{self.namespace}:{query}")
        stats = {"enabled": True, "query_id": query_id, "candidates": 0, "recalled": 0, "selected": 0}
        if time_sensitive(query):
            stats["status"] = "fresh_search_required"
            self.repo.log_access(query_id, state["run_id"], self.namespace, stats)
            return [], stats
        candidates = self.repo.candidates(self.namespace, self.identity, now())
        candidates = [r for r in candidates if r["source_run_id"] != state["run_id"] and not time_sensitive(r["claim"])]
        stats["candidates"] = len(candidates)
        if not candidates:
            stats["status"] = "empty"
            self.repo.log_access(query_id, state["run_id"], self.namespace, stats)
            return [], stats
        vector = self._vectors([query])[0]
        compatible = []
        vectors = []
        for row in candidates:
            arr = np.frombuffer(row["vector"], dtype="<f4")
            if len(arr) != row["dimension"] or len(arr) != len(vector):
                raise ValueError("stored embedding dimension mismatch")
            validate_vectors([arr])
            compatible.append(row)
            vectors.append(arr)
        matrix = np.stack(vectors)
        scores = (matrix / np.linalg.norm(matrix, axis=1, keepdims=True)) @ (vector / np.linalg.norm(vector))
        hits, remaining = [], self.options["char_budget"]
        for i in np.argsort(-scores, kind="stable"):
            if float(scores[i]) < self.options["min_score"] or len(hits) >= self.options["top_k"]:
                break
            row = compatible[i]
            # Recheck status after potentially slow embedding/network call.
            fresh = self.repo.get(row["id"], self.namespace)
            if not fresh or fresh["status"] != "active" or fresh["valid_until"] <= now():
                continue
            if len(row["content"]) > remaining:
                continue
            remaining -= len(row["content"])
            hits.append({"title": row["title"], "url": row["url"], "source": "memory", "origin": "memory",
                         "content": row["content"], "core_summary": row["content"],
                         "memory_id": row["id"], "memory_version": row["version"],
                         "memory_source_run_id": row["source_run_id"], "memory_observed_at": row["observed_at"],
                         "memory_valid_until": row["valid_until"], "memory_score": round(float(scores[i]), 6)})
        stats.update(status="ok", recalled=len(hits), recalled_ids=[x["memory_id"] for x in hits],
                     elapsed_seconds=round(time.monotonic() - started, 4))
        self.repo.log_access(query_id, state["run_id"], self.namespace, stats)
        return hits, stats

    def _eligible(self, state, completed_at):
        review = state.get("review_result", {})
        accepted = state.get("is_satisfactory") or (review.get("answer_status") == "limited" and review.get("quality_accepted") is True)
        if not accepted:
            return []
        sources = {str(c.get("citation_id")): c for c in state.get("retrieved_context", [])[:10] if isinstance(c, dict)}
        if state.get("reasoning_enabled"):
            sources.update({f"R{i}": c for i, c in enumerate(state.get("reasoning_contexts", [])[:15], 1) if isinstance(c, dict)})
        report = state.get("final_report") or state.get("draft", "")
        items = {}
        # Bound writes to avoid publication of a model-generated unbounded list.
        for verdict in review.get("evidence_verdicts", [])[:30]:
            if not isinstance(verdict, dict) or verdict.get("critical") is False:
                continue
            if str(verdict.get("status", "")).lower() not in {"supported", "verified", "已验证", "已支持"}:
                continue
            claim = str(verdict.get("claim") or "").strip()
            if not claim or time_sensitive(claim):
                continue
            for entry in evidence_entries(verdict):
                quote = str(entry.get("quote") or "").strip()
                cid = entry.get("citation_id")
                if not 8 <= len(normalize(quote)) <= 400 or not isinstance(cid, str) or f"[{cid}]" not in report:
                    continue
                context = sources.get(cid, {})
                if context.get("origin") == "memory" or context.get("source") == "memory":
                    continue  # No self-reinforcement or extending TTL by repeating old evidence.
                url = source_url(context.get("url", ""))
                # Store only excerpts genuinely present in retrieved content, not generated summaries.
                if not url or normalize(quote) not in normalize(context.get("content", "")):
                    continue
                content_hash = digest(normalize(quote))
                item_id = digest(json.dumps([self.namespace, url, content_hash], ensure_ascii=False))
                observed = datetime.fromisoformat(completed_at)
                if observed.tzinfo is None:
                    observed = observed.replace(tzinfo=timezone.utc)
                items[item_id] = {"id": item_id, "namespace": self.namespace, "url": url,
                    "title": str(context.get("title") or "")[:200], "content": quote, "claim": claim[:1000],
                    "topic": state["topic"], "content_hash": content_hash, "source_run_id": state["run_id"],
                    "observed_at": observed.astimezone(timezone.utc).isoformat(),
                    "valid_until": (observed.astimezone(timezone.utc) + timedelta(days=self.options["ttl_days"])).isoformat()}
        return list(items.values())

    def publish(self, state, checkpoint_id, completed_at):
        run_id = state["run_id"]
        stats = state.get("memory_stats", {})
        if stats.get("query_id"):
            self.repo.mark_used(stats["query_id"], run_id, self.namespace, stats.get("selected_ids", []), state.get("memory_used_ids", []))
        receipt = self.repo.publication(run_id, checkpoint_id, self.namespace)
        if receipt and receipt["status"] in {"completed", "skipped"}:
            return receipt
        if receipt is None:
            self.repo.prepare(run_id, checkpoint_id, self.namespace, self._eligible(state, completed_at))
            receipt = self.repo.publication(run_id, checkpoint_id, self.namespace)
        if not receipt["item_ids"]:
            review = state.get("review_result", {})
            accepted = state.get("is_satisfactory") or (review.get("answer_status") == "limited" and review.get("quality_accepted") is True)
            if not accepted:
                reason = "report_not_accepted"
            elif not review.get("evidence_verdicts"):
                reason = "no_reviewed_claims"
            elif all(time_sensitive(str(v.get("claim", ""))) for v in review["evidence_verdicts"] if isinstance(v, dict)):
                reason = "only_time_sensitive_claims"
            else:
                reason = "no_eligible_original_evidence"
            self.repo.finish(run_id, checkpoint_id, self.namespace, "skipped", skip_reason=reason)
            return self.repo.publication(run_id, checkpoint_id, self.namespace)
        try:
            for item_id in receipt["item_ids"]:
                if self.repo.has_embedding(item_id, self.identity):
                    continue
                item = self.repo.get(item_id, self.namespace)
                vector = self._vectors([item["claim"] + "\n" + item["content"]])[0]
                self.repo.put_embedding(item_id, self.identity, vector)
            self.repo.finish(run_id, checkpoint_id, self.namespace, "completed")
        except Exception as exc:
            self.repo.finish(run_id, checkpoint_id, self.namespace, "failed", type(exc).__name__)
        return self.repo.publication(run_id, checkpoint_id, self.namespace)


def recall_for_state(state):
    options = state.get("run_config", {}).get("memory", {})
    if not options.get("enabled") or not options.get("data_dir") or not state.get("run_id"):
        return [], {"enabled": False, "status": "disabled"}
    try:
        return MemoryService(state["run_config"]).recall(state)
    except Exception as exc:
        return [], {"enabled": True, "status": "failed", "error": type(exc).__name__}
