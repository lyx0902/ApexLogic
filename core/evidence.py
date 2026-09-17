"""Shared source-evidence schema; exact excerpts remain distinct from claims."""

SUPPORTED = {"supported", "verified", "已验证", "已支持"}


def reasoning_sources(chains, enabled=True):
    """Expose actual IRCoT output as a trusted channel, preserving hop IDs."""
    if not enabled:
        return {}
    return {f"推理链{i}": chain for i, chain in enumerate(chains or [], 1)
            if isinstance(chain, str) and chain.strip()}


def normalize(value):
    return "".join(str(value).split()).casefold()


def evidence_entries(verdict):
    if "source_evidence" in verdict:
        value = verdict["source_evidence"]
        return value if isinstance(value, list) and all(isinstance(x, dict) for x in value) else []
    ids = verdict.get("citation_ids")
    if not isinstance(ids, list):
        return []
    return [{"citation_id": cid, "quote": verdict.get("source_quote", "")} for cid in ids if isinstance(cid, str)]
