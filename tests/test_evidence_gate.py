from copy import deepcopy
import pytest
from agents.reviewer import _apply_evidence_gate

SOURCES = [{"citation_id": "S1", "content": "The CEO earned a computer science masters at Example University."}]
VERDICT = {"claim": "degree university", "critical": True, "status": "supported",
           "citation_ids": ["S1"], "source_quote": "computer science masters at Example University"}

def review(verdicts):
    return {"scores": dict.fromkeys(["S1", "S2", "S3", "S4"], 10),
            "is_satisfactory": True, "evidence_verdicts": verdicts}

@pytest.mark.parametrize("change", [
    {"status": "partial"}, {"status": "部分验证"}, {"status": "contradicted"},
    {"status": "unsupported"}, {"status": "unknown"},
    {"citation_ids": ["S99"]}, {"citation_ids": ["推理链1"]},
    {"citation_ids": "S1"}, {"source_quote": "fabricated quote"},
    {"source_quote": ""}, {"critical": False},
])
def test_high_score_cannot_override_missing_evidence(change):
    verdict = {**VERDICT, **change}
    result = _apply_evidence_gate(review([verdict]), SOURCES)
    assert not result["is_satisfactory"]
    assert result["needs_more_research"]
    assert result["evidence_gate_reasons"]
    assert result["info_gaps"]

@pytest.mark.parametrize("verdicts", [[], None, "malformed"])
def test_missing_schema_fails_closed(verdicts):
    assert _apply_evidence_gate(review(verdicts), SOURCES)["evidence_gate_blocked"]

def test_all_critical_claims_required_but_optional_claim_may_be_unresolved():
    good = deepcopy(VERDICT)
    optional = {"claim": "background", "critical": False, "status": "partial"}
    assert _apply_evidence_gate(review([good, optional]), SOURCES)["is_satisfactory"]
    optional["critical"] = True
    assert not _apply_evidence_gate(review([good, optional]), SOURCES)["is_satisfactory"]

def test_reasoning_source_requires_enabled_and_real_excerpt():
    verdict = {**VERDICT, "citation_ids": ["R1"]}
    assert _apply_evidence_gate(review([verdict]), [], SOURCES, True)["is_satisfactory"]
    assert not _apply_evidence_gate(review([verdict]), [], SOURCES, False)["is_satisfactory"]

def test_valid_evidence_does_not_override_score_failure():
    r = review([deepcopy(VERDICT)])
    r["is_satisfactory"] = False
    assert not _apply_evidence_gate(r, SOURCES)["is_satisfactory"]
