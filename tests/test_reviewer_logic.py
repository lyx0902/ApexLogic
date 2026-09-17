# -*- coding: utf-8 -*-
"""Reviewer 评审通过判定统一化的单元测试。

背景：#7 修复前，通过判定分散在 4 处且规则不一致——`_rule_based_review`
在 weighted=7.0 时硬编码 `is_satisfactory=True`，绕过 REVIEWER_PASS_THRESHOLD
（默认 7.5），导致「LLM 评审失败 / JSON 解析失败」会静默降低验收标准。

本测试锁定修复后的约定：
- 生产者（LLM / 语义兜底 / 规则）只提供 scores，判定统一由 _finalize_review 收口；
- 加权分以本地公式为准，不采用生产者自报的 weighted_score；
- 降级评审默认不放行，仅在 REVIEWER_ALLOW_DEGRADED_PASS=1 时显式放宽（并打标记）。

运行：python -m pytest tests/test_reviewer_logic.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import agents.reviewer as reviewer  # noqa: E402
from core.state import create_initial_state  # noqa: E402

LONG_DRAFT = "正文" * 350
TWO_CONTEXTS = [
    {"title": "a", "url": "u1", "source": "ddg", "content": "verified source evidence", "citation_id": "S1"},
    {"title": "b", "url": "u2", "source": "ddg", "content": "c"},
]
SEVEN_SEVENS = {"S1": 7, "S2": 7, "S3": 7, "S4": 7}


@pytest.fixture(autouse=True)
def _reset_review_env(monkeypatch):
    """每个用例都在默认阈值与默认降级策略下运行。"""

    monkeypatch.setenv("REVIEWER_PASS_THRESHOLD", "7.5")
    monkeypatch.delenv("REVIEWER_ALLOW_DEGRADED_PASS", raising=False)


def _make_state(draft: str = LONG_DRAFT) -> Dict[str, Any]:
    state = create_initial_state("测试主题")
    state["draft"] = draft
    state["retrieved_context"] = [dict(item) for item in TWO_CONTEXTS]
    return state


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


def _install_fake_llm(monkeypatch, content: str) -> None:
    """把 reviewer 模块内的 ChatOpenAI 替换为返回固定文本的假实现。"""

    class _FakeLLM:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def invoke(self, messages: Any) -> _FakeResponse:
            return _FakeResponse(content)

    monkeypatch.setattr(reviewer, "ChatOpenAI", _FakeLLM)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")


# ----------------------------------------------------------------------
# 1. 收口函数 _finalize_review
# ----------------------------------------------------------------------

class TestFinalizeReview:
    def test_above_threshold_passes(self):
        r = reviewer._finalize_review(
            {"scores": {"S1": 8, "S2": 8, "S3": 8, "S4": 8}}, mode="llm_scored"
        )
        assert r["is_satisfactory"] is True
        assert r["weighted_score"] == 8.0
        assert r["pass_threshold"] == 7.5
        assert r["review_mode"] == "llm_scored"
        assert r["degraded"] is False
        assert r["degraded_blocked"] is False

    def test_local_formula_is_authoritative(self):
        """生产者自报的 weighted_score 不参与判定。"""

        r = reviewer._finalize_review(
            {"scores": {"S1": 5, "S2": 5, "S3": 5, "S4": 5}, "weighted_score": 9.9},
            mode="llm_scored",
        )
        assert r["weighted_score"] == 5.0
        assert r["is_satisfactory"] is False

    def test_rule_score_7_below_threshold_not_passed(self):
        """修复核心：规则评审 7.0 分在默认阈值 7.5 下必须不放行。"""

        r = reviewer._finalize_review(dict(scores=SEVEN_SEVENS), mode="rule", degraded=True)
        assert r["is_satisfactory"] is False
        assert r["needs_more_research"] is False
        # 未达阈值 → 无需"拦截"，degraded_blocked 仅在降级本可放行时为 True
        assert r["degraded_blocked"] is False

    def test_degraded_blocked_even_if_threshold_lowered(self, monkeypatch):
        monkeypatch.setenv("REVIEWER_PASS_THRESHOLD", "6.5")
        r = reviewer._finalize_review(dict(scores=SEVEN_SEVENS), mode="rule", degraded=True)
        assert r["is_satisfactory"] is False
        assert r["degraded_blocked"] is True

    def test_degraded_pass_allowed_explicitly(self, monkeypatch):
        monkeypatch.setenv("REVIEWER_PASS_THRESHOLD", "6.5")
        monkeypatch.setenv("REVIEWER_ALLOW_DEGRADED_PASS", "1")
        r = reviewer._finalize_review(dict(scores=SEVEN_SEVENS), mode="rule", degraded=True)
        assert r["is_satisfactory"] is True
        assert r["degraded"] is True
        assert r["degraded_blocked"] is False

    def test_malformed_threshold_falls_back(self, monkeypatch):
        monkeypatch.setenv("REVIEWER_PASS_THRESHOLD", "abc")
        assert reviewer._resolve_pass_threshold() == 7.5

    def test_needs_more_research_from_low_scores(self):
        r = reviewer._finalize_review(
            {"scores": {"S1": 3, "S2": 6, "S3": 6, "S4": 6}}, mode="llm_scored"
        )
        assert r["needs_more_research"] is True

    def test_model_suggestion_is_kept(self):
        r = reviewer._finalize_review(
            {"scores": {"S1": 6, "S2": 6, "S3": 6, "S4": 6}, "needs_more_research": True},
            mode="llm_scored",
        )
        assert r["needs_more_research"] is True

    def test_passed_review_never_requests_more_research(self):
        r = reviewer._finalize_review(
            {"scores": {"S1": 9, "S2": 9, "S3": 9, "S4": 9}, "needs_more_research": True},
            mode="llm_scored",
        )
        assert r["is_satisfactory"] is True
        assert r["needs_more_research"] is False


# ----------------------------------------------------------------------
# 2. 规则评审三分支
# ----------------------------------------------------------------------

class TestRuleBasedReview:
    def test_branch_insufficient_context_routes_to_research(self):
        r = reviewer._rule_based_review("短", [{"title": "a", "url": "u1"}])
        assert r["is_satisfactory"] is False
        assert r["needs_more_research"] is True
        assert r["review_mode"] == "rule"
        assert r["degraded"] is True

    def test_branch_short_draft_routes_to_writer(self):
        r = reviewer._rule_based_review("短", TWO_CONTEXTS)
        assert r["is_satisfactory"] is False
        assert r["needs_more_research"] is False
        assert "深度不足" in r["critique_feedback"]

    def test_branch_qualified_but_not_passed(self):
        r = reviewer._rule_based_review(LONG_DRAFT, TWO_CONTEXTS)
        assert r["weighted_score"] == 7.0
        assert r["is_satisfactory"] is False
        assert r["degraded"] is True

    def test_branch_qualified_feedback_matches_routing(self):
        """未放行时文案不得声称"通过"，否则与打回 Writer 的路由矛盾。"""

        r = reviewer._rule_based_review(LONG_DRAFT, TWO_CONTEXTS)
        assert "基础标准" in r["critique_feedback"]
        assert "7.0/7.5" in r["critique_feedback"]

    def test_branch_qualified_feedback_when_explicitly_allowed(self, monkeypatch):
        monkeypatch.setenv("REVIEWER_ALLOW_DEGRADED_PASS", "1")
        r = reviewer._rule_based_review(LONG_DRAFT, TWO_CONTEXTS)
        assert r["is_satisfactory"] is False  # 7.0 仍低于 7.5，策略开关不改变阈值
        assert "基础标准" in r["critique_feedback"]


# ----------------------------------------------------------------------
# 3. 语义兜底路径
# ----------------------------------------------------------------------

class TestCoerceFallback:
    def test_claimed_pass_is_degraded_and_blocked(self):
        r = reviewer._coerce_text_review_to_json("[PASS] 事实问题: 无\n逻辑问题: 无")
        assert r["review_mode"] == "coerce_fallback"
        assert r["degraded"] is True
        assert r["is_satisfactory"] is False

    def test_substantive_issues_route_to_research(self):
        r = reviewer._coerce_text_review_to_json(
            "事实问题: 引用了不存在的来源\n逻辑问题: 结论与证据矛盾"
        )
        assert r["is_satisfactory"] is False
        assert r["needs_more_research"] is True
        assert r["weighted_score"] == 4.4


# ----------------------------------------------------------------------
# 4. _llm_review 全链路（mock LLM）
# ----------------------------------------------------------------------

class TestLlmReview:
    def test_normal_json_passes(self, monkeypatch):
        _install_fake_llm(
            monkeypatch,
            '```json\n{"scores":{"S1":9,"S2":9,"S3":9,"S4":9},"critique_feedback":"ok"}\n```',
        )
        r = reviewer._llm_review(topic="t", draft="d", retrieved_context=[])
        assert r["is_satisfactory"] is True
        assert r["review_mode"] == "llm_scored"
        assert r["degraded"] is False
        assert r["pass_threshold"] == 7.5

    def test_model_reported_weighted_score_ignored(self, monkeypatch):
        _install_fake_llm(
            monkeypatch,
            '```json\n{"scores":{"S1":5,"S2":5,"S3":5,"S4":5},"weighted_score":9.9}\n```',
        )
        r = reviewer._llm_review(topic="t", draft="d", retrieved_context=[])
        assert r["weighted_score"] == 5.0
        assert r["is_satisfactory"] is False

    def test_dirty_score_does_not_crash(self, monkeypatch):
        _install_fake_llm(
            monkeypatch,
            '```json\n{"scores":{"S1":9,"S2":9,"S3":"9分","S4":9}}\n```',
        )
        r = reviewer._llm_review(topic="t", draft="d", retrieved_context=[])
        assert r["scores"]["S3"] == 0
        assert r["is_satisfactory"] is False

    def test_scores_as_list_is_conservative(self, monkeypatch):
        _install_fake_llm(monkeypatch, '```json\n{"scores":[1,2,3,4]}\n```')
        r = reviewer._llm_review(topic="t", draft="d", retrieved_context=[])
        assert r["weighted_score"] == 0.0
        assert r["is_satisfactory"] is False

    def test_non_json_text_is_marked_degraded(self, monkeypatch):
        """回归：coerce 结果曾被 _llm_review 重标为 llm_scored，导致降级拦截失效。"""

        _install_fake_llm(monkeypatch, "[PASS] 事实问题: 无\n逻辑问题: 无")
        r = reviewer._llm_review(topic="t", draft="d", retrieved_context=[])
        assert r["review_mode"] == "coerce_fallback"
        assert r["degraded"] is True
        assert r["is_satisfactory"] is False

    def test_coerce_still_blocked_after_threshold_lowered(self, monkeypatch):
        monkeypatch.setenv("REVIEWER_PASS_THRESHOLD", "6.5")
        _install_fake_llm(monkeypatch, "[PASS] 无问题")
        r = reviewer._llm_review(topic="t", draft="d", retrieved_context=[])
        assert r["review_mode"] == "coerce_fallback"
        assert r["is_satisfactory"] is False
        assert r["degraded_blocked"] is True

    def test_coerce_passes_when_explicitly_allowed(self, monkeypatch):
        monkeypatch.setenv("REVIEWER_PASS_THRESHOLD", "6.5")
        monkeypatch.setenv("REVIEWER_ALLOW_DEGRADED_PASS", "1")
        _install_fake_llm(monkeypatch, "[PASS] 无问题")
        r = reviewer._llm_review(topic="t", draft="d", retrieved_context=[])
        assert r["is_satisfactory"] is True


# ----------------------------------------------------------------------
# 5. reviewer_node 端到端
# ----------------------------------------------------------------------

class TestReviewerNode:
    def test_llm_failure_degrades_conservatively(self, monkeypatch):
        def _boom(**kwargs):
            raise RuntimeError("模拟 LLM 不可用")

        monkeypatch.setattr(reviewer, "_llm_review", _boom)
        out = reviewer.reviewer_node(_make_state())

        assert out["review_result"]["review_mode"] == "rule"
        assert out["review_result"]["degraded"] is True
        assert out["is_satisfactory"] is False
        assert out["next_route"] == "researcher"
        assert "final_report" not in out
        assert len(out["errors"]) == 1
        assert out["review_stats"] == {"rounds_total": 1, "rule": 1, "degraded_rounds": 1}
        assert out["review_degraded_rounds"] == [1]

    def test_trace_carries_degradation_fields(self, monkeypatch):
        def _boom(**kwargs):
            raise RuntimeError("模拟 LLM 不可用")

        monkeypatch.setattr(reviewer, "_llm_review", _boom)
        out = reviewer.reviewer_node(_make_state())
        trace = out["execution_trace"][-1]
        assert trace["node"] == "reviewer"
        assert trace["mode"] == "rule"
        assert trace["degraded"] is True
        assert trace["degraded_blocked"] is False
        assert trace["pass_threshold"] == 7.5

    def test_recovery_accumulates_stats(self, monkeypatch):
        def _passing(**kwargs):
            return reviewer._finalize_review(
                {"scores": {"S1": 9, "S2": 9, "S3": 9, "S4": 9}, "evidence_verdicts": [{"claim": "test fact", "status": "supported", "citation_ids": ["S1"], "source_quote": "verified source evidence"}]}, mode="llm_scored"
            )

        monkeypatch.setattr(reviewer, "_llm_review", _passing)
        state = _make_state()
        state["review_stats"] = {"rounds_total": 1, "rule": 1, "degraded_rounds": 1}
        state["review_degraded_rounds"] = [1]
        out = reviewer.reviewer_node(state)

        assert out["is_satisfactory"] is True
        assert out["next_route"] == "end"
        assert "⚠️" not in out["final_report"]
        assert out["review_stats"] == {
            "rounds_total": 2,
            "rule": 1,
            "degraded_rounds": 1,
            "llm_scored": 1,
        }
        assert out["review_degraded_rounds"] == [1]

    def test_degraded_override_cannot_bypass_missing_evidence(self, monkeypatch):
        monkeypatch.setenv("REVIEWER_PASS_THRESHOLD", "6.5")
        monkeypatch.setenv("REVIEWER_ALLOW_DEGRADED_PASS", "1")

        def _coerced(**kwargs):
            return reviewer._coerce_text_review_to_json("[PASS] 无问题")

        monkeypatch.setattr(reviewer, "_llm_review", _coerced)
        out = reviewer.reviewer_node(_make_state())

        assert out["is_satisfactory"] is False
        assert out["review_result"]["evidence_gate_blocked"]
        assert out["next_route"] == "researcher"
        assert "final_report" not in out
        # 静默降级必须留下 errors 记录
        assert any("兜底" in e for e in out["errors"])


# ----------------------------------------------------------------------
# 6. State 契约与报告渲染
# ----------------------------------------------------------------------

class TestStateContract:
    def test_initial_state_has_review_fields(self):
        state = create_initial_state("t")
        assert state["review_stats"] == {}
        assert state["review_degraded_rounds"] == []


class TestExportReport:
    def test_report_shows_mode_stats_and_real_threshold(self, monkeypatch):
        from export_report import _render_markdown_debug

        def _passing(**kwargs):
            return reviewer._finalize_review(
                {"scores": {"S1": 9, "S2": 9, "S3": 9, "S4": 9}, "evidence_verdicts": [{"claim": "test fact", "status": "supported", "citation_ids": ["S1"], "source_quote": "verified source evidence"}]}, mode="llm_scored"
            )

        monkeypatch.setattr(reviewer, "_llm_review", _passing)
        state = _make_state()
        state["review_stats"] = {"rounds_total": 1, "rule": 1, "degraded_rounds": 1}
        state["review_degraded_rounds"] = [1]
        state.update(reviewer.reviewer_node(state))

        md = _render_markdown_debug(state)
        assert "评审模式统计" in md
        assert "降级评审轮次: [1]" in md
        # 历史遗留 bug：报告曾硬编码"通过阈值 8.0"
        assert "通过阈值 8.0" not in md
        assert "通过阈值: 8.0" not in md
        assert "通过阈值 7.5" in md

    def test_legacy_state_renders_without_crash(self):
        from export_report import _render_markdown_debug

        state = create_initial_state("旧格式兼容")
        state["draft"] = LONG_DRAFT
        state["review_result"] = {
            "scores": {"S1": 7, "S2": 7, "S3": 7, "S4": 7},
            "weighted_score": 7.0,
            "is_satisfactory": False,
            "review_mode": "rule",
            "critique_feedback": "x",
        }
        md = _render_markdown_debug(state)
        assert "通过阈值 N/A" in md
