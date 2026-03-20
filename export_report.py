from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

from core.graph import compile_graph
from core.state import ResearchState, create_initial_state


def parse_args() -> argparse.Namespace:
    """解析导出脚本参数。"""

    parser = argparse.ArgumentParser(description="Export deep research report to markdown")
    parser.add_argument(
        "--topic",
        required=True,
        help="研究主题",
    )
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=None,
        help="最大迭代轮次（默认读取环境变量 MAX_REVISIONS）",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出 markdown 文件路径（可选）",
    )
    parser.add_argument(
        "--output-mode",
        choices=["user", "debug", "both", "user_only"],
        default="both",
        help=(
            "导出模式：user=仅用户版，debug=仅调试版，"
            "both=单次运行同时导出两版，user_only=运行完整流程但仅导出用户版"
        ),
    )
    return parser.parse_args()


def _safe_filename(text: str) -> str:
    """将主题转换为可用作文件名的短字符串。"""

    keep = []
    for ch in text.strip():
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        elif ch in (" ", "/", "\\"):
            keep.append("-")
    slug = "".join(keep).strip("-")
    return slug[:60] or "report"


def _render_markdown_user(state: ResearchState) -> str:
    """渲染面向读者的简版报告（仅最终正文）。"""

    topic = state.get("topic", "")
    report = state.get("final_report", "") or state.get("draft", "") or ""
    contexts = state.get("retrieved_context", []) or []
    lines: List[str] = [f"# 深度研究报告：{topic}", "", report if report else "(未生成正文)"]

    lines.append("")
    lines.append("## 参考文献")
    lines.append("")
    for item in contexts[:12]:
        if not isinstance(item, dict):
            continue
        citation_id = item.get("citation_id", "-")
        title = item.get("title", "")
        source = item.get("source", "")
        url = item.get("url", "")
        lines.append(f"- [{citation_id}] {title} ({source})")
        if url:
            lines.append(f"  - {url}")
    if not any(isinstance(x, dict) for x in contexts[:12]):
        lines.append("- 无")
    return "\n".join(lines)


def _render_markdown_debug(state: ResearchState) -> str:
    """渲染调试版报告（含过程、评审、轨迹与错误）。"""

    topic = state.get("topic", "")
    report = state.get("final_report", "") or state.get("draft", "") or ""
    review = state.get("review_result", {}) or {}
    contexts = state.get("retrieved_context", []) or []
    trace = state.get("execution_trace", []) or []
    errors = state.get("errors", []) or []
    history = state.get("iteration_history", []) or []
    quality_summary = state.get("source_quality_summary", {}) or {}
    bge_summary = quality_summary.get("bge_summary", {}) or {}

    lines: List[str] = []

    lines.append(f"# 深度研究报告：{topic}")
    lines.append("")
    lines.append("## 1. 运行元信息")
    lines.append("")
    lines.append(f"- 迭代轮次: {state.get('revision_step', 0)}")
    lines.append(f"- 是否通过评审: {state.get('is_satisfactory', False)}")
    lines.append(f"- 下一路由建议: {state.get('next_route', '')}")
    lines.append(f"- 评审模式: {review.get('review_mode', 'unknown')}")
    weighted = review.get("weighted_score")
    if weighted is not None:
        lines.append(f"- 评审加权总分: {weighted:.2f} / 10.00（通过阈值 8.0）")
        scores = review.get("scores", {})
        if scores:
            lines.append(
                f"- 各维度得分: S1事实={scores.get('S1','?')} "
                f"S2逻辑={scores.get('S2','?')} "
                f"S3覆盖={scores.get('S3','?')} "
                f"S4执行={scores.get('S4','?')}"
            )
    else:
        lines.append(f"- 评审置信度: {review.get('confidence', 'N/A')}")
    lines.append(f"- 输出模式: {state.get('output_mode', 'debug')}")
    if bge_summary:
        lines.append(f"- BGE 广搜目标总量: {sum((bge_summary.get('broad_targets', {}) or {}).values())}")
        lines.append(f"- BGE 广搜计划请求量: {bge_summary.get('broad_attempted', 0)}")
        lines.append(f"- BGE 广搜有效返回量: {bge_summary.get('broad_total', 0)}")
        lines.append(f"- BGE 去重后候选: {bge_summary.get('dedup_total', 0)}")
        lines.append(f"- BGE 配额来源: {bge_summary.get('quota_source', {})}")
        lines.append(f"- BGE Provider 统计: {bge_summary.get('broad_fetched', {})}")
        retr = bge_summary.get("retriever", {}) or {}
        rer = bge_summary.get("reranker", {}) or {}
        lines.append(
            f"- BGE Retriever: mode={retr.get('mode', 'unknown')}, model={retr.get('model', '')}, "
            f"selected={retr.get('selected', 0)}/{retr.get('input', 0)}"
        )
        lines.append(
            f"- BGE Reranker: mode={rer.get('mode', 'unknown')}, model={rer.get('model', '')}, "
            f"selected={rer.get('selected', 0)}/{rer.get('input', 0)}"
        )
        if retr.get("error"):
            lines.append(f"- BGE Retriever 错误: {retr.get('error')}")
        if rer.get("error"):
            lines.append(f"- BGE Reranker 错误: {rer.get('error')}")
        bge_config = bge_summary.get("config", {}) or {}
        if bge_config:
            lines.append(f"- BGE 配置快照: {bge_config}")
    lines.append("")

    lines.append("## 2. 评审结果（四维量化评分）")
    lines.append("")

    scores = review.get("scores", {})
    weighted = review.get("weighted_score")
    score_rationale = review.get("score_rationale", {}) or {}
    if scores:
        lines.append("### 2.1 评分总览")
        lines.append("")
        lines.append("| 维度 | 得分 | 权重 | 加权贡献 | 评分理由 |")
        lines.append("|------|------|------|---------|---------|")
        dim_meta = [
            ("S1", "事实准确性", 0.35),
            ("S2", "逻辑完整性", 0.25),
            ("S3", "信息覆盖广度", 0.25),
            ("S4", "结论可执行性", 0.15),
        ]
        for dim, label, w in dim_meta:
            s = scores.get(dim, "?")
            contrib = round(float(s) * w, 2) if isinstance(s, (int, float)) else "?"
            rationale = str(score_rationale.get(dim, "")).replace("|", "｜")[:60]
            lines.append(f"| {dim} {label} | {s}/10 | {int(w*100)}% | {contrib} | {rationale} |")
        lines.append(f"| **加权总分** | **{weighted:.2f}/10** | 100% | — | 通过阈值: 8.0 |" if weighted is not None else "")
        verdict = "✅ 通过" if review.get("is_satisfactory") else "❌ 未通过"
        lines.append(f"| **评审结论** | {verdict} | — | — | — |")
        lines.append("")

    lines.append("### 2.2 综合反馈")
    lines.append("")
    lines.append(review.get("critique_feedback", state.get("critique_feedback", "")) or "(无)")
    lines.append("")

    lines.append("### 2.3 事实问题（fact_issues）")
    fact_issues = review.get("fact_issues", []) or []
    if fact_issues:
        lines.extend([f"- {item}" for item in fact_issues])
    else:
        lines.append("- 无")
    lines.append("")

    lines.append("### 2.4 逻辑问题（logic_issues）")
    logic_issues = review.get("logic_issues", []) or []
    if logic_issues:
        lines.extend([f"- {item}" for item in logic_issues])
    else:
        lines.append("- 无")
    lines.append("")

    lines.append("### 2.5 信息缺口（info_gaps）")
    info_gaps = review.get("info_gaps", []) or []
    if info_gaps:
        lines.extend([f"- {item}" for item in info_gaps])
    else:
        lines.append("- 无")
    lines.append("")

    lines.append("### 2.6 引用核查（citation_checks）")
    citation_checks = review.get("citation_checks", []) or []
    if citation_checks:
        lines.append("")
        lines.append("| 引用ID | 声明片段 | 是否有原文支撑 | 原文证据 |")
        lines.append("|--------|---------|--------------|---------|")
        for cc in citation_checks:
            if not isinstance(cc, dict):
                continue
            cid = cc.get("citation_id", "?")
            claim = str(cc.get("claim", "")).replace("|", "｜")[:60]
            supported = "✅ 支撑" if cc.get("supported") else "❌ 未找到"
            evidence = str(cc.get("evidence", "")).replace("|", "｜")[:80]
            lines.append(f"| {cid} | {claim} | {supported} | {evidence} |")
    else:
        lines.append("- 无引用核查记录（规则回退模式或模型未输出）")
    lines.append("")

    lines.append("### 2.7 支持与质疑观点")
    supporter = review.get("supporter", {}) or {}
    skeptic = review.get("skeptic", {}) or {}
    lines.append("- 支持者观点:")
    strengths = supporter.get("strengths", []) or []
    if strengths:
        lines.extend([f"  - {item}" for item in strengths])
    else:
        lines.append("  - 无")
    lines.append("- 质疑者观点:")
    critical = skeptic.get("critical_issues", []) or []
    if critical:
        lines.extend([f"  - {item}" for item in critical])
    else:
        lines.append("  - 无")
    lines.append("- 争议点:")
    controversy_points = review.get("controversy_points", []) or []
    if controversy_points:
        lines.extend([f"  - {item}" for item in controversy_points])
    else:
        lines.append("  - 无")
    lines.append("- 证据裁决:")
    verdicts = review.get("evidence_verdicts", []) or []
    if verdicts:
        for item in verdicts:
            if isinstance(item, dict):
                lines.append(
                    "  - claim: {claim} | status: {status} | action: {action}".format(
                        claim=item.get("claim", ""),
                        status=item.get("status", ""),
                        action=item.get("action", ""),
                    )
                )
                lines.append(f"    - evidence: {item.get('evidence', '')}")
            else:
                lines.append(f"  - {item}")
    else:
        lines.append("  - 无")
    lines.append("")

    lines.append("## 3. 每轮草稿与评审历史")
    lines.append("")
    for item in history:
        round_id = item.get("round", "-")
        lines.append(f"### 3.{round_id} 第 {round_id} 轮")
        lines.append("")
        lines.append("- 草稿片段:")
        draft_text = str(item.get("draft", "")).strip()
        lines.append("")
        lines.append((draft_text[:800] + "...") if len(draft_text) > 800 else (draft_text or "(空)"))
        lines.append("")

        review_item = item.get("review", {}) or {}
        mapping = item.get("feedback_paragraph_mapping", []) or []
        lines.append("- 评审结论:")
        lines.append(f"  - is_satisfactory: {item.get('is_satisfactory', False)}")
        r_scores = review_item.get("scores", {})
        r_weighted = review_item.get("weighted_score")
        if r_scores and r_weighted is not None:
            lines.append(
                f"  - 加权总分: {r_weighted:.2f}/10  "
                f"(S1={r_scores.get('S1','?')} S2={r_scores.get('S2','?')} "
                f"S3={r_scores.get('S3','?')} S4={r_scores.get('S4','?')})"
            )
        lines.append(f"  - next_route: {item.get('next_route', '')}")
        lines.append(f"  - critique_feedback: {review_item.get('critique_feedback', '')}")
        lines.append("")

        lines.append("- 反馈修订映射（must_fix -> 章节）:")
        if mapping:
            for mp in mapping:
                lines.append(
                    f"  - issue: {mp.get('issue', '')} -> section: {mp.get('mapped_section', '')}"
                )
                lines.append(f"    - snippet: {mp.get('evidence_snippet', '')}")
        else:
            lines.append("  - 无")
        lines.append("")

        review_item = item.get("review", {}) or {}
        verdicts = review_item.get("evidence_verdicts", []) or []
        lines.append("- 本轮争议点与裁决:")
        controversies = review_item.get("controversy_points", []) or []
        if controversies:
            for cp in controversies:
                lines.append(f"  - 争议: {cp}")
        else:
            lines.append("  - 争议: 无")
        if verdicts:
            for vd in verdicts:
                if isinstance(vd, dict):
                    lines.append(
                        f"  - 裁决: {vd.get('claim', '')} -> {vd.get('status', '')} | action={vd.get('action', '')}"
                    )
                else:
                    lines.append(f"  - 裁决: {vd}")
        else:
            lines.append("  - 裁决: 无")
        lines.append("")
    if not history:
        lines.append("- 无")
    lines.append("")

    lines.append("## 4. 参考上下文摘录")
    lines.append("")
    for item in contexts[:12]:
        if isinstance(item, dict):
            citation_id = item.get("citation_id", "-")
            title = item.get("title", "")
            url = item.get("url", "")
            source = item.get("source", "")
            lines.append(f"- [{citation_id}] {title} ({source})")
            if url:
                lines.append(f"  - url: {url}")
        else:
            lines.append(f"- {str(item)[:260]}")
    if not contexts:
        lines.append("- 无")
    lines.append("")

    lines.append("## 5. 执行轨迹")
    lines.append("")
    for idx, item in enumerate(trace, start=1):
        lines.append(f"{idx}. {item}")
    if not trace:
        lines.append("- 无")
    lines.append("")

    lines.append("## 6. 错误与降级记录")
    lines.append("")
    if errors:
        lines.extend([f"- {item}" for item in errors])
    else:
        lines.append("- 无")

    # ── MAB 自适应检索预算 ──────────────────────────────────────────
    mab_state = state.get("mab_state", {}) or {}
    mab_data = bge_summary.get("mab", {}) or {}
    if mab_state or mab_data:
        lines.append("")
        lines.append("## 7. MAB 自适应检索预算（Thompson Sampling）")
        lines.append("")

        mab_round = mab_state.get("round", mab_data.get("round", 0))
        lines.append(f"- 累计更新轮次: {mab_round}")
        lines.append("")

        # 7.1 各信源当前状态
        arm_display = {"duckduckgo": "DuckDuckGo", "arxiv": "ArXiv", "tavily": "Tavily"}
        alphas = mab_state.get("alpha", {})
        betas = mab_state.get("beta", {})
        lines.append("### 7.1 各信源当前状态")
        lines.append("")
        lines.append("| 信源 | α | β | 期望奖励 E[θ] | 趋势 |")
        lines.append("|------|---|---|-------------|------|")
        for arm in ("duckduckgo", "arxiv", "tavily"):
            a_val = alphas.get(arm, 1.0)
            b_val = betas.get(arm, 1.0)
            ex = round(a_val / (a_val + b_val), 4) if (a_val + b_val) > 0 else 0.5
            if ex > 0.6:
                trend = "↑ 优选"
            elif ex < 0.4:
                trend = "↓ 降权"
            else:
                trend = "→ 中性"
            lines.append(f"| {arm_display[arm]} | {a_val} | {b_val} | {ex} | {trend} |")
        lines.append("")

        # 7.2 逐轮预算分配与奖励记录
        hist = mab_state.get("history", [])
        if hist:
            lines.append("### 7.2 逐轮预算分配与奖励记录")
            lines.append("")
            for rec in hist:
                r_id = rec.get("round", "-")
                lines.append(f"#### 第 {r_id} 轮")
                lines.append("")
                lines.append("| 信源 | 观测奖励 | α（更新后）| β（更新后）| E[θ]（更新后）|")
                lines.append("|------|---------|-----------|-----------|--------------|")
                for arm in ("duckduckgo", "arxiv", "tavily"):
                    r_val = rec.get("rewards", {}).get(arm)
                    r_str = f"{r_val:.4f}" if r_val is not None else "N/A"
                    pa = (rec.get("params_after") or {}).get(arm) or {}
                    if pa:
                        a_after = pa.get("alpha", "-")
                        b_after = pa.get("beta", "-")
                        e_after = pa.get("expected_reward", "-")
                    else:
                        a_after = b_after = e_after = "N/A"
                    lines.append(f"| {arm_display[arm]} | {r_str} | {a_after} | {b_after} | {e_after} |")
                lines.append("")

        # 7.3 本轮预算对比
        base = mab_data.get("base_budgets", {})
        alloc = mab_data.get("allocated_budgets", {})
        if base or alloc:
            lines.append("### 7.3 本轮（最后一次 Researcher 调用）预算对比")
            lines.append("")
            lines.append("| 信源 | 基础预算（env）| MAB 分配 | 变化 |")
            lines.append("|------|--------------|---------|------|")
            total_base = 0
            total_alloc = 0
            for arm in ("duckduckgo", "arxiv", "tavily"):
                b = base.get(arm, 0)
                a = alloc.get(arm, 0)
                diff = a - b
                diff_str = f"+{diff}" if diff > 0 else str(diff)
                lines.append(f"| {arm_display[arm]} | {b} | {a} | {diff_str} |")
                total_base += b
                total_alloc += a
            lines.append("")
            lines.append(
                f"> 本轮 MAB 轮次编号: {mab_round}  "
                f"总基础预算: {total_base}  总 MAB 预算: {total_alloc}"
            )
            lines.append("")

    return "\n".join(lines)


def _build_output_paths(output: str | None, topic: str, mode: str) -> List[Path]:
    """根据导出模式生成输出路径列表。"""

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    if output:
        base_path = Path(output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_path = reports_dir / f"{timestamp}-{_safe_filename(topic)}.md"

    stem = base_path.stem
    suffix = base_path.suffix or ".md"
    parent = base_path.parent

    if mode == "both":
        return [
            parent / f"{stem}-user{suffix}",
            parent / f"{stem}-debug{suffix}",
        ]
    return [base_path]


def _render_by_mode(state: ResearchState, mode: str) -> Dict[str, str]:
    """返回需要写出的报告内容映射。"""

    user_md = _render_markdown_user(state)
    debug_md = _render_markdown_debug(state)

    if mode == "user":
        return {"user": user_md}
    if mode == "debug":
        return {"debug": debug_md}
    if mode == "user_only":
        # 实际仍跑完整链路，仅隐藏 debug 输出文件。
        return {"user": user_md}
    return {"user": user_md, "debug": debug_md}


def _build_bge_details_payload(state: ResearchState) -> Dict[str, object]:
    """构建 BGE 过程明细 JSON 载荷。"""

    bge_summary = (state.get("source_quality_summary", {}) or {}).get("bge_summary", {}) or {}
    retr = bge_summary.get("retriever", {}) or {}
    rer = bge_summary.get("reranker", {}) or {}
    retr_query = str(retr.get("query", "") or state.get("topic", ""))
    rer_query = str(state.get("topic", ""))

    def _attach_query(records: List[Dict[str, object]], query: str) -> List[Dict[str, object]]:
        enriched: List[Dict[str, object]] = []
        for row in records:
            item = dict(row)
            item["query"] = query
            enriched.append(item)
        return enriched

    bge_summary_mab = bge_summary.get("mab", {}) or {}
    mab_state_payload = state.get("mab_state", {}) or {}

    return {
        "topic": state.get("topic", ""),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "search_queries": state.get("search_queries", []),
        "mab": {
            "state": mab_state_payload,
            "last_round": bge_summary_mab,
        },
        "bge": {
            "broad_targets": bge_summary.get("broad_targets", {}),
            "quota_source": bge_summary.get("quota_source", {}),
            "broad_fetched": bge_summary.get("broad_fetched", {}),
            "broad_attempted": bge_summary.get("broad_attempted", 0),
            "broad_total": bge_summary.get("broad_total", 0),
            "dedup_total": bge_summary.get("dedup_total", 0),
            "config": bge_summary.get("config", {}),
            "retriever": {
                "mode": retr.get("mode", "unknown"),
                "model": retr.get("model", ""),
                "query": retr_query,
                "input": retr.get("input", 0),
                "selected": retr.get("selected", 0),
                "dropped_count": retr.get("dropped_count", 0),
                "selected_records": _attach_query(retr.get("selected_records", []), retr_query),
                "dropped_records": _attach_query(retr.get("dropped_records", []), retr_query),
            },
            "reranker": {
                "mode": rer.get("mode", "unknown"),
                "model": rer.get("model", ""),
                "query": rer_query,
                "input": rer.get("input", 0),
                "selected": rer.get("selected", 0),
                "dropped_count": rer.get("dropped_count", 0),
                "selected_records": _attach_query(rer.get("selected_records", []), rer_query),
                "dropped_records": _attach_query(rer.get("dropped_records", []), rer_query),
            },
        },
    }


def run_and_export(
    topic: str,
    max_revisions: int | None,
    output: str | None,
    output_mode: str,
) -> List[str]:
    """执行图并导出 markdown 报告，返回输出路径列表。"""

    env_loaded = load_dotenv()
    if not env_loaded:
        load_dotenv(".env.example")

    actual_max_revisions = max_revisions
    if actual_max_revisions is None:
        actual_max_revisions = int(os.getenv("MAX_REVISIONS", "3"))

    app = compile_graph(max_revisions=actual_max_revisions)
    normalized_mode = output_mode if output_mode in {"user", "debug", "both", "user_only"} else "both"

    initial_state = create_initial_state(
        topic=topic,
        # 运行时统一按 debug 状态记录，导出层再决定展示与落盘。
        output_mode="debug",
    )
    state: ResearchState = app.invoke(initial_state)

    outputs = _render_by_mode(state, normalized_mode)
    paths = _build_output_paths(output=output, topic=topic, mode=normalized_mode)

    saved_paths: List[str] = []
    if normalized_mode == "both":
        user_path, debug_path = paths
        user_path.write_text(outputs["user"], encoding="utf-8")
        debug_path.write_text(outputs["debug"], encoding="utf-8")
        bge_json_path = user_path.parent / f"{user_path.stem.rsplit('-user', 1)[0]}-bge-details.json"
        bge_payload = _build_bge_details_payload(state)
        bge_json_path.write_text(json.dumps(bge_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        saved_paths.extend([str(user_path), str(debug_path), str(bge_json_path)])
    else:
        path = paths[0]
        key = "debug" if normalized_mode == "debug" else "user"
        path.write_text(outputs[key], encoding="utf-8")
        saved_paths.append(str(path))

    return saved_paths


if __name__ == "__main__":
    args = parse_args()
    paths = run_and_export(
        topic=args.topic,
        max_revisions=args.max_revisions,
        output=args.output,
        output_mode=args.output_mode,
    )
    for path in paths:
        print(f"[EXPORT] report saved: {path}")

