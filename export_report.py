from __future__ import annotations

import argparse
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
    filter_summary = quality_summary.get("filter_summary", {}) or {}

    lines: List[str] = []

    lines.append(f"# 深度研究报告：{topic}")
    lines.append("")
    lines.append("## 1. 运行元信息")
    lines.append("")
    lines.append(f"- 迭代轮次: {state.get('revision_step', 0)}")
    lines.append(f"- 是否通过评审: {state.get('is_satisfactory', False)}")
    lines.append(f"- 下一路由建议: {state.get('next_route', '')}")
    lines.append(f"- 评审模式: {review.get('review_mode', 'unknown')}")
    lines.append(f"- 评审置信度: {review.get('confidence', 'N/A')}")
    lines.append(f"- 输出模式: {state.get('output_mode', 'debug')}")
    lines.append(f"- 来源质量均分: {quality_summary.get('avg_score', 'N/A')}")
    lines.append(f"- 来源质量分层: {quality_summary.get('tier_counts', {})}")
    if filter_summary:
        lines.append(f"- 过滤阈值: {filter_summary.get('thresholds', {})}")
        lines.append(f"- 过滤权重: {filter_summary.get('weights', {})}")
        lines.append(f"- 过滤结果: kept={filter_summary.get('kept', 0)}, dropped={filter_summary.get('dropped', 0)}")
        lines.append(f"- 过滤均值: {filter_summary.get('avg_scores', {})}")
        lines.append(f"- 过滤原因统计: {filter_summary.get('reason_counts', {})}")
    lines.append("")

    lines.append("## 2. 研究正文")
    lines.append("")
    lines.append(report if report else "(未生成正文)")
    lines.append("")

    lines.append("## 3. 评审反馈")
    lines.append("")
    lines.append(review.get("critique_feedback", state.get("critique_feedback", "")) or "(无)")
    lines.append("")
    lines.append("### 3.1 fact_issues（事实问题）")
    fact_issues = review.get("fact_issues", []) or []
    if fact_issues:
        lines.extend([f"- {item}" for item in fact_issues])
    else:
        lines.append("- 无")
    lines.append("")

    lines.append("### 3.2 logic_issues（逻辑问题）")
    logic_issues = review.get("logic_issues", []) or []
    if logic_issues:
        lines.extend([f"- {item}" for item in logic_issues])
    else:
        lines.append("- 无")
    lines.append("")

    lines.append("### 3.3 info_gaps（信息缺口）")
    info_gaps = review.get("info_gaps", []) or []
    if info_gaps:
        lines.extend([f"- {item}" for item in info_gaps])
    else:
        lines.append("- 无")
    lines.append("")

    lines.append("### 3.4 内部辩论与证据裁决")
    supporter = review.get("supporter", {}) or {}
    skeptic = review.get("skeptic", {}) or {}
    judge = review.get("judge", {}) or {}
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
    lines.append(f"- 裁判结论: {judge.get('decision', '')}")
    lines.append(f"- 裁判依据: {judge.get('rationale', '')}")
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

    lines.append("## 4. 每轮草稿与评审历史")
    lines.append("")
    for item in history:
        round_id = item.get("round", "-")
        lines.append(f"### 4.{round_id} 第 {round_id} 轮")
        lines.append("")
        lines.append("- 草稿片段:")
        draft_text = str(item.get("draft", "")).strip()
        lines.append("")
        lines.append((draft_text[:800] + "...") if len(draft_text) > 800 else (draft_text or "(空)"))
        lines.append("")

        review_item = item.get("review", {}) or {}
        mapping = item.get("feedback_paragraph_mapping", []) or []
        round_quality = item.get("source_quality_summary", {}) or {}
        lines.append("- 评审结论:")
        lines.append(f"  - is_satisfactory: {item.get('is_satisfactory', False)}")
        lines.append(f"  - next_route: {item.get('next_route', '')}")
        lines.append(f"  - critique_feedback: {review_item.get('critique_feedback', '')}")
        lines.append(f"  - source_quality_summary: {round_quality}")
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

    lines.append("## 5. 参考上下文摘录")
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
            scores = item.get("filter_scores", {}) if isinstance(item, dict) else {}
            if scores:
                lines.append(
                    f"  - 过滤评分: quality={scores.get('quality', 'N/A')}, "
                    f"signal={scores.get('signal_ratio', 'N/A')}, relevance={scores.get('relevance', 'N/A')}, "
                    f"composite={scores.get('composite', 'N/A')}"
                )
        else:
            lines.append(f"- {str(item)[:260]}")
    if not contexts:
        lines.append("- 无")
    lines.append("")

    if filter_summary:
        lines.append("### 5.1 被过滤样本（最多5条）")
        dropped_samples = filter_summary.get("dropped_samples", []) or []
        if dropped_samples:
            for item in dropped_samples:
                lines.append(
                    f"- {item.get('title', '')} ({item.get('source', '')}) -> {item.get('reasons', [])}"
                )
        else:
            lines.append("- 无")
        lines.append("")

    lines.append("## 6. 执行轨迹")
    lines.append("")
    for idx, item in enumerate(trace, start=1):
        lines.append(f"{idx}. {item}")
    if not trace:
        lines.append("- 无")
    lines.append("")

    lines.append("## 7. 错误与降级记录")
    lines.append("")
    if errors:
        lines.extend([f"- {item}" for item in errors])
    else:
        lines.append("- 无")

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
        saved_paths.extend([str(user_path), str(debug_path)])
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

