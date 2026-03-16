from __future__ import annotations

import argparse
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List

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
        "--report-length",
        choices=["short", "medium", "long"],
        default="medium",
        help="报告篇幅控制",
    )
    parser.add_argument(
        "--output-mode",
        choices=["user", "debug"],
        default="debug",
        help="导出模式：user 仅最终报告；debug 包含轨迹、评审与错误信息",
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
    lines: List[str] = [f"# 深度研究报告：{topic}", "", report if report else "(未生成正文)"]
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
    length_meta = state.get("length_control_meta", {}) or {}

    lines: List[str] = []

    def summarize_core_content(text: str) -> str:
        """从上下文正文提炼核心句，避免固定前缀截取。"""

        normalized = re.sub(r"\s+", " ", (text or "").strip())
        if not normalized:
            return ""
        sentences = [s.strip() for s in re.split(r"(?<=[。！？.!?])\s+", normalized) if s.strip()]
        if not sentences:
            return normalized[:260]

        keywords = [
            "benchmark",
            "result",
            "conclusion",
            "method",
            "architecture",
            "performance",
            "latency",
            "power",
            "security",
            "差异",
            "性能",
            "结论",
            "对比",
            "实验",
            "优势",
            "局限",
        ]
        scored: List[tuple[float, str]] = []
        for s in sentences:
            lower = s.lower()
            hit = sum(1 for k in keywords if k in lower)
            score = hit * 1.6 + min(len(s) / 90.0, 1.2)
            scored.append((score, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = " ".join([s for _, s in scored[:2]]).strip()
        return selected[:320] if selected else normalized[:260]
    lines.append(f"# 深度研究报告：{topic}")
    lines.append("")
    lines.append("## 1. 运行元信息")
    lines.append("")
    lines.append(f"- 迭代轮次: {state.get('revision_step', 0)}")
    lines.append(f"- 是否通过评审: {state.get('is_satisfactory', False)}")
    lines.append(f"- 下一路由建议: {state.get('next_route', '')}")
    lines.append(f"- 评审模式: {review.get('review_mode', 'unknown')}")
    lines.append(f"- 评审置信度: {review.get('confidence', 'N/A')}")
    lines.append(f"- 报告篇幅: {state.get('report_length', 'medium')}")
    lines.append(f"- 输出模式: {state.get('output_mode', 'debug')}")
    lines.append(f"- 长度控制策略: {length_meta.get('strategy', 'unknown')}")
    lines.append(f"- 完整性检查: {length_meta.get('completeness', {}).get('is_complete', 'N/A')}")
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
        lines.append((draft_text[:2000] + "...") if len(draft_text) > 2000 else (draft_text or "(空)"))
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
            core_summary = str(item.get("core_summary", "")).strip()
            raw_content = str(item.get("content", ""))
            content = core_summary if core_summary else summarize_core_content(raw_content)
            content = content.replace("\n", " ")
            lines.append(f"- [{citation_id}] {title} ({source})")
            if url:
                lines.append(f"  - url: {url}")
            lines.append(f"  - 核心提炼: {content}")
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


def _render_markdown(state: ResearchState) -> str:
    """根据 output_mode 自动选择渲染模板。"""

    if state.get("output_mode", "debug") == "user":
        return _render_markdown_user(state)
    return _render_markdown_debug(state)


def run_and_export(
    topic: str,
    max_revisions: int | None,
    output: str | None,
    report_length: str,
    output_mode: str,
) -> str:
    """执行图并导出 markdown 报告，返回输出路径。"""

    env_loaded = load_dotenv()
    if not env_loaded:
        load_dotenv(".env.example")

    actual_max_revisions = max_revisions
    if actual_max_revisions is None:
        actual_max_revisions = int(os.getenv("MAX_REVISIONS", "3"))

    app = compile_graph(max_revisions=actual_max_revisions)
    initial_state = create_initial_state(
        topic=topic,
        report_length=report_length if report_length in {"short", "medium", "long"} else "medium",
        output_mode=output_mode if output_mode in {"user", "debug"} else "debug",
    )
    state: ResearchState = app.invoke(initial_state)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    if output:
        output_path = Path(output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = reports_dir / f"{timestamp}-{_safe_filename(topic)}.md"

    output_path.write_text(_render_markdown(state), encoding="utf-8")
    return str(output_path)


if __name__ == "__main__":
    args = parse_args()
    path = run_and_export(
        topic=args.topic,
        max_revisions=args.max_revisions,
        output=args.output,
        report_length=args.report_length,
        output_mode=args.output_mode,
    )
    print(f"[EXPORT] report saved: {path}")

