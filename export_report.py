from __future__ import annotations

import argparse
import os
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
    lines.append(f"- 报告篇幅: {state.get('report_length', 'medium')}")
    lines.append(f"- 输出模式: {state.get('output_mode', 'debug')}")
    lines.append(f"- 来源质量均分: {quality_summary.get('avg_score', 'N/A')}")
    lines.append(f"- 来源质量分层: {quality_summary.get('tier_counts', {})}")
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
            content = str(item.get("content", ""))[:220].replace("\n", " ")
            lines.append(f"- [{citation_id}] {title} ({source})")
            if url:
                lines.append(f"  - url: {url}")
            lines.append(f"  - 摘要: {content}")
        else:
            lines.append(f"- {str(item)[:260]}")
    if not contexts:
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

