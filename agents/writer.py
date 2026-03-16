from __future__ import annotations

import os
import re
from typing import Any, Dict, List

try:
    from core.state import ResearchState
except ModuleNotFoundError:  # 兼容直接脚本方式
    from state import ResearchState  # type: ignore

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None

from prompts.system_prompts import WRITER_SYSTEM_PROMPT, build_writer_user_prompt


def _append_error(errors: List[str], message: str) -> List[str]:
    """将错误信息追加到 errors，避免覆盖既有日志。"""

    updated = list(errors)
    updated.append(message)
    return updated


def _fallback_draft(
    topic: str,
    revision_step: int,
    critique_feedback: str,
    retrieved_context: List[Dict[str, Any] | str],
    report_length: str,
) -> str:
    """在无法调用 LLM 时生成可用的占位草稿。"""

    return (
        f"# {topic} 深度研究草稿\n\n"
        f"- 迭代轮次: {revision_step}\n"
        f"- 上下文数量: {len(retrieved_context)}\n"
        f"- 目标篇幅: {report_length}\n"
        f"- 修订依据: {critique_feedback or '首轮生成'}\n\n"
        "## 一、问题定义\n"
        "待补充：请在接入 LLM 后生成完整问题背景与边界。\n\n"
        "## 二、关键发现\n"
        "待补充：请基于检索上下文形成有引用的结构化结论。\n\n"
        "## 三、结论与建议\n"
        "待补充：请输出可执行建议与风险提示。"
    )


def _strip_markdown(text: str) -> str:
    """粗略移除常见 Markdown 标记，估算净字数。"""

    clean = re.sub(r"```[\s\S]*?```", "", text)
    clean = re.sub(r"`[^`]*`", "", clean)
    clean = re.sub(r"[#>*\-]", "", clean)
    clean = re.sub(r"\[[^]]+]\([^)]+\)", "", clean)
    clean = re.sub(r"\s+", " ", clean)
    return clean.strip()


def _get_length_limits(report_length: str) -> tuple[int, int]:
    """返回 short/medium/long 的净字数上下限。"""

    if report_length == "short":
        return 800, 1200
    if report_length == "long":
        return 3200, 4500
    return 1800, 2600


def _build_outline_plan(report_length: str, revision_directives: Dict[str, Any]) -> List[Dict[str, Any]]:
    """按目标篇幅构建提纲与段落预算。"""

    min_len, max_len = _get_length_limits(report_length)
    target_len = (min_len + max_len) // 2

    must_fix = revision_directives.get("must_fix", [])
    fix_weight = 1.1 if must_fix else 1.0

    base = [
        ("摘要", 0.10),
        ("背景与范围", 0.18),
        ("关键发现", 0.34 * fix_weight),
        ("风险与局限", 0.18 * fix_weight),
        ("结论与建议", 0.20),
    ]
    total = sum(weight for _, weight in base)

    plan: List[Dict[str, Any]] = []
    for title, weight in base:
        ratio = weight / total
        budget = max(int(target_len * ratio), 120)
        plan.append({"title": title, "budget": budget})
    return plan


def _plan_to_text(plan: List[Dict[str, Any]]) -> str:
    """将提纲预算转换为提示词可读文本。"""

    lines = ["提纲与预算:"]
    for idx, item in enumerate(plan, start=1):
        lines.append(f"{idx}. {item.get('title', '')}: 约 {item.get('budget', 0)} 字")
    return "\n".join(lines)


def _split_paragraphs(text: str) -> List[str]:
    """按空行切分段落。"""

    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    return parts if parts else ([text.strip()] if text.strip() else [])


def _fit_sections_to_range(
    sections: List[Dict[str, str]],
    min_len: int,
    max_len: int,
    revision_directives: Dict[str, Any],
) -> List[Dict[str, str]]:
    """按段落压缩/扩写，使正文落入目标区间，不做文末硬截断。"""

    def joined_net_len(current: List[Dict[str, str]]) -> int:
        merged = "\n\n".join([f"## {s['title']}\n\n{s['content']}" for s in current])
        return len(_strip_markdown(merged))

    adjusted = [dict(item) for item in sections]
    current_len = joined_net_len(adjusted)

    # 过长时优先逐节删除末尾段落，保留每节至少一段。
    while current_len > max_len:
        changed = False
        for idx in sorted(range(len(adjusted)), key=lambda i: len(adjusted[i].get("content", "")), reverse=True):
            paragraphs = _split_paragraphs(adjusted[idx].get("content", ""))
            if len(paragraphs) <= 1:
                continue
            paragraphs.pop()
            adjusted[idx]["content"] = "\n\n".join(paragraphs)
            current_len = joined_net_len(adjusted)
            changed = True
            if current_len <= max_len:
                break
        if not changed:
            break

    # 过短时按关键章节补一段，强调评审修订项。
    if current_len < min_len:
        need = min_len - current_len
        must_fix = revision_directives.get("must_fix", [])
        addon = (
            "补充说明：结合当前证据进一步展开方法假设、边界条件与可执行建议，"
            "并明确不确定性来源与验证路径。"
        )
        if must_fix:
            addon += f" 本轮重点修复项包括：{'; '.join([str(x) for x in must_fix[:3]])}。"

        for idx in [2, 3, 4, 1, 0]:
            if idx >= len(adjusted):
                continue
            adjusted[idx]["content"] = (adjusted[idx].get("content", "") + "\n\n" + addon).strip()
            current_len = joined_net_len(adjusted)
            if current_len >= min_len or need <= 0:
                break

    return adjusted


def _assemble_planned_draft(topic: str, sections: List[Dict[str, str]]) -> str:
    """将分段内容拼装为 Markdown 报告。"""

    lines = [f"# {topic} 深度研究报告", ""]
    for item in sections:
        lines.append(f"## {item.get('title', '未命名章节')}")
        lines.append("")
        lines.append(item.get("content", "（待补充）"))
        lines.append("")
    return "\n".join(lines).strip()


def _check_completeness(draft: str, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
    """完整性检查：必须包含所有规划章节且每节有实质内容。"""

    missing_titles: List[str] = []
    thin_titles: List[str] = []
    for item in plan:
        title = str(item.get("title", "")).strip()
        marker = f"## {title}"
        if marker not in draft:
            missing_titles.append(title)
            continue
        block = draft.split(marker, 1)[1]
        next_pos = block.find("\n## ")
        body = block[:next_pos] if next_pos != -1 else block
        if len(_strip_markdown(body)) < 80:
            thin_titles.append(title)
    return {
        "is_complete": not missing_titles and not thin_titles,
        "missing_titles": missing_titles,
        "thin_titles": thin_titles,
    }


def _generate_section_with_llm(
    llm: Any,
    topic: str,
    report_length: str,
    critique_feedback: str,
    revision_directives: Dict[str, Any],
    context_snippet: str,
    plan_text: str,
    section_title: str,
    section_budget: int,
) -> str:
    """按单节预算生成段落，降低一次性长文失控概率。"""

    prompt = (
        f"研究主题: {topic}\n"
        f"目标篇幅档位: {report_length}\n"
        f"当前章节: {section_title}\n"
        f"章节预算: 约 {section_budget} 字\n"
        f"评审反馈: {critique_feedback or '无'}\n"
        f"修订指令: {revision_directives or {}}\n\n"
        f"{plan_text}\n\n"
        f"检索上下文:\n{context_snippet}\n\n"
        "请只输出该章节正文，不要输出标题。"
        "内容必须与证据一致，避免重复其他章节。"
    )
    response = llm.invoke(
        [
            ("system", WRITER_SYSTEM_PROMPT),
            ("human", prompt),
        ]
    )
    return (getattr(response, "content", "") or "").strip()


def _extract_section_blocks(draft: str) -> List[Dict[str, str]]:
    """按 Markdown 二级标题切分，便于构建反馈-段落映射。"""

    lines = draft.splitlines()
    sections: List[Dict[str, str]] = []
    current_title = "引言"
    buffer: List[str] = []

    for line in lines:
        if line.startswith("## "):
            if buffer:
                sections.append({"section": current_title, "content": "\n".join(buffer).strip()})
                buffer = []
            current_title = line[3:].strip()
            continue
        buffer.append(line)

    if buffer:
        sections.append({"section": current_title, "content": "\n".join(buffer).strip()})

    return sections


def _build_feedback_mapping(draft: str, revision_directives: Dict[str, Any]) -> List[Dict[str, Any]]:
    """将 must_fix 项映射到草稿章节，便于 debug 报告展示可追踪修订。"""

    must_fix = revision_directives.get("must_fix", [])
    if not isinstance(must_fix, list) or not must_fix:
        return []

    sections = _extract_section_blocks(draft)
    if not sections:
        sections = [{"section": "正文", "content": draft}]

    mapping: List[Dict[str, Any]] = []
    for idx, issue in enumerate(must_fix, start=1):
        picked = sections[(idx - 1) % len(sections)]
        snippet = picked.get("content", "")[:220].replace("\n", " ")
        mapping.append(
            {
                "issue": str(issue),
                "mapped_section": picked.get("section", "正文"),
                "evidence_snippet": snippet,
            }
        )
    return mapping


def _build_context_snippet(retrieved_context: List[Dict[str, Any] | str]) -> str:
    """提取用于分段生成的上下文摘要。"""

    chunks: List[str] = []
    for idx, item in enumerate(retrieved_context[:8], start=1):
        if isinstance(item, dict):
            citation = item.get("citation_id", f"S{idx}")
            title = item.get("title", f"source-{idx}")
            content = str(item.get("content", ""))[:360]
            chunks.append(f"[{citation}] {title}: {content}")
        else:
            chunks.append(f"[S{idx}] {str(item)[:360]}")
    return "\n".join(chunks)


def writer_node(state: ResearchState) -> Dict[str, Any]:
    """主笔代理节点。

    输入: topic / retrieved_context / critique_feedback / revision_step
    输出: draft / errors
    """

    topic = state.get("topic", "")
    revision_step = state.get("revision_step", 0)
    critique_feedback = state.get("critique_feedback", "")
    retrieved_context = list(state.get("retrieved_context", []))
    report_length = str(state.get("report_length", "medium"))
    revision_directives = dict(state.get("revision_directives", {}) or {})

    errors = list(state.get("errors", []))

    if ChatOpenAI is None:
        errors = _append_error(errors, "langchain_openai 未安装，Writer 使用本地占位草稿。")
        draft = _fallback_draft(
            topic,
            revision_step,
            critique_feedback,
            retrieved_context,
            report_length,
        )
        plan = _build_outline_plan(report_length, revision_directives)
        completeness = _check_completeness(draft, plan)
        mapping = _build_feedback_mapping(draft, revision_directives)
        trace = list(state.get("execution_trace", []))
        trace.append(
            {
                "node": "writer",
                "revision_step": revision_step,
                "draft_len": len(draft),
                "mode": "fallback_no_langchain_openai",
                "planned_sections": len(plan),
                "complete": completeness.get("is_complete", False),
                "mapped_items": len(mapping),
            }
        )
        return {
            "draft": draft,
            "errors": errors,
            "feedback_paragraph_mapping": mapping,
            "execution_trace": trace,
            "length_control_meta": {
                "strategy": "fallback",
                "plan": plan,
                "completeness": completeness,
            },
        }

    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    if not deepseek_api_key:
        errors = _append_error(errors, "未检测到 DEEPSEEK_API_KEY，Writer 使用本地占位草稿。")
        draft = _fallback_draft(
            topic,
            revision_step,
            critique_feedback,
            retrieved_context,
            report_length,
        )
        plan = _build_outline_plan(report_length, revision_directives)
        completeness = _check_completeness(draft, plan)
        mapping = _build_feedback_mapping(draft, revision_directives)
        trace = list(state.get("execution_trace", []))
        trace.append(
            {
                "node": "writer",
                "revision_step": revision_step,
                "draft_len": len(draft),
                "mode": "fallback_no_api_key",
                "planned_sections": len(plan),
                "complete": completeness.get("is_complete", False),
                "mapped_items": len(mapping),
            }
        )
        return {
            "draft": draft,
            "errors": errors,
            "feedback_paragraph_mapping": mapping,
            "execution_trace": trace,
            "length_control_meta": {
                "strategy": "fallback",
                "plan": plan,
                "completeness": completeness,
            },
        }

    user_prompt = build_writer_user_prompt(
        topic=topic,
        revision_step=revision_step,
        critique_feedback=critique_feedback,
        retrieved_context=retrieved_context,
        report_length=report_length if report_length in {"short", "medium", "long"} else "medium",
        revision_directives=revision_directives,
    )

    try:
        plan = _build_outline_plan(report_length, revision_directives)
        min_len, max_len = _get_length_limits(report_length)
        plan_text = _plan_to_text(plan)
        context_snippet = _build_context_snippet(retrieved_context)

        llm = ChatOpenAI(
            model=deepseek_model,
            api_key=lambda: deepseek_api_key,
            base_url=deepseek_base_url,
            temperature=0.3,
        )

        sections: List[Dict[str, str]] = []
        for item in plan:
            title = str(item.get("title", ""))
            budget = int(item.get("budget", 200))
            section_text = _generate_section_with_llm(
                llm=llm,
                topic=topic,
                report_length=report_length,
                critique_feedback=critique_feedback,
                revision_directives=revision_directives,
                context_snippet=context_snippet,
                plan_text=plan_text,
                section_title=title,
                section_budget=budget,
            )
            if not section_text:
                section_text = "目前资料未充分覆盖该章节，建议继续补充证据并在下一轮修订。"
            sections.append({"title": title, "content": section_text})

        sections = _fit_sections_to_range(sections, min_len, max_len, revision_directives)
        draft = _assemble_planned_draft(topic, sections)

        # 完整性不足时，回退一次整文修订，确保结构齐全。
        completeness = _check_completeness(draft, plan)
        if not completeness.get("is_complete", False):
            repair_prompt = (
                f"请修复以下完整性问题: {completeness}。\n"
                f"目标字数范围: {min_len}-{max_len}。\n"
                f"原草稿:\n{draft}\n\n"
                f"约束提示:\n{user_prompt}"
            )
            repaired = llm.invoke(
                [
                    ("system", WRITER_SYSTEM_PROMPT),
                    ("human", repair_prompt),
                ]
            )
            repaired_text = (getattr(repaired, "content", "") or "").strip()
            if repaired_text:
                draft = repaired_text
            completeness = _check_completeness(draft, plan)

        length_meta = {
            "strategy": "planned_generation",
            "target": report_length,
            "min_len": min_len,
            "max_len": max_len,
            "net_len": len(_strip_markdown(draft)),
            "plan": plan,
            "completeness": completeness,
        }
        mapping = _build_feedback_mapping(draft, revision_directives)
        trace = list(state.get("execution_trace", []))
        trace.append(
            {
                "node": "writer",
                "revision_step": revision_step,
                "draft_len": len(draft),
                "mode": "llm",
                "planned_sections": len(plan),
                "complete": completeness.get("is_complete", False),
                "mapped_items": len(mapping),
            }
        )
        return {
            "draft": draft,
            "errors": errors,
            "feedback_paragraph_mapping": mapping,
            "execution_trace": trace,
            "length_control_meta": length_meta,
        }
    except Exception as exc:
        errors = _append_error(errors, f"DeepSeek 生成失败，Writer 已降级: {exc}")
        draft = _fallback_draft(
            topic,
            revision_step,
            critique_feedback,
            retrieved_context,
            report_length,
        )
        plan = _build_outline_plan(report_length, revision_directives)
        min_len, max_len = _get_length_limits(report_length)
        sections = [
            {"title": item.get("title", "章节"), "content": "目前资料未充分覆盖该章节，建议补充检索后继续修订。"}
            for item in plan
        ]
        if sections:
            sections[0]["content"] = draft
        sections = _fit_sections_to_range(sections, min_len, max_len, revision_directives)
        draft = _assemble_planned_draft(topic, sections)
        completeness = _check_completeness(draft, plan)
        mapping = _build_feedback_mapping(draft, revision_directives)
        trace = list(state.get("execution_trace", []))
        trace.append(
            {
                "node": "writer",
                "revision_step": revision_step,
                "draft_len": len(draft),
                "mode": "fallback_exception",
                "planned_sections": len(plan),
                "complete": completeness.get("is_complete", False),
                "mapped_items": len(mapping),
            }
        )
        return {
            "draft": draft,
            "errors": errors,
            "feedback_paragraph_mapping": mapping,
            "execution_trace": trace,
            "length_control_meta": {
                "strategy": "fallback_after_exception",
                "plan": plan,
                "completeness": completeness,
            },
        }

