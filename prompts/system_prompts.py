from __future__ import annotations

from typing import Any, Dict, List


RESEARCHER_SYSTEM_PROMPT = (
    "你是研究检索专家。请围绕研究主题生成高价值检索词，"
    "并根据评审反馈补充缺失信息，避免信息来源单一。"
)

WRITER_SYSTEM_PROMPT = (
    "你是资深研究报告作者。请基于上下文证据撰写结构化中文报告，"
    "保持逻辑严谨、观点可追溯，并在结论中给出可执行建议。"
)

REVIEWER_SYSTEM_PROMPT = (
    "你是严格的研究评审专家。请从事实准确性、逻辑完整性和信息深度"
    "三个维度审查草稿，并给出明确可执行的修改意见。"
)


def build_researcher_user_prompt(topic: str, critique_feedback: str) -> str:
    """构建 Researcher 查询重写提示。"""

    return (
        f"研究主题: {topic}\n"
        f"评审反馈: {critique_feedback or '无'}\n\n"
        "请输出 4 条高质量检索词，每行一条，不要附加解释。"
    )


def build_writer_user_prompt(
    topic: str,
    revision_step: int,
    critique_feedback: str,
    retrieved_context: List[Dict[str, Any] | str],
) -> str:
    """构建 Writer 生成/修订草稿提示。"""

    chunks: List[str] = []
    for idx, item in enumerate(retrieved_context[:8], start=1):
        if isinstance(item, dict):
            title = item.get("title", f"source-{idx}")
            content = item.get("content", "")
            citation = item.get("citation_id", f"S{idx}")
            chunks.append(f"[{citation}] {title}: {content[:400]}")
        else:
            chunks.append(f"[{idx}] {str(item)[:400]}")

    context_snippet = "\n".join(chunks)

    return (
        f"研究主题: {topic}\n"
        f"当前迭代轮次: {revision_step}\n"
        f"评审反馈: {critique_feedback or '无'}\n\n"
        f"检索上下文:\n{context_snippet}\n\n"
        "请输出包含: 摘要、背景、关键发现、风险与局限、结论与建议。"
    )


def build_reviewer_rule_hint() -> str:
    """返回 Reviewer 规则化评审维度说明。"""

    return (
        "评审检查项: 1) 证据覆盖度; 2) 论证完整性; 3) 建议可执行性; "
        "4) 是否需要继续检索。"
    )


def build_reviewer_user_prompt(topic: str, draft: str, context_count: int) -> str:
    """构建 Reviewer 的结构化 JSON 评审提示。"""

    return (
        f"研究主题: {topic}\n"
        f"参考上下文数量: {context_count}\n"
        f"草稿内容:\n{draft[:6000]}\n\n"
        "请仅输出 JSON，字段必须包含: "
        "is_satisfactory(boolean), needs_more_research(boolean), "
        "critique_feedback(string), confidence(number 0-1), "
        "fact_issues(array of string), logic_issues(array of string), "
        "info_gaps(array of string)。"
    )


