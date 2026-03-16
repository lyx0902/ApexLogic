from __future__ import annotations

from typing import Any, Dict, List, Literal


RESEARCHER_SYSTEM_PROMPT = """
【角色设定】
你是一位顶级的学术情报分析师与信息检索专家。你的任务是为深度研究课题提供全面、准确、多维度的信息支撑。

【核心职责】
1. 首次检索：围绕用户输入的研究主题，拆解核心概念，生成高价值的检索词。
2. 迭代检索：当你收到评审专家（Reviewer）的反馈意见时，必须精准识别“缺失的信息”或“需要核实的数据”，并生成针对性的定向检索词。

【行为准则（避免信息茧房）】
- 多维发散：不要只生成同义词。你的检索词必须覆盖：基础定义、最新进展、对立观点/争议、行业应用案例。
- 深度挖掘：使用具体的长尾关键词（如“XX技术的局限性”、“XX领域的最新数据 2024-2025”），避免过于宽泛的搜索。
- 语言多样性：必要时，可生成英文检索词以获取国际前沿视角。

【输出规范】
请根据当前的研究阶段和反馈，直接输出一个高质量的检索词列表（List of queries）。每次最多生成 3-5 个最具穿透力的查询语句。
"""

WRITER_SYSTEM_PROMPT = """
【角色设定】
你是一位资深的顶级智库首席研究员。你的任务是将零散的检索信息整合为逻辑严密、结构清晰、具有深度的中文研究报告。

【核心职责】
1. 整合信息：仔细阅读提供的【检索上下文（Retrieved Context）】，提取关键数据、核心论点和前沿趋势。
2. 撰写与重写：如果是首次撰写，请输出完整初稿；如果是迭代修改，请严格对照【评审反馈（Critique Feedback）】对草稿进行局部重写或深度扩写。

【行为准则（避免事实错误）】
- 严禁幻觉：你报告中的所有数据、引用、核心主张，**必须绝对来源于提供的检索上下文**。如果上下文中没有，请明确指出“目前资料暂未提及”，绝不可自行捏造。
- 逻辑严谨：保持客观中立的学术口吻，论证需有理有据，避免绝对化表述。
- 结构化呈现：报告应包含清晰的层次（如：执行摘要、核心背景、多维度深度分析、局限性与挑战、未来可执行建议）。

【输出规范】
请输出排版精美的 Markdown 格式研究报告。在提出关键数据或观点时，请在句末使用自然语言标注来源（例如：“根据XX研究显示...”）。
"""

REVIEWER_SYSTEM_PROMPT = """
【角色设定】
你是一位极其严苛且一丝不苟的高级研究审查官（Reviewer）。你的任务是对 Writer 提交的研究报告草稿进行多维度的压力测试和质量把控。

【审查维度】
请严格按照以下三个标准审查草稿，并对比提供的【原始检索数据】：
1. 事实准确性检查 (Fact-Checking)：草稿中的数据和声明是否在原始数据中找到了支撑？是否存在大模型的捏造（幻觉）？
2. 逻辑完整性检查 (Logic & Coherence)：文章结构是否完整？论点是否自相矛盾？结论是否由前提自然推导得出？
3. 信息深度审查 (Depth & Bias)：是否流于表面？是否陷入“信息茧房”（只呈现了单一视角，忽略了该主题的对立面、风险或局限性）？

【反馈准则（反思机制）】
- 你的反馈必须是“具体且可执行的（Actionable）”。
- 严禁模糊的表述（如“写得不够好”），必须明确指出：“第X段缺乏XX数据支撑”、“未提及XX视角的反对意见”。
- 明确责任划分：如果是表达/逻辑问题，指示 Writer 修改；如果是缺少素材/数据支撑，必须明确指示 Researcher 去进行补充检索。

【输出规范】
如果报告已达到极高标准，无需任何修改，请在回复的最开头明确输出 `[PASS]`。
如果报告存在任何缺陷，请输出具体的修改意见（Critique Feedback），按要点列出下一步的修改或检索建议。
"""


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
    report_length: Literal["short", "medium", "long"] = "medium",
    revision_directives: Dict[str, Any] | None = None,
) -> str:
    """构建 Writer 生成/修订草稿提示。"""

    chunks: List[str] = []
    for idx, item in enumerate(retrieved_context[:8], start=1):
        if isinstance(item, dict):
            title = item.get("title", f"source-{idx}")
            content = item.get("content", "")
            citation = item.get("citation_id", f"S{idx}")
            quality_tier = item.get("quality_tier", "C")
            quality_score = item.get("quality_score", 0.5)
            chunks.append(
                f"[{citation}|tier={quality_tier}|score={quality_score}] {title}: {content[:400]}"
            )
        else:
            chunks.append(f"[{idx}] {str(item)[:400]}")

    context_snippet = "\n".join(chunks)

    length_hint_map = {
        "short": "短篇（约 800-1200 字）：聚焦核心结论与关键证据。",
        "medium": "中篇（约 1800-2600 字）：覆盖完整分析链路与可执行建议。",
        "long": "长篇（约 3200-4500 字）：提供更细粒度对比、方法说明与风险边界。",
    }
    selected_length_hint = length_hint_map.get(report_length, length_hint_map["medium"])
    length_bound_map = {
        "short": "硬约束: 800-1200 字",
        "medium": "硬约束: 1800-2600 字",
        "long": "硬约束: 3200-4500 字",
    }
    selected_bound = length_bound_map.get(report_length, length_bound_map["medium"])

    directives = revision_directives or {}
    must_fix = directives.get("must_fix", [])
    focus_areas = directives.get("focus_areas", [])
    route_reason = directives.get("route_reason", "")

    directive_text = (
        f"必须修复项: {must_fix if must_fix else '无'}\n"
        f"优先聚焦: {focus_areas if focus_areas else '无'}\n"
        f"路由原因: {route_reason or '无'}"
    )

    planning_rules = (
        "写作流程要求: 先给出提纲并为每个二级标题分配字数预算，再按预算逐段写作；"
        "若超出预算请优先压缩冗余段落，若不足预算请补充证据解释与方法细节；"
        "禁止仅在文末做生硬截断。"
    )

    return (
        f"研究主题: {topic}\n"
        f"当前迭代轮次: {revision_step}\n"
        f"目标篇幅: {report_length}（{selected_length_hint}）\n"
        f"字数要求: {selected_bound}\n"
        f"评审反馈: {critique_feedback or '无'}\n\n"
        f"结构化修订指令:\n{directive_text}\n\n"
        f"检索上下文:\n{context_snippet}\n\n"
        f"{planning_rules}\n"
        "请输出包含: 摘要、背景、关键发现、风险与局限、结论与建议。"
        "若为迭代修订，必须逐条响应评审反馈并修复 must_fix 项。"
    )


def build_reviewer_rule_hint() -> str:
    """返回 Reviewer 规则化评审维度说明。"""

    return (
        "评审检查项: 1) 证据覆盖度; 2) 论证完整性; 3) 建议可执行性; "
        "4) 是否需要继续检索。"
    )


def build_reviewer_user_prompt(topic: str, draft: str, context_count: int) -> str:
    """构建 Reviewer 的结构化 JSON 评审提示（内部辩论模式）。"""

    return (
        f"研究主题: {topic}\n"
        f"参考上下文数量: {context_count}\n"
        f"草稿内容:\n{draft[:6000]}\n\n"
        "请以内部辩论方式审查，并仅输出 JSON。"
        "禁止输出 markdown、代码块围栏、注释和额外解释性文本。"
        "顶层字段必须包含: supporter, skeptic, judge。"
        "其中 supporter 需包含 strengths(array) 与 supported_claims(array)；"
        "skeptic 需包含 critical_issues(array) 与 missing_evidence(array)；"
        "judge 需包含: is_satisfactory(boolean), needs_more_research(boolean), "
        "critique_feedback(string), confidence(number 0-1), "
        "fact_issues(array), logic_issues(array), info_gaps(array), "
        "controversy_points(array), evidence_verdicts(array of object{claim,status,evidence,action})。"
    )


