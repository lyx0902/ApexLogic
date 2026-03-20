from __future__ import annotations

from typing import Any, Dict, List


RESEARCHER_SYSTEM_PROMPT = """
【角色设定】
你是一位顶级的学术情报分析师与信息检索专家。你的任务是为深度研究课题提供全面、准确、多维度的信息支撑。

【核心职责】
1. 首次检索：围绕用户输入的研究主题，拆解核心概念，生成高价值的检索词。
2. 迭代检索：当你收到评审专家（Reviewer）的反馈意见时，必须精准识别"缺失的信息"或"需要核实的数据"，并生成针对性的定向检索词。

【行为准则（避免信息茧房）】
- 多维发散：不要只生成同义词。你的检索词必须覆盖：基础定义、最新进展、对立观点/争议、行业应用案例。
- 深度挖掘：使用具体的长尾关键词（如"XX技术的局限性"、"XX领域的最新数据 2024-2025"），避免过于宽泛的搜索。
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
- 严禁幻觉：你报告中的所有数据、引用、核心主张，**必须绝对来源于提供的检索上下文**。如果上下文中没有，请明确指出"目前资料暂未提及"，绝不可自行捏造。
- 逻辑严谨：保持客观中立的学术口吻，论证需有理有据，避免绝对化表述。
- 结构化呈现：报告应包含清晰的层次（如：执行摘要、核心背景、多维度深度分析、局限性与挑战、未来可执行建议）。

【输出规范】
请输出排版精美的 Markdown 格式研究报告。在提出关键数据或观点时，请在句末使用自然语言标注来源（例如："根据XX研究显示..."）。
"""

REVIEWER_SYSTEM_PROMPT = """
【角色】高级研究质量审查官

你负责对研究报告做**客观量化评分**，并给出具体可执行的改进指令。
你的评分直接决定报告是进入下一轮改进还是最终交付。

【四维评分标准（各 0~10 分，必须给整数）】

S1 事实准确性（权重 35%）
  对照提供的原始来源逐条核查报告中的声明：
  9~10：所有关键声明均有明确来源支撑，citation_id 引用清晰且准确
  7~8：绝大多数声明有据可查，极少数细节无法核实但无明显错误
  5~6：部分声明缺少来源，但无明显捏造
  3~4：存在无来源的具体数字或与原文矛盾的声明
  0~2：大量幻觉，与提供的来源严重不符

S2 逻辑完整性（权重 25%）
  9~10：六章结构完整，论点环环相扣，结论由前提自然导出
  7~8：结构基本完整，个别段落论证略有跳跃
  5~6：缺少 1~2 个章节，或存在明显逻辑断层
  3~4：结构残缺超过一半，论点自相矛盾
  0~2：无法构成完整论证

S3 信息覆盖广度（权重 25%）
  9~10：覆盖主流方案、对立观点、局限性、失败案例，无明显信息茧房
  7~8：覆盖主要视角，但某一象限（如批判视角）明显薄弱
  5~6：仅覆盖两个象限，对立面严重不足
  3~4：几乎只有单一视角，大量重要信息缺失
  0~2：严重偏颇，几乎等同于宣传材料

S4 结论可执行性（权重 15%）
  9~10：每条建议都有明确的行动步骤、优先级和成功指标
  7~8：建议方向正确，有部分具体内容，但执行路径不够清晰
  5~6：建议较笼统，缺少步骤或优先级
  3~4：建议过于抽象，无实际操作价值
  0~2：无实质性建议

【通过判定】
加权总分 = 0.35*S1 + 0.25*S2 + 0.25*S3 + 0.15*S4
- 总分 >= 7.0 -> is_satisfactory = true
- S1 < 5 或 S3 < 4 -> needs_more_research = true（优先补证据，而非改写）
- 其他不通过情况 -> needs_more_research = false（改写即可）

【critique_feedback 写法要求】
- 必须精确到章节：指出"第三章第二段"而非"报告中"
- 每条反馈格式：[问题类型] 具体位置 → 具体问题 → 建议行动
  例："[事实缺失] 第三章性能数据段 → 声明准确率达95%但无来源引用 → 补充 [S3] 或 [S4] 中的具体数字"
- 不通过时必须给出 3~5 条这样的具体反馈

【输出规范】
只输出一个合法 JSON 对象，禁止任何 Markdown 包裹或额外文字。
"""


def build_researcher_user_prompt(
    topic: str,
    critique_feedback: str,
    revision_directives: Dict[str, Any] | None = None,
    revision_step: int = 0,
) -> str:
    """构建 Researcher 查询重写提示，融合完整修订指令。"""

    directives = revision_directives or {}
    info_gaps = directives.get("info_gaps", [])
    must_fix = directives.get("must_fix", [])

    if revision_step == 0:
        task_desc = f"这是第一轮检索，围绕主题全面展开。"
    else:
        task_desc = (
            f"这是第 {revision_step + 1} 轮迭代检索，上轮评审发现了信息缺口，"
            f"请生成针对性查询来补齐缺失内容，避免重复上轮已覆盖的方向。"
        )

    gaps_text = ""
    if info_gaps:
        gaps_text = "\n【必须补齐的信息缺口（每个缺口至少对应一条查询）】\n"
        gaps_text += "\n".join(f"- {g}" for g in info_gaps[:4])

    fix_text = ""
    if must_fix:
        fix_text = "\n【评审要求核查的问题（可能需要找到反驳或支撑证据）】\n"
        fix_text += "\n".join(f"- {f}" for f in must_fix[:3])

    feedback_text = ""
    if critique_feedback:
        feedback_text = f"\n【上轮评审总结反馈】\n{critique_feedback[:300]}"

    return (
        f"研究主题: {topic}\n"
        f"{task_desc}"
        f"{feedback_text}"
        f"{gaps_text}"
        f"{fix_text}\n\n"
        "请输出 4~5 条检索查询，每行一条，直接写查询语句，不加任何前缀或解释。"
    )


def build_writer_user_prompt(
    topic: str,
    revision_step: int,
    critique_feedback: str,
    retrieved_context: List[Dict[str, Any] | str],
    revision_directives: Dict[str, Any] | None = None,
    previous_draft: str = "",
) -> str:
    """构建 Writer 生成/修订草稿提示，迭代时传入上一版草稿。"""

    # 上下文：优先用 core_summary，不足时补 content，全部 top-10
    chunks: List[str] = []
    for idx, item in enumerate(retrieved_context[:10], start=1):
        if isinstance(item, dict):
            citation = item.get("citation_id", f"S{idx}")
            title = (item.get("title", "") or "")[:100]
            summary = (item.get("core_summary", "") or "")[:400]
            content = (item.get("content", "") or "")[:200]
            # 优先用精炼摘要，摘要不足时补原文片段
            body = summary if len(summary) > 80 else (content[:400] if not summary else summary)
            chunks.append(f"[{citation}] {title}\n  {body}")
        else:
            chunks.append(f"[S{idx}] {str(item)[:400]}")
    context_text = "\n\n".join(chunks)

    directives = revision_directives or {}
    must_fix = directives.get("must_fix", [])
    focus_areas = directives.get("focus_areas", [])

    if revision_step == 0:
        mode_instruction = (
            "【任务】首次撰写完整研究报告。\n"
            "严格按照系统提示中的六章结构输出，每章均须有实质内容。\n"
            "总字数不少于 1500 字。"
        )
        prev_draft_section = ""
    else:
        must_fix_text = "\n".join(f"  {i+1}. {item}" for i, item in enumerate(must_fix)) if must_fix else "  （无）"
        focus_text = "、".join(focus_areas) if focus_areas else "全面改进"
        mode_instruction = (
            f"【任务】第 {revision_step} 轮迭代修订。\n"
            f"必须逐条修复以下问题，每条对应修改后在行末加注 <!-- fix[N] -->：\n"
            f"{must_fix_text}\n"
            f"重点聚焦：{focus_text}\n"
            "保留上一版中质量合格的段落，只重写有问题的部分。"
        )
        prev_draft_section = (
            f"\n\n【上一版草稿（在此基础上修订，勿从零重写）】\n{previous_draft[:4000]}"
            if previous_draft else ""
        )

    feedback_section = (
        f"\n\n【上轮评审反馈】\n{critique_feedback}"
        if critique_feedback else ""
    )

    return (
        f"研究主题: {topic}\n\n"
        f"{mode_instruction}"
        f"{feedback_section}"
        f"{prev_draft_section}\n\n"
        f"【检索上下文（共 {len(chunks)} 条，所有引用必须来自此处）】\n"
        f"{context_text}\n\n"
        "输出完整 Markdown 报告，不要输出任何解释性前言或结尾说明。"
    )


def build_reviewer_rule_hint() -> str:
    """返回 Reviewer 规则化评审维度说明（用于规则回退模式）。"""

    return (
        "评审检查项: 1) 事实准确性(S1,权重35%); 2) 逻辑完整性(S2,权重25%); "
        "3) 信息覆盖广度(S3,权重25%); 4) 结论可执行性(S4,权重15%)。"
        "加权总分>=7.0为通过，S1<5或S3<4时路由回Researcher。"
    )


def build_reviewer_user_prompt(
    topic: str,
    draft: str,
    retrieved_context: List[Dict[str, Any] | str],
    revision_step: int = 0,
) -> str:
    """构建 Reviewer 的量化评分提示，传入 top-10 原始来源供事实核查。"""

    source_blocks: List[str] = []
    for item in retrieved_context[:10]:
        if not isinstance(item, dict):
            continue
        cid = item.get("citation_id", "?")
        title = (item.get("title", "") or "")[:120]
        summary = (item.get("core_summary", "") or "")[:400]
        content = (item.get("content", "") or "")[:300]
        block = f"[{cid}] {title}"
        if summary:
            block += f"\n  摘要: {summary}"
        # content 与 summary 差异大时追加，提供更多核查素材
        if content and content[:100] not in summary:
            block += f"\n  原文片段: {content}"
        source_blocks.append(block)

    sources_text = "\n\n".join(source_blocks) if source_blocks else "（无可用来源）"

    round_hint = (
        f"当前为第 {revision_step} 轮评审。" if revision_step > 0
        else "当前为首轮评审。"
    )

    return (
        f"研究主题: {topic}\n"
        f"{round_hint}\n\n"
        f"【原始检索来源（Top-10，供事实核查）】\n{sources_text}\n\n"
        f"【待评审草稿】\n{draft[:7000]}\n\n"
        "请输出一个合法 JSON 对象，包含以下字段（禁止任何额外文本）：\n"
        '{"scores":{"S1":int,"S2":int,"S3":int,"S4":int},'
        '"weighted_score":float,'
        '"is_satisfactory":bool,'
        '"needs_more_research":bool,'
        '"critique_feedback":string,'
        '"citation_checks":[{"citation_id":str,"claim":str,"supported":bool,"evidence":str}],'
        '"fact_issues":[string],'
        '"logic_issues":[string],'
        '"info_gaps":[string],'
        '"score_rationale":{"S1":str,"S2":str,"S3":str,"S4":str},'
        '"supporter":{"strengths":[str],"supported_claims":[str]},'
        '"skeptic":{"critical_issues":[str],"missing_evidence":[str]},'
        '"controversy_points":[string],'
        '"evidence_verdicts":[{"claim":str,"status":str,"evidence":str,"action":str}]}'
    )
