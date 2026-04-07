"""agents/extractor.py

Answer Extractor 节点：从 Writer 生成的长篇草稿中提取出简短的、
可直接用于 Benchmark Exact Match 打分的 final_answer，
同时保留 CoT 思维链供人工分析。

该节点不加入 LangGraph 图，由 eval_runner.py 在 app.invoke() 之后手动调用，
做到零侵入现有流程。
"""

from __future__ import annotations

import os
import re
import string
from typing import Any, Dict

# ─── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a precise answer extraction expert for multi-hop QA benchmarks.
You will be given a QUESTION and a RESEARCH REPORT. Your job is to extract \
the shortest possible answer that directly answers the question.

STRICT RULES:
1. LANGUAGE: The answer MUST be in the SAME language and script as the question. \
If the question is in English, your <answer> MUST be in English. \
NEVER translate proper nouns, names, titles, or technical terms. \
Keep original English spelling exactly as it appears in the source.
2. DATE FORMAT: If the answer is a date, output it as "Month Day, Year" \
(e.g., "June 20, 1837" or "April 30, 1789"). Do NOT convert to other formats.
3. LENGTH: The answer must be extremely concise — typically 1 to 5 words or a short phrase. \
No explanations, no sentences, no punctuation at the end unless part of the answer itself. \
Your <think> section must be under 150 characters — one concise sentence identifying the key fact.
4. OUTPUT FORMAT: You MUST follow this exact format — nothing else before or after:

<think>
Step-by-step reasoning: identify the key facts from the report that answer the question.
</think>
<answer>
[the final short answer here]
</answer>

IMPORTANT: The <answer> tag must appear EXACTLY ONCE in your response and contain ONLY \
the final answer — no prefixes like "Answer:", no quotes, no extra whitespace."""

# 强制猜测模式追加指令（最后一轮时注入）
_FORCE_GUESS_ADDENDUM = """\

CRITICAL OVERRIDE — LAST RESORT MODE:
The research has exhausted all retrieval rounds. You MUST provide your best guess.
- NEVER output "Not found", "Unknown", "Cannot determine", or any similar non-answer.
- If the report contains partial clues, infer the most likely answer from them.
- If truly no clue exists, output the most plausible answer based on general knowledge.
- A specific guess (even if uncertain) is always better than a non-answer for scoring purposes."""

_USER_TMPL = """\
QUESTION: {question}

RESEARCH REPORT (excerpt):
{draft_snippet}"""

# 草稿截取上限（字符），避免超出 LLM context 限制
_MAX_DRAFT_CHARS = 6000

# 正则：严格匹配 <answer>…</answer>，捕获组内容即为答案
_RE_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_RE_ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)

# 需要从提取结果中剥离的前缀模式（模型有时仍会输出 "Answer: ..."）
_RE_ANSWER_PREFIX = re.compile(
    r"^(answer\s*[:：]\s*|最终答案\s*[:：]\s*|答案\s*[:：]\s*)", re.IGNORECASE
)

# 末尾多余标点（保留问号、感叹号，去掉句号、逗号等）
_TRAILING_PUNCT = re.compile(r"[。，,\.]+$")


# ─── 核心提取逻辑 ───────────────────────────────────────────────────────────────

def _build_client():
    """构建 OpenAI 兼容的 DeepSeek 客户端，失败返回 None。"""
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    if not api_key:
        return None
    try:
        from openai import OpenAI  # type: ignore
        return OpenAI(api_key=api_key, base_url=base_url)
    except Exception:
        return None


def _clean_answer(raw: str) -> str:
    """对提取出的原始答案做标准化清洗：
    - 去首尾空白
    - 去除 "Answer: " / "答案：" 等多余前缀
    - 去除末尾多余标点（句号、逗号）
    """
    text = raw.strip()
    text = _RE_ANSWER_PREFIX.sub("", text).strip()
    text = _TRAILING_PUNCT.sub("", text).strip()
    return text


def _fallback_answer(draft: str) -> tuple[str, str]:
    """LLM 不可用时的降级提取：取草稿首个非标题、非提示词行。

    避免取到 "问题：..." / "用户的问题是：..." 之类的内容。
    """
    _SKIP_PREFIXES = (
        "#", "问题", "question", "用户", "topic", "research report",
        "answer", "答案", "<", "[",
    )
    lines = [ln.strip() for ln in draft.splitlines() if ln.strip()]
    for line in lines:
        low = line.lower()
        if any(low.startswith(p.lower()) for p in _SKIP_PREFIXES):
            continue
        if len(line) > 3:
            return line[:100], "[降级：从草稿首段提取，LLM 不可用]"
    return draft[:100], "[降级：从草稿头部截取]"


def extract_answer(
    question: str,
    draft: str,
    client: Any = None,
    model: str | None = None,
    force_guess: bool = False,
) -> Dict[str, str]:
    """从草稿中提取简短答案和 CoT 思维链。

    Args:
        question: 原始研究问题
        draft:    Writer 生成的完整草稿
        client:   可复用的 OpenAI 兼容客户端（为 None 时内部新建）
        model:    LLM 模型名（为 None 时读取环境变量）

    Returns:
        {"final_answer": str, "cot_reasoning": str}
    """
    _client = client or _build_client()
    _model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    draft_snippet = draft[:_MAX_DRAFT_CHARS]

    if _client is None:
        ans, cot = _fallback_answer(draft)
        return {"final_answer": ans, "cot_reasoning": cot}

    system_prompt = _SYSTEM_PROMPT + (_FORCE_GUESS_ADDENDUM if force_guess else "")
    user_msg = _USER_TMPL.format(question=question, draft_snippet=draft_snippet)
    try:
        resp = _client.chat.completions.create(
            model=_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        raw = resp.choices[0].message.content or ""

        # ── Step 1：严格用正则提取 <think> 和 <answer> ────────────────────────
        think_match = _RE_THINK.search(raw)
        answer_match = _RE_ANSWER.search(raw)

        cot = think_match.group(1).strip() if think_match else ""
        cot = cot[:150]
        final_answer = _clean_answer(answer_match.group(1)) if answer_match else ""

        # ── Step 2：<answer> 未匹配时的兜底策略 ──────────────────────────────
        if not final_answer:
            # 策略A：尝试从 </think> 之后取第一个非空、非标签行
            after_think = raw
            think_end = raw.lower().rfind("</think>")
            if think_end != -1:
                after_think = raw[think_end + len("</think>"):]

            for line in after_think.splitlines():
                line = line.strip()
                if not line:
                    continue
                # 跳过标签行、提示词前缀行
                if line.startswith("<") or line.lower().startswith(
                    ("question:", "research report", "问题", "用户")
                ):
                    continue
                final_answer = _clean_answer(line[:120])
                break

        # ── Step 3：最终兜底（仍为空）→ 从草稿降级提取 ───────────────────────
        if not final_answer:
            fallback_ans, fallback_cot = _fallback_answer(draft)
            return {
                "final_answer": fallback_ans,
                "cot_reasoning": cot or fallback_cot,
            }

        return {"final_answer": final_answer, "cot_reasoning": cot}

    except Exception as exc:  # noqa: BLE001
        ans, _ = _fallback_answer(draft)
        return {
            "final_answer": ans,
            "cot_reasoning": f"[提取失败: {exc}]",
        }


# ─── LangGraph 节点接口（备用，eval_runner.py 直接调用 extract_answer）──────────

def extractor_node(state: Dict[str, Any]) -> Dict[str, str]:
    """LangGraph 兼容节点包装（不注册进图，仅供按需调用）。

    输入字段：state["topic"]、state["draft"]
    输出字段：{"final_answer": str, "cot_reasoning": str}
    """
    question = state.get("topic", "")
    draft = state.get("draft", "") or state.get("final_report", "")
    return extract_answer(question=question, draft=draft)
