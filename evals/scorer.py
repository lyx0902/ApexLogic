"""evals/scorer.py

两种评分模式：
  - exact_match: 标准化后判断 gold 是否在 pred 中（参照 HotpotQA 官方评估逻辑）
  - llm_judge:   调用 DeepSeek LLM 判断语义等价，返回 {correct, confidence, reason}
"""

from __future__ import annotations

import re
import string
from typing import Any, Dict


# ─── Exact Match ───────────────────────────────────────────────────────────────

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize(text: str) -> str:
    """小写 → 去冠词 → 去标点 → 合并空白。"""
    text = text.lower()
    text = _ARTICLES.sub(" ", text)
    text = text.translate(_PUNCT_TABLE)
    text = " ".join(text.split())
    return text


def exact_match(pred: str, gold: str) -> bool:
    """判断预测答案是否与标准答案匹配（标准化后的包含关系）。

    采用"gold in pred"策略而非严格相等，以容纳短答案被包含在略长句子中的情况。
    """
    if not pred or not gold:
        return False
    norm_pred = _normalize(pred)
    norm_gold = _normalize(gold)
    return norm_gold in norm_pred or norm_pred == norm_gold


# ─── LLM Judge ─────────────────────────────────────────────────────────────────

_LLM_JUDGE_SYSTEM = """\
你是一位严格的问答评估专家。
用户会给你一道问题、一个模型的预测答案和标准答案。
请判断预测答案是否在语义上等价于标准答案（允许不同措辞）。
必须以 JSON 格式输出，包含以下字段：
{
  "correct": true 或 false,
  "confidence": 0.0~1.0 之间的浮点数,
  "reason": "一句话说明判断依据"
}
不要输出任何其他内容。"""

_LLM_JUDGE_USER_TMPL = """\
问题：{question}
预测答案：{pred}
标准答案：{gold}"""


def llm_judge(
    question: str,
    pred: str,
    gold: str,
    client: Any,
    model: str = "deepseek-chat",
) -> Dict[str, Any]:
    """用 LLM 判断预测答案是否与标准答案语义等价。

    Args:
        question: 原始问题
        pred:     模型预测答案
        gold:     标准答案
        client:   openai.OpenAI 兼容客户端（如 DeepSeek）
        model:    使用的模型名

    Returns:
        {"correct": bool, "confidence": float, "reason": str}
        发生任何错误时返回 {"correct": False, "confidence": 0.0, "reason": "<error>"}
    """
    import json

    user_msg = _LLM_JUDGE_USER_TMPL.format(question=question, pred=pred, gold=gold)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _LLM_JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        raw = resp.choices[0].message.content.strip()
        # 尝试直接解析 JSON
        result = json.loads(raw)
        return {
            "correct": bool(result.get("correct", False)),
            "confidence": float(result.get("confidence", 0.0)),
            "reason": str(result.get("reason", "")),
        }
    except Exception as exc:  # noqa: BLE001
        return {"correct": False, "confidence": 0.0, "reason": f"LLM judge 失败: {exc}"}
