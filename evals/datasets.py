"""evals/datasets.py

统一数据集加载接口，支持：
  - hotpot_qa  (distractor 子集，难度低)
  - chiayewken/bamboogle  (难度高)

返回格式：List[{"question": str, "answer": str}]
"""

from __future__ import annotations

from typing import List, Dict


# split 优先级：优先 test，其次 validation，最后 train
_SPLIT_CANDIDATES = ["test", "validation", "train"]


def _load_with_fallback(hf_path: str, hf_name: str | None = None) -> object:
    """尝试按 split 优先级加载，返回第一个成功的 Dataset 对象。"""
    try:
        from datasets import load_dataset as hf_load  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "缺少 `datasets` 依赖，请执行：pip install datasets"
        ) from exc

    last_exc: Exception | None = None
    for split in _SPLIT_CANDIDATES:
        try:
            if hf_name:
                ds = hf_load(hf_path, hf_name, split=split, trust_remote_code=True)
            else:
                ds = hf_load(hf_path, split=split, trust_remote_code=True)
            return ds
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue

    raise RuntimeError(
        f"无法从 {hf_path!r} 加载任何 split（尝试了 {_SPLIT_CANDIDATES}）。"
        f"最后一次错误：{last_exc}"
    )


def _load_hotpotqa(limit: int | None) -> List[Dict[str, str]]:
    """加载 HotpotQA distractor 验证集。"""
    ds = _load_with_fallback("hotpot_qa", "distractor")
    items: List[Dict[str, str]] = []
    for row in ds:
        items.append({"question": row["question"], "answer": row["answer"]})
        if limit and len(items) >= limit:
            break
    return items


def _load_bamboogle(limit: int | None) -> List[Dict[str, str]]:
    """加载 Bamboogle 数据集（多跳推理，难度高）。"""
    ds = _load_with_fallback("chiayewken/bamboogle")
    items: List[Dict[str, str]] = []
    for row in ds:
        # 字段名为 Question / Answer（首字母大写）
        q = row.get("Question") or row.get("question", "")
        a = row.get("Answer") or row.get("answer", "")
        if q and a:
            items.append({"question": q, "answer": a})
        if limit and len(items) >= limit:
            break
    return items


_DATASET_LOADERS = {
    "hotpotqa": _load_hotpotqa,
    "bamboogle": _load_bamboogle,
}


def load_eval_dataset(name: str, limit: int | None = None) -> List[Dict[str, str]]:
    """统一数据集加载入口。

    Args:
        name:  数据集名称，支持 "hotpotqa" / "bamboogle"
        limit: 截断条数；None 表示全量加载

    Returns:
        List[{"question": str, "answer": str}]
    """
    name = name.lower().strip()
    if name not in _DATASET_LOADERS:
        raise ValueError(
            f"不支持的数据集 {name!r}，可选值：{list(_DATASET_LOADERS)}"
        )
    return _DATASET_LOADERS[name](limit)


if __name__ == "__main__":
    # 快速验证
    import sys

    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "bamboogle"
    rows = load_eval_dataset(dataset_name, limit=3)
    for i, r in enumerate(rows):
        print(f"[{i}] Q: {r['question']}")
        print(f"    A: {r['answer']}")
