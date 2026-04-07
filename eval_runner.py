"""eval_runner.py

Benchmark 自动化评测主入口，与 main.py / export_report.py 完全独立。

用法示例：
    python eval_runner.py --dataset bamboogle --limit 50 --scorer em
    python eval_runner.py --dataset hotpotqa  --limit 100 --scorer llm --concurrency 2

输出：
    tests/results_<timestamp>.json  含每题结果 + 汇总指标
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

# ─── tenacity 重试配置 ──────────────────────────────────────────────────────────
try:
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )
    _TENACITY_AVAILABLE = True
except ImportError:
    _TENACITY_AVAILABLE = False
    # 无 tenacity 时提供空装饰器，保证代码可运行
    def retry(*args, **kwargs):  # type: ignore[misc]
        def decorator(fn):
            return fn
        return decorator

    def wait_exponential(**kwargs):  # type: ignore[misc]
        return None

    def stop_after_attempt(n):  # type: ignore[misc]
        return None

    def retry_if_exception_type(*args):  # type: ignore[misc]
        return None


# 重试装饰器：针对 HTTP 429 / 5xx 类错误（连接错误、超时、APIError 等）
_RETRY_KWARGS = dict(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(3),
    reraise=True,
)


# ─── 项目内部导入 ───────────────────────────────────────────────────────────────

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.graph import compile_graph  # noqa: E402
from core.state import create_initial_state  # noqa: E402
from evals.datasets import load_eval_dataset  # noqa: E402
from evals.scorer import exact_match, llm_judge  # noqa: E402
from agents.extractor import extract_answer  # noqa: E402


# ─── 辅助：构建 DeepSeek 客户端 ────────────────────────────────────────────────

def _build_llm_client():
    """构建 OpenAI 兼容客户端，失败返回 None。"""
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    if not api_key:
        return None
    try:
        from openai import OpenAI  # type: ignore
        return OpenAI(api_key=api_key, base_url=base_url)
    except Exception:
        return None


# ─── 带重试的 graph invoke ─────────────────────────────────────────────────────

@retry(**_RETRY_KWARGS)
def _invoke_graph(app: Any, initial_state: Dict[str, Any]) -> Dict[str, Any]:
    """执行 LangGraph 图，带指数退避重试（最多 3 次）。"""
    return app.invoke(initial_state)


# ─── 带重试的答案提取 ───────────────────────────────────────────────────────────

@retry(**_RETRY_KWARGS)
def _extract_answer_with_retry(
    question: str, draft: str, client: Any, model: str, force_guess: bool = False
) -> Dict[str, str]:
    """带重试的答案提取调用。"""
    return extract_answer(question=question, draft=draft, client=client, model=model, force_guess=force_guess)


# ─── 带重试的 LLM Judge ─────────────────────────────────────────────────────────

@retry(**_RETRY_KWARGS)
def _llm_judge_with_retry(
    question: str, pred: str, gold: str, client: Any, model: str
) -> Dict[str, Any]:
    """带重试的 LLM 语义判定调用。"""
    return llm_judge(question=question, pred=pred, gold=gold, client=client, model=model)


# ─── search_stats 裁剪：只保留 retriever top20 / reranker top10 元数据 ──────────

def _trim_search_stats(source_quality_summary: Dict[str, Any]) -> Dict[str, Any]:
    """从完整的 source_quality_summary 中裁剪出轻量版 search_stats。

    只保留：
      - retriever: 统计数字 + selected_records（top20 元数据：rank/title/source/url/score）
      - reranker:  统计数字 + selected_records（top10 元数据：rank/title/source/url/score）
    丢弃 dropped_records / dropped_samples / selected_samples 等冗余字段，节省 token。
    """
    bge: Dict[str, Any] = source_quality_summary.get("bge_summary", {})

    def _slim_stage(stage: Dict[str, Any]) -> Dict[str, Any]:
        """保留计数字段 + selected_records，去掉其余列表。"""
        slim: Dict[str, Any] = {}
        # 保留所有非列表的统计字段（enabled, mode, input, selected, top_k 等）
        for k, v in stage.items():
            if not isinstance(v, list):
                slim[k] = v
        # 只保留 selected_records（已是纯元数据，无 content 字段）
        if "selected_records" in stage:
            slim["selected_records"] = stage["selected_records"]
        return slim

    retriever_raw: Dict[str, Any] = bge.get("retriever", {})
    reranker_raw: Dict[str, Any] = bge.get("reranker", {})

    return {
        "retriever": _slim_stage(retriever_raw),
        "reranker": _slim_stage(reranker_raw),
        # 保留顶层轻量统计，便于论文表格使用
        "dedup_total": bge.get("dedup_total"),
        "final_contexts": bge.get("final_contexts"),
        "dropped": bge.get("dropped"),
    }


# ─── 单题评测逻辑 ───────────────────────────────────────────────────────────────

def _evaluate_single(
    idx: int,
    item: Dict[str, str],
    app: Any,
    llm_client: Any,
    model: str,
    scorer: str,
    max_revisions: int,
) -> Dict[str, Any]:
    """对单道题目执行完整评测流程，返回结果字典。"""
    question = item["question"]
    ground_truth = item["answer"]

    t_start = time.perf_counter()
    errors: List[str] = []

    # ── 1. 运行 ResearchGraph ──────────────────────────────────────────────────
    final_state: Dict[str, Any] = {}
    try:
        initial_state = create_initial_state(topic=question, output_mode="eval")
        final_state = _invoke_graph(app, initial_state)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"graph invoke 失败: {exc}")

    draft: str = (
        final_state.get("final_report")
        or final_state.get("draft")
        or ""
    )

    # ── 2. 提取 final_answer + CoT ────────────────────────────────────────────
    extraction: Dict[str, str] = {"final_answer": "", "cot_reasoning": ""}
    # 最后一轮（已达上限）时强制猜测，避免输出 "Not found in context"
    is_last_round = final_state.get("revision_step", 0) >= max_revisions
    try:
        extraction = _extract_answer_with_retry(
            question=question,
            draft=draft,
            client=llm_client,
            model=model,
            force_guess=is_last_round,
        )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"answer extraction 失败: {exc}")
        # 降级：截取草稿头部
        extraction = {"final_answer": draft[:80], "cot_reasoning": f"[提取失败: {exc}]"}

    final_answer = extraction.get("final_answer", "")
    cot_reasoning = extraction.get("cot_reasoning", "")

    # ── 3. 评分 ───────────────────────────────────────────────────────────────
    is_correct = False
    judge_detail: Dict[str, Any] = {}

    if scorer == "llm" and llm_client is not None:
        try:
            judge_detail = _llm_judge_with_retry(
                question=question,
                pred=final_answer,
                gold=ground_truth,
                client=llm_client,
                model=model,
            )
            is_correct = judge_detail.get("correct", False)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"llm judge 失败，回退 EM: {exc}")
            is_correct = exact_match(final_answer, ground_truth)
    else:
        is_correct = exact_match(final_answer, ground_truth)

    elapsed = time.perf_counter() - t_start

    # ── 4. 提取实验数据字段（毕设重点）───────────────────────────────────────
    mab_state: Dict[str, Any] = final_state.get("mab_state") or {}
    search_stats: Dict[str, Any] = _trim_search_stats(
        final_state.get("source_quality_summary") or {}
    )

    # ── 5. 组装结果字典 ───────────────────────────────────────────────────────
    result: Dict[str, Any] = {
        "id": idx,
        "question": question,
        "ground_truth": ground_truth,
        "final_answer": final_answer,
        "cot_reasoning": cot_reasoning,
        "is_correct": is_correct,
        "elapsed_seconds": round(elapsed, 2),
        "revision_step": final_state.get("revision_step", 0),
        "scorer": scorer,
        "errors": errors + (final_state.get("errors") or []),
        # ── 毕设专用实验数据字段 ──────────────────────────────────────────────
        "mab_state": mab_state,
        "search_stats": search_stats,
    }

    if judge_detail:
        result["judge_detail"] = judge_detail

    status = "✓" if is_correct else "✗"
    print(
        f"  [{idx:>4}] {status}  elapsed={elapsed:.1f}s  "
        f"pred={final_answer[:40]!r}  gold={ground_truth[:40]!r}"
    )

    return result


# ─── 主评测入口 ─────────────────────────────────────────────────────────────────

def run_eval(
    dataset_name: str,
    limit: int | None,
    scorer: str,
    max_revisions: int,
    concurrency: int,
    output_dir: Path,
    difficulty: str | None = None,
) -> Path:
    """执行完整评测流程，返回结果文件路径。"""

    print(f"[eval] 加载数据集: {dataset_name}  limit={limit}  difficulty={difficulty or 'all'}")
    items = load_eval_dataset(dataset_name, limit=limit, difficulty=difficulty)
    print(f"[eval] 共 {len(items)} 道题，scorer={scorer}，concurrency={concurrency}")

    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    llm_client = _build_llm_client()
    if llm_client is None:
        print("[eval] 警告：DEEPSEEK_API_KEY 未配置，LLM 相关功能将降级处理")
        if scorer == "llm":
            print("[eval] scorer 自动回退为 em")
            scorer = "em"

    print(f"[eval] 编译 ResearchGraph（max_revisions={max_revisions}）...")
    app = compile_graph(max_revisions=max_revisions)

    results: List[Dict[str, Any]] = [None] * len(items)  # type: ignore[list-item]

    print(f"[eval] 开始评测（线程并发数={concurrency}）...")
    t_total_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_idx = {
            executor.submit(
                _evaluate_single,
                idx,
                item,
                app,
                llm_client,
                model,
                scorer,
                max_revisions,
            ): idx
            for idx, item in enumerate(items)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:  # noqa: BLE001
                # 极端情况：单题彻底失败时记录占位结果
                results[idx] = {
                    "id": idx,
                    "question": items[idx]["question"],
                    "ground_truth": items[idx]["answer"],
                    "final_answer": "",
                    "cot_reasoning": "",
                    "is_correct": False,
                    "elapsed_seconds": 0.0,
                    "revision_step": 0,
                    "scorer": scorer,
                    "errors": [f"评测任务崩溃: {exc}"],
                    "mab_state": {},
                    "search_stats": {},
                }
                print(f"  [{idx:>4}] ✗  [CRASHED] {exc}")

    total_elapsed = time.perf_counter() - t_total_start
    correct_count = sum(1 for r in results if r and r.get("is_correct"))
    total_count = len(results)
    accuracy = correct_count / total_count if total_count else 0.0

    summary = {
        "dataset": dataset_name,
        "total": total_count,
        "correct": correct_count,
        "accuracy": round(accuracy, 4),
        "avg_elapsed_seconds": round(
            sum(r.get("elapsed_seconds", 0) for r in results if r) / max(total_count, 1),
            2,
        ),
        "total_elapsed_seconds": round(total_elapsed, 2),
        "scorer": scorer,
        "max_revisions": max_revisions,
        "concurrency": concurrency,
        "model": model,
    }

    output: Dict[str, Any] = {
        "meta": summary,
        "results": results,
    }

    # ── 写出结果文件 ───────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"results_{dataset_name}_{timestamp}.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"[eval] 完成！准确率: {correct_count}/{total_count} = {accuracy:.2%}")
    print(f"[eval] 总耗时: {total_elapsed:.1f}s  平均: {summary['avg_elapsed_seconds']}s/题")
    print(f"[eval] 结果已写入: {out_path}")
    print("=" * 60)

    return out_path


# ─── CLI 入口 ───────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ApexLogic Benchmark 评测脚本（与 main.py/export_report.py 完全独立）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["hotpotqa", "bamboogle"],
        default="bamboogle",
        help="评测数据集名称",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="截断题目数量（不设则全量，建议先用 --limit 10 验证）",
    )
    parser.add_argument(
        "--scorer",
        choices=["em", "llm"],
        default="em",
        help="评分模式：em=Exact Match；llm=LLM 语义判定",
    )
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=int(os.getenv("MAX_REVISIONS", "3")),
        help="每道题 Graph 最大迭代修订轮次",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="并发线程数（建议 ≤3，避免 API 速率限制）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests"),
        help="结果 JSON 输出目录",
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default=None,
        help="仅对 hotpotqa 生效：过滤指定难度题目",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(dotenv_path=".env", override=False)
    load_dotenv(dotenv_path=".env.example", override=False)

    args = _parse_args()

    if not _TENACITY_AVAILABLE:
        print("[eval] 提示：未安装 tenacity，重试机制不可用。建议：pip install tenacity")

    run_eval(
        dataset_name=args.dataset,
        limit=args.limit,
        scorer=args.scorer,
        max_revisions=args.max_revisions,
        concurrency=args.concurrency,
        output_dir=args.output_dir,
        difficulty=args.difficulty,
    )


if __name__ == "__main__":
    main()
