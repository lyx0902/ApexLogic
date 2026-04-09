import os
import time
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# 引入项目现有的评测工具
from evals.datasets import load_eval_dataset
from evals.scorer import exact_match, llm_judge
from agents.extractor import extract_answer


def _require_env(name: str) -> str:
    """读取并校验环境变量，避免后续请求时报错信息不明确。"""
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"缺少环境变量: {name}")
    return value

def get_commercial_client_and_model(provider: str):
    if provider == "qwen":
        api_key = _require_env("QWEN_API_KEY")
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        model = "qwen-max"
        return OpenAI(api_key=api_key, base_url=base_url), model
    elif provider == "doubao":
        api_key = _require_env("DOUBAO_API_KEY")
        base_url = "https://ark.cn-beijing.volces.com/api/v3"
        model = _require_env("DOUBAO_ENDPOINT_ID")
        return OpenAI(api_key=api_key, base_url=base_url), model
    else:
        raise ValueError("不支持的提供商")

def call_commercial_api(client, model, question, provider):
    system_prompt = "你是一个强大的深度研究助手。请必须利用你的联网搜索能力，针对用户的复杂问题进行多跳检索。最后请给出详细的推理过程和最终结论。"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]

    extra_params = {}
    if provider == "qwen":
        extra_params["extra_body"] = {"enable_search": True}

    response = client.chat.completions.create(model=model, messages=messages, temperature=0.1, **extra_params)
    return response.choices[0].message.content

def evaluate_single(idx, item, provider, scorer):
    question = item["question"]
    ground_truth = item["answer"]
    t_start = time.perf_counter()
    client, model = get_commercial_client_and_model(provider)

    try:
        raw_draft = call_commercial_api(client, model, question, provider)
        ds_client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"))
        extraction = extract_answer(question=question, draft=raw_draft, client=ds_client, model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
        final_answer = extraction.get("final_answer", "")
        cot = extraction.get("cot_reasoning", "")

        if scorer == "llm":
            judge = llm_judge(question, final_answer, ground_truth, ds_client, os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
            is_correct = judge.get("correct", False)
        else:
            is_correct = exact_match(final_answer, ground_truth)
    except Exception as e:
        print(f"[{idx}] 报错: {e}")
        raw_draft, final_answer, cot, is_correct = str(e), "", "", False

    elapsed = time.perf_counter() - t_start
    status = "✓" if is_correct else "✗"
    print(f"[{provider.upper()}] [{idx:>3}] {status} elapsed={elapsed:.1f}s pred={final_answer[:20]!r} gold={ground_truth[:20]!r}")

    return {"id": idx, "question": question, "ground_truth": ground_truth, "raw_draft": raw_draft, "final_answer": final_answer, "cot_reasoning": cot, "is_correct": is_correct, "elapsed_seconds": round(elapsed, 2)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["qwen", "doubao"], required=True)
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--level", choices=["easy", "medium", "hard"], default=None)
    parser.add_argument("--split", choices=["auto", "train", "validation", "test"], default="auto")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--scorer", choices=["em", "llm"], default="llm")
    parser.add_argument("--output-dir", type=Path, default=Path("tests"))
    args = parser.parse_args()

    load_dotenv()

    # 与 evals/datasets.py 接口保持一致：使用 difficulty 字段名。
    items = load_eval_dataset(
        args.dataset,
        limit=args.limit + args.skip,
        difficulty=args.level,
        split=None if args.split == "auto" else args.split,
    )
    if args.skip:
        items = items[args.skip:]

    if not items:
        print("[baseline] 未加载到题目，请检查 dataset/level/split 组合")
        return

    results = [None] * len(items)
    t_total_start = time.perf_counter()
    correct_count = 0

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures_to_idx = {executor.submit(evaluate_single, i, item, args.provider, args.scorer): i for i, item in enumerate(items)}
        for f in as_completed(futures_to_idx):
            idx = futures_to_idx[f]
            results[idx] = f.result()
            result_item = results[idx]
            if isinstance(result_item, dict) and result_item.get("is_correct"):
                correct_count += 1

    total_elapsed = time.perf_counter() - t_total_start
    accuracy = correct_count / len(items) if items else 0.0

    output = {
        "meta": {"provider": args.provider, "dataset": args.dataset, "level": args.level, "skip": args.skip, "total": len(items), "correct": correct_count, "accuracy": round(accuracy, 4), "total_elapsed_seconds": round(total_elapsed, 2)},
        "results": results
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.output_dir / f"baseline_{args.provider}_{args.dataset}_{timestamp}.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[{args.provider.upper()}] 测试完成! 准确率: {correct_count}/{len(items)} ({accuracy:.2%})")
    print(f"结果已写入: {out_path}")

if __name__ == "__main__":
    main()
