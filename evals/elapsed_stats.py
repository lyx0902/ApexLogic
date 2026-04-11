from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_rows(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return [row for row in payload["results"] if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    raise ValueError("Unsupported JSON structure: expected {'results': [...]} or a list of records")


def _pick_id(row: Dict[str, Any]) -> Any:
    if "id" in row:
        return row["id"]
    meta = row.get("meta")
    if isinstance(meta, dict) and "id" in meta:
        return meta["id"]
    return None


def _pick_elapsed(row: Dict[str, Any]) -> float | None:
    value = row.get("elapsed_seconds")
    if value is None:
        meta = row.get("meta")
        if isinstance(meta, dict):
            value = meta.get("elapsed_seconds")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_elapsed_stats(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, List[float]], List[Dict[str, Any]]]:
    buckets: Dict[str, List[float]] = {}
    invalid_rows: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows):
        qid = _pick_id(row)
        elapsed = _pick_elapsed(row)

        if qid is None or elapsed is None:
            invalid_rows.append(
                {
                    "row_index": idx,
                    "id": qid,
                    "elapsed_seconds": row.get("elapsed_seconds"),
                    "reason": "missing id or elapsed_seconds",
                }
            )
            continue

        key = str(qid)
        buckets.setdefault(key, []).append(elapsed)

    return buckets, invalid_rows


def _build_output(input_path: Path, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets, invalid_rows = compute_elapsed_stats(rows)

    by_id: List[Dict[str, Any]] = []
    all_values: List[float] = []

    for qid in sorted(buckets.keys(), key=lambda x: (len(x), x)):
        values = buckets[qid]
        all_values.extend(values)
        avg = sum(values) / len(values)
        by_id.append(
            {
                "id": qid,
                "count": len(values),
                "elapsed_seconds": round(avg, 4)
            }
        )

    overall_avg = round(sum(all_values) / len(all_values), 4) if all_values else 0.0

    return {
        "input_file": str(input_path),
        "total_rows": len(rows),
        "valid_rows": len(all_values),
        "unique_ids": len(buckets),
        "overall_avg_elapsed_seconds": overall_avg,
        "elapsed_by_id": by_id,
        "invalid_rows": invalid_rows,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute elapsed_seconds statistics grouped by question id from an eval JSON file."
    )
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON file not found: {input_path}")

    payload = _load_json(input_path)
    rows = _extract_rows(payload)
    output = _build_output(input_path, rows)

    # 输出四个核心指标，便于在命令行快速查看统计结果
    print(
        f"[elapsed-stats] total_rows={output['total_rows']} "
        f"valid_rows={output['valid_rows']} "
        f"unique_ids={output['unique_ids']} "
        f"overall_avg={output['overall_avg_elapsed_seconds']}s"
    )


if __name__ == "__main__":
    main()

