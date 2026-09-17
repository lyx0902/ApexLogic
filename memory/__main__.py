"""Local memory inspection and explicit conflict resolution. No network calls."""
import argparse
import json

from core.persistence import data_directory
from memory.repository import MemoryRepository


def main():
    parser = argparse.ArgumentParser(description="查看记忆或登记人工核实后的关系")
    parser.add_argument("--data-dir")
    parser.add_argument("--namespace", default="workspace/default")
    parser.add_argument("--relation", choices=["supports", "contradicts", "duplicates", "supersedes"])
    parser.add_argument("--left")
    parser.add_argument("--right")
    parser.add_argument("--reason")
    args = parser.parse_args()
    if args.relation and not all([args.left, args.right, args.reason]):
        parser.error("登记关系需要 --left --right --reason；supersedes 表示 left 替代 right")
    repo = MemoryRepository(data_directory(args.data_dir))
    if args.relation:
        repo.relate(args.left, args.right, args.namespace, args.relation, args.reason)
    with repo.connect() as db:
        items = [dict(r) for r in db.execute("SELECT id,title,url,claim,status,observed_at,valid_until FROM memory_items WHERE namespace=? ORDER BY observed_at DESC LIMIT 100", (args.namespace,))]
    print(json.dumps(items, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
