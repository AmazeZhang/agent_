#!/usr/bin/env python
"""M2: 任务集划分（dev/val/holdout），防泄漏核心。

用法:
  .venvs/tau2/bin/python data/partition.py --manifest data/datasets/task_manifest.json \
      --out data/datasets --dev 0.6 --val 0.2

规则（SPEC 02）:
- 按任务 id hash 划分，不依赖执行顺序或 seed 随机流。
- 输出 dev/val/holdout 三份 id 清单 + partition_manifest.json（含划分 hash）。
- holdout 的 id 集合在任何优化入口被断言拒绝（配合 guard 函数使用）。

重要: 本脚本在 M1 采集完成前定稿；划分一旦锁定不得改动。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def id_hash(task_id: str) -> int:
    return int(hashlib.sha256(f"p1-partition-v1:{task_id}".encode()).hexdigest(), 16)


def partition(task_ids: list[str], dev_frac: float, val_frac: float) -> dict[str, list[str]]:
    n = len(task_ids)
    n_dev = int(n * dev_frac)
    n_val = int(n * val_frac)
    ordered = sorted(task_ids, key=id_hash)
    return {
        "dev": ordered[:n_dev],
        "val": ordered[n_dev:n_dev + n_val],
        "holdout": ordered[n_dev + n_val:],
    }


def load_ids(manifest_path: Path) -> list[str]:
    manifest = json.loads(manifest_path.read_text())
    return list(manifest["tasks"].keys())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dev", type=float, default=0.6)
    ap.add_argument("--val", type=float, default=0.2)
    args = ap.parse_args()

    task_ids = load_ids(args.manifest)
    if len(task_ids) < 10:
        raise SystemExit(f"任务数过少（{len(task_ids)}），不足以划分")

    splits = partition(task_ids, args.dev, args.val)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        (args.out / f"{name}.jsonl").write_text(
            "\n".join(json.dumps({"id": i}) for i in ids) + "\n"
        )

    manifest = {
        "schema_version": 1,
        "created_at": None,  # 由调用方补时间戳（防重放/确定性）
        "num_tasks": len(task_ids),
        "fractions": {"dev": args.dev, "val": args.val, "holdout": 1 - args.dev - args.val},
        "splits": {k: len(v) for k, v in splits.items()},
        "partition_hash": hashlib.sha256(
            json.dumps(splits, sort_keys=True).encode()
        ).hexdigest()[:16],
        "note": "划分按任务 id 稳定 hash；holdout 不得进入任何优化调用。",
    }
    (args.out / "partition_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def is_holdout(task_id: str, partition_manifest_path: Path) -> bool:
    """供优化入口调用的 holdout 断言守卫。"""
    manifest = json.loads(Path(partition_manifest_path).read_text())
    split_files = {
        "dev": Path(partition_manifest_path).parent / "dev.jsonl",
        "val": Path(partition_manifest_path).parent / "val.jsonl",
        "holdout": Path(partition_manifest_path).parent / "holdout.jsonl",
    }
    # 直接从分片文件读取，保证与实际使用一致
    for name, f in split_files.items():
        for line in f.read_text().splitlines():
            if json.loads(line)["id"] == str(task_id):
                return name == "holdout"
    raise ValueError(f"task_id {task_id} 不在任何分片中")


if __name__ == "__main__":
    main()
