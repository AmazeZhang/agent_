#!/usr/bin/env python3
"""Phase 1a: split candidate pool into eval pool (10) + train pool.

Eval selection: spread across repos (max 1-2 per repo) and difficulty bands
(0.25/0.5/0.75), so the 10-task eval set is representative. The remaining
candidates become the train pool. No overlap.
"""
import collections
import json
import os
import random

random.seed(42)

SRC = "/media/imc/data/yzy/agent/project2/phase1/task_pool/candidates.jsonl"
OUT_DIR = "/media/imc/data/yzy/agent/project2/phase1/task_pool"
N_EVAL = 10

cands = [json.loads(l) for l in open(SRC)]
by_repo = collections.defaultdict(list)
for c in cands:
    by_repo[c["repo"].split("/")[-1]].append(c)

# one candidate per repo first (smallest repos first to maximize spread),
# then fill remaining slots with second candidates from repos that have them.
eval_pool = []
used_repos = set()
repos_with_1 = [r for r, v in by_repo.items() if len(v) >= 1]
# prefer repos with exactly 1 candidate (else we'd empty a repo entirely)
for r in sorted(repos_with_1, key=lambda r: len(by_repo[r])):
    if len(eval_pool) >= N_EVAL:
        break
    if r in used_repos:
        continue
    c = by_repo[r][0]
    eval_pool.append(c)
    used_repos.add(r)

# if still short, take from multi-candidate repos
for r in sorted(repos_with_1, key=lambda r: -len(by_repo[r])):
    if len(eval_pool) >= N_EVAL:
        break
    if r in used_repos:
        continue
    c = by_repo[r][0]
    eval_pool.append(c)
    used_repos.add(r)

eval_ids = {c["instance_id"] for c in eval_pool}
train_pool = [c for c in cands if c["instance_id"] not in eval_ids]

os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "eval_pool.jsonl"), "w") as f:
    for c in eval_pool:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
with open(os.path.join(OUT_DIR, "train_pool.jsonl"), "w") as f:
    for c in train_pool:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

print(f"eval pool: {len(eval_pool)}  (repos: {sorted(used_repos)})")
print(f"train pool: {len(train_pool)}")
print("eval difficulty:", [round(c['resolve_rate'], 2) for c in eval_pool])
