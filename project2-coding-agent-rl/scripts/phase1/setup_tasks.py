#!/usr/bin/env python3
"""Phase 1a: clone repos + checkout task commits for the task pool.

For each unique (owner, repo, commit) in the pool:
  - clone (blob:none, no checkout) into phase1/repos/<owner>__<repo>
  - fetch the exact commit, record it
Writes phase1/task_pool/env_manifest.jsonl: per task -> {repo_dir, commit, task_type}.

Run with the clash proxy configured for git:
  git config --global http.proxy http://127.0.0.1:7890  (or -c http.proxy per clone)
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

POOL_DIR = "/media/imc/data/yzy/agent/project2/phase1/task_pool"
REPOS_DIR = "/media/imc/data/yzy/agent/project2/phase1/repos"

# instance_id format: owner__repo.<commit>.<task_type>__<suffix>
# e.g. mahmoud__glom.fb3c4e76.combine_file__apwtp94w
# or   facebookresearch__hydra.0f03eb60.pr_2543  (real-PR inversion)
ID_RE = re.compile(
    r"^(?P<owner>[^_]+)__(?P<repo>[^\.]+)\.(?P<commit>[0-9a-f]{7,40})\."
    r"(?P<task_type>pr_\d+|[a-z_0-9]+__[a-z0-9]+)$")


def parse_instance(instance_id: str):
    m = ID_RE.match(instance_id)
    if not m:
        raise ValueError(f"cannot parse instance_id: {instance_id}")
    return m.group("owner"), m.group("repo"), m.group("commit"), m.group("task_type")


def clone_and_fetch(repo_key: str, owner: str, repo: str, commits: set[str]):
    """Clone once per repo (blob:none, all branches), resolve short commit hashes.

    Short hashes from the dataset are not refs; they resolve via rev-parse after
    fetching full branch history (blob:none keeps this cheap).
    """
    repo_dir = os.path.join(REPOS_DIR, repo_key)
    url = f"https://github.com/{owner}/{repo}.git"
    proxy = "http://127.0.0.1:7890"
    if not os.path.isdir(repo_dir):
        os.makedirs(REPOS_DIR, exist_ok=True)
        r = subprocess.run(
            ["git", "-c", f"http.proxy={proxy}", "clone", "--no-checkout",
             "--filter=blob:none", url, repo_dir],
            capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            shutil.rmtree(repo_dir, ignore_errors=True)
            return repo_key, None, r.stderr.strip()[-300:]
    # resolve short -> full hash; fetch the branch containing it if needed
    for c in commits:
        r = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "-q", "--verify", f"{c}^{{commit}}"],
            capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            return repo_key, repo_dir, f"commit {c} not reachable in {owner}/{repo}"
    return repo_key, repo_dir, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=os.path.join(POOL_DIR, "train_pool.jsonl"))
    ap.add_argument("--pool2", default=os.path.join(POOL_DIR, "eval_pool.jsonl"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    tasks = []
    for p in (args.pool, args.pool2):
        if os.path.exists(p):
            tasks += [json.loads(l) for l in open(p)]
    print(f"total tasks: {len(tasks)}")

    repo_commits = {}
    parsed = []
    for t in tasks:
        owner, repo, commit, task_type = parse_instance(t["instance_id"])
        repo_key = f"{owner}__{repo}"
        repo_commits.setdefault(repo_key, {"owner": owner, "repo": repo, "commits": set()})
        repo_commits[repo_key]["commits"].add(commit)
        parsed.append({**t, "owner": owner, "repo": repo, "commit": commit,
                       "task_type": task_type, "repo_key": repo_key})
    print(f"unique repos: {len(repo_commits)}")

    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(clone_and_fetch, k, v["owner"], v["repo"], v["commits"]): k
                for k, v in repo_commits.items()}
        for f in as_completed(futs):
            key, repo_dir, err = f.result()
            results[key] = {"repo_dir": repo_dir, "error": err}
            print(("OK  " if repo_dir else "FAIL") + f" {key}" + (f" :: {err}" if err else ""))

    # write manifest
    out_path = os.path.join(POOL_DIR, "env_manifest.jsonl")
    n_ok = 0
    with open(out_path, "w") as f:
        for t in parsed:
            info = results.get(t["repo_key"], {})
            if info.get("repo_dir"):
                n_ok += 1
            f.write(json.dumps({**t, "repo_dir": info.get("repo_dir"),
                                "error": info.get("error")}, ensure_ascii=False) + "\n")
    print(f"wrote {out_path}: {n_ok}/{len(parsed)} tasks with repo clone")


if __name__ == "__main__":
    main()
