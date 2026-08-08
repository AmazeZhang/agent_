#!/usr/bin/env python3
"""Phase 1a: build task environments and verify SWE-smith tasks (eval pool).

SWE-smith patch semantics are INVERTED vs SWE-bench: the stored patch
INTRODUCES the bug. The broken state the agent must fix is
`commit + apply patch`; the fix is the patch's reverse.

Gate G1 (task soundness) per eval-pool task:
  1. git worktree at the task commit (blob:none clones share the object store)
  2. task venv (eval-venvs/<task_id>); install repo editable, pytest, deps
  3. CLEAN state (commit as-is): F2P tests (element col2) must ALL pass
     -> proves the fix is reachable and the tests are valid
  4. apply the bug-introducing patch -> BROKEN state
  5. F2P on BROKEN state must have FAILURES -> proves the bug is real
  6. P2P (element col3): delta broken-vs-clean must be 0
Verdict OK <=> f2p_clean all pass AND f2p_broken has failures AND p2p_delta==0.

Writes phase1/stats/gold_review.jsonl (one row per task).
"""
import argparse
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow.parquet as pq

BASE = "/media/imc/data/yzy/agent/project2"
POOL = f"{BASE}/phase1/task_pool/eval_pool.jsonl"
MANIFEST = f"{BASE}/phase1/task_pool/env_manifest.jsonl"
TASK_DATA = f"{BASE}/datasets/swe-smith-tasks"
WORK = f"{BASE}/phase1/work"
VENVS = f"{BASE}/phase1/eval-venvs"
OUT = f"{BASE}/phase1/stats/gold_review.jsonl"
BASE_PY = "/home/imc/yzy/agent/project2-coding-agent-rl/.venvs/rllm-base/bin/python"
PROXY = "http://127.0.0.1:7890"

_worktree_lock = threading.Lock()  # serialise worktree add per repo


def find_test_spec(instance_id: str):
    """F2P = element col2 (must pass after patch), P2P = element col3."""
    for f in sorted(os.listdir(TASK_DATA)):
        t = pq.read_table(os.path.join(TASK_DATA, f))
        col2, col3 = t.column(2), t.column(3)
        ids = [str(x) for x in t["instance_id"].to_pylist()]
        f2ps = [[str(x) for x in row] for row in col2.to_pylist()]
        p2ps = [[str(x) for x in row] for row in col3.to_pylist()]
        for inst, f2p, p2p in zip(ids, f2ps, p2ps):
            if inst == instance_id:
                return f2p, p2p
    return None, None


def run(cmd, cwd=None, timeout=1800, env=None):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd,
                          timeout=timeout, env=env)


def pytest_parse(out: str) -> list:
    # pytest -q summary lines: "FAILED <node> - <reason>" / "ERROR <node>"
    nodes = []
    for ln in out.splitlines():
        if ln.startswith("FAILED ") or ln.startswith("ERROR "):
            nodes.append(ln.split()[1])
    return nodes


def build_and_verify(task: dict):
    tid = task["instance_id"]
    t0 = time.time()
    if not task.get("repo_dir"):
        return {"instance_id": tid, "verdict": "SKIP", "reason": "no repo clone"}
    f2p, p2p = find_test_spec(tid)
    if f2p is None:
        return {"instance_id": tid, "verdict": "SKIP", "reason": "no test spec"}

    workdir = os.path.join(WORK, tid)
    venv = os.path.join(VENVS, tid)
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(VENVS, exist_ok=True)
    py_bin = os.path.join(venv, "bin", "python")
    pip_bin = os.path.join(venv, "bin", "pip")
    pytest_bin = os.path.join(venv, "bin", "pytest")

    # 1. worktree at task commit (serialised: same repo may serve many tasks)
    if not os.path.isdir(workdir):
        with _worktree_lock:
            r = run(["git", "-c", f"http.proxy={PROXY}", "worktree", "add", "-f",
                     workdir, task["commit"]], cwd=task["repo_dir"], timeout=1200)
        if r.returncode != 0:
            return {"instance_id": tid, "verdict": "FAIL", "stage": "worktree",
                    "detail": r.stderr[-300:]}

    # 2. venv + install (editable; PYTHONPATH fallback for legacy setups)
    if not os.path.exists(py_bin):
        r = run([BASE_PY, "-m", "venv", venv], timeout=600)
        if r.returncode != 0:
            return {"instance_id": tid, "verdict": "FAIL", "stage": "venv",
                    "detail": r.stderr[-300:]}
    install_mode = "editable"
    run([pip_bin, "install", "-q", "--upgrade", "pip"], timeout=600)
    r = run([pip_bin, "install", "-q", "-e", workdir], timeout=1800)
    if r.returncode != 0:
        install_mode = "pathonly"
        r = run([pip_bin, "install", "-q", "pytest"], timeout=600)
        if r.returncode != 0:
            return {"instance_id": tid, "verdict": "FAIL", "stage": "install",
                    "detail": r.stderr[-500:]}
    else:
        run([pip_bin, "install", "-q", "pytest"], timeout=600)
    env = dict(os.environ)
    if install_mode == "pathonly":
        env["PYTHONPATH"] = workdir

    def git_clean():
        run(["git", "-C", workdir, "checkout", "-f", "--", "."], timeout=300)
        run(["git", "-C", workdir, "clean", "-fd", "--exclude=__pycache__"],
            timeout=300)

    def git_apply(patch_path):
        r = run(["git", "-C", workdir, "apply", patch_path], timeout=300)
        if r.returncode != 0:
            r = run(["git", "-C", workdir, "apply", "--3way", patch_path],
                    timeout=600)
        return r

    def install_missing(out: str) -> int:
        """Install modules named in ModuleNotFoundError lines (test deps)."""
        names = set()
        for m in __import__("re").finditer(
                r"ModuleNotFoundError: No module named '([^']+)'", out):
            names.add(m.group(1).split(".")[0])
        # stdlib aliases / our own venv package: never pip-install these
        skip = {"encodings", "posixpath", "pkg_resources"}
        names -= skip
        for n in names:
            run([pip_bin, "install", "-q", n], timeout=900)
        return len(names)

    def run_suite(nodes, label, timeout=3600):
        if not nodes:
            return {"n": 0, "failed": [], "rc": 0}
        for attempt in range(4):
            r = run([pytest_bin, "-q", "--tb=short", *nodes], cwd=workdir,
                    timeout=timeout, env=env)
            out = (r.stdout or "") + (r.stderr or "")
            if r.returncode == 0 or "ModuleNotFoundError" not in out:
                return {"n": len(nodes), "failed": pytest_parse(r.stdout or ""),
                        "rc": r.returncode}
            if not install_missing(out):
                return {"n": len(nodes), "failed": pytest_parse(r.stdout or ""),
                        "rc": r.returncode}
            print(f"  [{tid}] {label}: installed missing deps, retry "
                  f"{attempt + 2}/4")
        return {"n": len(nodes), "failed": pytest_parse(r.stdout or ""),
                "rc": r.returncode}

    patch_file = os.path.join(WORK, f"{tid}.gold.diff")
    with open(patch_file, "w") as f:
        f.write(task["gold_patch"])

    # 3. CLEAN state: F2P must all pass (fix reachable, tests valid);
    #    P2P on clean is the regression baseline
    git_clean()
    f2p_clean = run_suite(f2p, "f2p-clean")
    p2p_clean = run_suite(p2p, "p2p-clean")

    # 4-5. apply bug-introducing patch -> BROKEN state; F2P must now fail
    git_clean()
    r = git_apply(patch_file)
    if r.returncode != 0:
        return {"instance_id": tid, "verdict": "FAIL", "stage": "apply-patch",
                "detail": r.stderr[-300:],
                "f2p_clean_failed": f2p_clean["failed"]}
    f2p_broken = run_suite(f2p, "f2p-broken")
    p2p_broken = run_suite(p2p, "p2p-broken")

    f2p_clean_ok = f2p_clean["n"] > 0 and not f2p_clean["failed"]
    bug_real = len(f2p_broken["failed"]) > 0
    delta = [x for x in p2p_broken["failed"] if x not in p2p_clean["failed"]]
    verdict = "OK" if (f2p_clean_ok and bug_real and not delta) else "FAIL"
    return {"instance_id": tid, "verdict": verdict, "t_sec": round(time.time() - t0),
            "install_mode": install_mode,
            "f2p_clean": {"n": f2p_clean["n"], "failed": f2p_clean["failed"]},
            "f2p_broken": {"n": f2p_broken["n"], "failed": f2p_broken["failed"]},
            "p2p": {"n": p2p_broken["n"], "delta": len(delta),
                    "clean_failed": p2p_clean["failed"],
                    "broken_failed": p2p_broken["failed"]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=POOL)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    tasks = {json.loads(l)["instance_id"]: json.loads(l) for l in open(args.pool)}
    manifest = {json.loads(l)["instance_id"]: json.loads(l) for l in open(MANIFEST)}
    items = [{**tasks[t], **manifest.get(t, {})} for t in tasks]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(build_and_verify, t) for t in items]
        for f in as_completed(futs):
            res = f.result()
            results.append(res)
            print(json.dumps(res, ensure_ascii=False))
    with open(OUT, "w") as fo:
        for r in results:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = sum(1 for r in results if r["verdict"] == "OK")
    print(f"\nverdicts: {ok}/{len(results)} OK; wrote {OUT}")


if __name__ == "__main__":
    main()
