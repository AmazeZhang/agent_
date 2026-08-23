#!/usr/bin/env python
"""P3 Search-aware clean v2: 11-item engineering-smoke PASS checker (CPU-only).

Reads a completed 1-step eng-smoke run directory (run ID
p3-search-aware-clean-v2-eng-smoke-fsdp6-b66-n5-s0-20260822a) and enforces the
11 PASS criteria of the user directive §5:

  S1  real optimizer + real global step (optim rank files, latest checkpointed
      iteration)
  S2  checkpoint complete (model/optim/extra rank files, tokenizer files,
      data.pt loadable)
  S3  at least one valid search trajectory in the rollout audit
  S4  at least one multi-search trajectory with a DIFFERENT query that brought
      at least one NEW document -- or the honest absence report
  S5  v2 reward components landed correctly: search_v1/search_v1_episode/
      search_v1_group carry "version":"v2", format_reward_c == 0 everywhere,
      per-trajectory record_score sum == episode total, per-uid episode sums ==
      group sums (exact cents, fail-closed)
  S6  true_redundant fires ONLY on duplicate query / no-new-document: the
      env's on-the-fly verdict is recomputed from the audit records with the
      shared library (searchr1_repro.search_v2_reward) and must match exactly
  S7  trajectory advantage broadcast correct: per-traj_uid audit value equals
      the per-uid GRPO recomputation ((return - mean) / (std + 1e-6))
  S8  at least one useful-search trajectory (evidence credit / sce) has a
      POSITIVE trajectory advantage
  S9  Observation loss mask: observations live in the prompt region, which is
      fully excluded from the policy loss (prompt_policy_loss_tokens == 0 for
      every record); the response region contains only model tokens
      (policy_loss_tokens <= active_response_tokens)
  S10 no OOM / NaN / Xid / dropped-GPU / Retriever-timeout in the run logs
  S11 no leftover ray/vllm/main_ppo processes after exit; GPUs back to baseline

§8 memory reporting (never conflate): the nvidia-smi sampler's
peak_memory_nvidia_smi.json holds per-GPU PHYSICAL peaks; the verl log's
max_memory_reserved_gb is a worker-aggregated torch-allocator view and is
reported separately with that label.

Usage:
  python scripts/check_p3_v2_smoke.py --run-dir <dir> [--run-id <id>]
        [--gpus 1,2,3,4,6,7] [--steps 1]
Exit code 0 = all 11 items PASS; 1 = any failure.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from searchr1_repro.search_v2_reward import norm_query, search_step_components_v2  # noqa: E402

COMPONENTS = (
    "answer_reward_c",
    "format_reward_c",
    "evidence_hit_reward_c",
    "searched_correct_bonus_c",
    "invalid_penalty_c",
    "redundant_penalty_c",
    "answer_leak_penalty_c",
)
SUCCESS_STATUSES = {"success", "no_results"}

# S10 log patterns (calibrated on the clean v1 10-step run logs: zero matches)
LOG_PATTERNS = {
    "oom": re.compile(r"CUDA out of memory|OutOfMemoryError|out of memory", re.I),
    "nan_loss": re.compile(r"loss.{0,80}nan|nan.{0,80}loss", re.I),
    "xid": re.compile(r"\bXid\s+\d+"),
    "dropped_gpu": re.compile(
        r"CUDA_ERROR|NCCL.{0,30}(error|fail)|no longer (responding|running)|RuntimeError: NCCL|\bKilled\b"
    ),
    "retriever_timeout": re.compile(r"TimeoutError|timed out|retriev.{0,60}(fail|error|timeout)", re.I),
}


def load_records(run_dir: Path, steps: int) -> list[dict]:
    records = []
    for step in range(1, steps + 1):
        aud = run_dir / "rollouts" / f"{step}.audit.jsonl"
        if not aud.exists():
            raise SystemExit(f"missing rollout audit: {aud}")
        with aud.open(encoding="utf-8") as handle:
            for line in handle:
                records.append(json.loads(line))
    return records


def check_optimizer_and_global_step(run_dir: Path, steps: int) -> tuple[bool, dict]:
    ckpts = run_dir / "checkpoints"
    iteration_file = ckpts / "latest_checkpointed_iteration.txt"
    if not iteration_file.exists():
        return False, {"error": "latest_checkpointed_iteration.txt missing"}
    iteration = iteration_file.read_text().strip()
    step_dir = ckpts / f"global_step_{iteration}"
    if not step_dir.exists():
        return False, {"iteration": iteration, "error": f"global_step_{iteration} missing"}
    actor = step_dir / "actor"
    optim_files = sorted(actor.glob("optim_world_size_*_rank_*.pt")) if actor.exists() else []
    bad = []
    for f in optim_files:
        if f.stat().st_size == 0 or not zipfile.is_zipfile(f):
            bad.append(str(f.name))
    ok = iteration == str(steps) and len(optim_files) >= 6 and not bad
    return ok, {
        "iteration": iteration,
        "expected_iteration": str(steps),
        "n_optim_rank_files": len(optim_files),
        "optim_rank_files_bad": bad[:5],
    }


def check_checkpoint_complete(run_dir: Path, steps: int) -> tuple[bool, dict]:
    ckpts = run_dir / "checkpoints"
    iteration_file = ckpts / "latest_checkpointed_iteration.txt"
    if not iteration_file.exists():
        return False, {"error": "latest_checkpointed_iteration.txt missing"}
    iteration = iteration_file.read_text().strip()
    step_dir = ckpts / f"global_step_{iteration}"
    actor = step_dir / "actor"
    tokenizer_files = ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
                       "merges.txt", "added_tokens.json", "special_tokens_map.json", "generation_config.json")
    missing_tok = [t for t in tokenizer_files if not (actor / t).exists()]
    rank_files = {
        kind: sorted(actor.glob(f"{kind}_world_size_*_rank_*.pt"))
        for kind in ("model", "optim", "extra_state")
    }
    bad_zips = {
        kind: [f.name for f in files if f.stat().st_size == 0 or not zipfile.is_zipfile(f)]
        for kind, files in rank_files.items()
    }
    data_pt = step_dir / "data.pt"
    data_ok = False
    if data_pt.exists():
        try:
            import torch

            obj = torch.load(data_pt, map_location="cpu", weights_only=False)
            data_ok = isinstance(obj, dict)
        except Exception as exc:
            data_ok = False
    n_ranks = len(rank_files["model"])
    ok = (len(missing_tok) == 0 and n_ranks >= 6 and len(rank_files["optim"]) >= 6
          and len(rank_files["extra_state"]) >= 6 and not any(bad_zips.values()) and data_ok)
    return ok, {
        "step_dir": step_dir.name,
        "missing_tokenizer_files": missing_tok,
        "rank_file_counts": {k: len(v) for k, v in rank_files.items()},
        "bad_zip_rank_files": bad_zips,
        "data_pt_loadable": data_ok,
    }


def check_valid_search_trajectory(records: list[dict]) -> tuple[bool, dict]:
    by_uid: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_uid[r["metadata"]["traj_uid"]].append(r)
    n_valid_trajs = 0
    for rs in by_uid.values():
        if any((r["metadata"].get("retrieval") or {}).get("status") in SUCCESS_STATUSES for r in rs):
            n_valid_trajs += 1
    return n_valid_trajs >= 1, {"n_valid_search_trajectories": n_valid_trajs}


def check_multi_search_new_doc(records: list[dict]) -> tuple[bool, dict]:
    by_uid: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_uid[r["metadata"]["traj_uid"]].append(r)
    n_multi_new_doc = 0
    examples = []
    for rs in by_uid.values():
        searches = sorted(
            (r for r in rs if r["metadata"].get("search_v1") is not None
             and not r["metadata"]["search_v1"].get("terminal")),
            key=lambda r: r["metadata"]["env_step"],
        )
        prior_queries: set[str] = set()
        prior_doc_ids: set[str] = set()
        for i, r in enumerate(searches):
            sv = r["metadata"]["search_v1"]
            retrieval = r["metadata"].get("retrieval") or {}
            nq = norm_query(sv.get("query"))
            doc_ids = {str(d) for d in (retrieval.get("document_ids") or [])}
            if i >= 1 and nq and nq not in prior_queries and doc_ids - prior_doc_ids:
                n_multi_new_doc += 1
                if len(examples) < 3:
                    examples.append({"traj_uid": r["metadata"]["traj_uid"],
                                     "n_searches": len(searches), "env_step": r["metadata"]["env_step"]})
            if nq:
                prior_queries.add(nq)
            prior_doc_ids |= doc_ids
    ok = n_multi_new_doc >= 1
    return ok, {
        "n_multi_search_different_query_new_doc": n_multi_new_doc,
        "examples": examples,
        "honest_absence_note": "0 means the sample contained no such trajectory; "
                               "the count above IS the honest report",
    }


def check_v2_components(records: list[dict]) -> tuple[bool, dict]:
    by_uid: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_uid[r["metadata"]["traj_uid"]].append(r)
    issues = []
    n_episodes = 0
    version_bad = 0
    format_nonzero = 0
    record_score_mismatch = []
    schema_bad = 0
    for traj_uid, rs in by_uid.items():
        n_episodes += 1
        ep = rs[0]["metadata"].get("search_v1_episode")
        grp = rs[0]["metadata"].get("search_v1_group")
        if ep is None or grp is None:
            issues.append(f"{traj_uid}: missing episode/group metadata")
            continue
        if ep.get("version") != "v2" or grp.get("version") != "v2":
            version_bad += 1
        if ep.get("format_reward_c") != 0 or grp.get("format_reward_c") != 0:
            format_nonzero += 1
        if any(k not in ep for k in COMPONENTS + ("total_reward_c",)):
            schema_bad += 1
        # per-trajectory record_score sum == episode total (cents)
        score_sum = sum(float(r.get("record_score") or 0.0) for r in rs)
        expected = ep["total_reward_c"] / 100.0
        if abs(score_sum - expected) > 1e-4:
            record_score_mismatch.append({"traj_uid": traj_uid, "score_sum": score_sum,
                                          "episode_total": expected})
    # per-uid: episode component sums == group component sums
    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        g = r["metadata"].get("search_v1_group")
        if g is not None:
            by_group[g["uid"]].append(g)
    group_mismatch = []
    for gid, groups in by_group.items():
        g = groups[0]
        ep_sums = Counter()
        for uid_recs in by_uid.values():
            ep = uid_recs[0]["metadata"].get("search_v1_episode")
            if ep is not None and ep.get("uid") == gid:
                for c in COMPONENTS + ("total_reward_c",):
                    ep_sums[c] += ep[c]
        for c in COMPONENTS + ("total_reward_c",):
            if ep_sums.get(c, 0) != g.get(c, 0):
                group_mismatch.append({"uid": gid, "component": c,
                                       "episode_sum": ep_sums.get(c, 0), "group_value": g.get(c, 0)})
    ok = (not issues and version_bad == 0 and format_nonzero == 0 and schema_bad == 0
          and not record_score_mismatch and not group_mismatch)
    return ok, {
        "n_episodes": n_episodes,
        "version_not_v2": version_bad,
        "format_reward_nonzero": format_nonzero,
        "schema_bad": schema_bad,
        "record_score_sum_mismatch": record_score_mismatch[:5],
        "n_record_score_mismatch": len(record_score_mismatch),
        "group_component_mismatch": group_mismatch[:5],
        "n_group_component_mismatch": len(group_mismatch),
        "metadata_issues": issues[:5],
    }


def check_true_redundant(records: list[dict]) -> tuple[bool, dict]:
    """Recompute the v2 redundant verdict from the audit with the shared
    library and compare against the env's on-the-fly verdict.

    Scope (P3 v2, 2026-08-23): only `redundant_search` is fully recomputable
    offline -- it depends on query/status/doc_ids and the step's own prior
    state, all of which the audit stores. `answer_leak` and `evidence_hit`
    need the question, gt_aliases and the observation text, which the audit
    does not store, so they are NOT recomputed here: their placement plumbing
    is covered by S5's sum-consistency (component sums == placed sums in exact
    cents). Audit leak=True steps are counted and reported as not offline
    verifiable."""
    by_uid: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_uid[r["metadata"]["traj_uid"]].append(r)
    mismatches = []
    n_search_steps = 0
    n_redundant = 0
    n_audit_leak = 0
    for traj_uid, rs in by_uid.items():
        searches = sorted(
            (r for r in rs if r["metadata"].get("search_v1") is not None
             and not r["metadata"]["search_v1"].get("terminal")),
            key=lambda r: r["metadata"]["env_step"],
        )
        state = {"prior_queries": set(), "prior_doc_ids": set(), "prior_content_hashes": set()}
        for i, r in enumerate(searches):
            sv = r["metadata"]["search_v1"]
            retrieval = r["metadata"].get("retrieval") or {}
            doc_ids = retrieval.get("document_ids")
            # redundant_search: fully recomputable (query/status/doc_ids/prior
            # state are all in the audit). The content-hash fallback only
            # fires when doc_ids is None, which never happens in this pipeline.
            expected = search_step_components_v2(
                query=sv.get("query"),
                status=sv.get("status"),
                doc_ids=doc_ids,
                doc_text=None,
                gt_aliases=[],
                question=None,
                prior_queries=state["prior_queries"],
                prior_doc_ids=state["prior_doc_ids"],
                prior_content_hashes=state["prior_content_hashes"],
                is_first_search=(i == 0),
                had_evidence_credit=False,
            )
            n_search_steps += 1
            if expected["redundant_search"] != bool(sv.get("redundant_search")):
                mismatches.append({"traj_uid": traj_uid, "env_step": r["metadata"]["env_step"],
                                   "recomputed": expected["redundant_search"], "audit": sv.get("redundant_search"),
                                   "query": sv.get("query"), "status": sv.get("status"), "doc_ids": doc_ids})
            n_redundant += int(expected["redundant_search"])
            n_audit_leak += int(bool(sv.get("answer_leak")))
            state = expected["state"]
    ok = not mismatches
    return ok, {
        "n_search_steps_recomputed": n_search_steps,
        "n_v2_redundant_steps": n_redundant,
        "n_audit_leak_steps_not_offline_verifiable": n_audit_leak,
        "verdict_mismatches": mismatches[:5],
        "n_verdict_mismatches": len(mismatches),
        "limitation": "evidence_hit and answer_leak are NOT recomputable from the audit "
                      "(question/gt_aliases/observation text are not stored); their placement "
                      "is covered by S5 sum-consistency in exact cents; redundant_search is "
                      "fully recomputed and compared",
    }


def check_trajectory_advantage_broadcast(records: list[dict]) -> tuple[bool, dict]:
    by_uid: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_uid[r["metadata"]["traj_uid"]].append(r)
    # per-uid trajectory returns
    uid_returns: dict[str, list[float]] = defaultdict(list)
    uid_of_traj: dict[str, str] = {}
    for traj_uid, rs in by_uid.items():
        ep = rs[0]["metadata"].get("search_v1_episode")
        if ep is None:
            continue
        uid_returns[str(ep["uid"])].append(ep["total_reward_c"] / 100.0)
        uid_of_traj[traj_uid] = str(ep["uid"])
    stats: dict[str, tuple[float, float]] = {}
    for uid, returns in uid_returns.items():
        import numpy as np

        arr = np.array(returns)
        if len(arr) > 1:
            mean, std = float(arr.mean()), float(arr.std(ddof=1))
        else:
            mean, std = 0.0, 1.0
        stats[uid] = (mean, std)
    mismatches = []
    n_unverifiable = 0
    for traj_uid, rs in by_uid.items():
        ep = rs[0]["metadata"].get("search_v1_episode")
        if ep is None:
            n_unverifiable += 1
            continue
        ret = ep["total_reward_c"] / 100.0
        mean, std = stats[uid_of_traj[traj_uid]]
        expected = (ret - mean) / (std + 1e-6)
        # audit view (P3 v2, 2026-08-23): trajectory advantage over ACTIVE
        # policy tokens only -- the v2 advantage is broadcast to every active
        # token of every record of the trajectory, so the active-token max is
        # EXACTLY the trajectory advantage for every record (padded zeros no
        # longer mask negative values via max())
        values = {r["trajectory_advantage"] for r in rs if abs(r["trajectory_advantage"] or 0) > 1e-9}
        if len(values) > 1:
            mismatches.append({"traj_uid": traj_uid, "distinct_audit_values": sorted(values)[:5]})
            continue
        audit_adv = next(iter(values)) if values else 0.0
        if abs(audit_adv - expected) > 1e-5:
            mismatches.append({"traj_uid": traj_uid, "audit_adv": audit_adv, "expected_adv": expected,
                               "return": ret, "uid": uid_of_traj[traj_uid]})
    ok = not mismatches
    return ok, {
        "n_trajectories": len(by_uid),
        "n_unverifiable": n_unverifiable,
        "n_uid_groups": len(uid_returns),
        "adv_mismatches": mismatches[:5],
        "n_adv_mismatches": len(mismatches),
        "uid_group_sizes": dict(Counter(len(v) for v in uid_returns.values())),
    }


def check_useful_search_positive_adv(records: list[dict]) -> tuple[bool, dict]:
    by_uid: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_uid[r["metadata"]["traj_uid"]].append(r)
    n_useful = 0
    n_useful_positive = 0
    for traj_uid, rs in by_uid.items():
        ep = rs[0]["metadata"].get("search_v1_episode")
        if ep is None:
            continue
        useful = any(r["metadata"].get("search_v1", {}).get("evidence_credit") for r in rs) or \
            ep.get("searched_correct_bonus_c", 0) > 0
        if not useful:
            continue
        n_useful += 1
        values = {r["trajectory_advantage"] for r in rs if abs(r["trajectory_advantage"] or 0) > 1e-9}
        adv = next(iter(values)) if values else 0.0
        if adv > 0:
            n_useful_positive += 1
    ok = n_useful_positive >= 1
    return ok, {
        "n_useful_search_trajectories": n_useful,
        "n_useful_search_positive_adv": n_useful_positive,
    }


def check_observation_loss_mask(records: list[dict]) -> tuple[bool, dict]:
    issues = []
    n_records = len(records)
    n_prompt_loss_bad = 0
    n_response_mask_bad = 0
    n_mask_sum_bad = 0
    n_mask_source_unknown = 0
    for r in records:
        if r["prompt_policy_loss_tokens"] != 0:
            n_prompt_loss_bad += 1
        if r["policy_loss_tokens"] > r["active_response_tokens"]:
            n_response_mask_bad += 1
        recomputed = sum(r["policy_loss_mask"])
        if recomputed != r["policy_loss_tokens"]:
            n_mask_sum_bad += 1
        if r["mask_source"] not in ("loss_mask", "response_attention_mask"):
            n_mask_source_unknown += 1
    ok = (n_prompt_loss_bad == 0 and n_response_mask_bad == 0
          and n_mask_sum_bad == 0 and n_mask_source_unknown == 0)
    return ok, {
        "n_records": n_records,
        "n_prompt_policy_loss_nonzero": n_prompt_loss_bad,
        "n_response_mask_exceeding_active": n_response_mask_bad,
        "n_mask_sum_mismatch": n_mask_sum_bad,
        "n_mask_source_unknown": n_mask_source_unknown,
        "note": "observations live in the prompt region of each env-step record; "
                "the prompt region is fully excluded from the policy loss "
                "(prompt_policy_loss_tokens==0 asserted per record)",
    }


def check_logs(run_dir: Path) -> tuple[bool, dict]:
    matches = defaultdict(list)
    for name in ("stdout.log", "stderr.log", "cleanup.log"):
        log_file = run_dir / name
        if not log_file.exists():
            continue
        with log_file.open(encoding="utf-8", errors="replace") as handle:
            for lineno, line in enumerate(handle, 1):
                for kind, pattern in LOG_PATTERNS.items():
                    if pattern.search(line):
                        matches[kind].append({"file": name, "line": lineno, "text": line.strip()[:200]})
    ok = not any(matches.values())
    return ok, {kind: {"n": len(v), "samples": v[:3]} for kind, v in sorted(matches.items())}


def check_residue_and_gpu(run_dir: Path, gpus: list[str]) -> tuple[bool, dict]:
    # 1. leftover processes
    ps = subprocess.run(["ps", "-eo", "user=,pid=,cmd="], capture_output=True, text=True).stdout
    residue = [
        line.strip() for line in ps.splitlines()
        if re.search(r"ray::|vllm|main_ppo|ppo_trainer|verl\.trainer", line)
        and "grep" not in line and "check_p3_v2_smoke" not in line
    ]
    # 2. GPU baseline (two samples, 1s apart)
    gpu_report = {}
    baseline_ok = True
    for sample in (1, 2):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        ).stdout
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue
            index, used, util = parts[0], int(parts[1]), int(parts[2])
            if index in gpus:
                gpu_report.setdefault(index, []).append({"mem_mib": used, "util_pct": util})
    for index, samples in gpu_report.items():
        mem = max(s["mem_mib"] for s in samples)
        util = max(s["util_pct"] for s in samples)
        if mem >= 1024 or util >= 5:
            baseline_ok = False
    ok = not residue and baseline_ok
    return ok, {
        "leftover_processes": residue[:5],
        "n_leftover_processes": len(residue),
        "gpu_baseline_mem_util": gpu_report,
        "gpu_baseline_ok": baseline_ok,
        "threshold": "mem < 1024 MiB and util < 5% per GPU (two samples)",
    }


def check_memory_report(run_dir: Path) -> dict:
    report = {"nvidia_smi_physical_peaks": None, "torch_allocator_view": None}
    peak_file = run_dir / "peak_memory_nvidia_smi.json"
    if peak_file.exists():
        report["nvidia_smi_physical_peaks"] = json.loads(peak_file.read_text())
    # verl log torch view: worker-aggregated max_memory_reserved_gb (last value)
    for name in ("stdout.log", "stderr.log"):
        log_file = run_dir / name
        if not log_file.exists():
            continue
        values = []
        with log_file.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                m = re.search(r"max_memory_reserved_gb:([0-9.]+)", line)
                if m:
                    values.append(float(m.group(1)))
        if values:
            report["torch_allocator_view"] = {
                "label": "worker-aggregated torch-allocator view (NOT a physical per-GPU peak)",
                "last_value_gb": values[-1],
                "max_value_gb": max(values),
                "n_observations": len(values),
            }
            break
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--gpus", type=str, default="1,2,3,4,6,7")
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()

    gpus = args.gpus.split(",")
    records = load_records(args.run_dir, args.steps)

    items = {}
    items["S1_real_optimizer_and_global_step"] = check_optimizer_and_global_step(args.run_dir, args.steps)
    items["S2_checkpoint_complete"] = check_checkpoint_complete(args.run_dir, args.steps)
    items["S3_valid_search_trajectory"] = check_valid_search_trajectory(records)
    items["S4_multi_search_different_query_new_doc"] = check_multi_search_new_doc(records)
    items["S5_v2_reward_components_landed"] = check_v2_components(records)
    items["S6_true_redundant_only_on_duplicate_or_no_new_doc"] = check_true_redundant(records)
    items["S7_trajectory_advantage_broadcast"] = check_trajectory_advantage_broadcast(records)
    items["S8_useful_search_positive_adv"] = check_useful_search_positive_adv(records)
    items["S9_observation_loss_mask"] = check_observation_loss_mask(records)
    items["S10_no_oom_nan_xid_dropped_gpu_retriever_timeout"] = check_logs(args.run_dir)
    items["S11_no_residue_gpu_baseline"] = check_residue_and_gpu(args.run_dir, gpus)

    report = {
        "run_id": args.run_id,
        "run_dir": str(args.run_dir),
        "gpus": gpus,
        "steps": args.steps,
        "n_audit_records": len(records),
        "items": {k: {"pass": v[0], "detail": v[1]} for k, v in items.items()},
        "memory_report_sec8": check_memory_report(args.run_dir),
    }
    all_pass = all(v[0] for v in items.values())
    report["all_items_pass"] = all_pass

    out_path = args.run_dir / "check_p3_v2_smoke.json"
    partial = out_path.with_suffix(".json.partial")
    with partial.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(out_path)

    print(json.dumps({k: {"pass": v[0], **v[1]} for k, v in items.items()},
                     ensure_ascii=False, indent=2, default=str))
    print(f"[CHECK_SMOKE] report: {out_path}")
    if not all_pass:
        print("[CHECK_SMOKE] HARD GATE FAILURE", file=sys.stderr)
        return 1
    print("[CHECK_SMOKE] all 11 items PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
