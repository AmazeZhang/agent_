#!/usr/bin/env python3
"""Phase 1a: build SFT data from SWE-smith trajectories for train-pool tasks.

For each train-pool instance with resolved trajectories:
  1. parse messages (SWE-agent text style: reasoning + ```fence commands)
  2. normalize to our phase0 protocol:
     - system prompt -> phase0 SYSTEM_TEMPLATE (submission semantics identical)
     - assistant ```fence commands -> <execute><command>...</command></execute>
       (our swe_command parser's primary pattern); pure reasoning text kept
     - submit/`| submit |` -> exit (our submission action, intercepted by phase0)
     - user observations kept as-is
  3. write OpenRLHF multiturn JSONL: {"input": [{"role","content"}, ...]}
"""
import argparse
import json
import os
import re

import pyarrow.parquet as pq

TRAJ_DIR = "/media/imc/data/yzy/agent/project2/datasets/swe-smith-trajectories"
POOL = "/media/imc/data/yzy/agent/project2/phase1/task_pool/train_pool.jsonl"
OUT = "/media/imc/data/yzy/agent/project2/phase1/sft_data/sft_train.jsonl"
PHASE0_PY = "/home/imc/yzy/agent/project2-coding-agent-rl/scripts/phase0/phase0.py"

FENCE_RE = re.compile(r"```(.*?)```", re.S)
SUBMIT_RE = re.compile(r"submit", re.I)


def load_phase0_system() -> str:
    src = open(PHASE0_PY).read()
    # plain string search (this env's re module misbehaves on backslash+LF)
    start = src.find('SYSTEM_TEMPLATE = """\\\n')
    if start == -1:
        raise RuntimeError("cannot extract SYSTEM_TEMPLATE from phase0.py")
    end = src.find('"""', start + 25)
    if end == -1:
        raise RuntimeError("cannot find SYSTEM_TEMPLATE end")
    return src[start + 25:end].rstrip()


def normalize_assistant(content: str, is_last: bool) -> str:
    """SWE-agent text+fence -> reasoning + <execute><command> blocks."""
    c = content.strip()
    if not c:
        return ""
    if SUBMIT_RE.search(c):
        return "exit"
    out = []
    pos = 0
    for m in FENCE_RE.finditer(c):
        pre = c[pos:m.start()].strip()
        if pre:
            out.append(pre)
        code = m.group(1).strip()
        if code.startswith(("bash", "sh", "python")):
            code = code.split("\n", 1)[1].strip() if "\n" in code else ""
        if code:
            out.append(f"<execute>\n<command>{code}</command>\n</execute>")
        pos = m.end()
    tail = c[pos:].strip()
    if tail:
        out.append(tail)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=POOL)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--traj-dir", default=TRAJ_DIR)
    args = ap.parse_args()

    system_prompt = load_phase0_system()
    pool = {json.loads(l)["instance_id"] for l in open(args.pool)}
    print(f"train pool instances: {len(pool)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_kept = n_turns_total = n_dropped = 0
    n_exit = 0
    seen_instances = set()
    files = sorted(f for f in os.listdir(args.traj_dir) if f.endswith(".parquet"))
    with open(args.out, "w") as fout:
        for fn in files:
            t = pq.read_table(os.path.join(args.traj_dir, fn))
            for inst, res, msgs in zip(t["instance_id"], t["resolved"], t["messages"]):
                inst = str(inst)
                if inst not in pool or not bool(res):
                    continue
                try:
                    raw = json.loads(str(msgs))
                except json.JSONDecodeError:
                    n_dropped += 1
                    continue
                # skip trivially short / malformed trajectories
                if len(raw) < 6:
                    n_dropped += 1
                    continue
                msgs_out = []
                for i, m in enumerate(raw):
                    role = m.get("role")
                    content = m.get("content", "")
                    if role == "system":
                        msgs_out.append({"role": "system", "content": system_prompt})
                    elif role == "assistant":
                        norm = normalize_assistant(content, i == len(raw) - 1)
                        if not norm:
                            continue
                        if norm == "exit":
                            n_exit += 1
                        msgs_out.append({"role": "assistant", "content": norm})
                    else:
                        if content.strip():
                            msgs_out.append({"role": role, "content": content})
                if len(msgs_out) < 4:
                    n_dropped += 1
                    continue
                fout.write(json.dumps({"input": msgs_out}, ensure_ascii=False) + "\n")
                n_kept += 1
                n_turns_total += len(msgs_out)
                seen_instances.add(inst)
    print(f"kept trajectories: {n_kept} (instances: {len(seen_instances)}), "
          f"avg msgs: {n_turns_total/max(n_kept,1):.1f}, exit-submissions: {n_exit}, "
          f"dropped: {n_dropped}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
