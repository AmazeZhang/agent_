#!/usr/bin/env python3
"""Build the fixed 64-question behaviour-diagnosis set for P3 (dev64).

Samples 64 questions from the upstream Search-R1 test split using the same
per-source-quota / ascending-SHA256 selection as build_p3_heldout_eval.py, but
under the NEW selection domain "searchr1-p3-dev64-v1" so the draw differs from
every earlier one, and with the FULL exclusion closure demanded by the
Search-aware clean v2 directive (2026-08-22):

  excluded = upstream train  ∪  smoke train  ∪  smoke test
           ∪ confirm256  ∪  official-confirm256-v1
           ∪ final-confirm512  ∪  heldout32

The set is used ONLY for behaviour diagnosis (Step5 sampling rollouts,
temperature=1, 5 rollouts/question, no quality claims) per the v2 eval plan.

Quotas scale ×2 from the base 32: nq 16 / hotpotqa 16 / popqa 8 /
2wikimultihopqa 8 / triviaqa 8 / musique 4 / bamboogle 4 = 64.

Output (atomic writes, deterministic rebuild => byte-identical):
  <output-dir>/dev64.parquet   (schema identical to the smoke test parquet)
  <output-dir>/records.jsonl
  <output-dir>/manifest.json   (outputs.dev64.sha256 for run_p3_eval_v2.py)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_p3_heldout_eval import (  # noqa: E402
    HELDOUT_QUOTAS,
    UPSTREAM_REPO,
    UPSTREAM_REVISION,
    normalize_question,
    normalized_question_set,
    record_id,
    select_rows,
    sha256_file,
    to_builtin,
    write_atomic,
)

SELECTION_DOMAIN = "searchr1-p3-dev64-v1"
DATA_ROOT = Path("/media/imc/data/project3-search-agent-rl/datasets")

INPUTS = {
    "test": DATA_ROOT / "searchr1-upstream" / "test.parquet",
    "train": DATA_ROOT / "searchr1-upstream" / "train.parquet",
    "smoke_train": DATA_ROOT / "searchr1-smoke" / "train.parquet",
    "smoke_test": DATA_ROOT / "searchr1-smoke" / "test.parquet",
    "confirm256": DATA_ROOT / "searchr1-confirm256" / "heldout.parquet",
    "official_confirm256_v1": DATA_ROOT / "searchr1-official-confirm256-v1" / "heldout.parquet",
    "final_confirm512": DATA_ROOT / "searchr1-final-confirm512" / "heldout.parquet",
    "heldout32": DATA_ROOT / "searchr1-heldout32" / "heldout.parquet",
}
EXTRA_EXCLUSION_KEYS = ("confirm256", "official_confirm256_v1", "final_confirm512", "heldout32")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DATA_ROOT / "searchr1-p3-dev64-v1")
    args = parser.parse_args()

    for key, path in INPUTS.items():
        if not path.is_file():
            raise SystemExit(f"input parquet missing ({key}): {path}")

    import pandas as pd

    test_frame = pd.read_parquet(INPUTS["test"])
    train_frame = pd.read_parquet(INPUTS["train"])
    smoke_train = pd.read_parquet(INPUTS["smoke_train"])
    smoke_test = pd.read_parquet(INPUTS["smoke_test"])
    extra_frames = {key: pd.read_parquet(INPUTS[key]) for key in EXTRA_EXCLUSION_KEYS}

    upstream_train_questions = normalized_question_set(train_frame)
    smoke_train_questions = normalized_question_set(smoke_train)
    smoke_test_questions = normalized_question_set(smoke_test)
    extra_questions: set[str] = set()
    for frame in extra_frames.values():
        extra_questions |= normalized_question_set(frame)

    banned = upstream_train_questions | smoke_train_questions
    pool_exclusions = banned | smoke_test_questions | extra_questions

    total = 64
    scale = total // 32
    quotas = {source: quota * scale for source, quota in HELDOUT_QUOTAS.items()}
    assert sum(quotas.values()) == total

    dev64 = select_rows(test_frame, quotas, pool_exclusions, SELECTION_DOMAIN)
    dev64_questions = normalized_question_set(dev64)
    if not dev64_questions.isdisjoint(smoke_test_questions):
        raise RuntimeError("dev64 questions overlap smoke test questions")
    if not dev64_questions.isdisjoint(extra_questions):
        raise RuntimeError("dev64 questions overlap earlier eval-set questions")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_parquet = output_dir / "dev64.parquet"
    if output_parquet.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output_parquet}")
    dev64.to_parquet(output_parquet, index=False)

    records = []
    for _, row in dev64.iterrows():
        env_kwargs = to_builtin(row["env_kwargs"])
        records.append(
            {
                "id": record_id(row),
                "split": "dev64",
                "source": str(row["data_source"]),
                "original_index": int(row["extra_info"]["index"]),
                "question": env_kwargs["question"],
                "answers": env_kwargs["ground_truth"]["target"],
            }
        )
    records_path = output_dir / "records.jsonl"
    write_atomic(
        records_path,
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in records),
    )

    root_revision = subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
        text=True,
    ).strip()

    manifest = {
        "schema_version": 1,
        "purpose": (
            f"fixed behaviour-diagnosis set ({total} rows, domain={SELECTION_DOMAIN}); "
            "USED ONLY for decoding-behaviour diagnosis (5 sampling rollouts/question, "
            "temperature=1) per the Search-aware clean v2 directive 2026-08-22; "
            "never used for quality claims"
        ),
        "claim_boundary": "diagnostic-only; not an evaluation set, not preregistered evidence",
        "selection_domain": SELECTION_DOMAIN,
        "upstream_dataset": {
            "repo_id": UPSTREAM_REPO,
            "revision": UPSTREAM_REVISION,
            "license": "not declared in the Hugging Face dataset metadata; manual verification required",
        },
        "code_revisions": {"root": root_revision},
        "selection_algorithm": (
            f"per-source quota, ascending SHA256({SELECTION_DOMAIN}\\0source\\0normalized_question), "
            "excluding upstream-train, smoke-train, smoke-test, confirm256, "
            "official-confirm256-v1, final-confirm512 and heldout32 normalized questions"
        ),
        "selection_is_order_independent": True,
        "quotas": quotas,
        "leakage": {
            "upstream_train_normalized_overlap": len(dev64_questions & upstream_train_questions),
            "smoke_train_normalized_overlap": len(dev64_questions & smoke_train_questions),
            "smoke_test_normalized_overlap": len(dev64_questions & smoke_test_questions),
            "extra_exclusion_normalized_overlap": len(dev64_questions & extra_questions),
        },
        "inputs": {
            "test": {"path": str(INPUTS["test"].resolve()), "sha256": sha256_file(INPUTS["test"]), "rows": len(test_frame)},
            "train": {"path": str(INPUTS["train"].resolve()), "sha256": sha256_file(INPUTS["train"]), "rows": len(train_frame)},
            "smoke_train": {"path": str(INPUTS["smoke_train"].resolve()), "sha256": sha256_file(INPUTS["smoke_train"]), "rows": len(smoke_train)},
            "smoke_test": {"path": str(INPUTS["smoke_test"].resolve()), "sha256": sha256_file(INPUTS["smoke_test"]), "rows": len(smoke_test)},
            "extra_exclusions": [
                {"key": key, "path": str(path.resolve()), "sha256": sha256_file(path), "rows": len(frame)}
                for key, (path, frame) in ((key, (INPUTS[key], extra_frames[key])) for key in EXTRA_EXCLUSION_KEYS)
            ],
        },
        "outputs": {
            "dev64": {"path": str(output_parquet.resolve()), "sha256": sha256_file(output_parquet), "rows": len(dev64)},
            "records": {"path": str(records_path.resolve()), "sha256": sha256_file(records_path), "rows": len(records)},
        },
        "records": records,
    }
    manifest_path = output_dir / "manifest.json"
    write_atomic(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(f"dev64: {len(dev64)} rows -> {output_parquet}")
    print(f"manifest: {manifest_path}")
    print(f"leakage: {manifest['leakage']}")


if __name__ == "__main__":
    main()
