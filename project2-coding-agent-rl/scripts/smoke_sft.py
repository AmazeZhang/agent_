"""WP5: real minimal LoRA SFT run on the rllm verl backend (GPU 1-7 only).

Run from tmux with CUDA_VISIBLE_DEVICES set to a free GPU (never 0):
    CUDA_VISIBLE_DEVICES=1 python scripts/smoke_sft.py

Checks: data -> parquet -> SFTSpec(model, lora_rank=8) -> AgentSFTTrainer(verl)
-> checkpoint written under /media/imc/data/yzy/agent/project2/checkpoints/.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, "/home/imc/yzy/agent/project2-coding-agent-rl/vendor/rllm")

PROJECT = Path("/media/imc/data/yzy/agent/project2")
MODEL = PROJECT / "hf-cache/hub/models--Qwen--Qwen2.5-Coder-3B-Instruct/snapshots/488639f1ff808d1d3d0ba301aef8c11461451ec5"
# Env overrides (SFT#2, diff-format gold-patch data for the GRPO policy init):
#   P2_SFT_PARQUET=<path>   P2_SFT_OUT=<ckpt dir name>   P2_SFT_NAME=<wandb project>
TRAIN_PARQUET = Path(os.environ.get("P2_SFT_PARQUET", str(PROJECT / "smoke-data/sft-train.parquet")))
SFT_OUT = os.environ.get("P2_SFT_OUT", "smoke-sft")
SFT_NAME = os.environ.get("P2_SFT_NAME", "p2-smoke-sft")


def main() -> None:
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if "0" in gpu.split(","):
        raise SystemExit("REFUSED: GPU 0 is disabled by project policy")
    if not gpu:
        raise SystemExit("set CUDA_VISIBLE_DEVICES to a free GPU (1-7)")

    if not TRAIN_PARQUET.exists():
        raise SystemExit(f"missing smoke data: {TRAIN_PARQUET} (run build_smoke_sft_data.py)")

    from rllm.data import Dataset
    from rllm.trainer.agent_sft_trainer import AgentSFTTrainer
    from rllm.trainer.sft.spec import SFTSpec

    # load parquet rows (JSON-string messages column) into a Dataset
    import pyarrow.parquet as pq

    table = pq.read_table(TRAIN_PARQUET)
    rows = [{"messages": json.loads(m)} for m in table.column("messages").to_pylist()]
    ds = Dataset(rows, name="smoke-train")
    print(f"[smoke_sft] {len(ds)} rows, model={MODEL}, gpu={gpu}")

    spec = SFTSpec(
        model=str(MODEL),
        train_dataset=ds,
        lora_rank=8,
        epochs=1,
        batch_size=4,
        max_length=2048,
        tokenize_method="cumulative",
        save_freq=2,
        val_freq=-1,
        project=SFT_NAME,
        output_dir=str(PROJECT / "checkpoints" / SFT_OUT),
        overrides={"trainer": {"n_gpus_per_node": 1, "logger": ["console"]}},
    )
    AgentSFTTrainer(spec, backend="verl").train()
    print("[smoke_sft] DONE")


if __name__ == "__main__":
    import json  # noqa: E402

    main()
