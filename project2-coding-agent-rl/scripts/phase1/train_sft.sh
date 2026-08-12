#!/bin/bash
# Phase 1b: SFT training with OpenRLHF 0.10.4 (hierarchical args).
# Qwen2.5-Coder-7B-Instruct, LoRA r16, ZeRO-3, GPU 2/4/6/7.
# Run inside tmux session p2-phase1-sft.
#
# max_len=24576 (not 32768): loss-backward materializes dlogits=(seq, vocab),
# vocab=151936 → 8.67 GiB at 30.6K seq OOMs 24GB cards (flash_attn triton CE).
# Data pre-filtered (sft_train_24k.jsonl, 238 samples ≤ 24320 tok) so the
# dataset's head-truncation never fires and final fix patches stay intact.
set -euo pipefail

[ -n "${TMUX:-}" ] || {
  echo "REFUSED: GPU training must run inside tmux (session p2-phase1-sft)." >&2
  exit 2
}

export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890

BASE=/media/imc/data/yzy/agent/project2
MODEL=$BASE/phase1/models/Qwen2.5-Coder-7B-Instruct
DATA=$BASE/phase1/sft_data/sft_train_24k.jsonl
OUT=$BASE/phase1/checkpoints/sft-7b
VENV=/home/imc/yzy/agent/project2-coding-agent-rl/.venvs/phase1-openrlhf
ROOT=/home/imc/yzy/agent
export PATH="$VENV/bin:$PATH"   # deepspeed JIT (fused_adam) needs ninja on PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPENRLHF_FUSED_CE=1
export OPENRLHF_FUSED_CE_CHUNK_SIZE=512

# Mandatory physical-GPU preflight. GPU 0 is rejected by check_gpu.sh.
for gpu_id in 2 4 6 7; do
  bash "$ROOT/shared/scripts/check_gpu.sh" "$gpu_id"
done

bash "$ROOT/project2-coding-agent-rl/scripts/phase1/install_openrlhf_fused_ce.sh" "$VENV"

# Fail closed: an ignored/missing fused patch would otherwise fall back to the
# full logits path and OOM while looking like a valid experiment.
$VENV/bin/python -c \
  "from openrlhf.models.actor import Actor; import inspect; assert 'OPENRLHF_FUSED_CE_ACTIVE' in inspect.getsource(Actor.forward)"
$VENV/bin/python -c \
  "from openrlhf.models.fused_ce import _chunk_size; assert _chunk_size() == 512"

# Verify both the explicit physical mapping and a real NCCL collective before
# spending time and memory on model loading. Any unhealthy GPU/NVML state must
# fail the run here.
$VENV/bin/deepspeed --include localhost:2,4,6,7 \
  "$ROOT/project2-coding-agent-rl/scripts/phase1/nccl_preflight.py"

mkdir -p "$OUT"

$VENV/bin/deepspeed --include localhost:2,4,6,7 --module openrlhf.cli.train_sft \
  --model.model_name_or_path "$MODEL" \
  --data.dataset "$DATA" \
  --data.dataset_probs 1.0 \
  --data.max_len 24576 \
  --data.multiturn \
  --data.apply_chat_template \
  --data.max_samples 1000000 \
  --ckpt.output_dir "$OUT" \
  --ckpt.save_steps 50 \
  --train.max_epochs 3 \
  --train.batch_size 4 \
  --train.micro_batch_size 1 \
  --adam.lr 5e-5 \
  --ds.zero_stage 3 \
  --ds.param_dtype bf16 \
  --ds.attn_implementation flash_attention_2 \
  --ds.lora.rank 16 \
  --ds.lora.alpha 32 \
  --ds.lora.target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --model.gradient_checkpointing_enable \
  --logger.logging_steps 5 \
  --eval.steps 200 2>&1 | tee "$BASE/phase1/stats/sft_train.log"
