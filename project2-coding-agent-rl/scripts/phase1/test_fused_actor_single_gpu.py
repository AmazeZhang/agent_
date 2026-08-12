#!/usr/bin/env python3
"""Full Qwen2.5-7B Actor reference/fused comparison without NCCL/ZeRO."""

import argparse
import os

import torch

from openrlhf.models import Actor


LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def collect_lora_grads(actor):
    return {
        name: param.grad.detach().float().cpu().clone()
        for name, param in actor.named_parameters()
        if "lora_" in name and param.grad is not None
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqlen", type=int, default=32)
    parser.add_argument(
        "--model",
        default="/media/imc/data/yzy/agent/project2/phase1/models/Qwen2.5-Coder-7B-Instruct",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    torch.manual_seed(11)
    actor = Actor(
        args.model,
        attn_implementation="flash_attention_2",
        param_dtype="bf16",
        lora_rank=16,
        lora_alpha=32,
        target_modules=LORA_TARGETS,
        lora_dropout=0,
        device_map={"": torch.cuda.current_device()},
        packing_samples=False,
        use_liger_kernel=False,
    )
    actor.train()
    assert not actor.model.lm_head.weight.requires_grad, "lm_head must remain frozen for this fused path"

    ids = torch.randint(100, 50000, (1, args.seqlen), device="cuda")
    attention_mask = torch.ones_like(ids)
    torch.cuda.reset_peak_memory_stats()

    os.environ.pop("OPENRLHF_FUSED_CE", None)
    ref_logps, ref_output = actor(
        ids, attention_mask=attention_mask, return_output=True, return_logprobs=True
    )
    ref_loss = -ref_logps.mean()
    ref_loss.backward()
    ref_grads = collect_lora_grads(actor)
    ref_logps_cpu = ref_logps.detach().float().cpu().clone()
    ref_loss_value = ref_loss.detach().float().item()
    ref_peak = torch.cuda.max_memory_allocated() / 2**30
    del ref_logps, ref_output, ref_loss
    actor.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    os.environ["OPENRLHF_FUSED_CE"] = "1"
    os.environ.setdefault("OPENRLHF_FUSED_CE_CHUNK_SIZE", "512")
    calls_before = getattr(actor, "_fused_ce_calls", 0)
    fused_logps, fused_output = actor(
        ids, attention_mask=attention_mask, return_output=True, return_logprobs=True
    )
    fused_loss = -fused_logps.mean()
    fused_loss.backward()
    fused_grads = collect_lora_grads(actor)
    fused_logps_cpu = fused_logps.detach().float().cpu().clone()
    fused_loss_value = fused_loss.detach().float().item()
    fused_peak = torch.cuda.max_memory_allocated() / 2**30
    calls_after = getattr(actor, "_fused_ce_calls", 0)

    assert calls_after == calls_before + 1, "fused Actor branch was not executed"
    assert ref_grads.keys() == fused_grads.keys() and ref_grads, "LoRA gradient sets differ or are empty"
    loss_abs = abs(ref_loss_value - fused_loss_value)
    assert loss_abs < 5e-3, f"loss mismatch: {loss_abs:.3e}"
    logps_max_abs = (ref_logps_cpu - fused_logps_cpu).abs().max().item()
    assert logps_max_abs < 5e-2, f"per-token log-prob mismatch: {logps_max_abs:.3e}"

    worst = 0.0
    worst_name = None
    ref_flat = []
    fused_flat = []
    for name in ref_grads:
        ref = ref_grads[name]
        got = fused_grads[name]
        ref_flat.append(ref.reshape(-1))
        fused_flat.append(got.reshape(-1))
        scale = max(ref.abs().max().item(), 1e-8)
        diff = (ref - got).abs().max().item() / scale
        if diff > worst:
            worst = diff
            worst_name = name
    ref_flat = torch.cat(ref_flat)
    fused_flat = torch.cat(fused_flat)
    ref_flat64 = ref_flat.double()
    fused_flat64 = fused_flat.double()
    grad_rel_l2 = (ref_flat64 - fused_flat64).norm().item() / max(ref_flat64.norm().item(), 1e-8)
    grad_cosine = torch.nn.functional.cosine_similarity(ref_flat64, fused_flat64, dim=0).item()
    assert grad_rel_l2 < 3e-2, f"LoRA grad relative L2 mismatch: {grad_rel_l2:.3e}"
    assert grad_cosine > 0.999, f"LoRA grad cosine mismatch: {grad_cosine:.6f}"
    print(
        "FUSED_ACTOR_SINGLE_GPU_PASS "
        f"seqlen={args.seqlen} loss_abs={loss_abs:.3e} logps_max_abs={logps_max_abs:.3e} "
        f"grad_rel_l2={grad_rel_l2:.3e} grad_cosine={grad_cosine:.6f} "
        f"grad_max_rel={worst:.3e}@{worst_name} "
        f"ref_peak_gib={ref_peak:.2f} fused_peak_gib={fused_peak:.2f}"
    )


if __name__ == "__main__":
    main()
