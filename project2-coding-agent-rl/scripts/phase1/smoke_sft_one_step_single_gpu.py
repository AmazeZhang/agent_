#!/usr/bin/env python3
"""One real SFT optimizer step on an actual Phase 1 trajectory, without NCCL.

This is a host-fault fallback smoke only. It validates the real multiturn mask,
fused Actor path, LoRA gradients/parameter update, and PEFT adapter save
surface. It does not replace the planned multi-GPU ZeRO-3 training.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from openrlhf.models import Actor


LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_multiturn_sample(tokenizer, messages):
    response_ranges = []
    for idx, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prompt = tokenizer.apply_chat_template(messages[:idx], tokenize=False, add_generation_prompt=True)
        response = tokenizer.apply_chat_template(messages[: idx + 1], tokenize=False)[len(prompt) :]
        start = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        end = start + len(tokenizer(response, add_special_tokens=False)["input_ids"]) - 1
        response_ranges.append((start, end))

    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    response = tokenizer.apply_chat_template(messages, tokenize=False)[len(prompt) :]
    text = (prompt + response).rstrip("\n")
    if not text.endswith(tokenizer.eos_token):
        text += " " + tokenizer.eos_token
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    input_ids[0, -1] = tokenizer.eos_token_id

    loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)
    for start, end in response_ranges:
        loss_mask[0, start - 1 : end] = 1
    return input_ids, attention_mask, loss_mask, response_ranges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-index", type=int, default=230)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument(
        "--model",
        default="/media/imc/data/yzy/agent/project2/phase1/models/Qwen2.5-Coder-7B-Instruct",
    )
    parser.add_argument(
        "--data",
        default="/media/imc/data/yzy/agent/project2/phase1/sft_data/sft_train_24k.jsonl",
    )
    parser.add_argument(
        "--output",
        default="/media/imc/data/yzy/agent/project2/phase1/checkpoints/sft-single-step-smoke-20260810",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    with open(args.data) as handle:
        rows = [json.loads(line) for line in handle]
    messages = rows[args.sample_index]["input"]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    input_ids, attention_mask, loss_mask, response_ranges = build_multiturn_sample(tokenizer, messages)
    shifted_mask = loss_mask[:, :-1].bool()
    if not shifted_mask.any():
        raise RuntimeError("actual trajectory produced an empty assistant-token loss mask")

    torch.manual_seed(23)
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
    actor.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    actor.train()
    assert not actor.model.lm_head.weight.requires_grad
    trainable = [param for param in actor.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0)

    tracked_name, tracked_param = next(
        (name, param) for name, param in actor.named_parameters() if "lora_B" in name and param.requires_grad
    )
    tracked_before = tracked_param.detach().float().cpu().clone()
    input_ids = input_ids.cuda()
    attention_mask = attention_mask.cuda()
    shifted_mask = shifted_mask.cuda()
    os.environ["OPENRLHF_FUSED_CE"] = "1"
    os.environ.setdefault("OPENRLHF_FUSED_CE_CHUNK_SIZE", "512")
    torch.cuda.reset_peak_memory_stats()

    logps, _ = actor(
        input_ids,
        attention_mask=attention_mask,
        return_output=True,
        return_logprobs=True,
    )
    loss = -(logps * shifted_mask).sum() / shifted_mask.sum()
    loss.backward()
    grad_sq = sum(
        param.grad.detach().float().pow(2).sum().item()
        for param in trainable
        if param.grad is not None
    )
    grad_norm = grad_sq**0.5
    if not grad_norm > 0:
        raise RuntimeError(f"non-positive LoRA grad norm: {grad_norm}")
    optimizer.step()
    delta = (tracked_param.detach().float().cpu() - tracked_before).abs().max().item()
    if not delta > 0:
        raise RuntimeError(f"tracked LoRA parameter did not update: {tracked_name}")
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    fused_calls = getattr(actor, "_fused_ce_calls", 0)
    if fused_calls != 1:
        raise RuntimeError(f"expected one fused Actor call, observed {fused_calls}")

    actor.model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    result = {
        "status": "pass",
        "scope": "single_gpu_single_optimizer_step_not_zero3_training",
        "physical_gpu": int(os.environ.get("PHYSICAL_GPU_ID", "-1")),
        "sample_index": args.sample_index,
        "sequence_tokens": int(input_ids.shape[1]),
        "assistant_loss_tokens": int(shifted_mask.sum().item()),
        "assistant_ranges": response_ranges,
        "loss": float(loss.detach().float().item()),
        "grad_norm": grad_norm,
        "tracked_parameter": tracked_name,
        "tracked_max_abs_delta": delta,
        "peak_memory_gib": peak_gib,
        "fused_calls": fused_calls,
    }
    with open(output / "smoke_result.json", "w") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print("SFT_SINGLE_STEP_SMOKE_PASS " + json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
