#!/usr/bin/env python3
"""Minimal HF-vs-vLLM generation divergence diagnostic (GPU, managed run).

Same 8 prompts, generated greedily by HF transformers (do_sample=False) and
by the vLLM V0 engine (temperature=0) in the same process, no environment.
Reports, per prompt: first-divergence token index, token-level agreement
before divergence, and both decoded texts. The point is to localize the
backend difference to the LLM generation layer (as opposed to env /
retriever / data gates).

Managed run only (run_managed.sh), physical GPU != 0.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# --------------------------------------------------------------------------- #
# Gates (same semantics as run_p3_eval_vllm.py)
# --------------------------------------------------------------------------- #

def validate_managed_environment() -> None:
    run_id = os.environ.get("PROJECT3_RUN_ID", "")
    run_dir = os.environ.get("PROJECT3_RUN_DIR", "")
    if not run_id or not run_dir:
        raise RuntimeError("diagnostic must run under scripts/run_managed.sh")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("diagnostic must expose exactly one logical GPU")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible and any(token.strip() == "0" for token in visible.split(",")):
        raise RuntimeError("physical GPU 0 must never be exposed")
    if os.environ.get("VLLM_USE_V1", "0") != "0":
        raise RuntimeError("VLLM_USE_V1 must be \"0\" (training rollout engine path)")


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #

def hf_generate(tokenizer, model, chats: list[str], max_new_tokens: int) -> list[str]:
    inputs = tokenizer(
        chats, return_tensors="pt", padding=True, truncation=True, max_length=2048
    ).to(model.device)
    input_width = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        sequences = model.generate(
            **inputs,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    return tokenizer.batch_decode(sequences[:, input_width:], skip_special_tokens=True)


def vllm_generate(llm, tokenizer, chats: list[str], max_new_tokens: int) -> list[str]:
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    inputs = tokenizer(
        chats, return_tensors=None, padding=False, truncation=True, max_length=2048
    )
    sampling = SamplingParams(
        temperature=0.0, top_p=1.0, top_k=-1, max_tokens=max_new_tokens, ignore_eos=False
    )
    outputs = llm.generate(prompt_token_ids=inputs["input_ids"], sampling_params=sampling, use_tqdm=False)
    return [o.outputs[0].text for o in outputs]


def first_divergence(a: str, b: str) -> int:
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return i


def main() -> int:
    validate_managed_environment()
    data_dir = Path("/media/imc/data/project3-search-agent-rl")
    model_path = data_dir / "models/Qwen2.5-1.5B-Instruct"

    import pandas as pd

    frame = pd.read_parquet(data_dir / "datasets/searchr1-heldout32/heldout.parquet")
    questions = [str(row["env_kwargs"]["question"]) for _, row in frame.head(8).iterrows()]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    chats = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False
        )
        for q in questions
    ]

    started = time.monotonic()

    # HF side
    torch.backends.cuda.matmul.allow_tf32 = False
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).eval().to("cuda:0")
    hf_texts = hf_generate(tokenizer, hf_model, chats, 256)
    del hf_model
    torch.cuda.empty_cache()

    # vLLM side (same engine configuration as training rollout)
    import vllm

    llm = vllm.LLM(
        model=str(model_path),
        tokenizer=str(model_path),
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.6,
        enforce_eager=True,
        max_model_len=2304,
        seed=0,
        trust_remote_code=False,
    )
    vllm_texts = vllm_generate(llm, tokenizer, chats, 256)
    try:
        del llm
    except Exception:
        pass

    elapsed = time.monotonic() - started

    records = []
    for question, hf_text, vllm_text in zip(questions, hf_texts, vllm_texts):
        div = first_divergence(hf_text, vllm_text)
        agree = min(len(hf_text), len(vllm_text))
        records.append(
            {
                "question": question,
                "first_divergence_char": div,
                "agreement_chars_before_divergence": agree if div == agree else div,
                "hf_head": hf_text[:120],
                "vllm_head": vllm_text[:120],
            }
        )

    result = {
        "schema_version": 1,
        "kind": "p3-hf-vs-vllm-generation-diagnostic",
        "n_prompts": len(questions),
        "elapsed_seconds": elapsed,
        "backend_note": "HF: transformers generate(do_sample=False), tf32 off, eager attention. "
        "vLLM: V0 engine (VLLM_USE_V1=0), bfloat16, FlashAttention, gpu_memory_utilization 0.6, "
        "enforce_eager, max_model_len 2304, temperature 0 (same as training rollout).",
        "records": records,
    }
    run_dir = Path(os.environ["PROJECT3_RUN_DIR"]).resolve()
    out = run_dir / "diag.json"
    partial = out.with_name(out.name + ".partial")
    partial.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    partial.replace(out)

    identical = sum(
        1
        for r, a, b in zip(records, hf_texts, vllm_texts)
        if r["first_divergence_char"] == min(len(a), len(b))
    )
    diverged_early = sum(1 for r in records if r["first_divergence_char"] < 40)
    print(f"n={len(records)} diverged_within_first_40_chars={diverged_early}")
    for r in records:
        print(f"div@{r['first_divergence_char']:4d}  HF: {r['hf_head'][:80]!r}")
        print(f"                 VL: {r['vllm_head'][:80]!r}")
    print(f"output={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
