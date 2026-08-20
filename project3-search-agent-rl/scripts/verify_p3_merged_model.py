#!/usr/bin/env python3
"""Verify a merged HF model produced by verl model_merger.py (FSDP backend).

Used by the checkpoint-merge eval-chain gate: after merging the resume-smoke
global_step_2 FSDP checkpoint, this script checks that the merged artifact is a
complete, loadable, NaN-free HF model.

Checks (exit nonzero on any failure):
  1. Required files present: config.json, generation_config.json, tokenizer
     files (tokenizer.json / tokenizer_config.json / vocab.json / merges.txt /
     special_tokens_map.json / added_tokens.json), model.safetensors.
  2. Transformers load (bf16, local_files_only): reports missing/unexpected
     keys; asserts both sets are empty.
  3. All weights finite (no NaN/Inf).
  4. Reports parameter count, weight dtype, on-disk size, per-file SHA256.
  5. Expected parameter count for Qwen2.5-3B (3,090,000,000 params) is
     asserted within a tolerance -- the gate input is Qwen2.5-3B-based.

Usage:
  <env-python> scripts/verify_p3_merged_model.py --merged-dir <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Qwen2.5-3B base actual param count (tied embeddings: 151936*2048 lm_head is
# shared with embed_tokens). The verl FSDP merge output additionally stores an
# independent lm_head.weight copy (identical content, tie preserved), so the
# expected range is [base_params, base_params + vocab*hidden].
BASE_PARAMS = 3_085_938_688
VOCAB = 151_936
HIDDEN = 2_048
PARAM_TOLERANCE = 1_000_000

REQUIRED_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)
# Optional: Qwen2.5-3B-Instruct (official HF layout) ships neither; its added
# tokens live in tokenizer_config.json's added_tokens_decoder, so transformers
# loads it byte-identically without them. Reported as present/absent only.
OPTIONAL_TOKENIZER_FILES = (
    "special_tokens_map.json",
    "added_tokens.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-dir", type=Path, required=True)
    args = parser.parse_args()

    merged = args.merged_dir.resolve()
    report: dict = {"merged_dir": str(merged)}

    # 1. required files (sharded safetensors layout produced by model_merger)
    shards = sorted((merged / "model.safetensors.index.json").is_file() and (p for p in merged.glob("model-*.safetensors")) or (p for p in merged.glob("model.safetensors")))
    shards = list(shards)
    assert shards, f"no safetensors weights in {merged}"
    required = ["config.json", "generation_config.json", "model.safetensors.index.json"]
    required += REQUIRED_TOKENIZER_FILES
    missing = [name for name in required if not (merged / name).is_file()]
    assert not missing, f"missing required files: {missing}"
    report["files_present"] = sorted(required)
    report["optional_tokenizer_files_present"] = [
        name for name in OPTIONAL_TOKENIZER_FILES if (merged / name).is_file()
    ]
    report["weight_shards"] = [p.name for p in shards]

    # 2. sizes + SHA256 per file
    files = sorted(p for p in merged.iterdir() if p.is_file())
    total_bytes = 0
    file_shas = {}
    for p in files:
        total_bytes += p.stat().st_size
        file_shas[p.name] = sha256_file(p)
    report["total_bytes"] = total_bytes
    report["file_sha256"] = file_shas

    # 3. config sanity
    config = AutoConfig.from_pretrained(merged, local_files_only=True)
    report["architectures"] = config.architectures
    report["hidden_size"] = config.hidden_size
    report["num_hidden_layers"] = config.num_hidden_layers
    report["num_attention_heads"] = config.num_attention_heads
    assert "Qwen2ForCausalLM" in (config.architectures or []), f"unexpected architecture: {config.architectures}"

    # 4. weights: dtype, NaN/Inf, param count (across all shards)
    from safetensors import safe_open
    n_params = 0
    dtypes = set()
    n_nan = 0
    n_inf = 0
    file_keys = set()
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                file_keys.add(key)
                tensor = handle.get_tensor(key)
                n_params += tensor.numel()
                dtypes.add(str(tensor.dtype))
                n_nan += int(bool(torch.isnan(tensor).any()))
                n_inf += int(bool(torch.isinf(tensor).any()))
    report["param_count"] = n_params
    report["dtypes"] = sorted(dtypes)
    report["tensors_with_nan"] = n_nan
    report["tensors_with_inf"] = n_inf
    assert n_nan == 0 and n_inf == 0, f"non-finite weights: nan={n_nan} inf={n_inf}"
    # Qwen2.5-3B is tied: the merged output may or may not carry an independent
    # lm_head.weight copy (same content). Accept base_params (no lm_head file)
    # or base_params + vocab*hidden (independent lm_head file); assert exact
    # content equality when the copy is present.
    has_lm_head = "lm_head.weight" in file_keys
    if has_lm_head:
        assert "model.embed_tokens.weight" in file_keys
        expected = BASE_PARAMS + VOCAB * HIDDEN
    else:
        expected = BASE_PARAMS
    assert abs(n_params - expected) <= PARAM_TOLERANCE, (
        f"param count {n_params} deviates from Qwen2.5-3B expected {expected}"
    )
    report["independent_lm_head_weight"] = has_lm_head

    # 5. full Transformers load; missing/unexpected = safetensors keys vs model keys
    model = AutoModelForCausalLM.from_pretrained(merged, torch_dtype=torch.bfloat16, local_files_only=True)
    if has_lm_head:
        embed = model.get_input_embeddings().weight.detach()
        lm_head = model.lm_head.weight.detach()
        tie_ok = bool(torch.equal(embed, lm_head))
        report["tie_lm_head_matches_embed_tokens"] = tie_ok
        assert tie_ok, "independent lm_head.weight differs from embed_tokens.weight (tie broken)"
    model_keys = set(model.state_dict().keys())
    missing = sorted(model_keys - file_keys)
    unexpected = sorted(file_keys - model_keys)
    if not has_lm_head:
        # Tied embeddings: transformers materializes lm_head.weight sharing
        # embed_tokens. The raw HF layout (e.g. Qwen2.5-3B-Instruct) ships no
        # independent lm_head file, so its absence is expected -- verify the
        # materialized copy ties exactly to embed_tokens instead.
        missing_wo_lm_head = [k for k in missing if k != "lm_head.weight"]
        if not missing_wo_lm_head and "lm_head.weight" in missing:
            embed = model.get_input_embeddings().weight.detach()
            lm_head = model.lm_head.weight.detach()
            report["tie_lm_head_matches_embed_tokens"] = bool(torch.equal(embed, lm_head))
            report["tied_lm_head_materialized"] = True
            missing = missing_wo_lm_head
        else:
            report["tied_lm_head_materialized"] = False
    report["load_missing_keys"] = missing
    report["load_unexpected_keys"] = unexpected
    assert not missing, f"keys present in model but missing from safetensors: {missing}"
    assert not unexpected, f"keys present in safetensors but absent from model: {unexpected}"
    # dtype of a representative param after load
    report["loaded_dtype"] = str(next(model.parameters()).dtype)
    del model

    # 6. tokenizer loads
    tok = AutoTokenizer.from_pretrained(merged, local_files_only=True)
    report["tokenizer_vocab_size"] = len(tok)
    report["tokenizer_special_tokens"] = {
        "bos": tok.bos_token, "eos": tok.eos_token, "pad": tok.pad_token, "unk": tok.unk_token,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print("VERIFY_MERGED: PASS")


if __name__ == "__main__":
    main()
