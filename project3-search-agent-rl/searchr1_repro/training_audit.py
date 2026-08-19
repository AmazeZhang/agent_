from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def dump_jsonl_records(records: list[dict[str, Any]], output_path: str | Path) -> Path:
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite JSONL evidence: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    if partial.exists():
        raise FileExistsError(f"stale JSONL evidence partial exists: {partial}")
    with partial.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_jsonable(record), ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(output)
    return output


def build_rollout_audit_records(batch, non_tensor_batch: dict[str, Any], *, multi_turn: bool) -> list[dict[str, Any]]:
    prompts = batch["prompts"].detach().cpu()
    responses = batch["responses"].detach().cpu()
    input_ids = batch["input_ids"].detach().cpu()
    attention_mask = batch["attention_mask"].detach().cpu()
    # P3 v1 (patch 0007): per-record training score (GRPO sees
    # token_level_scores.sum(-1)) so the offline replay can cross-validate the
    # step-attributed reward against the recorded component sums.
    token_level_scores = (
        batch["token_level_scores"].detach().cpu() if "token_level_scores" in batch else None
    )
    prompt_width = prompts.shape[1]
    response_width = responses.shape[1]
    if input_ids.shape[1] != prompt_width + response_width:
        raise ValueError("input_ids width must equal prompt width plus response width")

    response_attention_mask = attention_mask[:, -response_width:]
    if multi_turn:
        if "loss_mask" not in batch:
            raise ValueError("multi-turn audit requires loss_mask")
        response_policy_mask = batch["loss_mask"].detach().cpu()[:, -response_width:]
        mask_source = "loss_mask"
    else:
        response_policy_mask = response_attention_mask
        mask_source = "response_attention_mask"

    selected_metadata = (
        "uid", "traj_uid", "env_step", "retrieval", "retrieval_failed", "is_action_valid",
        # P3 v1 (patch 0007): per-step shaping components + per-episode component
        # totals, for offline replay verification and sum-consistency checks.
        "search_v1", "search_v1_episode",
    )
    records = []
    for index in range(input_ids.shape[0]):
        full_policy_mask = torch.cat(
            (torch.zeros(prompt_width, dtype=response_policy_mask.dtype), response_policy_mask[index]), dim=0
        )
        if torch.any(full_policy_mask[:prompt_width] != 0):
            raise RuntimeError("prompt tokens unexpectedly participate in policy loss")
        if torch.any(response_policy_mask[index] > response_attention_mask[index]):
            raise RuntimeError("policy loss mask includes padded response tokens")

        metadata = {}
        for key in selected_metadata:
            if key in non_tensor_batch:
                metadata[key] = _jsonable(non_tensor_batch[key][index])
        records.append(
            {
                "record_index": index,
                "mask_source": mask_source,
                "prompt_width": prompt_width,
                "response_width": response_width,
                "active_prompt_tokens": int(attention_mask[index, :prompt_width].sum()),
                "active_response_tokens": int(response_attention_mask[index].sum()),
                "policy_loss_tokens": int(response_policy_mask[index].sum()),
                "prompt_policy_loss_tokens": int(full_policy_mask[:prompt_width].sum()),
                "input_ids": input_ids[index].tolist(),
                "attention_mask": attention_mask[index].tolist(),
                "policy_loss_mask": full_policy_mask.tolist(),
                "record_score": (
                    float(token_level_scores[index].sum()) if token_level_scores is not None else None
                ),
                "metadata": metadata,
            }
        )
    return records


def dump_rollout_audit(batch, output_path: str | Path, *, multi_turn: bool) -> Path:
    records = build_rollout_audit_records(batch.batch, batch.non_tensor_batch, multi_turn=multi_turn)
    return dump_jsonl_records(records, output_path)
