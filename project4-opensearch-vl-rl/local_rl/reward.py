"""Rule-based reward and fatal-aware advantage helpers with no external judge."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

ACCURACY_WEIGHT = 0.8
QUERY_WEIGHT = 0.2
STRICT_SUCCESS_WEIGHT = 0.5
TITLE_WEIGHT = 0.2
EVIDENCE_F1_WEIGHT = 0.3
EVIDENCE_RE = re.compile(r"(?:^|\n)Evidence:\s*(.+?)\s*$", re.DOTALL)


def _tool_calls(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [turn["tool_call"] for turn in result["turns"] if "tool_call" in turn]


def _target_entity(task: dict[str, Any]) -> str | None:
    if task.get("gold_entity_id") is None and str(task["task_type"]).startswith(
        "no-match"
    ):
        return None
    if task.get("gold_entity_id") is not None:
        return str(task["gold_entity_id"])
    for candidate in task["retrieval_results"]:
        if str(candidate["title"]) == str(task["gold_title"]):
            return str(candidate["entity_id"])
    raise ValueError(f"gold title has no retrieval candidate: {task['task_id']}")


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def _token_f1(prediction: str, reference: str) -> float:
    predicted = Counter(_tokens(prediction))
    expected = Counter(_tokens(reference))
    overlap = sum((predicted & expected).values())
    if not predicted or not expected or overlap == 0:
        return 0.0
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(expected.values())
    return 2 * precision * recall / (precision + recall)


def _final_evidence(text: str) -> str:
    match = EVIDENCE_RE.search(text)
    return "" if match is None else match.group(1)


def _observation_has_gold_evidence(
    result: dict[str, Any], target_entity: str, gold_evidence: str
) -> bool:
    expected = " ".join(_tokens(gold_evidence))
    for turn in result["turns"]:
        call = turn.get("tool_call")
        if not call or not expected:
            continue
        observation = " ".join(_tokens(str(turn.get("observation", ""))))
        if expected not in observation:
            continue
        if call.get("name") == "text_search":
            return True
        if call.get("name") == "text_lookup" and str(
            call.get("arguments", {}).get("entity_id")
        ) == target_entity:
            return True
    return False


def score_trajectory(result: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    """Score a saved rollout without answer-serving APIs or learned judges."""

    if result["task_id"] != task["task_id"] or result["task_type"] != task["task_type"]:
        raise ValueError("result/task identity mismatch")
    calls = _tool_calls(result)
    tool_names = [str(call["name"]) for call in calls]
    fatal = result.get("fatal")
    final_score = result["score"]
    r_accuracy = 1.0 if final_score["full_success"] else 0.0
    r_format = 1.0 if fatal is None and final_score["format_valid"] else 0.0

    task_type = str(task["task_type"])
    if task_type == "no-match":
        evidence_path_valid = tool_names == ["image_search"]
    else:
        target = _target_entity(task)
        looked_up = {
            str(call["arguments"].get("entity_id"))
            for call in calls
            if call["name"] == "text_lookup"
        }
        evidence_path_valid = target in looked_up
        if task_type == "transient-tool-failure":
            evidence_path_valid = evidence_path_valid and tool_names[:2] == [
                "image_search",
                "image_search",
            ]
    oracle_tool_count = len(task["oracle_steps"]) - 1
    efficiency = min(1.0, oracle_tool_count / max(1, len(tool_names)))
    r_query = efficiency if evidence_path_valid else 0.0
    total = r_format * (ACCURACY_WEIGHT * r_accuracy + QUERY_WEIGHT * r_query)

    is_fatal = fatal is not None
    prefix_turns = (
        max(0, len(result["turns"]) - 1) if is_fatal else len(result["turns"])
    )
    return {
        "r_accuracy": r_accuracy,
        "r_query": r_query,
        "r_format": r_format,
        "reward": total,
        "evidence_path_valid": evidence_path_valid,
        "tool_efficiency": efficiency,
        "is_fatal": is_fatal,
        "fatal_reason": fatal,
        "learnable_prefix_turns": prefix_turns,
        "hard_mask": is_fatal and prefix_turns == 0,
    }


def score_evidence_fidelity_trajectory(
    result: dict[str, Any], task: dict[str, Any]
) -> dict[str, Any]:
    """Score exact success plus graded final-answer evidence fidelity."""

    if result["task_id"] != task["task_id"] or result["task_type"] != task["task_type"]:
        raise ValueError("result/task identity mismatch")
    calls = _tool_calls(result)
    tool_names = [str(call["name"]) for call in calls]
    fatal = result.get("fatal")
    final_score = result["score"]
    r_exact_success = 1.0 if final_score["full_success"] else 0.0
    r_title = 1.0 if final_score.get("title_exact", False) else 0.0
    evidence_f1 = _token_f1(
        _final_evidence(str(result.get("final", ""))),
        str(task["gold_evidence_sentence"]),
    )
    r_answer = (
        STRICT_SUCCESS_WEIGHT * r_exact_success
        + TITLE_WEIGHT * r_title
        + EVIDENCE_F1_WEIGHT * evidence_f1
    )
    r_format = 1.0 if fatal is None and final_score["format_valid"] else 0.0

    target = _target_entity(task)
    if target is None:
        evidence_path_valid = tool_names == task["oracle_steps"][:-1]
    else:
        evidence_path_valid = _observation_has_gold_evidence(
            result, target, str(task["gold_evidence_sentence"])
        )
        failures = int(task.get("image_search_failures_before_success", 0))
        if failures:
            evidence_path_valid = evidence_path_valid and tool_names[: failures + 1] == [
                "image_search"
            ] * (failures + 1)
    oracle_tool_count = len(task["oracle_steps"]) - 1
    efficiency = min(1.0, oracle_tool_count / max(1, len(tool_names)))
    r_query = efficiency if evidence_path_valid else 0.0
    total = r_format * (ACCURACY_WEIGHT * r_answer + QUERY_WEIGHT * r_query)

    is_fatal = fatal is not None
    prefix_turns = max(0, len(result["turns"]) - 1) if is_fatal else len(result["turns"])
    return {
        "r_exact_success": r_exact_success,
        "r_title": r_title,
        "r_evidence_f1": evidence_f1,
        "r_answer": r_answer,
        "r_query": r_query,
        "r_format": r_format,
        "reward": total,
        "evidence_path_valid": evidence_path_valid,
        "tool_efficiency": efficiency,
        "is_fatal": is_fatal,
        "fatal_reason": fatal,
        "learnable_prefix_turns": prefix_turns,
        "hard_mask": is_fatal and prefix_turns == 0,
    }


def compute_group_advantages(
    rewards: list[float], fatal_flags: list[bool]
) -> dict[str, list[float]]:
    """Mean-center a rollout group and apply the one-sided fatal clamp."""

    if not rewards or len(rewards) != len(fatal_flags):
        raise ValueError("reward group must be non-empty and align with fatal flags")
    if any(not 0.0 <= reward <= 1.0 for reward in rewards):
        raise ValueError("rewards must be in [0, 1]")
    mean = sum(rewards) / len(rewards)
    raw = [reward - mean for reward in rewards]
    clamped = [
        max(0.0, advantage) if fatal else advantage
        for advantage, fatal in zip(raw, fatal_flags, strict=True)
    ]
    return {"raw": raw, "fatal_clamped": clamped}
