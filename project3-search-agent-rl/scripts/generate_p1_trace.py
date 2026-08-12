#!/usr/bin/env python3
"""Generate one deterministic, model-free P1 Search-R1 protocol trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from searchr1_repro.fixture_retriever import FixtureRetriever
from agent_system.environments.env_package.search.third_party.skyrl_gym.envs.search.utils import compute_score
from agent_system.environments.env_package.search.third_party.skyrl_gym.tools.search import _passages2string


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = [json.loads(line) for line in (args.smoke_dir / "records.jsonl").read_text().splitlines()]
    record = next(item for item in records if item["split"] == "test")
    retriever = FixtureRetriever(args.smoke_dir / "fixture_corpus.jsonl")

    search_action = f"<search>{record['question']}</search>"
    api_response = retriever.api_response(record["question"], topk=3, return_scores=True)
    formatted_retrieval = _passages2string(api_response["result"][0])
    observation = f"\n<information>{json.dumps({'result': formatted_retrieval})}</information>\n"
    answer_action = f"<answer>{record['answers'][0]}</answer>"
    solution = search_action + observation + answer_action
    reward = compute_score(solution, {"target": record["answers"]})

    trace = {
        "schema_version": 1,
        "trace_type": "deterministic_model_free_protocol_smoke",
        "benchmark_result": False,
        "fixture_warning": "retrieved documents are ground-truth-derived P1 fixtures, not Wikipedia evidence",
        "record": record,
        "steps": [
            {
                "turn": 1,
                "action": search_action,
                "retriever_status": "success",
                "retriever_response": api_response,
                "observation": observation,
                "reward": 0,
                "done": False,
            },
            {"turn": 2, "action": answer_action, "observation": None, "reward": reward, "done": True},
        ],
        "final": {"predicted_answer": record["answers"][0], "ground_truth": record["answers"], "exact_match_reward": reward},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trace, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(f"wrote deterministic trace to {args.output}; reward={reward}")


if __name__ == "__main__":
    main()
