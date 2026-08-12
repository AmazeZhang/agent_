#!/usr/bin/env python3
"""Run a bounded model-driven Search-R1 protocol check on the P1 fixture.

The fixture corpus contains ground-truth-derived documents. Results from this
script validate integration and protocol behavior only; they are not benchmark
or model-quality evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from searchr1_repro.fixture_retriever import FixtureRetriever
from agent_system.environments.env_manager import SearchEnvironmentManager
from agent_system.environments.env_package.search.envs import SearchMultiProcessEnv
from agent_system.environments.env_package.search.projection import search_projection


FIXTURE_WARNING = (
    "GROUND-TRUTH-DERIVED FIXTURE: protocol/integration evidence only; "
    "prohibited for benchmark or model-quality claims."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--count", type=int, choices=(1, 4, 16), required=True)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument(
        "--require-search-first",
        action="store_true",
        help="diagnostic only: append a first-turn requirement to exercise the retrieval loop",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


@contextmanager
def fixture_server(retriever: FixtureRetriever):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/retrieve":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                response = retriever.api_response(
                    query=payload["query"],
                    topk=payload.get("topk", 3),
                    return_scores=payload.get("return_scores", True),
                )
                encoded = json.dumps(response).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
            except Exception as exc:
                self.send_error(400, str(exc))

        def log_message(self, *_):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, name="fixture-retriever", daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/retrieve"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def load_records(path: Path, count: int) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    test_records = [record for record in records if record["split"] == "test"]
    if len(test_records) < count:
        raise ValueError(f"requested {count} test records, found {len(test_records)}")
    return test_records[:count]


def generate_actions(model, tokenizer, prompts: list[str], args: argparse.Namespace) -> list[str]:
    chats = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for prompt in prompts
    ]
    inputs = tokenizer(
        chats,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_input_tokens,
    ).to(model.device)
    input_width = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        sequences = model.generate(
            **inputs,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
    return tokenizer.batch_decode(sequences[:, input_width:], skip_special_tokens=True)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the P2 model check")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "managed P2 must expose exactly one logical GPU; use scripts/run_managed.sh with one physical GPU"
        )
    run_dir = Path(os.environ.get("PROJECT3_RUN_DIR", ".")).resolve()
    output_path = args.output or run_dir / "p2_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = load_records(args.records, args.count)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval().to("cuda:0")
    torch.cuda.reset_peak_memory_stats()

    retriever = FixtureRetriever(args.corpus)
    episode_steps: list[list[dict[str, Any]]] = [[] for _ in records]
    final_rewards = np.zeros(len(records), dtype=float)
    done = np.zeros(len(records), dtype=bool)

    with fixture_server(retriever) as search_url:
        env_config = OmegaConf.create(
            {
                "max_steps": args.max_steps,
                "history_length": args.history_length,
                "search": {
                    "search_url": search_url,
                    "topk": args.topk,
                    "timeout": 2,
                    "log_requests": False,
                },
            }
        )
        raw_envs = SearchMultiProcessEnv(
            seed=args.seed,
            env_num=len(records),
            group_n=1,
            is_train=False,
            env_config=env_config,
        )
        manager = SearchEnvironmentManager(raw_envs, search_projection, OmegaConf.create({"env": env_config}))
        kwargs = [
            {
                "question": record["question"],
                "ground_truth": {"target": record["answers"]},
                "data_source": record["source"],
            }
            for record in records
        ]
        observations, _ = manager.reset(kwargs=kwargs)
        if args.require_search_first:
            diagnostic_instruction = (
                "\nIntegration diagnostic requirement: for the first step, you MUST use "
                "exactly one <search>query</search> action and MUST NOT answer directly.\n"
            )
            observations["text"] = [text + diagnostic_instruction for text in observations["text"]]
        try:
            for step_index in range(args.max_steps):
                active_before = ~done
                if not active_before.any():
                    break
                generation_started = time.monotonic()
                raw_actions = generate_actions(model, tokenizer, observations["text"], args)
                projected_actions, projected_valids = search_projection(raw_actions)
                next_observations, rewards, step_done, infos = manager.step(raw_actions)
                generation_seconds = time.monotonic() - generation_started
                for index in range(len(records)):
                    if not active_before[index]:
                        continue
                    episode_steps[index].append(
                        {
                            "step": step_index + 1,
                            "prompt": observations["text"][index],
                            "raw_action": raw_actions[index],
                            "projected_action": projected_actions[index],
                            "is_action_valid": bool(projected_valids[index]),
                            "observation": next_observations["anchor"][index],
                            "reward": float(rewards[index]),
                            "done": bool(step_done[index]),
                            "info": jsonable(infos[index]),
                            "batch_generation_seconds": generation_seconds,
                        }
                    )
                    final_rewards[index] += float(rewards[index])
                done |= np.asarray(step_done, dtype=bool)
                observations = next_observations
        finally:
            raw_envs.close()

    elapsed = time.monotonic() - started
    all_active_steps = [step for episode in episode_steps for step in episode]
    projected_search_steps = [
        step for step in all_active_steps if step["projected_action"].startswith("<search>")
    ]
    executed_search_steps = [step for step in projected_search_steps if step["info"].get("tool_calling")]
    retrieval_statuses = [
        step["info"].get("retrieval", {}).get("status") for step in executed_search_steps
    ]
    result = {
        "warning": FIXTURE_WARNING,
        "diagnostic_prompt_modified": bool(args.require_search_first),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": os.environ.get("PROJECT3_RUN_ID"),
        "model_path": str(args.model.resolve()),
        "model_revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "physical_gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_cuda_device": torch.cuda.get_device_name(0),
        "parameters": vars(args) | {"model": str(args.model), "records": str(args.records), "corpus": str(args.corpus), "output": str(output_path)},
        "metrics": {
            "episodes": len(records),
            "completed": int(done.sum()),
            "reward_one": int((final_rewards >= 1.0).sum()),
            "valid_actions": int(sum(step["is_action_valid"] for step in all_active_steps)),
            "total_actions": len(all_active_steps),
            "search_actions": len(projected_search_steps),
            "executed_search_calls": len(executed_search_steps),
            "answer_actions": int(sum(step["projected_action"].startswith("<answer>") for step in all_active_steps)),
            "retrieval_statuses": retrieval_statuses,
            "retrieval_failures": int(sum(status not in {"success", "no_results"} for status in retrieval_statuses)),
            "elapsed_seconds": elapsed,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "episodes": [
            {
                "record": record,
                "reward": float(final_rewards[index]),
                "done": bool(done[index]),
                "steps": episode_steps[index],
            }
            for index, record in enumerate(records)
        ],
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(FIXTURE_WARNING)
    print(f"output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
