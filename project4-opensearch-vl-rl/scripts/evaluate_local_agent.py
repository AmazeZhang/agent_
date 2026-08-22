#!/usr/bin/env python3
"""Evaluate base or LoRA models in the fixed local two-tool environment."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from local_retrieval import LocalTextIndex, entity_tool_observation, text_tool_observation  # noqa: E402
from local_retrieval.image_search_backend import resolve_local_image  # noqa: E402
from local_retrieval.resnet50_encoder import sha256_file  # noqa: E402

PROJECT_DATA = Path("/media/imc/data/yzy/agent/project4-opensearch-vl-rl")
MODEL_ROOT = PROJECT_DATA / "models/Qwen3-VL-8B-Instruct"
DATASET_ROOT = PROJECT_DATA / "datasets/processed/wit-agentic-challenge-v5"
RUN_ROOT = PROJECT_DATA / "runs"
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "image_search",
            "description": (
                "Find local Wikipedia entity candidates. The provided query image is "
                "registered under the literal runtime handle img_1."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 3},
                },
                "required": ["image"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "text_lookup",
            "description": "Look up local Wikipedia evidence by entity ID.",
            "parameters": {
                "type": "object",
                "properties": {"entity_id": {"type": "string"}},
                "required": ["entity_id"],
            },
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--max-tasks", type=int, default=5)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    return parser.parse_args()


def parse_tool_call(text: str) -> dict[str, Any] | None:
    matches = TOOL_CALL_RE.findall(text)
    if not matches:
        return None
    if len(matches) != 1 or TOOL_CALL_RE.sub("", text).strip():
        raise ValueError("assistant turn must contain exactly one standalone tool call")
    call = json.loads(matches[0])
    if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
        raise ValueError("tool call must contain exactly name and arguments")
    if not isinstance(call["name"], str) or not isinstance(call["arguments"], dict):
        raise ValueError("tool call name/arguments have invalid types")
    return call


def normalise(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


def score_final(text: str, task: dict[str, Any]) -> dict[str, bool]:
    title_match = re.search(r"(?:^|\n)Title:\s*(.+?)(?:\n|$)", text)
    evidence_match = re.search(r"(?:^|\n)Evidence:\s*(.+?)\s*$", text, re.DOTALL)
    format_valid = title_match is not None and evidence_match is not None
    if not format_valid:
        return {"format_valid": False, "title_exact": False, "evidence_exact": False}
    assert title_match is not None and evidence_match is not None
    return {
        "format_valid": True,
        "title_exact": normalise(title_match.group(1)) == normalise(task["gold_title"]),
        "evidence_exact": normalise(evidence_match.group(1))
        == normalise(task["gold_evidence_sentence"]),
    }


def is_full_success(fatal: str | None, final_score: dict[str, bool]) -> bool:
    return (
        fatal is None
        and final_score["format_valid"]
        and final_score["title_exact"]
        and final_score["evidence_exact"]
    )


def require_managed_run(environment: dict[str, str]) -> tuple[Path, str]:
    run_id = environment.get("PROJECT4_RUN_ID", "")
    run_token = environment.get("PROJECT4_RUN_TOKEN", "")
    raw_run_dir = environment.get("PROJECT4_RUN_DIR", "")
    visible = environment.get("CUDA_VISIBLE_DEVICES", "")
    if not run_id or not run_token or not raw_run_dir:
        raise RuntimeError("agent evaluation must run inside scripts/run_managed.sh")
    if len(visible.split(",")) != 1 or visible in {"0", "5"}:
        raise RuntimeError("agent evaluation requires one stable GPU, excluding GPU0/GPU5")
    run_dir = Path(raw_run_dir).resolve()
    if run_dir != (RUN_ROOT / run_id).resolve() or not run_dir.is_dir():
        raise RuntimeError(f"unexpected managed Run directory: {run_dir}")
    return run_dir, visible


def load_tasks(split: str, max_tasks: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not 1 <= max_tasks <= 20:
        raise ValueError("max_tasks must be between 1 and 20")
    with (DATASET_ROOT / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if (
        manifest.get("status") != "challenge-ready"
        or manifest.get("image_observation_contains_text_summary") is not False
        or manifest.get("image_runtime_handle") != "img_1"
        or manifest.get("final_response_format")
        != "Title: <exact title>\\nEvidence: <first sentence-or-no-match>"
        or manifest.get("evidence_extraction")
        != "first_terminal_punctuation_or_360_characters"
        or manifest.get("maximum_agent_turns") != 5
        or manifest.get("text_lookup_summary_max_characters") != 360
        or manifest.get("image_search_top_k_maximum") != 3
    ):
        raise ValueError("evaluation dataset is not a verified no-leak pilot")
    with (DATASET_ROOT / "tasks.jsonl").open(encoding="utf-8") as handle:
        tasks = [
            item
            for line in handle
            if line.strip()
            and (item := json.loads(line)).get("split") == split
        ]
    return manifest, tasks[:max_tasks]


def validate_adapter(raw_adapter: Path | None) -> Path | None:
    if raw_adapter is None:
        return None
    adapter = raw_adapter.resolve()
    if not adapter.is_relative_to(RUN_ROOT.resolve()):
        raise ValueError("adapter must belong to a project4 managed Run")
    if not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("adapter_model.safetensors is missing")
    return adapter


def generate_turn(
    model: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> tuple[str, int, int]:
    import torch

    inputs = processor.apply_chat_template(
        messages,
        tools=TOOLS,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda:0")
    generation = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generation.update({"temperature": temperature, "top_p": top_p})
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            **generation,
        )
    suffix = generated[:, inputs["input_ids"].shape[1] :]
    text = processor.batch_decode(suffix, skip_special_tokens=True)[0].strip()
    return text, int(inputs["input_ids"].shape[1]), int(suffix.shape[1])


def execute_call(
    call: dict[str, Any],
    task: dict[str, Any],
    text_index: LocalTextIndex,
    *,
    image_search_call_count: int,
) -> tuple[str, bool]:
    name = call["name"]
    arguments = call["arguments"]
    if name == "image_search":
        if arguments.get("image") != "img_1":
            raise ValueError("image_search must reference img_1")
        top_k = int(arguments.get("top_k", 3))
        if not 1 <= top_k <= 3:
            raise ValueError("cached image_search top_k must be between 1 and 3")
        failures = int(task.get("image_search_failures_before_success", 0))
        if image_search_call_count <= failures:
            return (
                "Tool execution result:\n"
                + json.dumps(
                    {"error": {"code": "TRANSIENT_FAILURE", "retryable": True}},
                    sort_keys=True,
                ),
                True,
            )
        results = task["retrieval_results"][:top_k]
        return entity_tool_observation(results), True
    if name == "text_lookup":
        if set(arguments) != {"entity_id"}:
            raise ValueError("text_lookup accepts only entity_id")
        result = text_index.lookup(str(arguments["entity_id"]))
        return text_tool_observation([] if result is None else [result]), False
    raise ValueError(f"unsupported tool: {name}")


def evaluate_task(
    model: Any,
    processor: Any,
    task: dict[str, Any],
    text_index: LocalTextIndex,
    *,
    max_new_tokens: int,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    seed: int | None = None,
) -> dict[str, Any]:
    if seed is not None:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    image_path = resolve_local_image(Path(task["query_image"]), DATASET_ROOT)
    with Image.open(image_path) as image:
        image.load()
        query_image = image.convert("RGB")
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Use one tool call per turn. Pass the literal handle img_1 to "
                        "image_search; do not replace it with a filename or image description. "
                        "Image search returns entity candidates only; text_lookup supplies "
                        "answer evidence. Retry a tool only when its error says retryable=true. "
                        "Do not invent missing evidence. The final response must contain exactly "
                        "two lines named Title and Evidence."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": query_image},
                {
                    "type": "text",
                    "text": str(task["user_prompt"]),
                },
            ],
        },
    ]
    turns = []
    tool_names = []
    fatal = None
    final_text = ""
    image_search_call_count = 0
    for _ in range(5):
        output, input_tokens, output_tokens = generate_turn(
            model,
            processor,
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
        turn = {
            "assistant": output,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        turns.append(turn)
        try:
            call = parse_tool_call(output)
        except (ValueError, json.JSONDecodeError) as error:
            fatal = f"invalid-tool-format:{error}"
            break
        if call is None:
            final_text = output
            break
        try:
            if call["name"] == "image_search":
                image_search_call_count += 1
            observation, used_image_cache = execute_call(
                call,
                task,
                text_index,
                image_search_call_count=image_search_call_count,
            )
        except (ValueError, TypeError) as error:
            fatal = f"tool-error:{error}"
            break
        tool_names.append(call["name"])
        turn["tool_call"] = call
        turn["observation"] = observation
        turn["image_search_cache"] = used_image_cache
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": output}]}
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"<tool_response>\n{observation}\n</tool_response>",
                    }
                ],
            }
        )
    else:
        fatal = "maximum-turns-exceeded"
    final_score = score_final(final_text, task)
    oracle_path_exact = tool_names == task["oracle_steps"][:-1]
    return {
        "task_id": task["task_id"],
        "task_type": task["task_type"],
        "split": task["split"],
        "fatal": fatal,
        "tool_names": tool_names,
        "oracle_path_exact": oracle_path_exact,
        "final": final_text,
        "score": {
            **final_score,
            "full_success": is_full_success(fatal, final_score),
        },
        "turns": turns,
    }


def aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        raise ValueError("cannot aggregate an empty result set")
    metric_names = ("format_valid", "title_exact", "evidence_exact", "full_success")
    metrics = {
        name: sum(bool(result["score"][name]) for result in results) / len(results)
        for name in metric_names
    }
    metrics.update(
        {
            "oracle_path_exact": sum(result["oracle_path_exact"] for result in results)
            / len(results),
            "fatal_rate": sum(result["fatal"] is not None for result in results)
            / len(results),
        }
    )
    return metrics


def main() -> int:
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    import torch

    args = parse_args()
    if not 32 <= args.max_new_tokens <= 256:
        raise ValueError("max_new_tokens must be between 32 and 256")
    run_dir, physical_gpu = require_managed_run(dict(os.environ))
    adapter = validate_adapter(args.adapter)
    output = args.output.resolve()
    if output != (run_dir / "evaluation.json").resolve():
        raise ValueError("evaluation output must be the managed Run evaluation.json")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {output}")
    dataset_manifest, tasks = load_tasks(args.split, args.max_tasks)
    processor = AutoProcessor.from_pretrained(
        MODEL_ROOT, local_files_only=True, trust_remote_code=False
    )
    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_ROOT,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda:0",
        local_files_only=True,
        trust_remote_code=False,
    )
    model = (
        base.eval()
        if adapter is None
        else PeftModel.from_pretrained(base, adapter, is_trainable=False).eval()
    )
    results = []
    text_path = Path(dataset_manifest["text_index"])
    with LocalTextIndex(text_path) as text_index:
        for index, task in enumerate(tasks, start=1):
            result = evaluate_task(
                model,
                processor,
                task,
                text_index,
                max_new_tokens=args.max_new_tokens,
            )
            results.append(result)
            print(
                json.dumps(
                    {
                        "task": f"{index}/{len(tasks)}",
                        "task_id": result["task_id"],
                        "fatal": result["fatal"],
                        "tools": result["tool_names"],
                        "score": result["score"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    metrics = aggregate_metrics(results)
    task_types = sorted({str(result["task_type"]) for result in results})
    metrics_by_task_type = {
        task_type: aggregate_metrics(
            [result for result in results if result["task_type"] == task_type]
        )
        for task_type in task_types
    }
    report = {
        "schema_version": 1,
        "model": "base" if adapter is None else "lora-adapter",
        "adapter": str(adapter) if adapter else None,
        "adapter_sha256": (
            sha256_file(adapter / "adapter_model.safetensors") if adapter else None
        ),
        "dataset_manifest_sha256": sha256_file(DATASET_ROOT / "manifest.json"),
        "tasks_sha256": sha256_file(DATASET_ROOT / "tasks.jsonl"),
        "split": args.split,
        "task_count": len(results),
        "task_ids": [result["task_id"] for result in results],
        "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "image_search_backend": "frozen-verified-cache",
        "physical_gpu": physical_gpu,
        "metrics": metrics,
        "metrics_by_task_type": metrics_by_task_type,
        "results": results,
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"metrics": metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
