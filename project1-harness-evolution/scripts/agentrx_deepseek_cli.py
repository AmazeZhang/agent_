"""Run the upstream AgentRx pipeline through a DeepSeek OpenAI endpoint.

AgentRx currently exposes only ``copilot``, ``azure`` and ``trapi`` endpoint
names even though every pipeline stage ultimately uses the OpenAI chat
completions interface.  This project-owned launcher supplies a compatible
client at runtime and leaves the vendored repository unchanged.
"""

from __future__ import annotations

import os
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = PROJECT_ROOT / "vendor" / "AgentRx"
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")


class _CompletionsProxy:
    def __init__(self, completions: Any) -> None:
        self._completions = completions

    def create(self, **kwargs: Any) -> Any:
        # V4 Flash can spend a large part of a small output budget on internal
        # reasoning. AgentRx needs concise JSON, so disable thinking here.
        kwargs.setdefault("max_tokens", 8192)
        extra_body = dict(kwargs.pop("extra_body", {}) or {})
        extra_body.setdefault("thinking", {"type": "disabled"})
        kwargs["extra_body"] = extra_body
        response = self._completions.create(**kwargs)
        if kwargs.get("response_format") == {"type": "json_object"}:
            content = response.choices[0].message.content or ""
            normalized = _normalize_agentrx_json(content)
            if normalized is not None:
                response.choices[0].message.content = normalized
        return response


def _normalize_agentrx_json(content: str) -> str | None:
    """Repair common schema-shape drift without inventing invariant content."""
    try:
        obj = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None

    if not isinstance(obj, dict):
        return None

    # Static generation occasionally returns one invariant directly although
    # the checker requires an ``invariant(s)`` collection.
    if "assertion_name" in obj and "invariant" not in obj and "invariants" not in obj:
        return json.dumps({"invariants": [obj]}, ensure_ascii=False)

    # One-shot dynamic generation can close its invariant array too early and
    # leave check fields on the wrapper. Merge only those existing fields back
    # into the sole invariant; no rule text or verdict is synthesized here.
    payload = obj.get("invariant")
    if isinstance(payload, dict):
        payload = [payload]
        obj["invariant"] = payload
    leaked_fields = ("check_type", "python_check", "nl_check")
    if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
        changed = False
        for field in leaked_fields:
            if field in obj and field not in payload[0]:
                payload[0][field] = obj.pop(field)
                changed = True
        if changed:
            return json.dumps(obj, ensure_ascii=False)

    return None


class _ChatProxy:
    def __init__(self, chat: Any) -> None:
        self.completions = _CompletionsProxy(chat.completions)


class _ClientProxy:
    def __init__(self) -> None:
        key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY or OPENAI_API_KEY is required")
        raw = OpenAI(api_key=key, base_url=BASE_URL, timeout=180.0, max_retries=2)
        self.chat = _ChatProxy(raw.chat)


def _new_client() -> _ClientProxy:
    return _ClientProxy()


def _pop_project_option(*names: str) -> str | None:
    """Remove one project-owned option before delegating to upstream argparse."""
    for name in names:
        if name in sys.argv:
            index = sys.argv.index(name)
            try:
                value = sys.argv[index + 1]
            except IndexError as exc:
                raise SystemExit(f"{name} requires a value") from exc
            del sys.argv[index : index + 2]
            return value
    return None


def install_adapter() -> None:
    sys.path.insert(0, str(VENDOR_ROOT))

    # Populate the names expected by AgentRx's existing "azure" branch before
    # its globals module is imported. No Azure credential is actually used.
    os.environ["AGENT_VERIFY_ENDPOINT"] = BASE_URL
    os.environ["AGENT_VERIFY_DEPLOYMENT"] = MODEL
    os.environ["AGENT_VERIFY_MODEL_NAME"] = MODEL
    os.environ["AGENT_VERIFY_ENDPOINT_TYPE"] = "azure"

    import agentrx.pipeline.globals as globals_module
    import agentrx.llm_clients.azure as azure_module
    import agentrx.reports.metrics as metrics

    globals_module.ENDPOINT = BASE_URL
    globals_module.DEPLOYMENT = MODEL
    globals_module.MODEL_NAME = MODEL
    globals_module.DEFAULT_ENDPOINT = "azure"

    class DeepSeekLLMAgent:
        def __init__(
            self,
            api_version: str,
            model_name: str,
            model_version: str,
            deployment_name: str,
        ) -> None:
            del api_version, model_version
            self.model_name = deployment_name or model_name or MODEL
            self.endpoint = BASE_URL
            self.client = _new_client()
            self.last_call_telemetry = None

        def get_llm_response(self, messages: list[dict[str, Any]]) -> Any:
            started_at = datetime.now()
            started = time.perf_counter()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_format={"type": "json_object"},
            )
            ended_at = datetime.now()
            usage = response.usage
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            self.last_call_telemetry = metrics.LLMCallTelemetry(
                tokens=metrics.TokenUsage(
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
                ),
                time=metrics.TimingInfo(
                    start_time=started_at,
                    end_time=ended_at,
                    execution_time_sec=round(time.perf_counter() - started, 4),
                ),
                model_name=self.model_name,
                instance=self.endpoint,
            )
            return response

        @staticmethod
        def azure_mk_client() -> _ClientProxy:
            return _new_client()

    azure_module.LLMAgent = DeepSeekLLMAgent


def main() -> None:
    examples_dir = _pop_project_option("--examples-dir", "--examples_dir")
    install_adapter()
    import run as agentrx_run

    if examples_dir:
        original_run_judge = agentrx_run.run_judge

        def run_judge_with_examples(*args: Any, **kwargs: Any) -> str:
            import agentrx.judge.judge as judge_module

            judge_module.EXAMPLES_DIR = str(Path(examples_dir).resolve())
            judge_module.FEW_SHOT_EXAMPLES = None
            return original_run_judge(*args, **kwargs)

        agentrx_run.run_judge = run_judge_with_examples

    agentrx_run.main()


if __name__ == "__main__":
    main()
