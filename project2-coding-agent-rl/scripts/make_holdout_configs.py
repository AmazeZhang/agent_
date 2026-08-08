"""WP7: generate SWE-agent run-batch configs for holdout evaluation.

Same protocol as WP1 (run8 template: 40 calls, cost limit $0.05, temperature
0.0, single worker) but the model section points at the LOCAL vLLM endpoint
serving base / sft / grpo variants:

    model.name        = openai/qwen25-coder-3b-<variant>
    api_base          = http://127.0.0.1:8011/v1  (vLLM OpenAI server)
    api_key           = dummy (vLLM ignores auth unless --api-key is set)
    completion_kwargs = {} (the DeepSeek "thinking" extra_body is removed —
                         vLLM rejects unknown fields with 400)
    registry          = configs/local-model-registry.json (zero-cost entries;
                         the per_instance_cost_limit never trips, the 40-call
                         limit is the governing constraint — same as WP1)

Usage:
    python scripts/make_holdout_configs.py
"""

from __future__ import annotations

import json
import yaml
from pathlib import Path

PILOT_ROOT = Path("/media/imc/data/yzy/agent/project2/swesmith-pilot20")
TEMPLATE = PILOT_ROOT / "runs/deepseek-v4-flash-run8-sanitized/run_batch.config.yaml"
# The local vLLM endpoint cannot serve SWE-agent's OpenAI function-calling
# requests: vLLM 0.22.1 has no built-in tool parser matching Qwen2.5's
# ```json output (hermes wants <tool_call> XML, llama3_json requires the
# <|python_tag|> token, qwen3_xml wants Qwen3 XML). All three WP7 variants
# therefore run the classic SWE-agent text protocol (thought_action: one
# discussion + one ``` code block per step) with the official 0.7-style
# system template that documents {{command_docs}} and the response format.
# The three variants share this exact protocol, so the base/sft/grpo
# comparison stays valid; the difference vs the WP1 DeepSeek run (function
# calling) is documented in reports/.
TA_SYSTEM_TEMPLATE = yaml.safe_load(
    (Path(__file__).resolve().parent.parent / "vendor/SWE-agent/config/sweagent_0_7/07_thought_action.yaml").read_text()
)["agent"]["templates"]["system_template"]
OUT_DIR = Path("/media/imc/data/yzy/agent/project2/swesmith-pilot20/holdout-eval/configs")

BASE_URL = "http://127.0.0.1:8011/v1"
API_KEY = "sk-local-dummy"

# Corrected frozen split (documented in reports/: oauthlib-signature-1bsv3m8l
# was excluded because the WP5 SFT smoke trained on it by mistake).
HOLDOUTS = {
    "funcy-curry-compose-3u9hti2d": "funcy-curry-compose-3u9hti2d-sanitized.json",
    "pygments-groff-0jqqr58z": "pygments-groff-0jqqr58z-sanitized.json",
    "stackprinter-1i9gep13": "stackprinter-1i9gep13-sanitized.json",
    "funcy-lookuper-3y0j7te5": "funcy-lookuper-3y0j7te5-sanitized.json",
    "boltons-7nlifqzn": "boltons-7nlifqzn-sanitized.json",
}
# Model names must match what the vLLM endpoint actually serves: the base
# served-model-name, and the --lora-modules names (sft/grpo) for the adapters.
# litellm strips the "openai/" provider prefix before sending, so the request
# body carries model=<lora name>, which vLLM resolves to the LoRA adapter.
VARIANTS = {
    "base": "openai/qwen25-coder-3b-base",
    "sft": "openai/sft",
    "grpo": "openai/grpo",
}


def main() -> None:
    raw = TEMPLATE.read_text(encoding="utf-8")
    cfg = yaml.safe_load(raw)
    if not isinstance(cfg, dict):
        cfg = json.loads(cfg)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for short, inst_file in HOLDOUTS.items():
        inst_path = PILOT_ROOT / "local-instances" / inst_file
        if not inst_path.exists():
            raise SystemExit(f"missing instance file: {inst_path}")
        for variant, model_name in VARIANTS.items():
            c = json.loads(json.dumps(cfg))  # deep copy
            c["instances"]["path"] = str(inst_path.resolve())
            c["agent"]["tools"]["parse_function"] = {"type": "thought_action"}
            c["agent"]["templates"]["system_template"] = TA_SYSTEM_TEMPLATE
            m = c["agent"]["model"]  # model section nests under agent in this template
            m["name"] = model_name
            m["api_base"] = BASE_URL
            m["api_key"] = API_KEY
            m["completion_kwargs"] = {}
            m["litellm_model_registry"] = str(
                Path(__file__).resolve().parent.parent / "configs/local-model-registry.json"
            )
            c["output_dir"] = str(PILOT_ROOT / "holdout-eval" / "runs" / f"{short}-{variant}")
            out = OUT_DIR / f"{short}-{variant}.config.yaml"
            out.write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest.append({"instance": short, "variant": variant, "model": model_name, "config": str(out)})
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(manifest)} configs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
