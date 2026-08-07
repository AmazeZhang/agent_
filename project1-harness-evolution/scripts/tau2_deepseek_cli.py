"""Project-owned τ²-bench CLI adapter for the DeepSeek V4 API.

The upstream τ²-bench defaults its NL assertion judge to GPT-4.1 and the
bundled LiteLLM price table predates DeepSeek V4. This adapter changes only
runtime configuration; vendor source remains untouched.
"""

import os

import litellm


MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
MODEL_WITH_PROVIDER = f"openai/{MODEL}"
API_BASE = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")

# Conservative cache-miss prices from the official DeepSeek pricing page on
# 2026-08-06. Costs are USD per token.
litellm.register_model(
    {
        MODEL: {
            "max_input_tokens": 1_000_000,
            "max_output_tokens": 384_000,
            "max_tokens": 384_000,
            "input_cost_per_token": 0.14 / 1_000_000,
            "output_cost_per_token": 0.28 / 1_000_000,
            "litellm_provider": "openai",
            "mode": "chat",
        }
    }
)

# The evaluator imports these names into its own module namespace, so patch the
# evaluator module directly before entering the CLI.
from tau2.evaluator import evaluator_nl_assertions

evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS = MODEL_WITH_PROVIDER
evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {
    "api_base": API_BASE,
    "max_tokens": 4096,
    "temperature": 0,
}

from tau2.cli import main


if __name__ == "__main__":
    main()

