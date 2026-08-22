"""P3 patch 0009 config-schema fix: real hydra compose + struct path tests.

Reproduces the exact startup blocker from
docs/P3_V1_GPU_SMOKE_STOP_REPORT_2026-08-19.md §2:

    ray_trainer.py:658 OmegaConf.set_struct(config, True)
    -> make_envs -> build_search_envs -> SearchMultiProcessEnv.__init__
    -> per-env deepcopy of env.search
    -> write search_aware_step_reward (patch 0008 propagation)

Before patch 0009 the env.search schema only had log_requests / search_url /
topk / timeout, so the struct write raised:

    ConfigAttributeError: Key 'search_aware_step_reward' is not in struct

All tests are CPU-only (no GPU, no real retriever; a fake local retriever
serves step() only where needed). The wrapper's REAL overrides are obtained
by running scripts/run_p3_grpo_search_aware_v1.sh --dump-overrides, so the
composed config is exactly what the training launch uses.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = PROJECT_ROOT / "vendor" / "verl-agent"
CONFIG_DIR = VENDOR_DIR / "verl" / "trainer" / "config"
sys.path.insert(0, str(PROJECT_ROOT))

from agent_system.environments.env_package.search.envs import SearchMultiProcessEnv  # noqa: E402

# Fake retriever: every query returns one document containing the answer.
DOC_CONTENTS = "Paris is the capital of France and the most populous city."


class _RetrieverHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        resp = {"result": [[{"document": {"id": "0", "contents": DOC_CONTENTS}}]]}
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # keep test output clean
        pass


@pytest.fixture(scope="module")
def retriever_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RetrieverHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/retrieve"
    finally:
        server.shutdown()
        server.server_close()


def _recursive_struct(cfg) -> None:
    """Mirror the real struct state: ray_trainer set_struct(config, True) plus
    the struct flag hydra propagates to nested nodes of a composed config."""
    from omegaconf import DictConfig

    OmegaConf.set_struct(cfg, True)
    for value in cfg.values():
        if isinstance(value, DictConfig):
            _recursive_struct(value)


@pytest.fixture(scope="module")
def wrapper_overrides() -> list[str]:
    """Real overrides from the v1 wrapper (--dump-overrides)."""
    script = PROJECT_ROOT / "scripts" / "run_p3_grpo_search_aware_v1.sh"
    result = subprocess.run(
        ["bash", str(script), "--dump-overrides"],
        capture_output=True,
        text=True,
        check=False,
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, (
        f"wrapper --dump-overrides failed (rc={result.returncode}): "
        f"{result.stderr[-2000:]}"
    )
    lines = [ln for ln in result.stdout.splitlines() if ln.startswith("[V1_EXP]") is False]
    overrides = [ln for ln in lines if "=" in ln and not ln.startswith("__config_fp__")]
    assert overrides, "no overrides parsed from wrapper output"
    return overrides


@pytest.fixture(scope="module")
def v1_cfg(wrapper_overrides):
    """Composed config from the REAL ppo_trainer.yaml + the wrapper's real
    overrides (v1 switches ON), in struct mode exactly like the training run."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="ppo_trainer", overrides=wrapper_overrides)
    _recursive_struct(cfg)
    # patch 0009: the schema key exists under env.search with default false,
    # while the top-level master switch is on via +env.search_aware_step_reward
    assert cfg.env.search.search_aware_step_reward is False
    assert cfg.env.search_aware_step_reward is True
    return cfg


@pytest.fixture(scope="module")
def default_cfg():
    """Default official-loose path: same wrapper overrides but WITHOUT the
    three +v1 switches (0007/0008 off), like a non-v1 launch."""
    from hydra import compose, initialize_config_dir

    script = PROJECT_ROOT / "scripts" / "run_p3_grpo_search_aware_v1.sh"
    result = subprocess.run(
        ["bash", str(script), "--dump-overrides"],
        capture_output=True,
        text=True,
        check=False,
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0
    overrides = [ln for ln in result.stdout.splitlines()
                 if "=" in ln and not ln.startswith("__config_fp__") and not ln.startswith("[V1_EXP]")]
    overrides = [o for o in overrides
                 if o not in ("+env.search_aware_step_reward=true",
                              "+reward_model.search_aware_step_reward=true",
                              "+algorithm.search_v1_trajectory_return=true")]
    with initialize_config_dir(config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="ppo_trainer", overrides=overrides)
    _recursive_struct(cfg)
    return cfg


def _make_env(env_config, retriever_url: str, *, env_num: int = 2) -> SearchMultiProcessEnv:
    # point the composed config's search_url at the fake retriever, then build
    # through the REAL struct path (deepcopy + propagation write of 0008)
    env_config.search.search_url = retriever_url
    return SearchMultiProcessEnv(
        seed=0, env_num=env_num, group_n=1, is_train=False, env_config=env_config
    )


class TestStructSchema:
    def test_v1_config_constructs_without_config_attribute_error(self, v1_cfg, retriever_url):
        """The exact blocker: struct write of search_aware_step_reward must not
        raise under the real composed+structed config."""
        env = _make_env(v1_cfg.env, retriever_url)
        try:
            assert len(env.envs) == 2
        finally:
            env.close()

    def test_flag_true_propagates_to_every_env(self, v1_cfg, retriever_url):
        env = _make_env(v1_cfg.env, retriever_url, env_num=4)
        try:
            flags = [e.search_aware_step_reward for e in env.envs]
            assert flags == [True] * 4, flags
        finally:
            env.close()

    def test_default_path_flag_false_and_key_present(self, default_cfg, retriever_url):
        """official-loose default: schema key exists (patch 0009) defaulting to
        false, top-level switch absent -> propagation writes false."""
        assert default_cfg.env.search.search_aware_step_reward is False
        assert "search_aware_step_reward" not in default_cfg.env
        env = _make_env(default_cfg.env, retriever_url, env_num=3)
        try:
            flags = [e.search_aware_step_reward for e in env.envs]
            assert flags == [False] * 3, flags
        finally:
            env.close()

    def test_default_path_behavior_unchanged_no_v1_metadata(self, default_cfg, retriever_url):
        """official-loose step with flag off: plain observation, no search_v1
        metadata, normal validity -- identical to pre-0009 behavior."""
        env = _make_env(default_cfg.env, retriever_url)
        try:
            obs, info = env.reset([{"question": "What is the capital of France?",
                                    "ground_truth": {"target": ["Paris"]},
                                    "data_source": "confirm256"}])
            assert len(obs) == 1
            obs, reward, done, info = env.step(["<search>Paris capital</search>"])
            assert info[0]["retrieval"]["status"] == "success"
            assert "search_v1" not in info[0], "v1 metadata must not exist with flag off"
            # plain observation carries the retrieval result, unchanged semantics
            assert "Paris" in obs[0]
            obs, reward, done, info = env.step(["<answer>Paris</answer>"])
            assert done[0] and reward[0] == 1.0
        finally:
            env.close()
