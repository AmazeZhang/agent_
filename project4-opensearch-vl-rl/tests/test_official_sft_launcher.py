"""CPU-only policy tests for the official SFT launcher."""

import importlib.util
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = PROJECT_ROOT / "scripts/run_official_sft.py"
    spec = importlib.util.spec_from_file_location("run_official_sft", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load official SFT launcher")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    launcher = load_module()
    with tempfile.TemporaryDirectory(prefix="p4-official-sft-launcher.") as temporary:
        run_root = Path(temporary) / "runs"
        run_dir = run_root / "run-1"
        run_dir.mkdir(parents=True)
        launcher.RUN_ROOT = run_root

        environment = {
            "PROJECT4_RUN_ID": "run-1",
            "PROJECT4_RUN_TOKEN": "token",
            "PROJECT4_RUN_DIR": str(run_dir),
            "CUDA_VISIBLE_DEVICES": "1",
        }
        assert launcher.require_managed_run(environment) == (run_dir.resolve(), "1")
        for forbidden in ("0", "5", "1,2"):
            environment["CUDA_VISIBLE_DEVICES"] = forbidden
            try:
                launcher.require_managed_run(environment)
            except RuntimeError:
                pass
            else:
                raise AssertionError(f"forbidden GPU selection passed: {forbidden}")

        config = launcher.training_config(run_dir, 5, None)
        assert config["max_steps"] == 5
        assert config["save_steps"] == 5
        assert config["template"] == "qwen3_vl"
        assert config["dataset"] == "wiki_en_official_1000"
        assert config["finetuning_type"] == "lora"
        assert config["freeze_vision_tower"] is True
        assert config["freeze_multi_modal_projector"] is True
        assert config["overwrite_output_dir"] is False

    print("official SFT launcher tests: PASS")


if __name__ == "__main__":
    main()
