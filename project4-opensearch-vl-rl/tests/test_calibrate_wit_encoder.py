"""CPU-only numeric tests for WIT encoder alignment decisions."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "calibrate_wit_encoder", PROJECT_ROOT / "scripts/calibrate_wit_encoder.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load calibrate_wit_encoder.py")
CALIBRATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CALIBRATION)


class CalibrateWitEncoderTest(unittest.TestCase):
    def test_accepts_same_space(self) -> None:
        published = np.eye(3, dtype=np.float32)
        computed = published * 2.0
        metrics = CALIBRATION.alignment_metrics(published, computed)
        self.assertTrue(metrics["aligned"])
        self.assertEqual(metrics["identity_top1_rate"], 1.0)

    def test_rejects_permuted_space(self) -> None:
        published = np.eye(3, dtype=np.float32)
        computed = published[[1, 2, 0]]
        metrics = CALIBRATION.alignment_metrics(published, computed)
        self.assertFalse(metrics["aligned"])
        self.assertEqual(metrics["identity_top1_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
