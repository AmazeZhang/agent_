from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from searchr1_repro.training_audit import build_rollout_audit_records, dump_rollout_audit


class TrainingAuditTest(unittest.TestCase):
    def fixture(self):
        batch = {
            "prompts": torch.tensor([[0, 11, 12], [21, 22, 23]]),
            "responses": torch.tensor([[31, 32], [41, 0]]),
            "input_ids": torch.tensor([[0, 11, 12, 31, 32], [21, 22, 23, 41, 0]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 0]]),
        }
        metadata = {
            "uid": np.array(["question-a", "question-b"], dtype=object),
            "traj_uid": np.array(["trajectory-a", "trajectory-b"], dtype=object),
            "env_step": np.array([0, 1]),
            "retrieval": np.array([{"status": "success", "document_ids": ["7", "9"]}, None], dtype=object),
            "retrieval_failed": np.array([False, False]),
            "is_action_valid": np.array([True, True]),
        }
        return batch, metadata

    def test_single_turn_prompt_tokens_are_zero_masked(self):
        batch, metadata = self.fixture()
        records = build_rollout_audit_records(batch, metadata, multi_turn=False)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["policy_loss_mask"], [0, 0, 0, 1, 1])
        self.assertEqual(records[1]["policy_loss_mask"], [0, 0, 0, 1, 0])
        self.assertEqual(records[0]["prompt_policy_loss_tokens"], 0)
        self.assertEqual(records[0]["metadata"]["retrieval"]["document_ids"], ["7", "9"])

    def test_multi_turn_uses_explicit_loss_mask_and_atomic_dump(self):
        batch, metadata = self.fixture()
        batch["loss_mask"] = torch.tensor([[0, 0, 0, 1, 0], [0, 0, 0, 1, 0]])
        records = build_rollout_audit_records(batch, metadata, multi_turn=True)
        self.assertEqual(records[0]["mask_source"], "loss_mask")
        self.assertEqual(records[0]["policy_loss_tokens"], 1)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "1.audit.jsonl"
            dump_rollout_audit(SimpleNamespace(batch=batch, non_tensor_batch=metadata), output, multi_turn=True)
            saved = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(saved, records)
            with self.assertRaises(FileExistsError):
                dump_rollout_audit(SimpleNamespace(batch=batch, non_tensor_batch=metadata), output, multi_turn=True)


if __name__ == "__main__":
    unittest.main()
