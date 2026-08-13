from __future__ import annotations

import unittest
from types import SimpleNamespace

from searchr1_repro.training_lifecycle import shutdown_worker_groups


class FakeRemoteMethod:
    def __init__(self, ref):
        self.ref = ref
        self.calls = 0

    def remote(self):
        self.calls += 1
        return self.ref


class FakeWorker:
    def __init__(self, actor_id):
        self._actor_id = SimpleNamespace(hex=lambda: actor_id)
        self.graceful_shutdown = FakeRemoteMethod(f"ref-{actor_id}")


class FakeRay:
    def __init__(self, pending=()):
        self.pending = list(pending)

    def wait(self, refs, *, num_returns, timeout):
        self.args = (list(refs), num_returns, timeout)
        return [ref for ref in refs if ref not in self.pending], self.pending


class TrainingLifecycleTest(unittest.TestCase):
    def test_shared_physical_actor_is_shutdown_once(self):
        worker = FakeWorker("actor-a")
        groups = [SimpleNamespace(workers=[worker]), SimpleNamespace(workers=[worker])]
        ray_api = FakeRay()
        state_getter = lambda _: {"State": "DEAD"}
        self.assertEqual(
            shutdown_worker_groups(groups, ray_api, timeout_seconds=9, actor_state_getter=state_getter),
            ["actor-a"],
        )
        self.assertEqual(worker.graceful_shutdown.calls, 1)
        self.assertEqual(ray_api.args, (["ref-actor-a"], 1, 9))

    def test_timeout_names_only_pending_actor(self):
        first = FakeWorker("actor-a")
        second = FakeWorker("actor-b")
        with self.assertRaisesRegex(TimeoutError, "actor-b"):
            shutdown_worker_groups(
                [SimpleNamespace(workers=[first, second])],
                FakeRay(pending=["ref-actor-b"]),
                timeout_seconds=3,
                actor_state_getter=lambda _: {"state": "DEAD"},
            )

    def test_empty_groups_are_a_noop(self):
        self.assertEqual(shutdown_worker_groups([None], FakeRay()), [])

    def test_acknowledged_actor_must_reach_dead_state(self):
        worker = FakeWorker("actor-a")
        with self.assertRaisesRegex(TimeoutError, "did not reach DEAD"):
            shutdown_worker_groups(
                [SimpleNamespace(workers=[worker])],
                FakeRay(),
                timeout_seconds=0,
                actor_state_getter=lambda _: {"state": "ALIVE"},
            )


if __name__ == "__main__":
    unittest.main()
