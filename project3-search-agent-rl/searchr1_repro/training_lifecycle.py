from __future__ import annotations

from collections.abc import Iterable
import time
from typing import Any


def _actor_id(worker: Any) -> str:
    actor_id = getattr(worker, "_actor_id", None)
    if actor_id is None or not hasattr(actor_id, "hex"):
        raise TypeError("Ray worker handle is missing _actor_id.hex()")
    return actor_id.hex()


def shutdown_worker_groups(
    worker_groups: Iterable[Any],
    ray_api: Any,
    *,
    timeout_seconds: float = 30,
    actor_state_getter: Any = None,
) -> list[str]:
    """Request one intentional exit per physical worker actor and wait for it."""
    unique_workers: dict[str, Any] = {}
    for worker_group in worker_groups:
        if worker_group is None:
            continue
        for worker in worker_group.workers:
            unique_workers.setdefault(_actor_id(worker), worker)

    return shutdown_actors(
        unique_workers.values(),
        ray_api,
        timeout_seconds=timeout_seconds,
        actor_state_getter=actor_state_getter,
    )


def shutdown_actors(
    actors: Iterable[Any],
    ray_api: Any,
    *,
    timeout_seconds: float = 30,
    actor_state_getter: Any = None,
) -> list[str]:
    """Request one intentional exit per unique Ray actor and wait for DEAD."""
    unique_workers: dict[str, Any] = {}
    for actor in actors:
        unique_workers.setdefault(_actor_id(actor), actor)

    if not unique_workers:
        return []

    deadline = time.monotonic() + timeout_seconds
    exit_refs = [worker.graceful_shutdown.remote() for worker in unique_workers.values()]
    ready, pending = ray_api.wait(exit_refs, num_returns=len(exit_refs), timeout=timeout_seconds)
    if pending:
        pending_ids = [actor_id for actor_id, ref in zip(unique_workers, exit_refs) if ref in pending]
        raise TimeoutError(f"Ray actors did not exit within {timeout_seconds}s: {pending_ids}")
    if len(ready) != len(exit_refs):
        raise RuntimeError("Ray returned an inconsistent graceful-shutdown wait result")
    if actor_state_getter is None:
        from ray._private.state import state

        actor_state_getter = state.actor_table
    remaining_ids = set(unique_workers)
    while remaining_ids and time.monotonic() < deadline:
        alive_ids = set()
        for actor_id in remaining_ids:
            actor_state = actor_state_getter(actor_id) or {}
            state_name = actor_state.get("state", actor_state.get("State"))
            if state_name != "DEAD":
                alive_ids.add(actor_id)
        remaining_ids = alive_ids
        if remaining_ids:
            time.sleep(0.05)
    if remaining_ids:
        raise TimeoutError(f"Ray actors acknowledged exit but did not reach DEAD state: {sorted(remaining_ids)}")
    return list(unique_workers)
