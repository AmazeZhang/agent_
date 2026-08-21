"""Deterministic offline RL contracts for the local visual-search agent."""

from .reward import compute_group_advantages, score_trajectory

__all__ = ["compute_group_advantages", "score_trajectory"]
