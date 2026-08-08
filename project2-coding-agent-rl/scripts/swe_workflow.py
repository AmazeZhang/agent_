"""WP6: single-turn SWE workflow for the Agentic RL (GRPO) smoke.

The model receives the problem statement and must produce a candidate
patch; the hidden-test reward function (SweHiddenTestReward) grades it
against the eval repo. One Step per episode, terminated ENV_DONE —
mirrors rllm's SimpleWorkflow (countdown) shape so the verl agent
workflow path handles it unchanged.

The reward function executes in the workflow process (CPU side) and
spawns per-repo pytest via subprocess.
"""

from __future__ import annotations

from rllm.engine import ModelOutput, RolloutEngine
from rllm.rewards.reward_fn import RewardFunction
from rllm.types import Action, Step
from rllm.workflows.workflow import TerminationEvent, TerminationReason, Workflow

from swe_reward import SweHiddenTestReward

SYSTEM_PROMPT = (
    "You are an expert software engineer. You are given a GitHub issue "
    "description. Produce a minimal patch (unified diff) that fixes the "
    "issue. Output ONLY the diff, inside a ```diff code block, with no "
    "surrounding commentary."
)


class SweSingleTurnWorkflow(Workflow):
    def __init__(self, rollout_engine: RolloutEngine, reward_fn: RewardFunction | SweHiddenTestReward, **kwargs):
        super().__init__(rollout_engine, **kwargs)
        self.reward_fn = reward_fn

    async def run(self, task: dict, uid: str, **kwargs):
        self.reset(task, uid)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task["problem_statement"]},
        ]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages, application_id=uid, **kwargs)

        # Robustness: rely on the raw decoded completion when the chat parser
        # misroutes content to reasoning (e.g. a <think> tag in a non-thinking
        # model's output). verl's GRPO update always uses completion_ids, so
        # only the reward link is affected — reward must see the real text.
        action = Action(action=output.content or output.text)
        reward = float(self.reward_fn(task, action))

        # Append a single-step trajectory so verl sees the response + reward.
        from rllm.agents.agent import BaseAgent

        agent = SimpleTrajAgent()
        agent.trajectory.steps.append(
            Step(
                chat_completions=messages + [{"role": "assistant", "content": output.content}],
                thought=output.reasoning,
                action=action,
                reward=reward,
                model_output=output,
            )
        )
        self.commit(agent=agent, reset=True)

        if output.finish_reason == "length":
            raise TerminationEvent(TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED)

        raise TerminationEvent(TerminationReason.ENV_DONE)


class SimpleTrajAgent:
    """Minimal agent-shaped holder matching the BaseAgent interface the
    workflow path consumes (trajectory + reset)."""

    def __init__(self):
        from rllm.types import Trajectory

        self._trajectory = Trajectory()

    @property
    def trajectory(self):
        return self._trajectory

    def reset(self):
        from rllm.types import Trajectory

        self._trajectory = Trajectory()
