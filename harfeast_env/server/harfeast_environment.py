"""
HarFeast Environment - OpenEnv server implementation.
Management consulting tasks with file, spreadsheet, and data actions.
"""

import json
import os
from uuid import uuid4

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
except ImportError:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State

# Import our core logic - use path relative to project root
import sys
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from harfeast_openenv.environment import HarFeastOpenEnv
from harfeast_openenv.schemas import StepResult
from harfeast_env.models import HarFeastAction, HarFeastObservation


class HarFeastEnvironment(Environment[HarFeastAction, HarFeastObservation, State]):
    """
    OpenEnv wrapper for HarFeast management consulting environment.
    Supports files.list, files.read, spreadsheet.read_range, data actions, submit.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = False  # Session state (filtered datasets)

    def __init__(self, world_path: str | None = None, worlds_base: str | None = None):
        self._world_path = world_path or os.path.join(_project_root, "harfeast_world")
        self._worlds_base = (worlds_base or os.environ.get("HARFEAST_WORLDS_BASE") or "").strip() or None
        self._env = HarFeastOpenEnv(
            world_path=self._world_path,
            worlds_base=os.path.abspath(self._worlds_base) if self._worlds_base else None,
        )
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
        **kwargs,
    ) -> HarFeastObservation:
        """Reset environment and load a task. Supports task_index for augmented dataset."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        result: StepResult = self._env.reset(
            seed=seed,
            task_id=task_id or kwargs.get("task_id"),
            task_index=kwargs.get("task_index"),
            **{k: v for k, v in kwargs.items() if k not in ("task_id", "task_index")},
        )
        return self._step_result_to_obs(result)

    def step(
        self,
        action: HarFeastAction,
        timeout_s: float | None = None,
        **kwargs,
    ) -> HarFeastObservation:
        """Execute action (action_json) and return observation."""
        try:
            action_dict = json.loads(action.action_json)
        except json.JSONDecodeError as e:
            return HarFeastObservation(
                observation=f"Invalid action JSON: {e}",
                prompt=self._env._prompt,
                step_count=self._env._step_count,
                datasets_available=json.dumps(list(self._env._filtered_datasets.keys())),
                done=False,
                reward=0.0,
                metadata={"error": str(e)},
            )
        result: StepResult = self._env.step(action_dict)
        self._state.step_count = result.step_count
        return self._step_result_to_obs(result)

    def _step_result_to_obs(self, r: StepResult) -> HarFeastObservation:
        """Convert our StepResult to HarFeastObservation."""
        return HarFeastObservation(
            observation=r.observation,
            prompt=r.prompt,
            step_count=r.step_count,
            datasets_available=json.dumps(r.info.get("datasets_available", [])),
            done=r.done,
            reward=r.reward,
            metadata={
                "action_taken": r.info.get("action_taken"),
                "last_error": r.info.get("last_error"),
                "task_id": self._env.state.get("task_id"),
            },
        )

    @property
    def state(self) -> State:
        """Current episode state."""
        return self._state
