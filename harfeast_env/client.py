"""
HarFeast Environment Client.
Connects to HarFeast OpenEnv server via WebSocket/HTTP.
"""

from typing import Any, Dict

try:
    from openenv.core.client_types import StepResult
    from openenv.core.env_server.types import State
    from openenv.core.env_client import EnvClient
    from harfeast_env.models import HarFeastAction, HarFeastObservation
except ImportError:
    from openenv.core.client_types import StepResult
    from openenv.core.env_server.types import State
    from openenv.core.env_client import EnvClient
    from models import HarFeastAction, HarFeastObservation


class HarFeastEnv(EnvClient[HarFeastAction, HarFeastObservation, State]):
    """
    Client for the HarFeast management consulting environment.
    """

    def _step_payload(self, action: HarFeastAction) -> Dict[str, Any]:
        """Convert HarFeastAction to JSON payload."""
        return {"action_json": action.action_json}

    def _parse_result(self, payload: Dict) -> StepResult[HarFeastObservation]:
        """Parse server response into StepResult."""
        obs_data = payload.get("observation", {})
        observation = HarFeastObservation(
            observation=obs_data.get("observation", ""),
            prompt=obs_data.get("prompt", ""),
            step_count=obs_data.get("step_count", 0),
            datasets_available=obs_data.get("datasets_available", "[]"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse state from server."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
