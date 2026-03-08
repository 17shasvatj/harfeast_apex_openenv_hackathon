"""Action and observation schemas for HarFeast OpenEnv."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ActionResult:
    """Result of executing an action."""
    observation: str
    success: bool = True
    error: str | None = None


@dataclass
class StepResult:
    """Result returned by environment.step()."""
    observation: str
    prompt: str
    step_count: int
    done: bool
    reward: float
    info: dict[str, Any] = field(default_factory=dict)


def parse_action(action: dict | str) -> tuple[str, dict]:
    """
    Parse action from dict or JSON string.
    Returns (action_name, params).
    """
    if isinstance(action, str):
        import json
        action = json.loads(action)
    
    if not isinstance(action, dict) or "action" not in action:
        raise ValueError("Action must be a dict with 'action' key")
    
    name = action["action"]
    params = {k: v for k, v in action.items() if k != "action"}
    return name, params
