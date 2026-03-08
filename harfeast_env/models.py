"""
Data models for the HarFeast Environment.
Actions are JSON-serialized calls: {"action": "files.list", "path": "."}
"""

from pydantic import Field

try:
    from openenv.core.env_server.types import Action, Observation
except ImportError:
    from openenv.core.env_server.types import Action, Observation


class HarFeastAction(Action):
    """
    Action for HarFeast - JSON string encoding the action call.
    Example: '{"action": "files.list", "path": "."}'
    """

    action_json: str = Field(
        ...,
        min_length=2,
        description="JSON action: {\"action\": \"<name>\", ...params}. "
        "Actions: files.list, files.read, spreadsheet.read_range, "
        "data.filter, data.group_by, data.add_columns, data.compute, submit",
    )


class HarFeastObservation(Observation):
    """Observation from HarFeast - text result + metadata."""

    observation: str = Field(
        ...,
        description="Text output from the action (file list, table, confirmation, etc.)",
    )
    prompt: str = Field(
        default="",
        description="Current task prompt",
    )
    step_count: int = Field(
        default=0,
        ge=0,
        description="Number of steps taken",
    )
    datasets_available: str = Field(
        default="[]",
        description="JSON list of filtered dataset names available for chaining",
    )
