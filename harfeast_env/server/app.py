"""
FastAPI application for HarFeast Environment.
Exposes the environment over HTTP/WebSocket for OpenEnv clients.
"""

import os
import sys

# Ensure project root is on path for harfeast_openenv imports
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from openenv.core.env_server.http_server import create_app
from harfeast_env.models import HarFeastAction, HarFeastObservation
from harfeast_env.server.harfeast_environment import HarFeastEnvironment

# World path - use env var or default to project harfeast_world
WORLD_PATH = os.environ.get("HARFEAST_WORLD_PATH") or os.path.join(_project_root, "harfeast_world")
# Augmented dataset - base dir with all_tasks.json (e.g. harfeast_worlds)
WORLDS_BASE = os.environ.get("HARFEAST_WORLDS_BASE")


def _env_factory():
    """Environment factory for create_app."""
    return HarFeastEnvironment(world_path=WORLD_PATH, worlds_base=WORLDS_BASE)


app = create_app(
    _env_factory,
    HarFeastAction,
    HarFeastObservation,
    env_name="harfeast_env",
)


@app.get("/")
def root():
    return {
        "name": "HarFeast OpenEnv",
        "description": "Management consulting RL environment with 14 APEX-style analytical tasks",
        "version": "0.1.0",
        "tasks": 14,
        "tools": [
            "files.list", "files.read", "spreadsheet.read_range",
            "data.filter", "data.group_by", "data.add_columns",
            "data.compute", "submit",
        ],
        "endpoints": {
            "info": "/info",
            "reset": "/reset",
            "step": "/step",
            "health": "/health",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
