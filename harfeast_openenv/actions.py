"""Action handlers for HarFeast OpenEnv."""

import os
from harfeast_openenv.schemas import ActionResult


def handle_files_list(world_path: str, path: str = ".") -> ActionResult:
    """
    List files and directories at the given path.
    Path can be ".", "data", "documents", or a subpath like "documents".
    """
    base = os.path.normpath(os.path.join(world_path, path))
    if not os.path.isdir(base):
        return ActionResult(
            observation=f"Path '{path}' does not exist or is not a directory.",
            success=False,
            error=f"Invalid path: {path}",
        )
    
    # Ensure we don't escape world_path
    world_abs = os.path.abspath(world_path)
    base_abs = os.path.abspath(base)
    if not base_abs.startswith(world_abs):
        return ActionResult(
            observation="Access denied: path outside world directory.",
            success=False,
            error="Path traversal not allowed",
        )
    
    items = sorted(os.listdir(base))
    files = []
    for name in items:
        full = os.path.join(base, name)
        if os.path.isfile(full):
            files.append({"name": name, "type": "file"})
        else:
            files.append({"name": name + "/", "type": "directory"})
    
    import json
    return ActionResult(
        observation=json.dumps({"path": path, "items": files}, indent=2),
    )


def handle_files_read(world_path: str, path: str) -> ActionResult:
    """
    Read a text document. Only allows .txt files in documents/.
    Rejects CSV paths with a message to use spreadsheet.read_range.
    """
    # Normalize path: accept "scrap_rate_report.txt", "documents/scrap_rate_report.txt", etc.
    path = path.strip().lstrip("/")
    if not path.startswith("documents"):
        path = "documents/" + path
    
    full_path = os.path.normpath(os.path.join(world_path, path))
    
    # Security: ensure within world_path
    world_abs = os.path.abspath(world_path)
    full_abs = os.path.abspath(full_path)
    if not full_abs.startswith(world_abs):
        return ActionResult(
            observation="Access denied: path outside world directory.",
            success=False,
            error="Path traversal not allowed",
        )
    
    # Reject CSV files
    if path.endswith(".csv") or "data/" in path:
        return ActionResult(
            observation=(
                "CSV files cannot be read with files.read. "
                "Use spreadsheet.read_range(file, range) to read CSV data."
            ),
            success=False,
            error="Use spreadsheet.read_range for CSV files",
        )
    
    if not os.path.isfile(full_path):
        return ActionResult(
            observation=f"File not found: {path}",
            success=False,
            error=f"File not found: {path}",
        )
    
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
        return ActionResult(observation=content)
    except Exception as e:
        return ActionResult(
            observation=f"Error reading file: {e}",
            success=False,
            error=str(e),
        )
