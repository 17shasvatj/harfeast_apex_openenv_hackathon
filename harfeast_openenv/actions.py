"""Action handlers for HarFeast OpenEnv."""

import ast
import csv
import json
import operator
import os
import re
from collections import defaultdict
from statistics import median as stat_median

from .schemas import ActionResult


# ── Observation size limits ──────────────────────────────────────
MAX_TABLE_ROWS = 20

# ── Safe arithmetic evaluator (replaces eval) ────────────────────
_SAFE_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
}

def _safe_eval_expr(node, namespace=None):
    """Evaluate an AST node containing only arithmetic on numbers (and optionally named vars)."""
    if isinstance(node, ast.Expression):
        return _safe_eval_expr(node.body, namespace)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BINOPS:
        left = _safe_eval_expr(node.left, namespace)
        right = _safe_eval_expr(node.right, namespace)
        return _SAFE_BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_safe_eval_expr(node.operand, namespace)
    if isinstance(node, ast.Name) and namespace is not None:
        if node.id in namespace:
            return namespace[node.id]
        raise ValueError(f"Unknown variable: {node.id}")
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")
MAX_DOCUMENT_CHARS = 2000
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
        if len(content) > MAX_DOCUMENT_CHARS:
            total = len(content)
            content = content[:MAX_DOCUMENT_CHARS] + (
                f"\n\n[Truncated — showing first {MAX_DOCUMENT_CHARS} of {total} characters.]"
            )
        return ActionResult(observation=content)
    except Exception as e:
        return ActionResult(
            observation=f"Error reading file: {e}",
            success=False,
            error=str(e),
        )


def _resolve_csv_path(world_path: str, file_or_dataset: str) -> str:
    """Resolve file/dataset name to full CSV path. Reject path traversal."""
    file_or_dataset = file_or_dataset.strip()
    if not file_or_dataset.lower().endswith(".csv"):
        file_or_dataset = file_or_dataset + ".csv"
    if not file_or_dataset.lower().startswith("data"):
        file_or_dataset = "data/" + file_or_dataset.lstrip("/")
    full = os.path.normpath(os.path.join(world_path, file_or_dataset))
    world_abs = os.path.abspath(world_path)
    full_abs = os.path.abspath(full)
    if not full_abs.startswith(world_abs) or not full_abs.endswith(".csv"):
        raise ValueError(f"Invalid path: {file_or_dataset}")
    return full


def _load_csv_rows(path: str) -> tuple[list[str], list[dict]]:
    """Load CSV as (columns, rows)."""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        rows = list(reader)
    return columns, rows


def _get_table(world_path: str, dataset: str, filtered_datasets: dict) -> tuple[list[str], list[dict]]:
    """Load table (columns, rows) from CSV file or filtered dataset."""
    if dataset in filtered_datasets:
        stored = filtered_datasets[dataset]
        cols = stored["columns"]
        rows = [dict(r) for r in stored["rows"]]
        return cols, rows
    path = _resolve_csv_path(world_path, dataset)
    return _load_csv_rows(path)


def _format_table(columns: list[str], rows: list[dict], max_rows: int | None = None) -> str:
    """Format as text table. Defaults to MAX_TABLE_ROWS."""
    if max_rows is None:
        max_rows = MAX_TABLE_ROWS
    if not rows:
        return " | ".join(columns) + "\n(0 rows)"
    lines = [" | ".join(columns)]
    for r in rows[:max_rows]:
        lines.append(" | ".join(str(r.get(c, "")) for c in columns))
    if len(rows) > max_rows:
        lines.append(f"\n[Showing {max_rows} of {len(rows)} rows. Use data.filter to narrow results.]")
    return "\n".join(lines)


def handle_spreadsheet_read_range(
    world_path: str,
    file: str,
    range_spec: str,
) -> ActionResult:
    """
    Read rows from a CSV file.
    range: "columns" (headers only), "1:10" (rows 1-10), "all" (everything).
    """
    try:
        path = _resolve_csv_path(world_path, file)
    except ValueError as e:
        return ActionResult(observation=str(e), success=False, error=str(e))
    if not os.path.isfile(path):
        return ActionResult(
            observation=f"File not found: {file}",
            success=False,
            error=f"File not found: {file}",
        )
    try:
        columns, rows = _load_csv_rows(path)
    except Exception as e:
        return ActionResult(
            observation=f"Error reading CSV: {e}",
            success=False,
            error=str(e),
        )
    range_spec = str(range_spec).strip().lower()
    if range_spec == "columns":
        obs = json.dumps({"columns": columns}, indent=2)
        return ActionResult(observation=obs)
    if range_spec == "all":
        table = _format_table(columns, rows)
        return ActionResult(observation=table)
    # Parse "1:10" format (1-indexed inclusive)
    m = re.match(r"(\d+)\s*:\s*(\d+)", range_spec)
    if m:
        start, end = int(m.group(1)), int(m.group(2))
        start = max(1, start)
        end = min(len(rows), end)
        if start > end:
            return ActionResult(
                observation="Invalid range: start > end",
                success=False,
                error="Invalid range",
            )
        subset = rows[start - 1 : end]
        table = _format_table(columns, subset, max_rows=len(subset))
        return ActionResult(observation=table)
    return ActionResult(
        observation=f"Invalid range: '{range_spec}'. Use 'columns', 'all', or 'start:end' (e.g. '1:10').",
        success=False,
        error="Invalid range",
    )


def _try_float(x: str) -> float | str:
    """Try to parse as float, else return string."""
    try:
        return float(x)
    except (ValueError, TypeError):
        return str(x).strip()


def _row_matches(row: dict, column: str, op: str, compare_val: float | str) -> bool:
    """Check if row matches filter."""
    raw = row.get(column, "")
    is_numeric = isinstance(compare_val, (int, float))
    if op == "contains":
        return str(compare_val).lower() in str(raw).lower()
    if is_numeric:
        try:
            cell = float(raw) if raw != "" else float("nan")
        except (ValueError, TypeError):
            return False
    else:
        cell = str(raw).strip()
    if op == "eq":
        return cell == compare_val
    if op == "neq":
        return cell != compare_val
    if op == "gt":
        return is_numeric and cell > compare_val
    if op == "lt":
        return is_numeric and cell < compare_val
    if op == "gte":
        return is_numeric and cell >= compare_val
    if op == "lte":
        return is_numeric and cell <= compare_val
    return False


def handle_data_filter(
    world_path: str,
    dataset: str,
    column: str,
    operator: str,
    value: str,
    filtered_datasets: dict,
) -> ActionResult:
    """
    Filter rows. Operators: eq, neq, gt, lt, gte, lte, contains.
    Stores result as filtered_0, filtered_1, ... in filtered_datasets.
    """
    try:
        columns, rows = _get_table(world_path, dataset, filtered_datasets)
    except Exception as e:
        return ActionResult(observation=str(e), success=False, error=str(e))
    if column not in columns:
        return ActionResult(
            observation=f"Column '{column}' not found. Available: {columns}",
            success=False,
            error=f"Column not found: {column}",
        )
    op = operator.strip().lower()
    if op not in ("eq", "neq", "gt", "lt", "gte", "lte", "contains"):
        return ActionResult(
            observation=f"Unknown operator: {operator}. Use: eq, neq, gt, lt, gte, lte, contains.",
            success=False,
            error=f"Unknown operator: {operator}",
        )
    compare_val = str(value).strip() if op == "contains" else _try_float(value)
    try:
        filtered = [r for r in rows if _row_matches(r, column, op, compare_val)]
    except Exception as e:
        return ActionResult(
            observation=f"Filter error: {e}",
            success=False,
            error=str(e),
        )
    next_idx = len([k for k in filtered_datasets if k.startswith("filtered_")])
    store_name = f"filtered_{next_idx}"
    filtered_datasets[store_name] = {"columns": columns, "rows": filtered}
    return ActionResult(
        observation=json.dumps({"rows": len(filtered), "stored_as": store_name}, indent=2),
    )


def handle_data_group_by(
    world_path: str,
    dataset: str,
    column: str,
    aggregation: str,
    target_column: str,
    filtered_datasets: dict,
) -> ActionResult:
    """Group by column and aggregate target_column. Aggregations: sum, mean, median, count, min, max."""
    try:
        columns, rows = _get_table(world_path, dataset, filtered_datasets)
    except Exception as e:
        return ActionResult(observation=str(e), success=False, error=str(e))
    if column not in columns:
        return ActionResult(
            observation=f"Column '{column}' not found. Available: {columns}",
            success=False,
            error=f"Column not found: {column}",
        )
    if target_column not in columns:
        return ActionResult(
            observation=f"Column '{target_column}' not found. Available: {columns}",
            success=False,
            error=f"Column not found: {target_column}",
        )
    agg = aggregation.strip().lower()
    if agg not in ("sum", "mean", "median", "count", "min", "max"):
        return ActionResult(
            observation=f"Unknown aggregation: {aggregation}. Use: sum, mean, median, count, min, max.",
            success=False,
            error=f"Unknown aggregation: {aggregation}",
        )
    try:
        groups: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            key = str(r.get(column, ""))
            raw = r.get(target_column, "")
            try:
                val = float(raw)
            except (ValueError, TypeError):
                if agg == "count":
                    val = 1
                else:
                    continue
            groups[key].append(val)
        result_rows = []
        for key in sorted(groups.keys()):
            vals = groups[key]
            if agg == "sum":
                v = sum(vals)
            elif agg == "mean":
                v = sum(vals) / len(vals) if vals else 0
            elif agg == "median":
                v = stat_median(vals) if vals else 0
            elif agg == "count":
                v = len(vals)
            elif agg == "min":
                v = min(vals) if vals else 0
            else:  # max
                v = max(vals) if vals else 0
            result_rows.append({column: key, f"{agg}({target_column})": round(v, 2) if isinstance(v, float) else v})
        table = _format_table([column, f"{agg}({target_column})"], result_rows, max_rows=1000)
        return ActionResult(observation=table)
    except Exception as e:
        return ActionResult(
            observation=f"Group-by error: {e}",
            success=False,
            error=str(e),
        )


def handle_data_add_columns(
    world_path: str,
    dataset: str,
    new_column: str,
    expression: str,
    filtered_datasets: dict,
) -> ActionResult:
    """Create derived column from expression (e.g. 'a + b + c')."""
    try:
        columns, rows = _get_table(world_path, dataset, filtered_datasets)
    except Exception as e:
        return ActionResult(observation=str(e), success=False, error=str(e))
    # Restrict expression to column names and arithmetic
    allowed = set("abcdefghijklmnopqrstuvwxyz_0123456789.+-*/() ")
    if not all(c in allowed for c in expression.lower().replace(" ", "")):
        return ActionResult(
            observation="Expression may only contain column names and +, -, *, /, (, ).",
            success=False,
            error="Invalid expression",
        )
    # Verify all names in expression are columns
    try:
        tree = ast.parse(expression, mode="eval")
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        for n in names:
            if n not in columns:
                return ActionResult(
                    observation=f"Column '{n}' in expression not found. Available: {columns}",
                    success=False,
                    error=f"Column not found: {n}",
                )
    except SyntaxError as e:
        return ActionResult(
            observation=f"Invalid expression syntax: {e}",
            success=False,
            error=str(e),
        )
    try:
        new_rows = []
        for r in rows:
            row = dict(r)
            ns = {}
            for c in columns:
                v = _try_float(row.get(c, ""))
                ns[c] = v if isinstance(v, (int, float)) else 0
            try:
                row[new_column] = round(_safe_eval_expr(tree, namespace=ns), 2)
            except Exception:
                row[new_column] = 0
            new_rows.append(row)
        new_columns = columns + [new_column]
        next_idx = len([k for k in filtered_datasets if k.startswith("filtered_")])
        store_name = f"filtered_{next_idx}"
        filtered_datasets[store_name] = {"columns": new_columns, "rows": new_rows}
        return ActionResult(
            observation=json.dumps({"rows": len(new_rows), "stored_as": store_name, "new_column": new_column}, indent=2),
        )
    except Exception as e:
        return ActionResult(
            observation=f"Expression error: {e}",
            success=False,
            error=str(e),
        )


def handle_data_compute(expression: str) -> ActionResult:
    """Evaluate a math expression. Only numbers and +, -, *, /, (, )."""
    expr = expression.strip()
    safe_pattern = re.compile(r"^[\d\s+\-*/().]+$")
    if not safe_pattern.match(expr):
        return ActionResult(
            observation="Expression may only contain numbers and +, -, *, /, (, ).",
            success=False,
            error="Invalid expression",
        )
    try:
        tree = ast.parse(expr, mode="eval")
        result = _safe_eval_expr(tree)
        if isinstance(result, float) and not result.is_integer():
            return ActionResult(observation=str(round(result, 2)))
        return ActionResult(observation=str(result))
    except Exception as e:
        return ActionResult(
            observation=f"Compute error: {e}",
            success=False,
            error=str(e),
        )
