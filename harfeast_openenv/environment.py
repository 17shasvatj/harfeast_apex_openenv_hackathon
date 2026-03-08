"""HarFeast OpenEnv environment."""

import json
import os
import random
from .rubric import score_answer
from .schemas import ActionResult, StepResult, parse_action
from . import actions


class HarFeastOpenEnv:
    """
    OpenEnv environment for HarFeast management consulting tasks.
    Phase 1-3: files, spreadsheet, data actions, submit with rubric scoring.
    """

    def __init__(self, world_path: str | None = None, worlds_base: str | None = None):
        """
        Args:
            world_path: Single world directory (harfeast_world or world_XXXX).
            worlds_base: Base dir with manifest.json + all_tasks.json for augmented dataset.
                        When set, reset() samples from all task instances.
        """
        self._worlds_base = os.path.abspath(worlds_base) if worlds_base else None
        self._all_tasks: list[dict] = []
        if self._worlds_base:
            at_path = os.path.join(self._worlds_base, "all_tasks.json")
            if os.path.isfile(at_path):
                with open(at_path) as f:
                    self._all_tasks = json.load(f)

        self.world_path = world_path or os.path.join(
            os.path.dirname(__file__), "..", "harfeast_world"
        )
        self.world_path = os.path.abspath(self.world_path)

        self._task: dict | None = None
        self._tasks: list = []
        self._prompt: str = ""
        self._step_count: int = 0
        self._done: bool = False
        self._submitted_answer: str | None = None
        self._rubric_score: float | None = None
        self._filtered_datasets: dict = {}
        self._rng: random.Random | None = None
        self._history: list[dict] = []

    @property
    def state(self) -> dict:
        """Current environment state."""
        return {
            "task_id": self._task["task_id"] if self._task else None,
            "task_name": self._task["task_name"] if self._task else None,
            "prompt": self._prompt,
            "step_count": self._step_count,
            "done": self._done,
            "submitted_answer": self._submitted_answer,
            "rubric_score": self._rubric_score,
            "filtered_datasets": list(self._filtered_datasets.keys()),
            "history": self._history,
        }

    def reset(
        self,
        task_id: str | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> StepResult:
        """
        Reset environment and load a task.
        If task_id is None, pick a random task.
        """
        self._step_count = 0
        self._done = False
        self._submitted_answer = None
        self._rubric_score = None
        self._filtered_datasets = {}
        self._rng = random.Random(seed) if seed is not None else random.Random()
        self._history = []
        # Augmented dataset: sample from all_tasks or use specific task_index
        task_index = kwargs.get("task_index")
        if self._all_tasks:
            if task_index is not None and 0 <= task_index < len(self._all_tasks):
                entry = self._all_tasks[task_index]
            else:
                entry = self._rng.choice(self._all_tasks)
            wp = entry["world_path"]
            if not os.path.isabs(wp):
                # e.g. "./harfeast_worlds/world_0000" -> world_0000
                wp = os.path.join(self._worlds_base, os.path.basename(wp.rstrip("/")))
            self.world_path = os.path.abspath(wp)
            tasks_path = os.path.join(self.world_path, "tasks.json")
            with open(tasks_path) as f:
                self._tasks = json.load(f)
            self._task = next(t for t in self._tasks if t["task_id"] == entry["task_id"])
        else:
            # Single world
            tasks_path = os.path.join(self.world_path, "tasks.json")
            if not os.path.isfile(tasks_path):
                raise FileNotFoundError(f"Tasks not found: {tasks_path}. Run world generator first.")
            with open(tasks_path, "r", encoding="utf-8") as f:
                self._tasks = json.load(f)
            if task_id:
                matches = [t for t in self._tasks if t["task_id"] == task_id]
                if not matches:
                    raise ValueError(f"Task not found: {task_id}")
                self._task = matches[0]
            else:
                self._task = self._rng.choice(self._tasks)
        
        self._prompt = self._task["prompt"]
        
        return StepResult(
            observation=f"Task: {self._task['task_name']}\n\nPrompt:\n{self._prompt}\n\nYou can use files.list(path), files.read(path), or other actions. What would you like to do?",
            prompt=self._prompt,
            step_count=0,
            done=False,
            reward=0.0,
            info={"task_id": self._task["task_id"], "action_taken": "reset"},
        )

    def step(self, action: dict | str) -> StepResult:
        """
        Execute one action and return the result.
        Action format: {"action": "files.list", "path": "."} or JSON string.
        """
        if self._done:
            return StepResult(
                observation="Episode already ended. Call reset() to start a new episode.",
                prompt=self._prompt,
                step_count=self._step_count,
                done=True,
                reward=self._rubric_score or 0.0,
                info={"action_taken": "none", "last_error": "Episode already ended"},
            )
        
        try:
            name, params = parse_action(action)
        except (ValueError, json.JSONDecodeError) as e:
            return self._make_step_result(
                observation=f"Invalid action format: {e}",
                action_taken="parse_error",
                success=False,
                last_error=str(e),
            )
        
        # Dispatch to handler
        result = self._dispatch(name, params)
        self._step_count += 1
        # Record in history for training context reconstruction
        self._history.append({
            "step": self._step_count,
            "action": {"action": name, **params},
            "observation": result.observation,
            "success": result.success,
        })
        step_result = self._make_step_result(
            observation=result.observation,
            action_taken=name,
            success=result.success,
            last_error=result.error,
        )
        if name == "submit":
            step_result.info["rubric_score"] = self._rubric_score
        return step_result

    def _dispatch(self, name: str, params: dict) -> ActionResult:
        """Dispatch action to handler."""
        if name == "files.list":
            path = params.get("path", ".")
            return actions.handle_files_list(self.world_path, path)
        
        if name == "files.read":
            path = params.get("path")
            if path is None:
                return ActionResult(
                    observation="files.read requires 'path' parameter.",
                    success=False,
                    error="Missing path",
                )
            return actions.handle_files_read(self.world_path, path)
        
        # Phase 2: spreadsheet and data actions
        if name == "spreadsheet.read_range":
            file = params.get("file")
            range_spec = params.get("range", "columns")
            if file is None:
                return ActionResult(
                    observation="spreadsheet.read_range requires 'file' parameter.",
                    success=False,
                    error="Missing file",
                )
            return actions.handle_spreadsheet_read_range(self.world_path, file, range_spec)
        
        if name == "data.filter":
            dataset = params.get("dataset")
            column = params.get("column")
            operator = params.get("operator")
            value = params.get("value")
            if None in (dataset, column, operator, value):
                return ActionResult(
                    observation="data.filter requires dataset, column, operator, value.",
                    success=False,
                    error="Missing parameters",
                )
            return actions.handle_data_filter(
                self.world_path, dataset, column, operator, str(value), self._filtered_datasets
            )
        
        if name == "data.group_by":
            dataset = params.get("dataset")
            column = params.get("column")
            aggregation = params.get("aggregation")
            target_column = params.get("target_column")
            if None in (dataset, column, aggregation, target_column):
                return ActionResult(
                    observation="data.group_by requires dataset, column, aggregation, target_column.",
                    success=False,
                    error="Missing parameters",
                )
            return actions.handle_data_group_by(
                self.world_path, dataset, column, aggregation, target_column, self._filtered_datasets
            )
        
        if name == "data.add_columns":
            dataset = params.get("dataset")
            new_column = params.get("new_column")
            expression = params.get("expression")
            if None in (dataset, new_column, expression):
                return ActionResult(
                    observation="data.add_columns requires dataset, new_column, expression.",
                    success=False,
                    error="Missing parameters",
                )
            return actions.handle_data_add_columns(
                self.world_path, dataset, new_column, expression, self._filtered_datasets
            )
        
        if name == "data.compute":
            expression = params.get("expression")
            if expression is None:
                return ActionResult(
                    observation="data.compute requires 'expression' parameter.",
                    success=False,
                    error="Missing expression",
                )
            return actions.handle_data_compute(str(expression))
        if name == "submit":
            answer = params.get("answer")
            if answer is None or (isinstance(answer, str) and not answer.strip()):
                return ActionResult(
                    observation="submit requires non-empty 'answer' parameter.",
                    success=False,
                    error="Missing answer",
                )
            answer_text = str(answer).strip()
            rubric_list = self._task.get("rubric", [])
            score, results = score_answer(answer_text, rubric_list)
            self._submitted_answer = answer_text
            self._rubric_score = score
            self._done = True
            passed = sum(1 for _, p in results if p)
            total = len(results)
            obs = (
                f"Episode ended. Rubric score: {score:.1f}% ({passed}/{total} criteria met).\n"
                f"Details:\n" + "\n".join(f"  {'✓' if p else '✗'} {c[:70]}{'...' if len(c) > 70 else ''}" for c, p in results)
            )
            return ActionResult(observation=obs)
        
        return ActionResult(
            observation=f"Unknown action: {name}. Valid actions: files.list, files.read, spreadsheet.read_range, data.filter, data.group_by, data.add_columns, data.compute, submit.",
            success=False,
            error=f"Unknown action: {name}",
        )

    def _make_step_result(
        self,
        observation: str,
        action_taken: str,
        success: bool = True,
        last_error: str | None = None,
    ) -> StepResult:
        """Build StepResult from action outcome."""
        return StepResult(
            observation=observation,
            prompt=self._prompt,
            step_count=self._step_count,
            done=self._done,
            reward=self._rubric_score if self._done else 0.0,
            info={
                "action_taken": action_taken,
                "datasets_available": list(self._filtered_datasets.keys()),
                "last_error": last_error,
            },
        )
