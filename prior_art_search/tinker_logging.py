from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import tinker_cookbook.rl.train as tinker_rl_train
from tinker_cookbook.rl.types import Metrics


_TOOL_TRACE_FILE: Path | None = None
_TOOL_TRACE_LOCK = Lock()


@dataclass(frozen=True)
class RunArtifacts:
    run_id: str
    run_dir: Path
    metrics_dir: Path
    traces_dir: Path
    logs_file: Path
    tool_trace_file: Path


def load_json_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")
    return raw


def make_run_id(model_name: str) -> str:
    model_tag = model_name.replace("/", "-")
    ts = time.strftime("%Y%m%d-%H%M%S")
    return f"{model_tag}-{ts}"


def create_run_artifacts(
    *,
    model_name: str,
    training_logs_dir: str,
    log_path: str,
    run_id: str,
) -> RunArtifacts:
    root_dir = Path(training_logs_dir)
    run_dir = Path(log_path) if log_path else root_dir / "runs" / run_id
    metrics_dir = root_dir / "metrics" / run_id
    traces_dir = root_dir / "traces" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)
    return RunArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        metrics_dir=metrics_dir,
        traces_dir=traces_dir,
        logs_file=run_dir / "logs.log",
        tool_trace_file=traces_dir / "tool_trace.jsonl",
    )


def summarize(value: Any, max_len: int = 240) -> str:
    try:
        text = json.dumps(value, ensure_ascii=True)
    except Exception:
        text = str(value)
    if len(text) > max_len:
        return text[:max_len] + "...(truncated)"
    return text


def extract_thought(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Thought:"):
            return stripped
    return None


def set_tool_trace_file(path: Path | None) -> None:
    global _TOOL_TRACE_FILE
    _TOOL_TRACE_FILE = path


def append_tool_trace(
    *,
    publication_number: str,
    query: str,
    turn: int,
    assistant_text: str,
    parsed_action: dict[str, Any] | None,
    tool_name: str | None,
    tool_args: dict[str, Any] | None,
    tool_result: Any,
    reward: float,
    done: bool,
    metrics: Metrics,
) -> None:
    if _TOOL_TRACE_FILE is None:
        return
    record = {
        "event": "env_step",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "publication_number": publication_number,
        "query": query,
        "turn": turn,
        "thought": extract_thought(assistant_text),
        "assistant_text_preview": summarize(assistant_text, max_len=400),
        "parsed_action": parsed_action,
        "tool_name": tool_name,
        "tool_args": tool_args,
        "tool_result_preview": summarize(tool_result, max_len=1200),
        "reward": reward,
        "done": done,
        "metrics": metrics,
    }
    line = json.dumps(record, ensure_ascii=True)
    with _TOOL_TRACE_LOCK:
        assert _TOOL_TRACE_FILE is not None
        _TOOL_TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with _TOOL_TRACE_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def disable_cookbook_trajectory_logging() -> None:
    def _quiet_print_group(*_args: Any, **_kwargs: Any) -> None:
        return None

    tinker_rl_train.print_group = _quiet_print_group


def persist_run_artifacts(artifacts: RunArtifacts) -> None:
    _copy_if_exists(artifacts.run_dir / "metrics.jsonl", artifacts.metrics_dir / "metrics.jsonl")
    _copy_if_exists(
        artifacts.run_dir / "checkpoints.jsonl",
        artifacts.metrics_dir / "checkpoints.jsonl",
    )
    for pattern in ("train_iteration_*.html", "eval_*.html", "trace_events*.jsonl"):
        for path in artifacts.run_dir.glob(pattern):
            _copy_if_exists(path, artifacts.traces_dir / path.name)


def finalize_run(
    *,
    artifacts: RunArtifacts,
    training_logs_dir: str,
    run_metadata: dict[str, Any],
) -> None:
    persist_run_artifacts(artifacts)
    latest_checkpoint = _read_last_checkpoint(artifacts.run_dir / "checkpoints.jsonl")
    summary = {
        **run_metadata,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "latest_checkpoint": latest_checkpoint,
    }
    (artifacts.run_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    (Path(training_logs_dir) / "latest_run.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _read_last_checkpoint(checkpoints_file: Path) -> dict[str, Any] | None:
    if not checkpoints_file.exists():
        return None
    last_line = ""
    with checkpoints_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last_line = line.strip()
    if not last_line:
        return None
    try:
        return json.loads(last_line)
    except json.JSONDecodeError:
        return None
