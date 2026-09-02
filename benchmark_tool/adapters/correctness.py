"""Normalize lm-eval and BFCL correctness artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark_tool.results import CorrectnessResult


def parse_lm_eval(job_dir: Path, task_name: str, runner_task: str) -> tuple[CorrectnessResult, Path]:
    candidates = []
    for path in sorted((job_dir / "artifacts").rglob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and isinstance(value.get("results"), dict):
            candidates.append((path, value))
    if len(candidates) != 1:
        raise ValueError(f"expected one lm-eval results JSON, found {len(candidates)}")
    path, raw = candidates[0]
    task_results = raw["results"].get(runner_task)
    if not isinstance(task_results, dict):
        if len(raw["results"]) == 1:
            task_results = next(iter(raw["results"].values()))
        else:
            raise ValueError(f"lm-eval output has no task {runner_task!r}")
    metrics = {
        key: float(value)
        for key, value in task_results.items()
        if isinstance(value, (int, float)) and not key.endswith("_stderr") and ",stderr" not in key
    }
    preferred = (
        "exact_match,strict-match",
        "exact_match,flexible-extract",
        "exact_match,none",
        "acc,none",
        "acc_norm,none",
    )
    primary = next((name for name in preferred if name in metrics), None)
    if primary is None:
        primary = next(iter(sorted(metrics)), None)
    if primary is None:
        raise ValueError("lm-eval result has no numeric score")
    metric_name, separator, filter_name = primary.partition(",")
    stderr_keys = [f"{metric_name}_stderr"]
    if separator:
        stderr_keys = [
            f"{metric_name}_stderr,{filter_name}",
            f"{metric_name},stderr,{filter_name}",
            *stderr_keys,
        ]
    stderr = next((task_results[key] for key in stderr_keys if key in task_results), None)
    sample_data = raw.get("n-samples", {}).get(runner_task, {})
    sample_count = sample_data.get("effective") if isinstance(sample_data, dict) else None
    return (
        CorrectnessResult(
            task=task_name,
            runner_task=runner_task,
            primary_metric=primary,
            score=metrics[primary],
            stderr=float(stderr) if isinstance(stderr, (int, float)) else None,
            sample_count=int(sample_count) if isinstance(sample_count, (int, float)) else None,
            metrics=metrics,
        ),
        path,
    )


def parse_bfcl(job_dir: Path, task_name: str, runner_task: str) -> tuple[CorrectnessResult, Path]:
    candidates = sorted((job_dir / "artifacts" / "scores").rglob(f"*_{runner_task}_score.json"))
    if len(candidates) != 1:
        raise ValueError(f"expected one BFCL category score JSON, found {len(candidates)}")
    path = candidates[0]
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw or not isinstance(raw[0], dict):
        raise ValueError("invalid BFCL score artifact")
    header = raw[0]
    score = header.get("accuracy")
    if not isinstance(score, (int, float)):
        raise ValueError("BFCL score header has no accuracy")
    sample_count = header.get("total_count")
    metrics = {"accuracy": float(score)}
    correct = header.get("correct_count")
    if isinstance(correct, (int, float)):
        metrics["correct_count"] = float(correct)
    return (
        CorrectnessResult(
            task=task_name,
            runner_task=runner_task,
            primary_metric="accuracy",
            score=float(score),
            sample_count=int(sample_count) if isinstance(sample_count, (int, float)) else None,
            metrics=metrics,
        ),
        path,
    )
