"""Normalize raw run directories into canonical JSONL records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark_tool.adapters.aiperf import (
    agentx_metadata,
    load_aiperf,
    performance_metrics,
    reliability,
    token_usage,
)
from benchmark_tool.adapters.correctness import parse_bfcl, parse_lm_eval
from benchmark_tool.adapters.pricing import normalize_pricing
from benchmark_tool.config import AgentXWorkload, BenchmarkConfig, load_config
from benchmark_tool.io import read_json, write_jsonl
from benchmark_tool.results import (
    CanonicalResult,
    ModelDimension,
    ProviderDimension,
    RunMetadata,
    Workload,
)
from benchmark_tool.runners.lm_eval import TASK_ALIASES


def _base_result(
    run_id: str,
    run_metadata: RunMetadata,
    run_dir: Path,
    job_dir: Path,
    job: dict[str, Any],
) -> dict[str, Any]:
    phase = job["phase"]
    workload_name = (
        job.get("workload_type", "performance")
        if phase == "performance"
        else job.get("task_name", "unknown")
    )
    runner_task = job.get("runner_task")
    if phase == "correctness" and not runner_task:
        if job["runner"] == "lm-eval":
            runner_task = TASK_ALIASES.get(workload_name, workload_name)
        elif job["runner"] == "bfcl":
            runner_task = "simple_python"
    return {
        "run_id": run_id,
        "run_metadata": run_metadata,
        "job_id": job["job_id"],
        "config_hash": job["config_hash"],
        "model": ModelDimension(id=job["model_id"], openrouter_id=job["openrouter_model"]),
        "provider": ProviderDimension(
            id=job["provider_id"], openrouter_slug=job["provider_slug"]
        ),
        "phase": phase,
        "status": job["status"],
        "reason": job.get("reason"),
        "workload": Workload(
            name=workload_name,
            runner=job["runner"],
            concurrency=job.get("concurrency"),
            runner_task=runner_task,
        ),
        "source": {
            "runDirectory": str(run_dir),
            "jobDirectory": str(job_dir.relative_to(run_dir)),
            "job": str((job_dir / "job.json").relative_to(run_dir)),
        },
    }


def normalize_job(
    config: BenchmarkConfig,
    run_id: str,
    run_metadata: RunMetadata,
    run_dir: Path,
    job_path: Path,
) -> CanonicalResult:
    job_dir = job_path.parent
    raw = read_json(job_path)
    if not isinstance(raw, dict):
        raise ValueError(f"invalid job record: {job_path}")
    data = _base_result(run_id, run_metadata, run_dir, job_dir, raw)
    if raw["status"] != "completed":
        return CanonicalResult.model_validate(data)

    try:
        if raw["phase"] == "performance":
            summary, records, summary_path, records_path = load_aiperf(job_dir)
            metrics = performance_metrics(summary)
            phase = config.phases.performance
            if phase is None:
                raise ValueError("run configuration has no performance phase")
            if isinstance(phase.workload, AgentXWorkload):
                data["agentx"] = agentx_metadata(summary, phase.warmup.request_count)
                data["workload"].dataset = phase.workload.dataset.name
            data["workload"].duration_seconds = phase.duration_seconds
            data["performance"] = metrics
            if config.reliability.enabled:
                data["reliability"] = reliability(
                    records,
                    config.reliability.slo.min_success_rate,
                    config.reliability.slo.max_p95_ttft_ms,
                    metrics,
                    config.reliability.collect,
                )
            usage = token_usage(records)
            snapshot_path = job_dir.parents[2] / "endpoint" / "pricing-snapshot.json"
            snapshot = read_json(snapshot_path) if snapshot_path.exists() else {}
            request_count = data["reliability"].total_requests if data.get("reliability") else None
            data["pricing"] = normalize_pricing(
                snapshot, usage, config.pricing.enabled, request_count=request_count
            )
            sources = {
                "summary": str(summary_path.relative_to(run_dir)),
                "records": str(records_path.relative_to(run_dir)),
            }
            if snapshot_path.exists():
                sources["pricingSnapshot"] = str(snapshot_path.relative_to(run_dir))
            data["source"].update(sources)
        elif raw["runner"] == "lm-eval":
            runner_task = data["workload"].runner_task or data["workload"].name
            correctness, source_path = parse_lm_eval(job_dir, data["workload"].name, runner_task)
            data["correctness"] = correctness
            data["source"]["correctness"] = str(source_path.relative_to(run_dir))
        else:
            runner_task = data["workload"].runner_task or "simple_python"
            correctness, source_path = parse_bfcl(job_dir, data["workload"].name, runner_task)
            data["correctness"] = correctness
            data["source"]["correctness"] = str(source_path.relative_to(run_dir))
    except (OSError, ValueError, KeyError, TypeError) as error:
        data["status"] = "failed"
        data["reason"] = f"artifact normalization failed: {type(error).__name__}: {error}"
    return CanonicalResult.model_validate(data)


def normalize_run(run_dir: Path, output_path: Path) -> list[CanonicalResult]:
    run_dir = run_dir.resolve()
    manifest = read_json(run_dir / "manifest.json")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("runId"), str):
        raise ValueError("run manifest is missing runId")
    config = load_config(run_dir / "config.yaml")
    run_metadata = RunMetadata(
        name=str(manifest.get("runName", config.run_name)),
        started_at=str(manifest.get("startedAt", "")),
        completed_at=(
            str(manifest["completedAt"]) if manifest.get("completedAt") is not None else None
        ),
        benchmark_config_sha256=str(manifest.get("benchmarkConfigSha256", "")),
        versions={
            str(key): str(value) if value is not None else None
            for key, value in manifest.get("versions", {}).items()
        },
    )
    paths = sorted((run_dir / "models").rglob("job.json"))
    records = [
        normalize_job(config, manifest["runId"], run_metadata, run_dir, path)
        for path in paths
    ]
    write_jsonl(output_path, (record.json_record() for record in records))
    return records
