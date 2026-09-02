"""Benchmark orchestration and immutable raw-artifact layout."""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from benchmark_tool.config import AgentXWorkload, BenchmarkConfig, load_config
from benchmark_tool.io import slug, stable_hash, write_json, write_text
from benchmark_tool.matrix import Job, expand_matrix
from benchmark_tool.manifest import RunManifest
from benchmark_tool.openrouter import OpenRouterClient, pricing_snapshot
from benchmark_tool.runners import aiperf, bfcl, lm_eval


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def run_id(config: BenchmarkConfig) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{slug(config.run_name)}-{stamp}-{stable_hash(config.public_dump(), 8)}"


def job_directory(run_dir: Path, job: Job) -> Path:
    workload = (
        f"{job.workload_type or 'performance'}-c{job.concurrency}"
        if job.phase == "performance"
        else job.task_name or "task"
    )
    return (
        run_dir
        / "models"
        / slug(job.model_id)
        / "providers"
        / slug(job.provider_id)
        / job.phase
        / slug(workload)
        / job.job_id
    )


def _job_record(job: Job, status: str, **extra: Any) -> dict[str, Any]:
    return {
        "schemaVersion": "1.0",
        **job.dump(),
        "status": status,
        **{key: value for key, value in extra.items() if value is not None},
    }


def _write_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    manifest["updatedAt"] = utc_now()
    validated = RunManifest.model_validate(manifest)
    write_json(
        run_dir / "manifest.json",
        validated.model_dump(mode="json", by_alias=True, exclude_none=True),
    )


def _version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def execute_benchmark(
    config_path: Path,
    output_root: Path,
    *,
    dry_run: bool = False,
    preflight_during_dry_run: bool = False,
) -> Path:
    load_dotenv()
    config = load_config(config_path)
    jobs = expand_matrix(config)
    identifier = run_id(config)
    run_dir = output_root.resolve() / identifier
    run_dir.mkdir(parents=True, exist_ok=False)
    write_text(run_dir / "config.yaml", config_path.read_text(encoding="utf-8"))
    manifest: dict[str, Any] = {
        "schemaVersion": "1.0",
        "runId": identifier,
        "runName": config.run_name,
        "benchmarkConfigSha256": stable_hash(config.public_dump(), 64),
        "startedAt": utc_now(),
        "dryRun": dry_run,
        "apiKeyEnv": config.gateway.api_key_env,
        "versions": {
            "benchmarkTool": "0.1.0",
            "python": sys.version.split()[0],
            "aiperf": _version("aiperf"),
            "lmEval": _version("lm-eval"),
            "bfcl": _version("bfcl-eval"),
        },
        "matrix": {
            "models": len(config.models),
            "providers": len(config.providers),
            "endpointCombinations": len(config.models) * len(config.providers),
            "jobs": len(jobs),
        },
        "jobs": [],
    }
    _write_manifest(run_dir, manifest)

    api_key = os.getenv(config.gateway.api_key_env)
    if (not dry_run or preflight_during_dry_run) and not api_key:
        manifest["status"] = "failed"
        manifest["reason"] = f"required environment variable {config.gateway.api_key_env} is not set"
        _write_manifest(run_dir, manifest)
        raise RuntimeError(str(manifest["reason"]))

    performance = config.phases.performance
    correctness = config.phases.correctness
    uses_agentx = bool(
        performance
        and performance.enabled
        and isinstance(performance.workload, AgentXWorkload)
    )
    needs_plain = bool(
        correctness
        and correctness.enabled
        and any(task.runner == "lm-eval" for task in correctness.tasks)
    )
    needs_tools = bool(
        correctness
        and correctness.enabled
        and any(task.runner == "bfcl" for task in correctness.tasks)
    )
    client = (
        OpenRouterClient(
            str(config.gateway.base_url).rstrip("/"),
            api_key or "",
            config.reliability.slo.request_timeout_seconds,
        )
        if api_key and (not dry_run or preflight_during_dry_run)
        else None
    )
    endpoint_state: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for model in config.models:
        for provider in config.providers:
            key = (model.id, provider.id)
            endpoint_dir = (
                run_dir / "models" / slug(model.id) / "providers" / slug(provider.id) / "endpoint"
            )
            endpoint_dir.mkdir(parents=True, exist_ok=True)
            if client is None:
                skipped = {"status": "not-run", "reason": "dry run: no network requests made"}
                states = {"plain": skipped, "performance": skipped, "tools": skipped}
                write_json(endpoint_dir / "preflight.json", states)
                write_json(
                    endpoint_dir / "pricing-snapshot.json",
                    {"schemaVersion": "1.0", "status": "not-run", "reason": "dry run"},
                )
            else:
                snapshot: dict[str, Any] | None = None
                try:
                    metadata = client.endpoint_metadata(model.openrouter_model)
                    write_json(endpoint_dir / "endpoint-metadata.json", metadata)
                    snapshot = pricing_snapshot(
                        metadata, model.openrouter_model, provider.openrouter_slug
                    )
                    write_json(
                        endpoint_dir / "pricing-snapshot.json",
                        snapshot,
                    )
                except Exception as error:  # metadata failure must not hide the routing check
                    write_json(
                        endpoint_dir / "pricing-snapshot.json",
                        {
                            "schemaVersion": "1.0",
                            "status": "failed",
                            "reason": f"{type(error).__name__}: {error}",
                        },
                    )
                plain = (
                    client.preflight(
                        model.openrouter_model, provider.openrouter_slug, require_seed=True
                    )
                    if needs_plain
                    else {"status": "not-required"}
                )
                performance_state = (
                    client.preflight(
                        model.openrouter_model,
                        provider.openrouter_slug,
                        require_agentx=uses_agentx,
                        require_streaming=True,
                    )
                    if performance and performance.enabled
                    else {"status": "not-required"}
                )
                tools = (
                    client.preflight(
                        model.openrouter_model,
                        provider.openrouter_slug,
                        require_tools=True,
                        require_seed=True,
                    )
                    if needs_tools
                    else {"status": "not-required"}
                )
                states = {"plain": plain, "performance": performance_state, "tools": tools}
                if snapshot is not None:
                    matches = snapshot.get("matchingEndpoints", [])
                    for state in states.values():
                        state["metadataDetails"] = {"matchingEndpointCount": len(matches)}
                        if state.get("status") == "supported" and not matches:
                            state["status"] = "unsupported"
                            state["reason"] = (
                                "pinned provider does not expose this model in endpoint metadata"
                            )
                    if (
                        uses_agentx
                        and performance_state.get("status") == "supported"
                        and performance
                        and isinstance(performance.workload, AgentXWorkload)
                    ):
                        contexts = [
                            item.get("context_length")
                            for item in matches
                            if isinstance(item, dict) and isinstance(item.get("context_length"), int)
                        ]
                        required_context = performance.workload.dataset.max_context_length
                        if contexts and max(contexts) < required_context:
                            performance_state["status"] = "unsupported"
                            performance_state["reason"] = (
                                f"AgentX requires {required_context} tokens but pinned endpoint "
                                f"advertises at most {max(contexts)}"
                            )
                write_json(endpoint_dir / "preflight.json", states)
            endpoint_state[key] = states

    for job in jobs:
        directory = job_directory(run_dir, job)
        directory.mkdir(parents=True, exist_ok=True)
        states = endpoint_state[(job.model_id, job.provider_id)]
        state = (
            states["performance"]
            if job.phase == "performance"
            else states["tools"]
            if job.runner == "bfcl"
            else states["plain"]
        )
        write_json(directory / "preflight.json", state)
        if job.runner == "aiperf":
            runner_config = aiperf.prepare(config, job, directory)
        elif job.runner == "lm-eval":
            runner_config = lm_eval.prepare(config, job, directory)
        else:
            runner_config = bfcl.prepare(config, job, directory)

        if dry_run:
            validation = None
            if job.runner == "aiperf":
                validation = aiperf.validate(
                    config,
                    runner_config,
                    directory,
                    api_key=api_key or "dry-run-validation-placeholder",
                )
            if preflight_during_dry_run and state.get("status") != "supported":
                status = "unsupported" if state.get("status") == "unsupported" else "failed"
                result = _job_record(
                    job,
                    status,
                    reason=state.get("reason", "preflight failed"),
                    validation=validation,
                )
            elif validation is not None and validation["returnCode"] != 0:
                result = _job_record(
                    job,
                    "failed",
                    reason=(
                        "AIPerf native configuration validation failed"
                        + (
                            f": {validation['failureSummary']}"
                            if validation.get("failureSummary")
                            else ""
                        )
                    ),
                    validation=validation,
                )
            elif validation is not None:
                result = _job_record(
                    job,
                    "planned",
                    reason="dry run: configuration generated and AIPerf-validated",
                    validation=validation,
                )
            else:
                result = _job_record(job, "planned", reason="dry run: configuration generated")
        elif state.get("status") != "supported":
            status = "unsupported" if state.get("status") == "unsupported" else "failed"
            result = _job_record(job, status, reason=state.get("reason", "preflight failed"))
        else:
            started = utc_now()
            if job.runner == "aiperf":
                execution = aiperf.execute(config, job, directory, api_key or "")
            elif job.runner == "lm-eval":
                execution = lm_eval.execute(config, job, directory, api_key or "")
            else:
                execution = bfcl.execute(config, job, directory, api_key or "")
            result = _job_record(job, startedAt=started, finishedAt=utc_now(), **execution)
        write_json(directory / "job.json", result)
        manifest["jobs"].append(
            {
                "jobId": job.job_id,
                "path": str(directory.relative_to(run_dir)),
                "status": result["status"],
                "reason": result.get("reason"),
            }
        )
        _write_manifest(run_dir, manifest)

    if dry_run:
        manifest["status"] = (
            "planned"
            if all(item["status"] == "planned" for item in manifest["jobs"])
            else "completed-with-errors"
        )
    else:
        manifest["status"] = (
            "completed"
            if all(item["status"] == "completed" for item in manifest["jobs"])
            else "completed-with-errors"
        )
    manifest["completedAt"] = utc_now()
    _write_manifest(run_dir, manifest)
    return run_dir
