"""Benchmark orchestration and immutable raw-artifact layout."""

from __future__ import annotations

import os
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv

from benchmark_tool.config import AgentXWorkload, BenchmarkConfig, load_config
from benchmark_tool.io import read_json, read_jsonl, slug, stable_hash, write_json, write_text
from benchmark_tool.manifest import RunManifest
from benchmark_tool.matrix import Job, expand_matrix
from benchmark_tool.openrouter import OpenRouterClient, pricing_snapshot
from benchmark_tool.runners import aiperf, bfcl, lm_eval
from benchmark_tool.state_logging import configure_debug, log_error, log_state


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


def _preflight(
    client: OpenRouterClient,
    kind: str,
    model: str,
    provider: str,
    **requirements: bool,
) -> dict[str, Any]:
    started = time.monotonic()
    enabled_requirements = [name for name, enabled in requirements.items() if enabled]
    suffix = f" ({', '.join(enabled_requirements)})" if enabled_requirements else ""
    log_state("preflight", f"Starting {kind} check for {model} on {provider}{suffix}")
    try:
        result = client.preflight(model, provider, **requirements)
    except Exception as error:  # preserve the rest of the endpoint matrix
        reason = f"Unexpected preflight error: {type(error).__name__}: {error}"
        log_error(f"{kind} preflight for {model} on {provider} failed: {reason}")
        return {"status": "failed", "reason": reason}
    elapsed = time.monotonic() - started
    detail = f": {result['reason']}" if result.get("reason") else ""
    log_state(
        "preflight",
        f"{kind} check {result.get('status', 'unknown')} in {elapsed:.3f}s{detail}",
    )
    return result


def _save_trace_cost_breakdown_if_available(job_dir: Path, *, enabled: bool = True) -> None:
    if not enabled:
        return
    try:
        from benchmark_tool.adapters.aiperf import find_artifact
        from benchmark_tool.adapters.pricing import generate_trace_cost_breakdown

        records_path = find_artifact(job_dir, "profile_export.jsonl")
        summary_path = find_artifact(job_dir, "profile_export_aiperf.json")
        snapshot_path = job_dir.parents[2] / "endpoint" / "pricing-snapshot.json"
        if records_path.exists() and snapshot_path.exists():
            records = read_jsonl(records_path)
            snapshot = read_json(snapshot_path)
            breakdown = generate_trace_cost_breakdown(snapshot, records, enabled=enabled)
            if breakdown.get("traces"):
                output_path = summary_path.parent / "trace-cost-breakdown.json"
                write_json(output_path, breakdown)
                log_state("artifact", f"Saved trace cost breakdown: {output_path}")
    except Exception as error:
        log_state("artifact", f"Trace cost breakdown skipped: {error}")


def _execute_job(
    config: BenchmarkConfig,
    job: Job,
    directory: Path,
    state: dict[str, Any],
    *,
    api_key: str | None,
    dry_run: bool,
    preflight_during_dry_run: bool,
) -> dict[str, Any]:
    write_json(directory / "preflight.json", state)
    log_state("prepare", f"Generating {job.runner} configuration in {directory}")
    if job.runner == "aiperf":
        runner_config = aiperf.prepare(config, job, directory)
    elif job.runner == "lm-eval":
        runner_config = lm_eval.prepare(config, job, directory)
    else:
        runner_config = bfcl.prepare(config, job, directory)
    log_state("prepare", f"Generated runner configuration: {runner_config}")

    if dry_run:
        validation = None
        if job.runner == "aiperf":
            log_state("validate", f"Validating AIPerf configuration for {job.job_id}")
            provider = config.get_provider(job.provider_id)
            env_name = (
                provider.api_key_env
                if provider and provider.api_key_env
                else config.gateway.api_key_env
            )
            validation = aiperf.validate(
                config,
                runner_config,
                directory,
                api_key=api_key or "dry-run-validation-placeholder",
                api_key_env=env_name,
            )
        if preflight_during_dry_run and state.get("status") != "supported":
            status = "unsupported" if state.get("status") == "unsupported" else "failed"
            return _job_record(
                job,
                status,
                reason=state.get("reason", "preflight failed"),
                validation=validation,
            )
        if validation is not None and validation["returnCode"] != 0:
            return _job_record(
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
        if validation is not None:
            return _job_record(
                job,
                "planned",
                reason="dry run: configuration generated and AIPerf-validated",
                validation=validation,
            )
        return _job_record(job, "planned", reason="dry run: configuration generated")

    if state.get("status") != "supported":
        status = "unsupported" if state.get("status") == "unsupported" else "failed"
        return _job_record(job, status, reason=state.get("reason", "preflight failed"))

    started = utc_now()
    log_state("execute", f"Starting {job.runner} runner for {job.job_id}")
    if job.runner == "aiperf":
        execution = aiperf.execute(config, job, directory, api_key or "")
        if execution.get("status") == "completed" and config.pricing.enabled:
            _save_trace_cost_breakdown_if_available(directory, enabled=config.pricing.enabled)
    elif job.runner == "lm-eval":
        execution = lm_eval.execute(config, job, directory, api_key or "")
    else:
        execution = bfcl.execute(config, job, directory, api_key or "")
    return _job_record(
        job,
        startedAt=started,
        finishedAt=utc_now(),
        **cast(dict[str, Any], execution),
    )


def execute_benchmark(
    config_path: Path,
    output_root: Path,
    *,
    dry_run: bool = False,
    preflight_during_dry_run: bool = False,
) -> Path:
    configure_debug(False)
    load_dotenv()
    config = load_config(config_path)
    configure_debug(config.debug)
    log_state("config", f"Loaded configuration: {config_path.resolve()}")
    log_state("config", f"DEBUG={str(config.debug).lower()}")
    jobs = expand_matrix(config)
    log_state(
        "matrix",
        f"Expanded {len(jobs)} jobs across {len(config.models)} model(s) "
        f"and {len(config.providers)} provider(s)",
    )
    identifier = run_id(config)
    run_dir = output_root.resolve() / identifier
    run_dir.mkdir(parents=True, exist_ok=False)
    log_state("run", f"Created run directory: {run_dir}")
    write_text(run_dir / "config.yaml", config_path.read_text(encoding="utf-8"))
    log_state("artifact", f"Copied source configuration to {run_dir / 'config.yaml'}")
    manifest: dict[str, Any] = {
        "schemaVersion": "1.0",
        "runId": identifier,
        "runName": config.run_name,
        "benchmarkConfigSha256": stable_hash(config.public_dump(), 64),
        "startedAt": utc_now(),
        "dryRun": dry_run,
        "apiKeyEnv": config.gateway.api_key_env or "DIRECT_PROVIDERS",
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
    log_state("manifest", f"Initialized {run_dir / 'manifest.json'}")

    if not config.is_direct:
        api_key = os.getenv(config.gateway.api_key_env or "")
        if (not dry_run or preflight_during_dry_run) and not api_key:
            manifest["status"] = "failed"
            manifest["reason"] = f"required environment variable {config.gateway.api_key_env} is not set"
            _write_manifest(run_dir, manifest)
            raise RuntimeError(str(manifest["reason"]))
        credential_state = "available" if api_key else "not required for offline dry run"
        log_state(
            "credentials",
            f"{config.gateway.api_key_env} is {credential_state}; its value will not be logged",
        )
    else:
        api_key = None
        missing_envs = [
            p.api_key_env
            for p in config.providers
            if p.api_key_env and not os.getenv(p.api_key_env)
        ]
        if (not dry_run or preflight_during_dry_run) and missing_envs:
            manifest["status"] = "failed"
            manifest["reason"] = f"required environment variable(s) not set: {', '.join(missing_envs)}"
            _write_manifest(run_dir, manifest)
            raise RuntimeError(str(manifest["reason"]))
        log_state(
            "credentials",
            f"Direct provider credentials verified for {len(config.providers)} provider(s)",
        )

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
            max_retries=config.gateway.retries.max_retries,
            retry_delay_seconds=config.gateway.retries.retry_delay_seconds,
            backoff_factor=config.gateway.retries.backoff_factor,
        )
        if not config.is_direct and api_key and (not dry_run or preflight_during_dry_run)
        else None
    )
    endpoint_state: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    endpoint_total = len(config.models) * len(config.providers)
    endpoint_index = 0
    for model in config.models:
        for provider in config.providers:
            endpoint_index += 1
            log_state(
                "endpoint",
                f"[{endpoint_index}/{endpoint_total}] model={model.id} provider={provider.id}",
            )
            key = (model.id, provider.id)
            endpoint_dir = (
                run_dir / "models" / slug(model.id) / "providers" / slug(provider.id) / "endpoint"
            )
            endpoint_dir.mkdir(parents=True, exist_ok=True)
            states: dict[str, dict[str, Any]]
            if config.is_direct:
                status_reason = "direct provider endpoint configured"
                direct_status = "supported" if (not dry_run or preflight_during_dry_run) else "not-run"
                direct_state = {"status": direct_status, "reason": status_reason}
                states = {"plain": direct_state, "performance": direct_state, "tools": direct_state}
                write_json(endpoint_dir / "preflight.json", states)
                snapshot_data: dict[str, Any] = {
                    "schemaVersion": "1.0",
                    "status": direct_status,
                    "reason": status_reason,
                }
                if provider.pricing:
                    snapshot_data["matchingEndpoints"] = [
                        {
                            "name": f"{provider.id} direct",
                            "pricing": {
                                "prompt": provider.pricing.input_usd_per_million / 1_000_000,
                                "completion": provider.pricing.output_usd_per_million / 1_000_000,
                                "input_cache_read": (
                                    provider.pricing.cached_input_usd_per_million / 1_000_000
                                    if provider.pricing.cached_input_usd_per_million is not None
                                    else None
                                ),
                                "input_cache_write": (
                                    provider.pricing.cache_write_usd_per_million / 1_000_000
                                    if provider.pricing.cache_write_usd_per_million is not None
                                    else None
                                ),
                            },
                        }
                    ]
                write_json(endpoint_dir / "pricing-snapshot.json", snapshot_data)
                log_state("preflight", f"Direct provider endpoint configured for {model.id} on {provider.id}")
            elif client is None:
                skipped = {"status": "not-run", "reason": "dry run: no network requests made"}
                states = {"plain": skipped, "performance": skipped, "tools": skipped}
                write_json(endpoint_dir / "preflight.json", states)
                write_json(
                    endpoint_dir / "pricing-snapshot.json",
                    {"schemaVersion": "1.0", "status": "not-run", "reason": "dry run"},
                )
                log_state("preflight", "Skipped network checks for offline dry run")
            else:
                snapshot: dict[str, Any] | None = None
                try:
                    metadata_started = time.monotonic()
                    log_state(
                        "metadata",
                        f"Fetching endpoint metadata for {model.openrouter_model}",
                    )
                    metadata = client.endpoint_metadata(model.openrouter_model)
                    write_json(endpoint_dir / "endpoint-metadata.json", metadata)
                    snapshot = pricing_snapshot(
                        metadata, model.openrouter_model, provider.openrouter_slug
                    )
                    write_json(
                        endpoint_dir / "pricing-snapshot.json",
                        snapshot,
                    )
                    log_state(
                        "metadata",
                        f"Saved endpoint metadata and pricing snapshot in "
                        f"{time.monotonic() - metadata_started:.3f}s",
                    )
                except Exception as error:  # metadata failure must not hide the routing check
                    reason = f"{type(error).__name__}: {error}"
                    write_json(
                        endpoint_dir / "pricing-snapshot.json",
                        {
                            "schemaVersion": "1.0",
                            "status": "failed",
                            "reason": reason,
                        },
                    )
                    log_error(
                        f"Endpoint metadata for {model.openrouter_model} failed: {reason}"
                    )
                plain = (
                    _preflight(
                        client,
                        "plain",
                        model.openrouter_model,
                        provider.openrouter_slug,
                        require_seed=True,
                    )
                    if needs_plain
                    else {"status": "not-required"}
                )
                performance_state = (
                    _preflight(
                        client,
                        "performance",
                        model.openrouter_model,
                        provider.openrouter_slug,
                        require_agentx=uses_agentx,
                        require_streaming=True,
                    )
                    if performance and performance.enabled
                    else {"status": "not-required"}
                )
                tools = (
                    _preflight(
                        client,
                        "tools",
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
                            context
                            for item in matches
                            if isinstance(item, dict)
                            for context in [item.get("context_length")]
                            if isinstance(context, int)
                        ]
                        required_context = performance.workload.dataset.max_context_length
                        if contexts and max(contexts) < required_context:
                            performance_state["status"] = "unsupported"
                            performance_state["reason"] = (
                                f"AgentX requires {required_context} tokens but pinned endpoint "
                                f"advertises at most {max(contexts)}"
                            )
                write_json(endpoint_dir / "preflight.json", states)
                log_state("artifact", f"Saved endpoint checks: {endpoint_dir / 'preflight.json'}")
            endpoint_state[key] = states

    for job_index, job in enumerate(jobs, 1):
        job_started = time.monotonic()
        log_state(
            "job",
            f"[{job_index}/{len(jobs)}] Starting {job.job_id} "
            f"(phase={job.phase}, runner={job.runner})",
        )
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
        provider = config.get_provider(job.provider_id)
        job_api_key = (
            os.getenv(provider.api_key_env)
            if config.is_direct and provider and provider.api_key_env
            else api_key
        )
        try:
            result = _execute_job(
                config,
                job,
                directory,
                state,
                api_key=job_api_key,
                dry_run=dry_run,
                preflight_during_dry_run=preflight_during_dry_run,
            )
        except Exception as error:
            reason = f"Runner error: {type(error).__name__}: {error}"
            log_error(f"Job {job.job_id} failed: {reason}")
            result = _job_record(
                job,
                "failed",
                startedAt=utc_now(),
                finishedAt=utc_now(),
                reason=reason,
            )
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
        detail = f": {result['reason']}" if result.get("reason") else ""
        log_state(
            "job",
            f"[{job_index}/{len(jobs)}] {job.job_id} -> {result['status']} "
            f"in {time.monotonic() - job_started:.3f}s{detail}",
        )

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
    log_state("run", f"Finished with status {manifest['status']}: {run_dir}")
    if not dry_run and getattr(config, "normalize", False) and manifest["status"] not in {"failed"}:
        try:
            from benchmark_tool.normalization import normalize_run

            results_path = run_dir / "results.jsonl"
            normalize_run(run_dir, results_path)
            log_state("normalize", f"Generated canonical results: {results_path}")
        except Exception as error:
            log_error(f"Normalization failed: {type(error).__name__}: {error}")
    return run_dir
