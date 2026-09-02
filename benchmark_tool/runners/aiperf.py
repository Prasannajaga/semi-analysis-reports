"""AIPerf 0.12 synthetic and AgentX performance integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark_tool.config import AgentXWorkload, BenchmarkConfig, SyntheticWorkload
from benchmark_tool.io import write_yaml
from benchmark_tool.matrix import Job
from benchmark_tool.openrouter import provider_routing
from benchmark_tool.runners.common import run_process


def make_aiperf_config(config: BenchmarkConfig, job: Job, artifacts_dir: Path) -> dict[str, Any]:
    phase = config.phases.performance
    if phase is None or job.concurrency is None or job.tokenizer is None:
        raise ValueError("performance phase, concurrency, and tokenizer are required")
    endpoint_url = f"{str(config.gateway.base_url).rstrip('/')}/chat/completions"
    benchmark: dict[str, Any] = {
        "model": job.openrouter_model,
        "endpoint": {
            "url": endpoint_url,
            "type": "chat",
            "apiKey": f"${{{config.gateway.api_key_env}}}",
            "streaming": True,
            "timeout": config.reliability.slo.request_timeout_seconds,
            "useLegacyMaxTokens": True,
            "useServerTokenCount": phase.use_server_token_count,
            "extra": {"provider": provider_routing(job.provider_slug, config.gateway)},
        },
        "tokenizer": {
            "name": job.tokenizer,
            "trustRemoteCode": job.tokenizer_trust_remote_code,
        },
        "profiling": {
            "type": "concurrency",
            "concurrency": job.concurrency,
            "duration": phase.duration_seconds,
        },
        "artifacts": {
            "dir": str(artifacts_dir.resolve()),
            "summary": ["json"],
            "records": ["jsonl"],
        },
    }
    if isinstance(phase.workload, AgentXWorkload):
        benchmark["scenario"] = "inferencex-agentx-mvp"
        benchmark["dataset"] = {
            "type": "public",
            "dataset": phase.workload.dataset.name,
            "maxContextLength": phase.workload.dataset.max_context_length,
        }
    elif isinstance(phase.workload, SyntheticWorkload):
        benchmark["dataset"] = {
            "name": "synthetic-performance",
            "type": "synthetic",
            "entries": phase.workload.dataset_entries,
            "randomSeed": config.seed,
            "prompts": {
                "isl": phase.workload.input_tokens,
                "osl": phase.workload.output_tokens,
            },
        }
    if phase.warmup.request_count:
        benchmark["warmup"] = {
            "type": "concurrency",
            "requests": phase.warmup.request_count,
            "concurrency": job.concurrency,
        }
    return {"schemaVersion": "2.0", "randomSeed": config.seed, "benchmark": benchmark}


def prepare(config: BenchmarkConfig, job: Job, job_dir: Path) -> Path:
    path = job_dir / "aiperf-config.yaml"
    write_yaml(path, make_aiperf_config(config, job, job_dir / "artifacts"))
    return path


def validate(
    config: BenchmarkConfig,
    path: Path,
    job_dir: Path,
    api_key: str | None = None,
) -> dict[str, object]:
    environment = {config.gateway.api_key_env: api_key} if api_key is not None else None
    return run_process(
        ["aiperf", "config", "validate", str(path)],
        job_dir,
        environment,
        log_prefix="validation",
        redact_values=(api_key,) if api_key else (),
    )


def execute(
    config: BenchmarkConfig,
    job: Job,
    job_dir: Path,
    api_key: str,
) -> dict[str, object]:
    path = prepare(config, job, job_dir)
    validation = validate(config, path, job_dir, api_key)
    if validation["returnCode"] != 0:
        return {
            "status": "failed",
            "reason": (
                "AIPerf configuration validation failed"
                + (
                    f": {validation['failureSummary']}"
                    if validation.get("failureSummary")
                    else ""
                )
            ),
            "validation": validation,
        }
    process = run_process(
        ["aiperf", "profile", "--config", str(path)],
        job_dir,
        {config.gateway.api_key_env: api_key},
        log_prefix="aiperf",
        redact_values=(api_key,),
    )
    return_code = process["returnCode"]
    if not isinstance(return_code, int):
        raise TypeError("runner returnCode must be an integer")
    return {
        "status": "completed" if return_code == 0 else "failed",
        "reason": (
            None
            if return_code == 0
            else f"AIPerf exited with code {return_code}: {process['failureSummary']}"
        ),
        "validation": validation,
        "process": process,
    }
