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
    provider = config.get_provider(job.provider_id)
    is_direct = config.is_direct or (provider is not None and provider.base_url is not None)

    if is_direct and provider and provider.base_url:
        endpoint_base = str(provider.base_url).rstrip("/")
        endpoint_url = f"{endpoint_base}/chat/completions"
        api_key_env = provider.api_key_env or config.gateway.api_key_env or "API_KEY"
        model_name = provider.model or job.openrouter_model
    else:
        endpoint_url = f"{str(config.gateway.base_url).rstrip('/')}/chat/completions"
        api_key_env = config.gateway.api_key_env
        model_name = job.openrouter_model

    endpoint_payload: dict[str, Any] = {
        "url": endpoint_url,
        "type": "chat",
        "apiKey": f"${{{api_key_env}}}",
        "streaming": True,
        "timeout": config.reliability.slo.request_timeout_seconds,
        "useLegacyMaxTokens": True,
        "useServerTokenCount": phase.use_server_token_count,
    }
    if not is_direct:
        endpoint_payload["extra"] = {
            "provider": provider_routing(job.provider_slug, config.gateway)
        }

    benchmark: dict[str, Any] = {
        "model": model_name,
        "endpoint": endpoint_payload,
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
            "raw": True,
            "trace": True,
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
    api_key_env: str | None = None,
) -> dict[str, object]:
    env_name = api_key_env or config.gateway.api_key_env or "API_KEY"
    environment = {env_name: api_key} if api_key is not None else None
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
    api_key_env: str | None = None,
) -> dict[str, object]:
    path = prepare(config, job, job_dir)
    provider = config.get_provider(job.provider_id)
    env_name = (
        api_key_env
        or (provider.api_key_env if provider and provider.api_key_env else None)
        or config.gateway.api_key_env
        or "API_KEY"
    )
    validation = validate(config, path, job_dir, api_key, api_key_env=env_name)
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
        [
            "aiperf",
            "profile",
            "--config",
            str(path),
            "--export-level",
            "raw",
            "--export-http-trace",
        ],
        job_dir,
        {env_name: api_key},
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
