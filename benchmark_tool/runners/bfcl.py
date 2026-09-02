"""BFCL runner configuration and subprocess integration."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from benchmark_tool.config import BenchmarkConfig
from benchmark_tool.io import write_json
from benchmark_tool.matrix import Job
from benchmark_tool.openrouter import provider_routing
from benchmark_tool.runners.common import run_process


def resolved_category(job: Job) -> str:
    return job.runner_task or "simple_python"


def make_bfcl_config(config: BenchmarkConfig, job: Job, job_dir: Path) -> dict[str, Any]:
    correctness = config.phases.correctness
    if correctness is None:
        raise ValueError("correctness phase is required")
    return {
        "schemaVersion": "1.0",
        "model": job.openrouter_model,
        "registryName": f"openrouter-{job.config_hash}",
        "provider": provider_routing(job.provider_slug, config.gateway),
        "baseUrl": str(config.gateway.base_url).rstrip("/"),
        "temperature": correctness.generation.temperature,
        "maxTokens": correctness.generation.max_tokens,
        "seed": config.seed,
        "timeoutSeconds": config.reliability.slo.request_timeout_seconds,
        "testCategory": resolved_category(job),
        "limit": job.limit,
        "resultDir": str((job_dir / "artifacts" / "results").resolve()),
        "scoreDir": str((job_dir / "artifacts" / "scores").resolve()),
    }


def prepare(config: BenchmarkConfig, job: Job, job_dir: Path) -> Path:
    path = job_dir / "bfcl-config.json"
    write_json(path, make_bfcl_config(config, job, job_dir))
    return path


def execute(config: BenchmarkConfig, job: Job, job_dir: Path, api_key: str) -> dict[str, object]:
    path = prepare(config, job, job_dir)
    bridge = Path(__file__).resolve().with_name("bfcl_bridge.py")
    process = run_process(
        [sys.executable, str(bridge), str(path)],
        job_dir,
        {
            "OPENAI_API_KEY": api_key,
            "OPENAI_BASE_URL": str(config.gateway.base_url).rstrip("/"),
            "BENCHMARK_OPENROUTER_ROUTING": json.dumps(
                provider_routing(job.provider_slug, config.gateway), separators=(",", ":")
            ),
        },
        log_prefix="bfcl",
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
            else f"BFCL exited with code {return_code}: {process['failureSummary']}"
        ),
        "process": process,
    }
