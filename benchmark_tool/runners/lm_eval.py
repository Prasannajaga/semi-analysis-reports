"""lm-evaluation-harness 0.4.13 chat-completions integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark_tool.config import BenchmarkConfig
from benchmark_tool.io import write_yaml
from benchmark_tool.matrix import Job
from benchmark_tool.openrouter import provider_routing
from benchmark_tool.runners.common import run_process


TASK_ALIASES = {
    "gpqa_diamond": "gpqa_diamond_cot_zeroshot",
    "gsm8k": "gsm8k",
}


def resolved_task(job: Job) -> str:
    if job.runner_task:
        return job.runner_task
    return TASK_ALIASES.get(job.task_name or "", job.task_name or "")


def make_lm_eval_config(config: BenchmarkConfig, job: Job, output_dir: Path) -> dict[str, Any]:
    correctness = config.phases.correctness
    if correctness is None:
        raise ValueError("correctness phase is required")
    generation = correctness.generation
    provider = config.get_provider(job.provider_id)
    is_direct = config.is_direct or (provider is not None and provider.base_url is not None)

    if is_direct and provider and provider.base_url:
        endpoint_base = str(provider.base_url).rstrip("/")
        base_url = f"{endpoint_base}/chat/completions"
        model_name = provider.model or job.openrouter_model
    else:
        base_url = f"{str(config.gateway.base_url).rstrip('/')}/chat/completions"
        model_name = job.openrouter_model

    gen_kwargs: dict[str, Any] = {
        "temperature": generation.temperature,
        "max_tokens": generation.max_tokens,
    }
    if not is_direct:
        gen_kwargs["provider"] = provider_routing(job.provider_slug, config.gateway)

    value: dict[str, Any] = {
        "model": "local-chat-completions",
        "model_args": {
            "model": model_name,
            "base_url": base_url,
            "tokenizer_backend": "none",
            "tokenized_requests": False,
            "max_gen_toks": generation.max_tokens,
            "seed": config.seed,
            "timeout": config.reliability.slo.request_timeout_seconds,
        },
        "tasks": [resolved_task(job)],
        "apply_chat_template": True,
        "gen_kwargs": gen_kwargs,
        "seed": config.seed,
        "output_path": str(output_dir.resolve()),
        "log_samples": True,
    }
    if job.limit is not None:
        value["limit"] = job.limit
    return value


def build_lm_eval_request(config: BenchmarkConfig, job: Job, messages: list[dict[str, str]]) -> dict[str, Any]:
    """Mirror the runner's payload merge for a routing contract test."""

    correctness = config.phases.correctness
    if correctness is None:
        raise ValueError("correctness phase is required")
    generation = correctness.generation
    provider = config.get_provider(job.provider_id)
    is_direct = config.is_direct or (provider is not None and provider.base_url is not None)
    model_name = (
        provider.model
        if is_direct and provider and provider.model
        else job.openrouter_model
    )

    payload: dict[str, Any] = {
        "messages": messages,
        "model": model_name,
        "temperature": generation.temperature,
        "max_tokens": generation.max_tokens,
    }
    if not is_direct:
        payload["provider"] = provider_routing(job.provider_slug, config.gateway)
    return payload


def prepare(config: BenchmarkConfig, job: Job, job_dir: Path) -> Path:
    path = job_dir / "lm-eval-config.yaml"
    write_yaml(path, make_lm_eval_config(config, job, job_dir / "artifacts"))
    return path


def execute(config: BenchmarkConfig, job: Job, job_dir: Path, api_key: str) -> dict[str, object]:
    path = prepare(config, job, job_dir)
    process = run_process(
        ["lm-eval", "run", "--config", str(path)],
        job_dir,
        {"OPENAI_API_KEY": api_key},
        log_prefix="lm-eval",
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
            else f"lm-eval exited with code {return_code}: {process['failureSummary']}"
        ),
        "process": process,
    }
