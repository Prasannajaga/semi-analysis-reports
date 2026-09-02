"""Deterministic expansion of model/provider benchmark jobs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from benchmark_tool.config import BenchmarkConfig, CorrectnessTask, Model, Provider
from benchmark_tool.io import slug, stable_hash


@dataclass(frozen=True)
class Job:
    job_id: str
    config_hash: str
    phase: Literal["performance", "correctness"]
    runner: Literal["aiperf", "lm-eval", "bfcl"]
    model_id: str
    openrouter_model: str
    tokenizer: str | None
    tokenizer_trust_remote_code: bool
    provider_id: str
    provider_slug: str
    workload_type: str | None = None
    concurrency: int | None = None
    task_name: str | None = None
    runner_task: str | None = None
    limit: int | None = None

    def dump(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if value is not None}


def _job(
    config: BenchmarkConfig,
    model: Model,
    provider: Provider,
    phase: Literal["performance", "correctness"],
    runner: Literal["aiperf", "lm-eval", "bfcl"],
    concurrency: int | None = None,
    task: CorrectnessTask | None = None,
) -> Job:
    identity = {
        "schema_version": config.schema_version,
        "seed": config.seed,
        "gateway": config.gateway.model_dump(mode="json", by_alias=True),
        "model": model.model_dump(mode="json", by_alias=True),
        "provider": provider.model_dump(mode="json", by_alias=True),
        "phase": phase,
        "runner": runner,
        "concurrency": concurrency,
        "task": task.model_dump(mode="json", by_alias=True) if task else None,
        "phase_config": (
            config.phases.performance.model_dump(mode="json", by_alias=True)
            if phase == "performance" and config.phases.performance
            else config.phases.correctness.model_dump(mode="json", by_alias=True)
            if config.phases.correctness
            else None
        ),
    }
    digest = stable_hash(identity)
    workload_type = (
        config.phases.performance.workload.type
        if phase == "performance" and config.phases.performance
        else None
    )
    dimension = (
        f"{workload_type or phase}--c{concurrency}"
        if concurrency is not None
        else slug(task.name if task else "task")
    )
    job_id = "--".join((phase, slug(model.id), slug(provider.id), dimension, digest[:8]))
    return Job(
        job_id=job_id,
        config_hash=digest,
        phase=phase,
        runner=runner,
        model_id=model.id,
        openrouter_model=model.openrouter_model,
        tokenizer=model.tokenizer,
        tokenizer_trust_remote_code=model.tokenizer_trust_remote_code,
        provider_id=provider.id,
        provider_slug=provider.openrouter_slug,
        workload_type=workload_type,
        concurrency=concurrency,
        task_name=task.name if task else None,
        runner_task=task.runner_task if task else None,
        limit=task.limit if task else None,
    )


def expand_matrix(config: BenchmarkConfig) -> list[Job]:
    jobs: list[Job] = []
    performance = config.phases.performance
    correctness = config.phases.correctness
    for model in config.models:
        for provider in config.providers:
            if performance and performance.enabled:
                jobs.extend(
                    _job(config, model, provider, "performance", "aiperf", concurrency=value)
                    for value in performance.load.values
                )
            if correctness and correctness.enabled:
                jobs.extend(
                    _job(config, model, provider, "correctness", task.runner, task=task)
                    for task in correctness.tasks
                )
    return sorted(jobs, key=lambda item: item.job_id)
