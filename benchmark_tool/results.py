"""Versioned canonical result schema."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ResultModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelDimension(ResultModel):
    id: str
    openrouter_id: str


class ProviderDimension(ResultModel):
    id: str
    openrouter_slug: str


class RunMetadata(ResultModel):
    name: str
    started_at: str
    completed_at: str | None = None
    benchmark_config_sha256: str
    versions: dict[str, str | None]


class Workload(ResultModel):
    name: str
    runner: str
    concurrency: int | None = None
    runner_task: str | None = None
    dataset: str | None = None
    duration_seconds: int | None = None


class MetricStats(ResultModel):
    unit: str | None = None
    mean: float | None = None
    p50: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    stddev: float | None = None
    count: int | None = None
    total: float | None = None


class AgentX(ResultModel):
    scenario: str | None = None
    submission_valid: bool | None = None
    submission_invalid_reasons: list[str] = Field(default_factory=list)
    warmup_request_count: int | None = None


class SloCheck(ResultModel):
    value: float | None
    target: float
    passed: bool


class SloResult(ResultModel):
    passed: bool
    checks: dict[str, SloCheck]


class ReliabilityResult(ResultModel):
    successful_requests: int
    failed_requests: int
    total_requests: int
    success_rate: float | None
    error_rate: float | None
    errors: dict[str, int]
    slo: SloResult


class TokenUsage(ResultModel):
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0


class PricingResult(ResultModel):
    enabled: bool
    available: bool
    method: str
    currency: Literal["USD"] = "USD"
    input_usd_per_million: float | None = None
    output_usd_per_million: float | None = None
    cached_input_usd_per_million: float | None = None
    cache_write_usd_per_million: float | None = None
    token_usage: TokenUsage | None = None
    estimated_cost_usd: float | None = None
    cost_per_request_usd: float | None = None
    notes: list[str] = Field(default_factory=list)


class CorrectnessResult(ResultModel):
    task: str
    runner_task: str | None = None
    primary_metric: str
    score: float
    stderr: float | None = None
    sample_count: int | None = None
    metrics: dict[str, float]


class CanonicalResult(ResultModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    run_metadata: RunMetadata | None = None
    job_id: str
    config_hash: str
    model: ModelDimension
    provider: ProviderDimension
    phase: Literal["performance", "correctness"]
    status: Literal["completed", "failed", "unsupported", "planned"]
    reason: str | None = None
    workload: Workload
    agentx: AgentX | None = None
    performance: dict[str, MetricStats] | None = None
    reliability: ReliabilityResult | None = None
    pricing: PricingResult | None = None
    correctness: CorrectnessResult | None = None
    source: dict[str, Any]

    def json_record(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)
