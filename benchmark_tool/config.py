"""Strict user configuration for benchmark execution."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import AliasGenerator, BaseModel, ConfigDict, Field, HttpUrl, model_validator
from pydantic.alias_generators import to_camel


class StrictModel(BaseModel):
    """Base model accepting camelCase YAML while rejecting unknown keys."""

    model_config = ConfigDict(
        alias_generator=AliasGenerator(serialization_alias=to_camel, validation_alias=to_camel),
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )


PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
Probability = Annotated[float, Field(ge=0, le=1)]


class GatewayRouting(StrictModel):
    allow_fallbacks: Literal[False] = False
    require_parameters: Literal[True] = True


class Gateway(StrictModel):
    type: Literal["openrouter"] = "openrouter"
    base_url: HttpUrl = HttpUrl("https://openrouter.ai/api/v1")
    api_key_env: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    routing: GatewayRouting = Field(default_factory=GatewayRouting)


class Provider(StrictModel):
    id: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
    openrouter_slug: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class Model(StrictModel):
    id: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
    openrouter_model: str = Field(pattern=r"^[^\s/]+/[^\s/]+$")
    tokenizer: str | None = Field(default=None, min_length=1)
    tokenizer_trust_remote_code: bool = False
    context_length: PositiveInt | None = None


class Pricing(StrictModel):
    enabled: bool = True


class ReliabilityCollect(StrictModel):
    http_errors: bool = True
    timeouts: bool = True
    connection_errors: bool = True
    parse_errors: bool = True
    context_overflow: bool = True


class ReliabilitySlo(StrictModel):
    request_timeout_seconds: Annotated[float, Field(gt=0)] = 180.0
    max_p95_ttft_ms: Annotated[float, Field(gt=0)] | None = None
    min_success_rate: Probability = 0.99


class Reliability(StrictModel):
    enabled: bool = True
    collect: ReliabilityCollect = Field(default_factory=ReliabilityCollect)
    slo: ReliabilitySlo = Field(default_factory=ReliabilitySlo)


class AgentXDataset(StrictModel):
    name: str = "semianalysis_cc_traces_weka_062126"
    max_context_length: PositiveInt = 131_072


class PerformanceLoad(StrictModel):
    mode: Literal["concurrency"] = "concurrency"
    values: list[PositiveInt] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_values(self) -> PerformanceLoad:
        if len(set(self.values)) != len(self.values):
            raise ValueError("performance.load.values must be unique")
        return self


class Warmup(StrictModel):
    request_count: NonNegativeInt = 0


class AgentXWorkload(StrictModel):
    type: Literal["agentx"] = "agentx"
    dataset: AgentXDataset = Field(default_factory=AgentXDataset)


class SyntheticWorkload(StrictModel):
    type: Literal["synthetic"] = "synthetic"
    input_tokens: PositiveInt = 128
    output_tokens: PositiveInt = 16
    dataset_entries: PositiveInt = 64


class Performance(StrictModel):
    enabled: bool = True
    runner: Literal["aiperf"] = "aiperf"
    workload: Annotated[
        AgentXWorkload | SyntheticWorkload,
        Field(discriminator="type"),
    ] = Field(default_factory=AgentXWorkload)
    load: PerformanceLoad
    duration_seconds: PositiveInt = 1800
    warmup: Warmup = Field(default_factory=Warmup)
    use_server_token_count: bool = True


class Generation(StrictModel):
    temperature: Annotated[float, Field(ge=0, le=2)] = 0
    max_tokens: PositiveInt = 1024


class CorrectnessTask(StrictModel):
    name: str = Field(min_length=1)
    runner: Literal["lm-eval", "bfcl"]
    runner_task: str | None = None
    limit: PositiveInt | None = None


class Correctness(StrictModel):
    enabled: bool = True
    generation: Generation = Field(default_factory=Generation)
    tasks: list[CorrectnessTask] = Field(min_length=1)


class Phases(StrictModel):
    performance: Performance | None = None
    correctness: Correctness | None = None


class BenchmarkMetadata(StrictModel):
    name: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
    description: str = ""
    seed: NonNegativeInt = 42


class BenchmarkConfig(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    benchmark: BenchmarkMetadata
    gateway: Gateway
    providers: list[Provider] = Field(min_length=1)
    models: list[Model] = Field(min_length=1)
    pricing: Pricing = Field(default_factory=Pricing)
    reliability: Reliability = Field(default_factory=Reliability)
    phases: Phases

    @model_validator(mode="after")
    def cross_validate(self) -> BenchmarkConfig:
        for label, values in (("provider", self.providers), ("model", self.models)):
            ids = [value.id for value in values]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {label} id")

        performance = self.phases.performance
        correctness = self.phases.correctness
        if not ((performance and performance.enabled) or (correctness and correctness.enabled)):
            raise ValueError("at least one phase must be enabled")
        if performance and performance.enabled:
            missing = [model.id for model in self.models if not model.tokenizer]
            if missing:
                raise ValueError(
                    "performance workloads require an explicit tokenizer for every model; missing: "
                    + ", ".join(missing)
                )
            if isinstance(performance.workload, AgentXWorkload):
                if performance.duration_seconds < 900:
                    raise ValueError("AgentX performance.durationSeconds must be at least 900")
                required_context = performance.workload.dataset.max_context_length
                too_small = [
                    model.id
                    for model in self.models
                    if model.context_length is not None
                    and model.context_length < required_context
                ]
                if too_small:
                    raise ValueError(
                        f"AgentX maxContextLength {required_context} exceeds configured "
                        "model context: " + ", ".join(too_small)
                    )
        if correctness and correctness.enabled:
            identities = [(task.runner, task.name) for task in correctness.tasks]
            if len(identities) != len(set(identities)):
                raise ValueError("duplicate correctness task")
            limited_bfcl = [
                task.name
                for task in correctness.tasks
                if task.runner == "bfcl" and task.limit
            ]
            if limited_bfcl:
                raise ValueError(
                    "BFCL does not expose a stable row limit; select a smaller runnerTask instead: "
                    + ", ".join(limited_bfcl)
                )
        return self

    @property
    def run_name(self) -> str:
        return self.benchmark.name

    @property
    def seed(self) -> int:
        return self.benchmark.seed

    def public_dump(self) -> dict[str, object]:
        """Return reproducibility metadata; the API key itself is never in this model."""

        return self.model_dump(mode="json", by_alias=True, exclude_none=True)


def load_config(path: Path) -> BenchmarkConfig:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a YAML mapping")
    return BenchmarkConfig.model_validate(raw)
