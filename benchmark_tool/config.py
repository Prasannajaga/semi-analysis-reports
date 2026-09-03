"""Strict user configuration for benchmark execution."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

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


class GatewayRetries(StrictModel):
    max_retries: NonNegativeInt = 3
    retry_delay_seconds: Annotated[float, Field(gt=0)] = 2.0
    backoff_factor: Annotated[float, Field(ge=1.0)] = 2.0


class GatewayRouting(StrictModel):
    allow_fallbacks: Literal[False] = False
    require_parameters: Literal[True] = True


class Gateway(StrictModel):
    type: Literal["openrouter", "directProviders", "direct"] = "openrouter"
    base_url: HttpUrl = HttpUrl("https://openrouter.ai/api/v1")
    api_key_env: str | None = Field(default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    retries: GatewayRetries = Field(default_factory=GatewayRetries)
    routing: GatewayRouting = Field(default_factory=GatewayRouting)
    providers: Any | None = None
    openrouter: Any | None = None
    direct_providers: Any | None = Field(default=None, alias="directProviders")

    @model_validator(mode="before")
    @classmethod
    def normalize_gateway(cls, data: Any) -> Any:
        if isinstance(data, dict):
            raw = dict(data)
            gt = raw.get("type")
            if gt in ("direct_providers", "direct"):
                raw["type"] = "directProviders"
            return raw
        return data

    @model_validator(mode="after")
    def validate_gateway(self) -> Gateway:
        if self.type == "openrouter" and not self.api_key_env:
            raise ValueError("apiKeyEnv is required for openrouter gateway")
        return self


class ProviderPricing(StrictModel):
    input_usd_per_million: Annotated[float, Field(ge=0)] = 0.0
    output_usd_per_million: Annotated[float, Field(ge=0)] = 0.0
    cached_input_usd_per_million: Annotated[float, Field(ge=0)] | None = None
    cache_write_usd_per_million: Annotated[float, Field(ge=0)] | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_pricing_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            raw = dict(data)
            if "inputUsdPerMillion" in raw:
                raw["input_usd_per_million"] = raw.pop("inputUsdPerMillion")
            elif "prompt" in raw:
                raw["input_usd_per_million"] = raw.pop("prompt")
            elif "input" in raw:
                raw["input_usd_per_million"] = raw.pop("input")

            if "outputUsdPerMillion" in raw:
                raw["output_usd_per_million"] = raw.pop("outputUsdPerMillion")
            elif "completion" in raw:
                raw["output_usd_per_million"] = raw.pop("completion")
            elif "output" in raw:
                raw["output_usd_per_million"] = raw.pop("output")

            if "cachedInputUsdPerMillion" in raw:
                raw["cached_input_usd_per_million"] = raw.pop("cachedInputUsdPerMillion")
            elif "cachedPrompt" in raw or "cached_prompt" in raw:
                raw["cached_input_usd_per_million"] = raw.pop("cachedPrompt", None) or raw.pop("cached_prompt", None)
            elif "cachedInput" in raw or "cached_input" in raw:
                raw["cached_input_usd_per_million"] = raw.pop("cachedInput", None) or raw.pop("cached_input", None)
            elif "cached" in raw:
                raw["cached_input_usd_per_million"] = raw.pop("cached")

            if "cacheWriteUsdPerMillion" in raw:
                raw["cache_write_usd_per_million"] = raw.pop("cacheWriteUsdPerMillion")
            elif "cacheWrite" in raw or "cache_write" in raw:
                raw["cache_write_usd_per_million"] = raw.pop("cacheWrite", None) or raw.pop("cache_write", None)
            return raw
        return data


class Provider(StrictModel):
    id: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
    openrouter_slug: str = Field(default="", pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$|^$")
    base_url: HttpUrl | None = None
    api_key_env: str | None = Field(default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    model: str | None = None
    pricing: ProviderPricing | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            raw = dict(data)
            slug = raw.pop("slug", None)
            if slug is not None and "openrouterSlug" not in raw and "openrouter_slug" not in raw:
                raw["openrouterSlug"] = slug

            endpoint = raw.pop("endpoints", None)
            if endpoint is None:
                endpoint = raw.pop("endpoint", None)
            if endpoint is not None and "baseUrl" not in raw and "base_url" not in raw:
                raw["baseUrl"] = endpoint

            api_key = (
                raw.pop("apikeyENV", None)
                or raw.pop("apiKey_env", None)
                or raw.pop("apikeyEnv", None)
                or raw.pop("apiKeyEnv", None)
            )
            if api_key is not None and "apiKeyEnv" not in raw and "api_key_env" not in raw:
                raw["apiKeyEnv"] = api_key

            model = raw.pop("modelName", None) or raw.pop("model_name", None)
            if model is not None and "model" not in raw:
                raw["model"] = model
            return raw
        return data

    @model_validator(mode="after")
    def _default_slug(self) -> Provider:
        if not self.openrouter_slug:
            self.openrouter_slug = self.id
        return self

    @property
    def slug(self) -> str:
        return self.openrouter_slug

    @property
    def endpoints(self) -> HttpUrl | None:
        return self.base_url

    @property
    def endpoint(self) -> HttpUrl | None:
        return self.base_url

    @property
    def apikey_env(self) -> str | None:
        return self.api_key_env


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
    eos_string: str | None = Field(default=None, min_length=1)


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
    pricing: Pricing = Field(default_factory=Pricing)
    reliability: Reliability = Field(default_factory=Reliability)


class BenchmarkMetadata(StrictModel):
    name: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
    description: str = ""
    seed: NonNegativeInt = 42


class BenchmarkConfig(StrictModel):
    debug: bool = Field(default=False, alias="DEBUG")
    schema_version: Literal["1.0"] = "1.0"
    benchmark: BenchmarkMetadata
    gateway: Gateway
    providers: list[Provider] = Field(min_length=1)
    openrouter_providers: list[Provider] = Field(default_factory=list)
    direct_providers: list[Provider] = Field(default_factory=list)
    models: list[Model] = Field(min_length=1)
    phases: Phases

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_phases_and_providers(cls, data: Any) -> Any:
        if isinstance(data, dict):
            phases = data.setdefault("phases", {})
            if isinstance(phases, dict):
                if "pricing" in data and "pricing" not in phases:
                    phases["pricing"] = data.pop("pricing")
                if "reliability" in data and "reliability" not in phases:
                    phases["reliability"] = data.pop("reliability")

            raw_gateway = data.get("gateway")
            gateway_type = "openrouter"
            if isinstance(raw_gateway, dict):
                gt = raw_gateway.get("type", "openrouter")
                if gt in ("directProviders", "direct_providers", "direct"):
                    gateway_type = "directProviders"

            openrouter_raw: list[Any] | None = None
            direct_raw: list[Any] | None = None

            if isinstance(raw_gateway, dict):
                gw_providers = raw_gateway.get("providers")
                if isinstance(gw_providers, dict):
                    openrouter_raw = gw_providers.get("openrouter")
                    direct_raw = (
                        gw_providers.get("directProviders")
                        or gw_providers.get("direct_providers")
                        or gw_providers.get("direct")
                    )
                elif isinstance(gw_providers, list):
                    if gateway_type == "openrouter":
                        openrouter_raw = gw_providers
                    else:
                        direct_raw = gw_providers
                    data["providers"] = gw_providers

                if "openrouter" in raw_gateway:
                    or_val = raw_gateway["openrouter"]
                    if isinstance(or_val, dict) and "providers" in or_val:
                        openrouter_raw = or_val["providers"]
                    elif isinstance(or_val, list):
                        openrouter_raw = or_val

                for k in ("directProviders", "direct_providers", "direct"):
                    if k in raw_gateway:
                        dp_val = raw_gateway[k]
                        if isinstance(dp_val, dict) and "providers" in dp_val:
                            direct_raw = dp_val["providers"]
                        elif isinstance(dp_val, list):
                            direct_raw = dp_val

            top_providers = data.get("providers")
            if isinstance(top_providers, dict):
                if openrouter_raw is None:
                    openrouter_raw = top_providers.get("openrouter")
                if direct_raw is None:
                    direct_raw = (
                        top_providers.get("directProviders")
                        or top_providers.get("direct_providers")
                        or top_providers.get("direct")
                    )

            if gateway_type == "openrouter":
                if openrouter_raw is not None:
                    data["providers"] = openrouter_raw
            elif gateway_type == "directProviders":
                if direct_raw is not None:
                    data["providers"] = direct_raw

            if openrouter_raw is not None:
                data["openrouterProviders"] = openrouter_raw
            if direct_raw is not None:
                data["directProviders"] = direct_raw

        return data

    @property
    def pricing(self) -> Pricing:
        return self.phases.pricing

    @property
    def reliability(self) -> Reliability:
        return self.phases.reliability

    @property
    def is_direct(self) -> bool:
        return self.gateway.type in ("directProviders", "direct")

    @property
    def is_openrouter(self) -> bool:
        return self.gateway.type == "openrouter"

    def get_provider(self, provider_id: str) -> Provider | None:
        for provider in self.providers:
            if provider.id == provider_id:
                return provider
        return None

    @property
    def grouped_providers(self) -> dict[str, list[Provider]]:
        return {
            "openrouter": self.openrouter_providers,
            "directProviders": self.direct_providers,
        }

    @model_validator(mode="after")
    def cross_validate(self) -> BenchmarkConfig:
        is_direct = self.gateway.type in ("directProviders", "direct")
        direct_to_validate = list(self.direct_providers)
        if is_direct:
            for p in self.providers:
                if p not in direct_to_validate:
                    direct_to_validate.append(p)

        for p in direct_to_validate:
            if not p.base_url:
                raise ValueError(
                    f"Direct provider '{p.id}' is missing required 'endpoints' (or 'baseUrl')"
                )
            if not p.api_key_env:
                raise ValueError(
                    f"Direct provider '{p.id}' is missing required 'apiKeyEnv' (or 'apikeyENV')"
                )

        for label, values in (("provider", self.providers), ("model", self.models)):
            ids = [value.id for value in values]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {label} id")

        if self.openrouter_providers:
            or_ids = [p.id for p in self.openrouter_providers]
            if len(or_ids) != len(set(or_ids)):
                raise ValueError("duplicate provider id in openrouter providers")
        if self.direct_providers:
            dp_ids = [p.id for p in self.direct_providers]
            if len(dp_ids) != len(set(dp_ids)):
                raise ValueError("duplicate provider id in direct providers")

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

        dump = self.model_dump(mode="json", by_alias=True, exclude_none=True)
        if not self.openrouter_providers:
            dump.pop("openrouterProviders", None)
        if not self.direct_providers:
            dump.pop("directProviders", None)
        return dump


def load_config(path: Path) -> BenchmarkConfig:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a YAML mapping")
    return BenchmarkConfig.model_validate(raw)
