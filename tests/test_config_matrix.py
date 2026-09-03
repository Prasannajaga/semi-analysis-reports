from __future__ import annotations

from copy import deepcopy

import pytest
import yaml
from pydantic import ValidationError

from benchmark_tool.config import BenchmarkConfig, load_config
from benchmark_tool.matrix import expand_matrix


def test_yaml_parsing(config_dict, tmp_path):
    path = tmp_path / "benchmark.yaml"
    path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")
    config = load_config(path)
    assert config.benchmark.name == "test-run"
    assert config.gateway.api_key_env == "OPENROUTER_API_KEY"
    assert config.phases.pricing.enabled is True
    assert config.pricing.enabled is True
    assert config.phases.reliability.slo.request_timeout_seconds == 300
    assert config.reliability.slo.request_timeout_seconds == 300


def test_unknown_yaml_key_is_rejected(config_dict):
    config_dict["unexpected"] = True
    with pytest.raises(ValidationError, match="unexpected"):
        BenchmarkConfig.model_validate(config_dict)


@pytest.mark.parametrize("dimension", ["providers", "models"])
def test_duplicate_ids_are_rejected(config_dict, dimension):
    config_dict[dimension].append(deepcopy(config_dict[dimension][0]))
    with pytest.raises(ValidationError, match="duplicate"):
        BenchmarkConfig.model_validate(config_dict)


@pytest.mark.parametrize(("key", "value"), [("allowFallbacks", True), ("requireParameters", False)])
def test_invalid_routing_is_rejected(config_dict, key, value):
    config_dict["gateway"]["routing"][key] = value
    with pytest.raises(ValidationError):
        BenchmarkConfig.model_validate(config_dict)


def test_api_key_environment_declaration_is_required(config_dict):
    del config_dict["gateway"]["apiKeyEnv"]
    with pytest.raises(ValidationError, match="apiKeyEnv"):
        BenchmarkConfig.model_validate(config_dict)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("phases", "reliability", "slo", "requestTimeoutSeconds"), 0),
        (("phases", "reliability", "slo", "maxP95TtftMs"), -1),
        (("phases", "reliability", "slo", "minSuccessRate"), 1.01),
        (("phases", "performance", "durationSeconds"), 899),
        (("phases", "performance", "load", "values"), [0]),
    ],
)
def test_invalid_slo_and_agentx_values_are_rejected(config_dict, path, value):
    target = config_dict
    for segment in path[:-1]:
        target = target[segment]
    target[path[-1]] = value
    with pytest.raises(ValidationError):
        BenchmarkConfig.model_validate(config_dict)


def test_agentx_requires_explicit_tokenizer(config_dict):
    del config_dict["models"][0]["tokenizer"]
    with pytest.raises(ValidationError, match="explicit tokenizer"):
        BenchmarkConfig.model_validate(config_dict)


def test_three_by_three_endpoint_matrix(config_dict):
    config_dict["providers"] = [
        {"id": f"p{index}", "openrouterSlug": f"provider-{index}"} for index in range(3)
    ]
    config_dict["models"] = [
        {
            "id": f"m{index}",
            "openrouterModel": f"example/model-{index}",
            "tokenizer": "builtin",
        }
        for index in range(3)
    ]
    config_dict["phases"]["performance"]["load"]["values"] = [4]
    del config_dict["phases"]["correctness"]
    jobs = expand_matrix(BenchmarkConfig.model_validate(config_dict))
    assert len(jobs) == 9
    assert len({(job.model_id, job.provider_id) for job in jobs}) == 9


def test_concurrency_and_correctness_expansion(benchmark_config):
    jobs = expand_matrix(benchmark_config)
    assert [job.concurrency for job in jobs if job.phase == "performance"] == [4, 8]
    assert {job.task_name for job in jobs if job.phase == "correctness"} == {"gsm8k", "bfcl"}


def test_job_ids_and_hashes_are_deterministic(benchmark_config):
    first = [(job.job_id, job.config_hash) for job in expand_matrix(benchmark_config)]
    second = [(job.job_id, job.config_hash) for job in expand_matrix(benchmark_config)]
    assert first == second
    assert len(set(first)) == len(first)


def test_gateway_retries_parsing(config_dict):
    config_dict["gateway"]["retries"] = {
        "maxRetries": 5,
        "retryDelaySeconds": 3.5,
        "backoffFactor": 2.5,
    }
    config = BenchmarkConfig.model_validate(config_dict)
    assert config.gateway.retries.max_retries == 5
    assert config.gateway.retries.retry_delay_seconds == 3.5
    assert config.gateway.retries.backoff_factor == 2.5


def test_gateway_openrouter_type_with_slug(config_dict):
    config_dict["gateway"] = {
        "type": "openrouter",
        "baseUrl": "https://openrouter.ai/api/v1",
        "apiKeyEnv": "OPENROUTER_API_KEY",
        "retries": {
            "maxRetries": 3,
            "retryDelaySeconds": 2.0,
            "backoffFactor": 2.0,
        },
        "routing": {
            "allowFallbacks": False,
            "requireParameters": True,
        },
        "providers": [
            {"id": "fireworks", "slug": "fireworks"},
            {"id": "together", "slug": "together"},
            {"id": "baseten", "slug": "baseten"},
        ],
    }
    del config_dict["providers"]
    config = BenchmarkConfig.model_validate(config_dict)
    assert config.is_openrouter is True
    assert config.is_direct is False
    assert len(config.providers) == 3
    assert [p.id for p in config.providers] == ["fireworks", "together", "baseten"]
    assert [p.slug for p in config.providers] == ["fireworks", "together", "baseten"]
    assert [p.openrouter_slug for p in config.providers] == ["fireworks", "together", "baseten"]


def test_gateway_direct_providers_type_with_slug_endpoints_apikey(config_dict):
    config_dict["gateway"] = {
        "type": "directProviders",
        "providers": [
            {
                "id": "fireworks",
                "slug": "fireworks",
                "endpoints": "https://api.fireworks.ai/inference/v1",
                "apikeyENV": "FIREWORKS_API_KEY",
            },
            {
                "id": "together",
                "slug": "together",
                "endpoints": "https://api.together.ai/v1",
                "apikeyENV": "TOGETHER_API_KEY",
            },
            {
                "id": "baseten",
                "slug": "baseten",
                "endpoints": "https://inference.baseten.co/v1",
                "apikeyENV": "BASETEN_API_KEY",
            },
        ],
    }
    del config_dict["providers"]
    config = BenchmarkConfig.model_validate(config_dict)
    assert config.is_openrouter is False
    assert config.is_direct is True
    assert len(config.providers) == 3
    assert [p.id for p in config.providers] == ["fireworks", "together", "baseten"]
    assert str(config.providers[0].endpoints).rstrip("/") == "https://api.fireworks.ai/inference/v1"
    assert config.providers[0].apikey_env == "FIREWORKS_API_KEY"
    assert config.providers[0].slug == "fireworks"
    assert str(config.providers[1].endpoints).rstrip("/") == "https://api.together.ai/v1"
    assert config.providers[1].apikey_env == "TOGETHER_API_KEY"


def test_direct_provider_missing_endpoint_is_rejected(config_dict):
    config_dict["gateway"] = {
        "type": "directProviders",
        "providers": [
            {
                "id": "fireworks",
                "slug": "fireworks",
                "apikeyENV": "FIREWORKS_API_KEY",
            }
        ],
    }
    del config_dict["providers"]
    with pytest.raises(ValidationError, match="missing required 'endpoints'"):
        BenchmarkConfig.model_validate(config_dict)


def test_direct_provider_missing_api_key_env_is_rejected(config_dict):
    config_dict["gateway"] = {
        "type": "directProviders",
        "providers": [
            {
                "id": "fireworks",
                "slug": "fireworks",
                "endpoints": "https://api.fireworks.ai/inference/v1",
            }
        ],
    }
    del config_dict["providers"]
    with pytest.raises(ValidationError, match="missing required 'apiKeyEnv'"):
        BenchmarkConfig.model_validate(config_dict)


def test_direct_provider_invalid_url_is_rejected(config_dict):
    config_dict["gateway"] = {
        "type": "directProviders",
        "providers": [
            {
                "id": "fireworks",
                "endpoints": "invalid-url",
                "apikeyENV": "FIREWORKS_API_KEY",
            }
        ],
    }
    del config_dict["providers"]
    with pytest.raises(ValidationError):
        BenchmarkConfig.model_validate(config_dict)


def test_direct_provider_invalid_api_key_env_format_is_rejected(config_dict):
    config_dict["gateway"] = {
        "type": "directProviders",
        "providers": [
            {
                "id": "fireworks",
                "endpoints": "https://api.fireworks.ai/inference/v1",
                "apikeyENV": "123-INVALID-ENV",
            }
        ],
    }
    del config_dict["providers"]
    with pytest.raises(ValidationError):
        BenchmarkConfig.model_validate(config_dict)


def test_kimi_yaml_file_loads_validly():
    from pathlib import Path
    kimi_yaml_path = Path("/data/semi-analysis-report/configs/kimi.yaml")
    config = load_config(kimi_yaml_path)
    assert config.is_direct is True
    assert len(config.providers) == 3
    assert [p.id for p in config.providers] == ["fireworks", "together", "baseten"]
    assert [p.slug for p in config.providers] == ["fireworks", "together", "baseten"]
    assert {p.api_key_env for p in config.providers} == {
        "FIREWORKS_API_KEY",
        "TOGETHER_API_KEY",
        "BASETEN_API_KEY",
    }
    assert [p.model for p in config.providers] == [
        "accounts/fireworks/models/kimi-k3",
        "moonshotai/Kimi-K3",
        "moonshotai/Kimi-K3",
    ]


def test_kimi_direct_yaml_file_loads_validly():
    from pathlib import Path
    kimi_direct_path = Path("/data/semi-analysis-report/configs/kimi.yaml")
    config = load_config(kimi_direct_path)
    assert config.is_direct is True
    assert len(config.providers) == 3
    assert [p.id for p in config.providers] == ["fireworks", "together", "baseten"]
    assert [p.slug for p in config.providers] == ["fireworks", "together", "baseten"]
    assert {p.api_key_env for p in config.providers} == {
        "FIREWORKS_API_KEY",
        "TOGETHER_API_KEY",
        "BASETEN_API_KEY",
    }
    assert [p.model for p in config.providers] == [
        "accounts/fireworks/models/kimi-k3",
        "moonshotai/Kimi-K3",
        "moonshotai/Kimi-K3",
    ]
    assert all(p.pricing is not None for p in config.providers)
    assert [p.pricing.input_usd_per_million for p in config.providers] == [3.0, 3.0, 3.0]
    assert [p.pricing.output_usd_per_million for p in config.providers] == [15.0, 15.0, 15.0]
    assert [p.pricing.cached_input_usd_per_million for p in config.providers] == [0.3, 0.3, 0.3]


def test_deepseek_direct_yaml_file_loads_validly():
    from pathlib import Path
    deepseek_path = Path("/data/semi-analysis-report/configs/deepseek.yaml")
    config = load_config(deepseek_path)
    assert config.is_direct is True
    assert len(config.providers) == 3
    assert [p.id for p in config.providers] == ["fireworks", "together", "baseten"]
    assert [p.slug for p in config.providers] == ["fireworks", "together", "baseten"]
    assert {p.api_key_env for p in config.providers} == {
        "FIREWORKS_API_KEY",
        "TOGETHER_API_KEY",
        "BASETEN_API_KEY",
    }
    assert [p.model for p in config.providers] == [
        "accounts/fireworks/models/deepseek-v4-pro-0813",
        "deepseek-ai/DeepSeek-V4-Pro-0813",
        "deepseek-ai/DeepSeek-V4-Pro",
    ]
    assert all(p.pricing is not None for p in config.providers)
    assert [p.pricing.input_usd_per_million for p in config.providers] == [1.32, 1.32, 1.32]
    assert [p.pricing.output_usd_per_million for p in config.providers] == [3.96, 3.96, 3.96]
    assert [p.pricing.cached_input_usd_per_million for p in config.providers] == [0.044, 0.13, 0.132]


def test_glm_direct_yaml_file_loads_validly():
    from pathlib import Path
    glm_path = Path("/data/semi-analysis-report/configs/glm.yaml")
    config = load_config(glm_path)
    assert config.is_direct is True
    assert len(config.providers) == 3
    assert [p.id for p in config.providers] == ["fireworks", "together", "baseten"]
    assert [p.slug for p in config.providers] == ["fireworks", "together", "baseten"]
    assert {p.api_key_env for p in config.providers} == {
        "FIREWORKS_API_KEY",
        "TOGETHER_API_KEY",
        "BASETEN_API_KEY",
    }
    assert [p.model for p in config.providers] == [
        "accounts/fireworks/models/glm-5p3-flash",
        "zai-org/GLM-5.3-Flash",
        "zai-org/GLM-5.3-Flash"]
    assert all(p.pricing is not None for p in config.providers)
    assert [p.pricing.input_usd_per_million for p in config.providers] == [0.15, 0.15, 0.15]
    assert [p.pricing.output_usd_per_million for p in config.providers] == [0.50, 0.50, 0.50]
    assert [p.pricing.cached_input_usd_per_million for p in config.providers] == [0.03, 0.03, 0.03]
