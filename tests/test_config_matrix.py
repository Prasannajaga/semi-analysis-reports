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


def test_unknown_yaml_key_is_rejected(config_dict):
    config_dict["unexpected"] = True
    with pytest.raises(ValidationError, match="unexpected"):
        BenchmarkConfig.model_validate(config_dict)


@pytest.mark.parametrize("dimension", ["providers", "models"])
def test_duplicate_ids_are_rejected(config_dict, dimension):
    config_dict[dimension].append(deepcopy(config_dict[dimension][0]))
    with pytest.raises(ValidationError, match="duplicate"):
        BenchmarkConfig.model_validate(config_dict)


def test_invalid_routing_cannot_enable_fallback(config_dict):
    config_dict["gateway"]["routing"]["allowFallbacks"] = True
    with pytest.raises(ValidationError):
        BenchmarkConfig.model_validate(config_dict)


def test_invalid_routing_cannot_disable_required_parameters(config_dict):
    config_dict["gateway"]["routing"]["requireParameters"] = False
    with pytest.raises(ValidationError):
        BenchmarkConfig.model_validate(config_dict)


def test_api_key_environment_declaration_is_required(config_dict):
    del config_dict["gateway"]["apiKeyEnv"]
    with pytest.raises(ValidationError, match="apiKeyEnv"):
        BenchmarkConfig.model_validate(config_dict)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("reliability", "slo", "requestTimeoutSeconds"), 0),
        (("reliability", "slo", "maxP95TtftMs"), -1),
        (("reliability", "slo", "minSuccessRate"), 1.01),
        (("phases", "performance", "durationSeconds"), 899),
    ],
)
def test_invalid_slo_and_agentx_values_are_rejected(config_dict, path, value):
    target = config_dict
    for segment in path[:-1]:
        target = target[segment]
    target[path[-1]] = value
    with pytest.raises(ValidationError):
        BenchmarkConfig.model_validate(config_dict)


def test_non_positive_concurrency_is_rejected(config_dict):
    config_dict["phases"]["performance"]["load"]["values"] = [0]
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
