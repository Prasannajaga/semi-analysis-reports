from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from benchmark_tool.config import BenchmarkConfig


@pytest.fixture
def config_dict() -> dict[str, Any]:
    return {
        "schemaVersion": "1.0",
        "benchmark": {"name": "test-run", "description": "fixture", "seed": 42},
        "gateway": {
            "type": "openrouter",
            "baseUrl": "https://openrouter.ai/api/v1",
            "apiKeyEnv": "OPENROUTER_API_KEY",
            "routing": {"allowFallbacks": False, "requireParameters": True},
        },
        "providers": [{"id": "provider-a", "openrouterSlug": "provider-a"}],
        "models": [
            {
                "id": "model-a",
                "openrouterModel": "example/model-a",
                "tokenizer": "builtin",
            }
        ],
        "pricing": {"enabled": True},
        "reliability": {
            "enabled": True,
            "slo": {
                "requestTimeoutSeconds": 300,
                "maxP95TtftMs": 5000,
                "minSuccessRate": 0.5,
            },
        },
        "phases": {
            "performance": {
                "enabled": True,
                "runner": "aiperf",
                "workload": {
                    "type": "agentx",
                    "dataset": {
                        "name": "semianalysis_cc_traces_weka_062126",
                        "maxContextLength": 32768,
                    },
                },
                "load": {"mode": "concurrency", "values": [4, 8]},
                "durationSeconds": 1800,
                "warmup": {"requestCount": 10},
                "useServerTokenCount": True,
            },
            "correctness": {
                "enabled": True,
                "generation": {"temperature": 0, "maxTokens": 128},
                "tasks": [
                    {"name": "gsm8k", "runner": "lm-eval"},
                    {"name": "bfcl", "runner": "bfcl", "runnerTask": "simple_python"},
                ],
            },
        },
    }


@pytest.fixture
def benchmark_config(config_dict: dict[str, Any]) -> BenchmarkConfig:
    return BenchmarkConfig.model_validate(deepcopy(config_dict))
