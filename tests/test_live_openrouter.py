from __future__ import annotations

import os

import pytest

from benchmark_tool.openrouter import OpenRouterClient


@pytest.mark.integration
def test_live_provider_pin_only_when_explicitly_enabled():
    if os.getenv("RUN_OPENROUTER_INTEGRATION") != "1":
        pytest.skip("set RUN_OPENROUTER_INTEGRATION=1 to authorize a paid request")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY is not set")
    model = os.getenv("OPENROUTER_TEST_MODEL", "qwen/qwen3-32b")
    provider = os.getenv("OPENROUTER_TEST_PROVIDER", "deepinfra")
    result = OpenRouterClient("https://openrouter.ai/api/v1", api_key, 60).preflight(
        model, provider, require_agentx=False
    )
    assert result["status"] == "supported", result
