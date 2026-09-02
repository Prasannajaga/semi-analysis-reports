from __future__ import annotations

import json

import httpx
import respx
import yaml

from benchmark_tool.config import BenchmarkConfig
from benchmark_tool.matrix import expand_matrix
from benchmark_tool.openrouter import OpenRouterClient, provider_routing, routed_request_body
from benchmark_tool.runners import aiperf as aiperf_runner
from benchmark_tool.runners.aiperf import make_aiperf_config
from benchmark_tool.runners.bfcl import make_bfcl_config
from benchmark_tool.runners.common import run_process
from benchmark_tool.runners.lm_eval import build_lm_eval_request, make_lm_eval_config


def performance_job(config):
    return next(job for job in expand_matrix(config) if job.phase == "performance")


def correctness_job(config):
    return next(job for job in expand_matrix(config) if job.runner == "lm-eval")


def bfcl_job(config):
    return next(job for job in expand_matrix(config) if job.runner == "bfcl")


def test_openrouter_provider_body_is_top_level():
    body = routed_request_body("example/model", "fireworks", messages=[])
    assert body["provider"] == {
        "only": ["fireworks"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }


def test_locked_routing_defaults():
    assert provider_routing("baseten") == {
        "only": ["baseten"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }


def test_aiperf_config_uses_endpoint_extra_and_env_reference(benchmark_config, tmp_path):
    native = make_aiperf_config(benchmark_config, performance_job(benchmark_config), tmp_path)
    endpoint = native["benchmark"]["endpoint"]
    assert endpoint["extra"]["provider"]["only"] == ["provider-a"]
    assert endpoint["extra"]["provider"]["allow_fallbacks"] is False
    assert endpoint["extra"]["provider"]["require_parameters"] is True
    assert endpoint["apiKey"] == "${OPENROUTER_API_KEY}"
    assert native["benchmark"]["tokenizer"]["trustRemoteCode"] is False
    assert "secret-value" not in yaml.safe_dump(native)


def test_simple_performance_config_does_not_enable_agentx(benchmark_config, tmp_path):
    raw = benchmark_config.model_dump(mode="json", by_alias=True)
    raw["phases"]["performance"]["workload"] = {
        "type": "synthetic",
        "inputTokens": 128,
        "outputTokens": 16,
        "datasetEntries": 64,
    }
    raw["phases"]["performance"]["durationSeconds"] = 10
    config = BenchmarkConfig.model_validate(raw)
    native = make_aiperf_config(config, performance_job(config), tmp_path)
    assert "scenario" not in native["benchmark"]
    assert native["benchmark"]["dataset"] == {
        "name": "synthetic-performance",
        "type": "synthetic",
        "entries": 64,
        "randomSeed": 42,
        "prompts": {"isl": 128, "osl": 16},
    }


def test_aiperf_config_forwards_explicit_tokenizer_trust(benchmark_config, tmp_path):
    raw = benchmark_config.model_dump(mode="json", by_alias=True)
    raw["models"][0]["tokenizer"] = "moonshotai/Kimi-K3"
    raw["models"][0]["tokenizerTrustRemoteCode"] = True
    config = BenchmarkConfig.model_validate(raw)
    native = make_aiperf_config(config, performance_job(config), tmp_path)
    assert native["benchmark"]["tokenizer"] == {
        "name": "moonshotai/Kimi-K3",
        "trustRemoteCode": True,
    }


def test_current_aiperf_chat_formatter_emits_top_level_provider():
    from aiperf.common.models import RequestInfo, Turn
    from aiperf.common.models.model_endpoint_info import (
        EndpointInfo,
        ModelEndpointInfo,
        ModelInfo,
        ModelListInfo,
    )
    from aiperf.endpoints.openai_chat import ChatEndpoint

    routing = provider_routing("fireworks")
    endpoint_info = EndpointInfo(streaming=True, extra=[("provider", routing)])
    model_info = ModelEndpointInfo.model_construct(
        models=ModelListInfo.model_construct(models=[ModelInfo(name="example/model")]),
        endpoint=endpoint_info,
    )
    request = RequestInfo.model_construct(
        model_endpoint=model_info,
        turns=[Turn(raw_messages=[{"role": "user", "content": "hello"}], max_tokens=1)],
        system_message=None,
        user_context_message=None,
    )
    payload = ChatEndpoint(model_info).format_payload(request)
    assert payload["provider"] == routing


def test_lm_eval_config_and_payload_preserve_routing(benchmark_config, tmp_path):
    job = correctness_job(benchmark_config)
    native = make_lm_eval_config(benchmark_config, job, tmp_path)
    assert native["gen_kwargs"]["provider"] == provider_routing("provider-a")
    body = build_lm_eval_request(
        benchmark_config, job, [{"role": "user", "content": "2+2?"}]
    )
    assert body["provider"] == provider_routing("provider-a")


def test_bfcl_bridge_config_preserves_routing_without_secret(benchmark_config, tmp_path):
    native = make_bfcl_config(benchmark_config, bfcl_job(benchmark_config), tmp_path)
    assert native["provider"] == provider_routing("provider-a")
    assert "OPENROUTER_API_KEY" not in json.dumps(native)
    assert "secret" not in json.dumps(native)


def test_aiperf_native_validation_wrapper(monkeypatch, benchmark_config, tmp_path):
    captured = {}

    def fake_run(
        command,
        work_dir,
        environment=None,
        log_prefix="process",
        redact_values=(),
    ):
        captured.update(
            command=command,
            environment=environment,
            log_prefix=log_prefix,
            redact_values=redact_values,
        )
        return {"returnCode": 0}

    monkeypatch.setattr(aiperf_runner, "run_process", fake_run)
    path = tmp_path / "aiperf-config.yaml"
    result = aiperf_runner.validate(benchmark_config, path, tmp_path, "runtime-only")
    assert result["returnCode"] == 0
    assert captured["command"] == ["aiperf", "config", "validate", str(path)]
    assert captured["environment"] == {"OPENROUTER_API_KEY": "runtime-only"}
    assert captured["redact_values"] == ("runtime-only",)


def test_subprocess_artifacts_redact_secret(monkeypatch, tmp_path):
    secret = "runtime-secret-value"

    def fake_subprocess_run(*args, **kwargs):
        del args, kwargs
        return type(
            "Completed",
            (),
            {
                "stdout": f"debug output {secret}",
                "stderr": f"request failed with {secret}",
                "returncode": 1,
            },
        )()

    monkeypatch.setattr("benchmark_tool.runners.common.subprocess.run", fake_subprocess_run)
    result = run_process(
        ["runner", "--credential", secret],
        tmp_path,
        {"OPENROUTER_API_KEY": secret},
        redact_values=(secret,),
    )
    serialized = json.dumps(result)
    logs = (tmp_path / "process-stdout.log").read_text() + (
        tmp_path / "process-stderr.log"
    ).read_text()
    assert secret not in serialized
    assert secret not in logs
    assert "[REDACTED]" in serialized


@respx.mock
def test_preflight_preserves_rate_limit_reason_and_retry_metadata():
    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(
            429,
            json={
                "error": {
                    "code": 429,
                    "message": "Provider returned error",
                    "metadata": {
                        "provider_name": "BaseTen",
                        "raw": "Upstream capacity limit reached",
                    },
                },
                "openrouter_metadata": {"attempt": 1},
            },
            headers={"retry-after": "30"},
        )
    )
    client = OpenRouterClient("https://openrouter.ai/api/v1", "not-persisted", 10)
    result = client.preflight(
        "moonshotai/kimi-k3",
        "baseten",
        require_streaming=True,
    )
    assert result["status"] == "failed"
    assert result["failureCategory"] == "rate_limit"
    assert result["retryable"] is True
    assert result["error"]["providerMetadata"]["provider_name"] == "BaseTen"
    assert "HTTP 429 Too Many Requests" in result["reason"]
    assert "Upstream capacity limit reached" in result["reason"]
    assert "Retry-After: 30" in result["reason"]
    assert route.calls[0].request.headers["X-OpenRouter-Metadata"] == "enabled"
    assert "not-persisted" not in json.dumps(result)


@respx.mock
def test_mocked_preflight_checks_exposed_provider():
    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            text=(
                'data: {"id":"generation-1","choices":[{"delta":{"content":"OK"}}]}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"x-openrouter-provider": "Fireworks"},
        )
    )
    client = OpenRouterClient("https://openrouter.ai/api/v1", "not-persisted", 10)
    result = client.preflight("example/model", "fireworks", require_agentx=True)
    assert result["status"] == "supported"
    sent = json.loads(route.calls[0].request.content)
    assert sent["provider"] == provider_routing("fireworks")
    assert sent["ignore_eos"] is True
    assert "not-persisted" not in json.dumps(result)
