"""Inspect plot-ready metrics from direct provider responses and AIPerf traces.

Normal pytest runs are offline. To issue two cache-probing requests directly to
each provider and print a JSON report:

    uv run python tests/test_provider_metrics.py --live --provider all

To analyze an existing AIPerf job directory or profile_export.jsonl:

    uv run python tests/test_provider_metrics.py --aiperf PATH
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv


@dataclass(frozen=True)
class Provider:
    endpoint: str
    api_key_env: str
    auth_scheme: str
    model: str


PROVIDERS = {
    "together": Provider(
        endpoint="https://api.together.ai/v1/chat/completions",
        api_key_env="TOGETHER_API_KEY",
        auth_scheme="Bearer",
        model="deepseek-ai/DeepSeek-V4-Pro-0813",
    ),
    "baseten": Provider(
        endpoint="https://inference.baseten.co/v1/chat/completions",
        api_key_env="BASETEN_API_KEY",
        auth_scheme="Api-Key",
        model="deepseek-ai/DeepSeek-V4-Pro",
    ),
    "fireworks": Provider(
        endpoint="https://api.fireworks.ai/inference/v1/chat/completions",
        api_key_env="FIREWORKS_API_KEY",
        auth_scheme="Bearer",
        model="accounts/fireworks/models/deepseek-v4-pro-0813",
    ),
}


# These are the response fields useful for charts. Provider-specific values are
# normalized into these names before plotting.
PLOT_METRICS = {
    "latency": [
        "client_request_latency_ms",
        "server_ttft_ms",
        "server_processing_ms",
    ],
    "tokens": [
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "total_tokens",
        "cached_prompt_tokens",
        "cache_write_tokens",
    ],
    "cache": ["cache_hit_rate_pct"],
    "capacity": [
        "rate_limit_requests",
        "rate_limit_remaining_requests",
        "rate_limit_tokens",
        "rate_limit_remaining_tokens",
    ],
}


# AIPerf adds distribution and transport data that a single non-streaming HTTP
# response cannot provide reliably.
AIPERF_PLOT_METRICS = {
    "latency": [
        "request_latency",
        "time_to_first_token",
        "time_to_first_output_token",
        "inter_token_latency",
        "time_to_second_token",
    ],
    "throughput": [
        "output_token_throughput_per_user",
        "e2e_output_token_throughput",
        "prefill_throughput_per_user",
    ],
    "tokens": [
        "input_sequence_length",
        "output_sequence_length",
        "usage_prompt_tokens",
        "usage_completion_tokens",
        "usage_reasoning_tokens",
        "usage_total_tokens",
        "usage_prompt_cache_read_tokens",
        "usage_prompt_cache_write_tokens",
    ],
    "transport": [
        "http_req_connecting",
        "http_req_dns_lookup",
        "http_req_waiting",
        "http_req_sending",
        "http_req_receiving",
        "http_req_total",
        "http_req_data_sent",
        "http_req_data_received",
    ],
}


def _nested(body: dict[str, Any], *path: str) -> Any:
    value: Any = body
    for key in path:
        if isinstance(value, dict):
            value = value.get(key)
        elif isinstance(value, list) and key.isdigit() and int(key) < len(value):
            value = value[int(key)]
        else:
            return None
    return value


def _first_number(*values: Any) -> float | None:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            match = re.search(r"-?\d+(?:\.\d+)?", value)
            if match:
                return float(match.group())
    return None


def _header(headers: dict[str, str], name: str) -> str | None:
    lowered = {key.lower(): value for key, value in headers.items()}
    return lowered.get(name.lower())


def _duration_ms(milliseconds: Any, seconds: Any) -> float | None:
    value = _first_number(milliseconds)
    if value is not None:
        return value
    value = _first_number(seconds)
    return value * 1_000 if value is not None else None


def response_schema(value: Any, prefix: str = "") -> dict[str, str]:
    """Return response field paths and types without persisting generated content."""

    result: dict[str, str] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(response_schema(child, path))
    elif isinstance(value, list):
        result[prefix] = "array"
        if value:
            result.update(response_schema(value[0], f"{prefix}[]"))
    else:
        result[prefix] = type(value).__name__
    return result


def extract_provider_metrics(
    provider: str,
    body: dict[str, Any],
    headers: dict[str, str] | None = None,
    client_latency_ms: float | None = None,
) -> dict[str, Any]:
    """Normalize Together, Baseten, and Fireworks response metrics."""

    headers = headers or {}
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    perf = body.get("perf_metrics") if isinstance(body.get("perf_metrics"), dict) else {}

    prompt_tokens = _first_number(
        usage.get("prompt_tokens"),
        usage.get("input_tokens"),
        perf.get("prompt-tokens"),
        _header(headers, "fireworks-prompt-tokens"),
    )
    cached_tokens = _first_number(
        _nested(usage, "prompt_tokens_details", "cached_tokens"),
        _nested(usage, "input_tokens_details", "cached_tokens"),
        usage.get("cached_tokens"),
        usage.get("prompt_cache_hit_tokens"),
        usage.get("cache_read_input_tokens"),
        perf.get("cached-prompt-tokens"),
        _header(headers, "fireworks-cached-prompt-tokens"),
    )
    cache_write_tokens = _first_number(
        _nested(usage, "prompt_tokens_details", "cache_write_tokens"),
        usage.get("cache_write_tokens"),
        usage.get("prompt_cache_write_tokens"),
    )
    completion_tokens = _first_number(usage.get("completion_tokens"), usage.get("output_tokens"))
    reasoning_tokens = _first_number(
        _nested(usage, "completion_tokens_details", "reasoning_tokens"),
        usage.get("reasoning_tokens"),
    )
    total_tokens = _first_number(usage.get("total_tokens"))
    hit_rate = (
        cached_tokens * 100.0 / prompt_tokens
        if cached_tokens is not None and prompt_tokens
        else 0.0
    )

    metrics = {
        "provider": provider,
        "model": body.get("model"),
        "finish_reason": _nested(body, "choices", "0", "finish_reason"),
        "client_request_latency_ms": client_latency_ms,
        "server_ttft_ms": _duration_ms(
            perf.get("server-time-to-first-token-ms"),
            perf.get("server-time-to-first-token")
            or _header(headers, "fireworks-server-time-to-first-token"),
        ),
        "server_processing_ms": _duration_ms(
            perf.get("server-processing-time-ms"),
            perf.get("server-processing-time")
            or _header(headers, "fireworks-server-processing-time"),
        ),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
        "cached_prompt_tokens": cached_tokens or 0.0,
        "cache_write_tokens": cache_write_tokens or 0.0,
        "cache_hit_rate_pct": hit_rate,
        "cost": _first_number(usage.get("cost"), body.get("cost")),
        "rate_limit_requests": _first_number(
            _header(headers, "x-ratelimit-limit-requests"),
            _header(headers, "x-ratelimit-limit"),
        ),
        "rate_limit_remaining_requests": _first_number(
            _header(headers, "x-ratelimit-remaining-requests"),
            _header(headers, "x-ratelimit-remaining"),
        ),
        "rate_limit_tokens": _first_number(
            _header(headers, "x-ratelimit-limit-tokens")
        ),
        "rate_limit_remaining_tokens": _first_number(
            _header(headers, "x-ratelimit-remaining-tokens")
        ),
        "rate_limit_reset": _header(headers, "x-ratelimit-reset"),
        "request_id": (
            _header(headers, "x-request-id")
            or _header(headers, "x-baseten-request-id")
            or _header(headers, "request-id")
            or _header(headers, "x-correlation-id")
        ),
        "provider_trace": {
            name: value
            for name in (
                "x-api-received",
                "x-cluster",
                "x-together-routing",
                "x-baseten-accepted-at",
                "x-baseten-model-id",
                "x-baseten-model-prediction-attempts",
                "x-baseten-model-version-id",
                "x-baseten-received-at",
                "x-baseten-route-trail",
                "x-baseten-sent-at",
                "x-baseten-session-id",
                "x-ratelimit-over-limit",
            )
            if (value := _header(headers, name)) is not None
        },
    }
    return metrics


def _metric_value(record: dict[str, Any], name: str) -> float | None:
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(name)
    if isinstance(value, dict):
        value = value.get("value")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def analyze_aiperf_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Inventory and summarize plot-ready fields from AIPerf JSONL records."""

    profiling = []
    for record in records:
        metadata = record.get("metadata")
        phase = str(metadata.get("phase_kind", "")) if isinstance(metadata, dict) else ""
        if not phase or "profil" in phase.lower():
            profiling.append(record)

    successful = [record for record in profiling if not record.get("error")]
    available = sorted(
        {
            key
            for record in successful
            for key in (record.get("metrics") or {})
            if isinstance(record.get("metrics"), dict)
        }
    )
    summaries: dict[str, dict[str, float | int | None]] = {}
    for metric in available:
        values = [
            value
            for record in successful
            if (value := _metric_value(record, metric)) is not None
        ]
        if values:
            summaries[metric] = {
                "count": len(values),
                "mean": statistics.fmean(values),
                "p50": _percentile(values, 0.50),
                "p95": _percentile(values, 0.95),
                "minimum": min(values),
                "maximum": max(values),
            }

    prompt_tokens = sum(
        _metric_value(record, "usage_prompt_tokens")
        or _metric_value(record, "input_sequence_length")
        or 0
        for record in successful
    )
    cached_tokens = sum(
        _metric_value(record, "usage_prompt_cache_read_tokens")
        or _metric_value(record, "cached_input_tokens")
        or _metric_value(record, "cache_read_input_tokens")
        or 0
        for record in successful
    )
    errors = Counter(
        str((record.get("error") or {}).get("type", "unknown"))
        for record in profiling
        if record.get("error")
    )
    trace_fields = sorted(
        {
            key
            for record in profiling
            for key in (record.get("metadata") or {})
            if isinstance(record.get("metadata"), dict)
            and key
            in {
                "x_request_id",
                "x_correlation_id",
                "conversation_id",
                "session_num",
                "turn_index",
                "benchmark_phase",
                "phase_kind",
                "worker_id",
            }
        }
    )
    return {
        "records": len(profiling),
        "successful_records": len(successful),
        "failed_records": len(profiling) - len(successful),
        "errors": dict(errors),
        "available_metric_fields": available,
        "plot_metric_groups": {
            group: [name for name in names if name in available]
            for group, names in AIPERF_PLOT_METRICS.items()
        },
        "metric_summaries": summaries,
        "cache": {
            "prompt_tokens": prompt_tokens,
            "cached_prompt_tokens": cached_tokens,
            "cache_hit_rate_pct": cached_tokens * 100.0 / prompt_tokens if prompt_tokens else 0.0,
        },
        "trace_metadata_fields": trace_fields,
    }


def analyze_aiperf(path: Path) -> dict[str, Any]:
    if path.is_dir():
        direct = path / "artifacts" / "profile_export.jsonl"
        matches = [direct] if direct.is_file() else sorted(path.rglob("profile_export.jsonl"))
        if not matches:
            raise FileNotFoundError(f"no profile_export.jsonl found under {path}")
        path = matches[0]
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {"source": str(path.resolve()), **analyze_aiperf_records(records)}


def _live_payload(
    provider_name: str,
    provider: Provider,
    prefix_repetitions: int,
) -> dict[str, Any]:
    prefix = "Stable cache-test reference material. " * prefix_repetitions
    payload: dict[str, Any] = {
        "model": provider.model,
        "messages": [
            {"role": "system", "content": prefix},
            {"role": "user", "content": "Reply with OK."},
        ],
        "max_tokens": 1,
        "temperature": 0,
        "user": "provider-metrics-cache-test",
    }
    if provider_name == "fireworks":
        payload["prompt_cache_key"] = "provider-metrics-cache-test"
        payload["perf_metrics_in_response"] = True
    return payload


def probe_provider(
    provider_name: str,
    *,
    repeat: int = 2,
    prefix_repetitions: int = 1_500,
    timeout_seconds: float = 180,
) -> dict[str, Any]:
    """Issue sequential, non-streaming requests directly to one provider."""

    provider = PROVIDERS[provider_name]
    api_key = os.getenv(provider.api_key_env)
    if not api_key:
        return {
            "provider": provider_name,
            "status": "skipped",
            "reason": f"{provider.api_key_env} is not set",
        }

    headers = {
        "Authorization": f"{provider.auth_scheme} {api_key}",
        "Content-Type": "application/json",
    }
    if provider_name == "fireworks":
        headers["x-session-affinity"] = "provider-metrics-cache-test"
    payload = _live_payload(provider_name, provider, prefix_repetitions)
    attempts = []
    with httpx.Client(timeout=timeout_seconds) as client:
        for index in range(repeat):
            started = time.perf_counter()
            try:
                response = client.post(provider.endpoint, headers=headers, json=payload)
            except httpx.RequestError as exc:
                attempts.append(
                    {
                        "attempt": index + 1,
                        "outcome": "transport_error",
                        "error_type": type(exc).__name__,
                    }
                )
                if index + 1 < repeat:
                    time.sleep(2)
                continue
            elapsed_ms = (time.perf_counter() - started) * 1_000
            try:
                body = response.json()
            except json.JSONDecodeError:
                body = {"non_json_response": True}
            if not isinstance(body, dict):
                body = {"unexpected_response_type": type(body).__name__}
            attempts.append(
                {
                    "attempt": index + 1,
                    "http_status": response.status_code,
                    "outcome": "success" if response.is_success else "http_error",
                    "metrics": extract_provider_metrics(
                        provider_name,
                        body,
                        dict(response.headers),
                        elapsed_ms,
                    ),
                    "response_schema": response_schema(body),
                    "response_header_names": sorted(response.headers),
                }
            )
            if index + 1 < repeat:
                time.sleep(2)
    return {
        "provider": provider_name,
        "endpoint": provider.endpoint,
        "model": provider.model,
        "status": (
            "completed"
            if all(attempt["outcome"] == "success" for attempt in attempts)
            else "completed-with-errors"
        ),
        "plot_metric_groups": PLOT_METRICS,
        "attempts": attempts,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--live", action="store_true", help="make paid direct-provider requests")
    value.add_argument(
        "--provider",
        choices=["all", *PROVIDERS],
        default="all",
        help="direct provider to probe",
    )
    value.add_argument("--repeat", type=int, default=2)
    value.add_argument("--prefix-repetitions", type=int, default=1_500)
    value.add_argument("--timeout", type=float, default=180)
    value.add_argument("--aiperf", type=Path, help="AIPerf job directory or JSONL trace")
    value.add_argument("--output", type=Path, help="optional JSON report destination")
    return value


def main() -> int:
    args = parser().parse_args()
    if not args.live and args.aiperf is None:
        raise SystemExit("select --live and/or provide --aiperf PATH")
    if args.repeat < 1 or args.prefix_repetitions < 1 or args.timeout <= 0:
        raise SystemExit("repeat, prefix repetitions, and timeout must be positive")

    load_dotenv()
    report: dict[str, Any] = {"direct_provider_probes": [], "aiperf": None}
    if args.live:
        names = list(PROVIDERS) if args.provider == "all" else [args.provider]
        report["direct_provider_probes"] = [
            probe_provider(
                name,
                repeat=args.repeat,
                prefix_repetitions=args.prefix_repetitions,
                timeout_seconds=args.timeout,
            )
            for name in names
        ]
    if args.aiperf is not None:
        report["aiperf"] = analyze_aiperf(args.aiperf)

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


def test_together_nested_usage_is_normalized():
    metrics = extract_provider_metrics(
        "together",
        {
            "model": "deepseek-ai/DeepSeek-V4-Pro-0813",
            "usage": {
                "prompt_tokens": 10_000,
                "completion_tokens": 50,
                "total_tokens": 10_050,
                "prompt_tokens_details": {"cached_tokens": 8_000},
                "completion_tokens_details": {"reasoning_tokens": 20},
            },
        },
    )
    assert metrics["cached_prompt_tokens"] == 8_000
    assert metrics["cache_hit_rate_pct"] == 80
    assert metrics["reasoning_tokens"] == 20


def test_baseten_flat_cache_usage_is_normalized():
    metrics = extract_provider_metrics(
        "baseten",
        {"usage": {"input_tokens": 4_000, "output_tokens": 20, "cached_tokens": 3_000}},
    )
    assert metrics["prompt_tokens"] == 4_000
    assert metrics["completion_tokens"] == 20
    assert metrics["cache_hit_rate_pct"] == 75


def test_fireworks_body_and_headers_are_normalized():
    metrics = extract_provider_metrics(
        "fireworks",
        {"perf_metrics": {"prompt-tokens": 5_000, "cached-prompt-tokens": 4_000}},
        {
            "fireworks-server-time-to-first-token": "0.1235",
            "x-ratelimit-remaining": "17",
        },
        250,
    )
    assert metrics["client_request_latency_ms"] == 250
    assert metrics["server_ttft_ms"] == 123.5
    assert metrics["cache_hit_rate_pct"] == 80
    assert metrics["rate_limit_remaining_requests"] == 17


def test_baseten_capacity_and_trace_headers_are_normalized():
    metrics = extract_provider_metrics(
        "baseten",
        {"usage": {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 50}}},
        {
            "x-baseten-request-id": "request-123",
            "x-baseten-session-id": "session-123",
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "99",
            "x-ratelimit-limit-tokens": "10000",
            "x-ratelimit-remaining-tokens": "9000",
        },
    )
    assert metrics["request_id"] == "request-123"
    assert metrics["rate_limit_requests"] == 100
    assert metrics["rate_limit_remaining_requests"] == 99
    assert metrics["rate_limit_tokens"] == 10_000
    assert metrics["rate_limit_remaining_tokens"] == 9_000
    assert metrics["provider_trace"]["x-baseten-session-id"] == "session-123"


def test_aiperf_analysis_uses_real_cache_metric_and_excludes_warmup():
    records = [
        {
            "metadata": {"phase_kind": "warmup"},
            "metrics": {"usage_prompt_cache_read_tokens": {"value": 999}},
        },
        {
            "metadata": {
                "phase_kind": "profiling",
                "x_request_id": "request-1",
                "conversation_id": "session-1",
            },
            "metrics": {
                "usage_prompt_tokens": {"value": 8_000},
                "usage_prompt_cache_read_tokens": {"value": 6_000},
                "time_to_first_token": {"value": 100, "unit": "ms"},
            },
        },
        {
            "metadata": {"phase_kind": "profiling"},
            "error": {"type": "Too Many Requests", "code": 429},
        },
    ]
    analysis = analyze_aiperf_records(records)
    assert analysis["records"] == 2
    assert analysis["successful_records"] == 1
    assert analysis["cache"]["cached_prompt_tokens"] == 6_000
    assert analysis["cache"]["cache_hit_rate_pct"] == 75
    assert analysis["plot_metric_groups"]["latency"] == ["time_to_first_token"]
    assert analysis["errors"] == {"Too Many Requests": 1}
    assert analysis["trace_metadata_fields"] == ["conversation_id", "phase_kind", "x_request_id"]


def test_canonical_aiperf_adapter_recognizes_usage_cache_fields():
    from benchmark_tool.adapters.aiperf import token_usage

    usage = token_usage(
        [
            {
                "metadata": {"phase_kind": "profiling"},
                "metrics": {
                    "input_sequence_length": {"value": 8_000},
                    "output_sequence_length": {"value": 100},
                    "usage_prompt_cache_read_tokens": {"value": 6_000},
                    "usage_prompt_cache_write_tokens": {"value": 2_000},
                },
            }
        ]
    )
    assert usage.input_tokens == 8_000
    assert usage.output_tokens == 100
    assert usage.cached_input_tokens == 6_000
    assert usage.cache_write_tokens == 2_000


def test_response_schema_records_shapes_not_values():
    schema = response_schema(
        {
            "usage": {"prompt_tokens": 10},
            "choices": [{"message": {"content": "secret"}}],
        }
    )
    assert schema["usage.prompt_tokens"] == "int"
    assert schema["choices"] == "array"
    assert schema["choices[].message.content"] == "str"
    assert "secret" not in json.dumps(schema)


def test_provider_registry_uses_direct_endpoints_only():
    assert set(PROVIDERS) == {"together", "baseten", "fireworks"}
    assert PROVIDERS["together"].endpoint == "https://api.together.ai/v1/chat/completions"
    assert PROVIDERS["baseten"].endpoint == "https://inference.baseten.co/v1/chat/completions"
    assert (
        PROVIDERS["fireworks"].endpoint
        == "https://api.fireworks.ai/inference/v1/chat/completions"
    )
    assert all("openrouter" not in provider.endpoint for provider in PROVIDERS.values())
    assert PROVIDERS["baseten"].auth_scheme == "Api-Key"
    assert PROVIDERS["together"].auth_scheme == "Bearer"
    assert PROVIDERS["fireworks"].auth_scheme == "Bearer"


if __name__ == "__main__":
    raise SystemExit(main())
