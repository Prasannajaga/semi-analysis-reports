from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from benchmark_tool.adapters.aiperf import (
    agentx_metadata,
    classify_error,
    load_aiperf,
    performance_metrics,
    reliability,
    token_usage,
)
from benchmark_tool.adapters.correctness import parse_bfcl, parse_lm_eval
from benchmark_tool.adapters.pricing import (
    calculate_trace_costs,
    generate_trace_cost_breakdown,
    normalize_pricing,
)
from benchmark_tool.results import MetricStats, TokenUsage


FIXTURES = Path(__file__).parent / "fixtures"


def test_aiperf_summary_and_request_artifact_parsing(tmp_path):
    artifacts = tmp_path / "artifacts"
    shutil.copytree(FIXTURES / "aiperf", artifacts)
    summary, records, _, _ = load_aiperf(tmp_path)
    metrics = performance_metrics(summary)
    assert metrics["ttft"].p95 == 410.0
    assert metrics["e2e_latency"].mean == 900.0
    assert len(records) == 8


def test_agentx_invalidity_reasons_are_preserved():
    result = agentx_metadata(
        {
            "metadata": {
                "scenario": "inferencex-agentx-mvp",
                "submission_valid": False,
                "submission_invalid_reasons": ["context_overflow_rate_exceeded"],
            }
        },
        10,
    )
    assert result.submission_valid is False
    assert result.submission_invalid_reasons == ["context_overflow_rate_exceeded"]


def test_reliability_uses_only_profiling_population(tmp_path):
    artifacts = tmp_path / "artifacts"
    shutil.copytree(FIXTURES / "aiperf", artifacts)
    summary, records, _, _ = load_aiperf(tmp_path)
    result = reliability(records, 0.4, 500.0, performance_metrics(summary))
    assert result.total_requests == 7
    assert result.successful_requests == 3
    assert result.failed_requests == 4
    assert result.success_rate == pytest.approx(3 / 7)
    assert result.errors["http_429"] == 1
    assert result.errors["http_4xx"] == 1
    assert result.errors["http_5xx"] == 1
    assert result.errors["timeouts"] == 1
    assert result.error_rate == pytest.approx(4 / 7)
    assert result.slo.passed is True


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        ({"code": 429, "message": "rate limit"}, "http_429"),
        ({"code": 401, "message": "unauthorized"}, "http_4xx"),
        ({"code": 502, "message": "bad gateway"}, "http_5xx"),
        ({"type": "ReadTimeout", "message": "timed out"}, "timeouts"),
        ({"type": "JSONDecodeError", "message": "parse malformed JSON"}, "parse_errors"),
        ({"type": "ConnectError", "message": "socket disconnected"}, "connection_errors"),
        ({"message": "maximum context length exceeded"}, "context_overflow"),
        ({"code": 400, "message": "maximum context length exceeded"}, "context_overflow"),
        ({"code": 408, "message": "request timed out"}, "timeouts"),
        ({"code": 504, "message": "gateway timeout"}, "timeouts"),
        ({"code": 413, "message": "payload too large"}, "context_overflow"),
    ],
)
def test_error_classification(error, expected):
    assert classify_error(error, {}) == expected


def test_token_usage_excludes_warmup(tmp_path):
    artifacts = tmp_path / "artifacts"
    shutil.copytree(FIXTURES / "aiperf", artifacts)
    _, records, _, _ = load_aiperf(tmp_path)
    usage = token_usage(records)
    assert usage.input_tokens == 4900
    assert usage.output_tokens == 270


def test_pricing_calculation_from_snapshot():
    snapshot = {
        "matchingEndpoints": [
            {
                "provider_name": "Provider A",
                "pricing": {
                    "prompt": "0.000002",
                    "completion": "0.000006",
                    "input_cache_read": "0.0000005",
                    "input_cache_write": "0.0000025",
                },
            }
        ]
    }
    usage = TokenUsage(
        input_tokens=1_000_000,
        output_tokens=100_000,
        cached_input_tokens=250_000,
        cache_write_tokens=20_000,
    )
    result = normalize_pricing(snapshot, usage, True)
    assert result.input_usd_per_million == 2.0
    assert result.output_usd_per_million == 6.0
    assert result.estimated_cost_usd == pytest.approx(2.275)


def test_trace_cost_calculation_multi_trace():
    snapshot = {
        "matchingEndpoints": [
            {
                "provider_name": "Provider A",
                "pricing": {
                    "prompt": "0.000003",
                    "completion": "0.000015",
                    "input_cache_read": "0.0000003",
                    "input_cache_write": "0.0000015",
                },
            }
        ]
    }
    records = [
        # Warmup record: must be excluded
        {
            "metadata": {"benchmark_phase": "warmup", "source_trace_id": "trace-1"},
            "metrics": {
                "input_sequence_length": {"value": 50000},
                "usage_prompt_cache_read_tokens": {"value": 40000},
                "output_sequence_length": {"value": 500},
            },
        },
        # Trace 1, turn 1
        {
            "metadata": {"phase_kind": "profiling", "source_trace_id": "trace-1"},
            "metrics": {
                "input_sequence_length": {"value": 10000},
                "usage_prompt_cache_read_tokens": {"value": 8000},
                "output_sequence_length": {"value": 200},
                "usage_reasoning_tokens": {"value": 50},
            },
        },
        # Trace 1, turn 2
        {
            "metadata": {"phase_kind": "profiling", "source_trace_id": "trace-1"},
            "metrics": {
                "input_sequence_length": {"value": 15000},
                "usage_prompt_cache_read_tokens": {"value": 14000},
                "output_sequence_length": {"value": 300},
                "usage_reasoning_tokens": {"value": 100},
            },
        },
        # Trace 2, turn 1
        {
            "metadata": {"phase_kind": "profiling", "source_trace_id": "trace-2"},
            "metrics": {
                "input_sequence_length": {"value": 20000},
                "usage_prompt_cache_read_tokens": {"value": 5000},
                "output_sequence_length": {"value": 1000},
                "usage_reasoning_tokens": {"value": 0},
            },
        },
    ]

    trace_costs = calculate_trace_costs(snapshot, records, enabled=True)
    assert len(trace_costs) == 2

    t1 = trace_costs[0]
    assert t1.trace_id == "trace-1"
    assert t1.request_count == 2
    assert t1.input_tokens == 25000
    assert t1.cached_input_tokens == 22000
    assert t1.uncached_input_tokens == 3000
    assert t1.output_tokens == 500
    assert t1.reasoning_tokens == 150
    assert t1.total_tokens == 25500
    assert t1.cache_hit_rate == pytest.approx(22000 / 25000, abs=1e-4)
    # uncached: 3000 * 3.0 / 1e6 = 0.009
    # cached: 22000 * 0.3 / 1e6 = 0.0066
    # output: 500 * 15.0 / 1e6 = 0.0075
    # total: 0.009 + 0.0066 + 0.0075 = 0.0231
    assert t1.estimated_cost_usd == pytest.approx(0.0231)
    assert t1.cost_per_request_usd == pytest.approx(0.0231 / 2)

    t2 = trace_costs[1]
    assert t2.trace_id == "trace-2"
    assert t2.request_count == 1
    assert t2.input_tokens == 20000
    assert t2.cached_input_tokens == 5000
    assert t2.uncached_input_tokens == 15000
    assert t2.output_tokens == 1000
    assert t2.cache_hit_rate == pytest.approx(5000 / 20000, abs=1e-4)
    # uncached: 15000 * 3.0 / 1e6 = 0.045
    # cached: 5000 * 0.3 / 1e6 = 0.0015
    # output: 1000 * 15.0 / 1e6 = 0.015
    # total: 0.045 + 0.0015 + 0.015 = 0.0615
    assert t2.estimated_cost_usd == pytest.approx(0.0615)
    assert t2.cost_per_request_usd == pytest.approx(0.0615)


def test_trace_cost_fallback_to_conversation_id():
    snapshot = {
        "matchingEndpoints": [
            {
                "pricing": {"prompt": "0.000001", "completion": "0.000002"},
            }
        ]
    }
    records = [
        {
            "metadata": {"phase_kind": "profiling", "conversation_id": "conv-xyz"},
            "metrics": {
                "input_sequence_length": {"value": 1000},
                "output_sequence_length": {"value": 100},
            },
        }
    ]
    trace_costs = calculate_trace_costs(snapshot, records, enabled=True)
    assert len(trace_costs) == 1
    assert trace_costs[0].trace_id == "conv-xyz"


def test_generate_trace_cost_breakdown_schema():
    snapshot = {
        "matchingEndpoints": [
            {
                "pricing": {"prompt": "0.000002", "completion": "0.000005"},
            }
        ]
    }
    records = [
        {
            "metadata": {"phase_kind": "profiling", "source_trace_id": "t1"},
            "metrics": {
                "input_sequence_length": {"value": 1000},
                "output_sequence_length": {"value": 100},
            },
        }
    ]
    breakdown = generate_trace_cost_breakdown(snapshot, records, enabled=True)
    assert breakdown["schemaVersion"] == "1.0"
    assert breakdown["totalTraces"] == 1
    assert breakdown["totalRequests"] == 1
    assert breakdown["totalEstimatedCostUsd"] is not None
    assert breakdown["avgCostPerTraceUsd"] is not None
    assert "pricingRates" in breakdown
    assert len(breakdown["traces"]) == 1
    assert breakdown["traces"][0]["traceId"] == "t1"


def test_normalize_pricing_with_trace_records():
    snapshot = {
        "matchingEndpoints": [
            {
                "pricing": {"prompt": "0.000002", "completion": "0.000005"},
            }
        ]
    }
    usage = TokenUsage(input_tokens=1000, output_tokens=100)
    records = [
        {
            "metadata": {"phase_kind": "profiling", "source_trace_id": "trace-a"},
            "metrics": {
                "input_sequence_length": {"value": 1000},
                "output_sequence_length": {"value": 100},
            },
        }
    ]
    result = normalize_pricing(snapshot, usage, True, request_count=1, records=records)
    assert len(result.trace_costs) == 1
    assert result.trace_costs[0].trace_id == "trace-a"


def test_lm_eval_correctness_normalization(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    shutil.copy(FIXTURES / "lm_eval" / "results.json", artifacts / "results.json")
    result, _ = parse_lm_eval(tmp_path, "gsm8k", "gsm8k")
    assert result.score == 0.75
    assert result.primary_metric == "exact_match,strict-match"
    assert result.sample_count == 100


def test_bfcl_correctness_normalization(tmp_path):
    score_dir = tmp_path / "artifacts" / "scores" / "model"
    score_dir.mkdir(parents=True)
    path = score_dir / "BFCL_v4_simple_python_score.json"
    path.write_text(
        json.dumps(
            [
                {"accuracy": 0.8, "correct_count": 80, "total_count": 100},
                {"id": "simple_python_0", "valid": True},
            ]
        ),
        encoding="utf-8",
    )
    result, source = parse_bfcl(tmp_path, "bfcl", "simple_python")
    assert source == path
    assert result.score == 0.8
    assert result.sample_count == 100


def test_pricing_breakdown_disabled():
    snapshot = {
        "matchingEndpoints": [{"pricing": {"prompt": "0.000001", "completion": "0.000002"}}]
    }
    records = [
        {
            "metadata": {"phase_kind": "profiling", "source_trace_id": "t1"},
            "metrics": {"input_sequence_length": {"value": 1000}, "output_sequence_length": {"value": 100}},
        }
    ]
    breakdown = generate_trace_cost_breakdown(snapshot, records, enabled=False)
    assert breakdown["totalTraces"] == 0
    assert len(breakdown["traces"]) == 0
    assert breakdown["totalEstimatedCostUsd"] is None


def test_normalize_pricing_derives_request_count_without_reliability():
    snapshot = {
        "matchingEndpoints": [{"pricing": {"prompt": "0.000002", "completion": "0.000004"}}]
    }
    usage = TokenUsage(input_tokens=1000, output_tokens=100)
    records = [
        {
            "metadata": {"phase_kind": "profiling", "source_trace_id": "t1"},
            "metrics": {"input_sequence_length": {"value": 500}, "output_sequence_length": {"value": 50}},
        },
        {
            "metadata": {"phase_kind": "profiling", "source_trace_id": "t2"},
            "metrics": {"input_sequence_length": {"value": 500}, "output_sequence_length": {"value": 50}},
        },
    ]
    result = normalize_pricing(snapshot, usage, enabled=True, request_count=None, records=records)
    assert result.cost_per_request_usd is not None
    assert result.cost_per_request_usd == pytest.approx(result.estimated_cost_usd / 2)


def test_token_usage_aggregates_error_isl():
    records = [
        {
            "metadata": {"phase_kind": "profiling"},
            "metrics": {"input_sequence_length": {"value": 1000}, "output_sequence_length": {"value": 100}},
        },
        {
            "metadata": {"phase_kind": "profiling"},
            "metrics": {"error_isl": {"value": 2500}},
            "error": {"code": 500, "message": "server error"},
        },
    ]
    usage = token_usage(records)
    assert usage.input_tokens == 3500
    assert usage.output_tokens == 100
    assert usage.missing_input_records == 0


def test_trace_ids_without_metadata_not_merged():
    snapshot = {
        "matchingEndpoints": [{"pricing": {"prompt": "0.000001", "completion": "0.000002"}}]
    }
    records = [
        {
            "metadata": {"phase_kind": "profiling", "session_num": 0, "x_request_id": "req-1"},
            "metrics": {"input_sequence_length": {"value": 1000}, "output_sequence_length": {"value": 100}},
        },
        {
            "metadata": {"phase_kind": "profiling", "session_num": 0, "x_request_id": "req-2"},
            "metrics": {"input_sequence_length": {"value": 1000}, "output_sequence_length": {"value": 100}},
        },
    ]
    trace_costs = calculate_trace_costs(snapshot, records, enabled=True)
    assert len(trace_costs) == 2
    assert trace_costs[0].trace_id == "req-1"
    assert trace_costs[1].trace_id == "req-2"


def test_prompt_cache_read_ratio_field():
    snapshot = {
        "matchingEndpoints": [
            {"pricing": {"prompt": "0.000002", "completion": "0.000004", "input_cache_read": "0.0000005"}}
        ]
    }
    records = [
        {
            "metadata": {"phase_kind": "profiling", "source_trace_id": "t1"},
            "metrics": {
                "input_sequence_length": {"value": 1000},
                "usage_prompt_cache_read_tokens": {"value": 400},
                "output_sequence_length": {"value": 100},
            },
        }
    ]
    breakdown = generate_trace_cost_breakdown(snapshot, records, enabled=True)
    assert breakdown["traces"][0]["cacheHitRate"] == 0.4
    assert breakdown["traces"][0]["promptCacheReadRatio"] == 0.4

