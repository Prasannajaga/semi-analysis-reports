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
from benchmark_tool.adapters.pricing import normalize_pricing
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
