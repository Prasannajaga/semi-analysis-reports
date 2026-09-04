"""Tolerant parser for AIPerf 0.12 summary and per-request artifacts."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from benchmark_tool.config import ReliabilityCollect
from benchmark_tool.io import read_json, read_jsonl
from benchmark_tool.results import (
    AgentX,
    MetricStats,
    ReliabilityResult,
    SloCheck,
    SloResult,
    TokenUsage,
)

PERFORMANCE_METRICS = {
    "request_latency": "e2e_latency",
    "time_to_first_token": "ttft",
    "inter_token_latency": "itl",
    "request_throughput": "request_throughput",
    "output_token_throughput": "output_token_throughput",
    "output_token_throughput_per_user": "output_token_throughput_per_user",
    "request_count": "request_count",
    "goodput": "goodput",
    "input_sequence_length": "input_sequence_length",
    "output_sequence_length": "output_sequence_length",
}


def find_artifact(job_dir: Path, name: str) -> Path:
    artifacts_dir = job_dir / "artifacts"
    # 1. Check direct file in root artifacts
    direct = artifacts_dir / name
    if direct.is_file():
        return direct
    # 2. Check main profiling phase artifact
    profiling = artifacts_dir / "phases" / "profiling" / name
    if profiling.is_file():
        return profiling
    # 3. Filter out warmup artifacts from rglob matches
    non_warmup = [p for p in sorted(artifacts_dir.rglob(name)) if "warmup" not in p.parts]
    if non_warmup:
        return non_warmup[0]
    matches = sorted(artifacts_dir.rglob(name))
    if not matches:
        raise ValueError(f"expected {name} under {artifacts_dir}, found 0")
    return matches[0]


def load_aiperf(job_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], Path, Path]:
    summary_path = find_artifact(job_dir, "profile_export_aiperf.json")
    records_path = find_artifact(job_dir, "profile_export.jsonl")
    summary = read_json(summary_path)
    records = read_jsonl(records_path)
    if not isinstance(summary, dict) or not all(isinstance(item, dict) for item in records):
        raise ValueError("invalid AIPerf artifact shape")
    return summary, records, summary_path, records_path


def metric_stats(raw: Any) -> MetricStats | None:
    if not isinstance(raw, dict):
        return None

    def numeric(name: str) -> float | None:
        return float(raw[name]) if isinstance(raw.get(name), (int, float)) else None

    count = raw.get("count")
    return MetricStats(
        unit=str(raw["unit"]) if raw.get("unit") is not None else None,
        mean=numeric("avg"),
        p50=numeric("p50"),
        p90=numeric("p90"),
        p95=numeric("p95"),
        p99=numeric("p99"),
        minimum=numeric("min"),
        maximum=numeric("max"),
        stddev=numeric("std"),
        count=int(count) if isinstance(count, (int, float)) else None,
        total=numeric("sum"),
    )


def performance_metrics(summary: dict[str, Any]) -> dict[str, MetricStats]:
    result = {}
    for source, target in PERFORMANCE_METRICS.items():
        value = metric_stats(summary.get(source))
        if value is not None:
            result[target] = value
    return result


def agentx_metadata(summary: dict[str, Any], warmup_request_count: int) -> AgentX:
    metadata = summary.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    scenario = metadata.get("scenario")
    validity = metadata.get("submission_valid")
    invalid_reasons = metadata.get("submission_invalid_reasons", [])
    return AgentX(
        scenario=str(scenario) if scenario is not None else "inferencex-agentx-mvp",
        submission_valid=validity if isinstance(validity, bool) else None,
        submission_invalid_reasons=(
            [str(reason) for reason in invalid_reasons]
            if isinstance(invalid_reasons, list)
            else []
        ),
        warmup_request_count=warmup_request_count,
    )


def profiling_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for record in records:
        metadata = record.get("metadata", {})
        phase = ""
        if isinstance(metadata, dict):
            phase = str(
                metadata.get("phase_kind")
                or metadata.get("benchmark_phase")
                or metadata.get("phase_name")
                or ""
            ).lower()
        if phase and "profil" not in phase:
            continue
        selected.append(record)
    return selected


def _error_text(error: Any) -> tuple[int | None, str]:
    if not isinstance(error, dict):
        return None, str(error or "")
    code = error.get("code")
    try:
        status = int(code) if code is not None else None
    except (TypeError, ValueError):
        status = None
    parts = [error.get("type"), error.get("message"), error.get("cause")]
    return status, " ".join(str(part) for part in parts if part).lower()


def classify_error(error: Any, record: dict[str, Any]) -> str:
    code, text = _error_text(error)
    metadata = record.get("metadata", {})
    if isinstance(metadata, dict) and metadata.get("was_cancelled"):
        return "cancelled_requests"
    if (
        (isinstance(metadata, dict) and metadata.get("context_overflow_skip"))
        or code == 413
        or ("context" in text and any(word in text for word in ("length", "window", "overflow", "long", "exceeded", "limit")))
        or "maximum context" in text
    ):
        return "context_overflow"
    if code in (408, 504) or "timeout" in text or "timed out" in text:
        return "timeouts"
    if any(word in text for word in ("connect", "disconnect", "dns", "socket", "connection reset", "connection refused")):
        return "connection_errors"
    if any(word in text for word in ("parse", "json", "decode", "malformed", "invalid response", "invalid inference result")):
        return "parse_errors"
    if code == 429 or "rate limit" in text:
        return "http_429"
    if code is not None and 400 <= code < 500:
        return "http_4xx"
    if code is not None and 500 <= code < 600:
        return "http_5xx"
    return "other"


def reliability(
    records: list[dict[str, Any]],
    minimum_success_rate: float,
    maximum_p95_ttft_ms: float | None,
    performance: dict[str, MetricStats],
    collect: ReliabilityCollect | None = None,
) -> ReliabilityResult:
    records = profiling_records(records)
    errors: Counter[str] = Counter()
    successful = 0
    for record in records:
        error = record.get("error")
        metadata = record.get("metadata", {})
        was_cancelled = isinstance(metadata, dict) and bool(metadata.get("was_cancelled"))
        if error or was_cancelled:
            errors[classify_error(error, record)] += 1
        else:
            successful += 1
    failed = sum(errors.values())
    total = successful + failed
    success_rate = successful / total if total else None
    checks = {
        "success_rate": SloCheck(
            value=success_rate,
            target=minimum_success_rate,
            passed=success_rate is not None and success_rate >= minimum_success_rate,
        )
    }
    ttft = performance.get("ttft")
    if maximum_p95_ttft_ms is not None:
        ttft_value = ttft.p95 if ttft else None
        checks["p95_ttft_ms"] = SloCheck(
            value=ttft_value,
            target=maximum_p95_ttft_ms,
            passed=ttft_value is not None and ttft_value <= maximum_p95_ttft_ms,
        )
    categories = (
        "http_4xx",
        "http_429",
        "http_5xx",
        "timeouts",
        "connection_errors",
        "parse_errors",
        "context_overflow",
        "cancelled_requests",
        "other",
    )
    collect = collect or ReliabilityCollect()
    enabled_categories = {
        "http_4xx": collect.http_errors,
        "http_429": collect.http_errors,
        "http_5xx": collect.http_errors,
        "timeouts": collect.timeouts,
        "connection_errors": collect.connection_errors,
        "parse_errors": collect.parse_errors,
        "context_overflow": collect.context_overflow,
        "cancelled_requests": True,
        "other": True,
    }
    return ReliabilityResult(
        successful_requests=successful,
        failed_requests=failed,
        total_requests=total,
        success_rate=success_rate,
        error_rate=failed / total if total else None,
        errors={name: errors[name] for name in categories if enabled_categories[name]},
        slo=SloResult(passed=all(check.passed for check in checks.values()), checks=checks),
    )


def _record_metric(
    record: dict[str, Any],
    *names: str,
    default: float | None = 0.0,
) -> float | None:
    metrics = record.get("metrics", {})
    if not isinstance(metrics, dict):
        return default
    for name in names:
        value = metrics.get(name)
        if isinstance(value, dict):
            value = value.get("value")
        if isinstance(value, (int, float)):
            return float(value)
    return default


def token_usage(records: list[dict[str, Any]]) -> TokenUsage:
    records = profiling_records(records)
    missing_input = 0
    input_total = 0.0
    for item in records:
        val = _record_metric(
            item,
            "input_sequence_length",
            "input_token_count",
            "error_isl",
            default=None,
        )
        if val is None:
            missing_input += 1
        else:
            input_total += val

    return TokenUsage(
        input_tokens=round(input_total),
        output_tokens=round(
            sum(
                _record_metric(item, "output_sequence_length", "output_token_count") or 0.0
                for item in records
            )
        ),
        cached_input_tokens=round(
            sum(
                _record_metric(
                    item,
                    "usage_prompt_cache_read_tokens",
                    "cached_input_tokens",
                    "cache_read_input_tokens",
                ) or 0.0
                for item in records
            )
        ),
        cache_write_tokens=round(
            sum(
                _record_metric(
                    item,
                    "usage_prompt_cache_write_tokens",
                    "cache_write_tokens",
                ) or 0.0
                for item in records
            )
        ),
        missing_input_records=missing_input,
    )
