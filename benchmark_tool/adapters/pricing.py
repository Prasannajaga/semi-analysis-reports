"""Pricing normalization and trace-level cost breakdown from pricing snapshots."""

from __future__ import annotations

from typing import Any

from benchmark_tool.adapters.aiperf import _record_metric, profiling_records
from benchmark_tool.results import PricingResult, TokenUsage, TraceCost


def _price(pricing: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = pricing.get(name)
        try:
            return float(value) * 1_000_000 if value is not None else None
        except (TypeError, ValueError):
            continue
    return None


def extract_pricing_rates(snapshot: dict[str, Any]) -> dict[str, float | None]:
    matches = snapshot.get("matchingEndpoints", [])
    if not isinstance(matches, list) or not matches or not isinstance(matches[0], dict):
        return {
            "input_price": None,
            "output_price": None,
            "cached_price": None,
            "write_price": None,
        }
    endpoint = matches[0]
    raw_pricing = endpoint.get("pricing", {})
    if not isinstance(raw_pricing, dict):
        raw_pricing = {}
    return {
        "input_price": _price(raw_pricing, "prompt", "input"),
        "output_price": _price(raw_pricing, "completion", "output"),
        "cached_price": _price(raw_pricing, "input_cache_read", "cached_prompt", "cache_read"),
        "write_price": _price(raw_pricing, "input_cache_write", "cache_write"),
    }


def _calculate_cost(
    input_tokens: int,
    cached_input_tokens: int,
    cache_write_tokens: int,
    output_tokens: int,
    rates: dict[str, float | None],
) -> float | None:
    input_price = rates.get("input_price")
    output_price = rates.get("output_price")
    cached_price = rates.get("cached_price")
    write_price = rates.get("write_price")
    if input_price is None or output_price is None:
        return None
    uncached = max(0, input_tokens - cached_input_tokens)
    cost = uncached * input_price / 1_000_000 + output_tokens * output_price / 1_000_000
    if cached_input_tokens:
        cost += cached_input_tokens * (cached_price or input_price) / 1_000_000
    if cache_write_tokens:
        cost += cache_write_tokens * (write_price or input_price) / 1_000_000
    return cost


def _record_trace_id(record: dict[str, Any], fallback_id: str = "unknown") -> str:
    metadata = record.get("metadata", {})
    if isinstance(metadata, dict):
        trace_id = (
            metadata.get("source_trace_id")
            or metadata.get("conversation_id")
            or metadata.get("session_id")
        )
        if trace_id is not None:
            return str(trace_id)
        req_id = metadata.get("x_request_id") or metadata.get("request_id") or record.get("request_id")
        if req_id is not None:
            return str(req_id)
    return fallback_id


def calculate_trace_costs(
    snapshot: dict[str, Any],
    records: list[dict[str, Any]],
    enabled: bool = True,
) -> list[TraceCost]:
    if not enabled:
        return []
    rates = extract_pricing_rates(snapshot)
    prof_records = profiling_records(records)
    if not prof_records:
        return []

    traces: dict[str, dict[str, Any]] = {}
    for idx, record in enumerate(prof_records):
        trace_id = _record_trace_id(record, fallback_id=f"untracked_{idx}")
        if trace_id not in traces:
            traces[trace_id] = {
                "request_count": 0,
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "cache_write_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
            }
        t = traces[trace_id]
        t["request_count"] += 1
        t["input_tokens"] += round(
            _record_metric(record, "input_sequence_length", "input_token_count", "error_isl") or 0.0
        )
        t["cached_input_tokens"] += round(
            _record_metric(
                record,
                "usage_prompt_cache_read_tokens",
                "cached_input_tokens",
                "cache_read_input_tokens",
            ) or 0.0
        )
        t["cache_write_tokens"] += round(
            _record_metric(
                record,
                "usage_prompt_cache_write_tokens",
                "cache_write_tokens",
            ) or 0.0
        )
        t["output_tokens"] += round(
            _record_metric(record, "output_sequence_length", "output_token_count") or 0.0
        )
        t["reasoning_tokens"] += round(
            _record_metric(record, "usage_reasoning_tokens", "reasoning_token_count") or 0.0
        )

    result: list[TraceCost] = []
    for trace_id, data in traces.items():
        inp = data["input_tokens"]
        cached = data["cached_input_tokens"]
        uncached = max(0, inp - cached)
        write = data["cache_write_tokens"]
        out = data["output_tokens"]
        reasoning = data["reasoning_tokens"]
        count = data["request_count"]
        hit_rate = cached / inp if inp > 0 else 0.0
        cost = _calculate_cost(inp, cached, write, out, rates)
        cost_per_req = cost / count if cost is not None and count > 0 else None
        result.append(
            TraceCost(
                trace_id=trace_id,
                request_count=count,
                input_tokens=inp,
                cached_input_tokens=cached,
                uncached_input_tokens=uncached,
                cache_write_tokens=write,
                cache_hit_rate=round(hit_rate, 4),
                prompt_cache_read_ratio=round(hit_rate, 4),
                output_tokens=out,
                reasoning_tokens=reasoning,
                total_tokens=inp + out,
                estimated_cost_usd=cost,
                cost_per_request_usd=cost_per_req,
            )
        )
    return result


def generate_trace_cost_breakdown(
    snapshot: dict[str, Any],
    records: list[dict[str, Any]],
    enabled: bool = True,
) -> dict[str, Any]:
    trace_costs = calculate_trace_costs(snapshot, records, enabled=enabled)
    rates = extract_pricing_rates(snapshot)
    has_costs = any(t.estimated_cost_usd is not None for t in trace_costs)
    total_cost = (
        sum(t.estimated_cost_usd for t in trace_costs if t.estimated_cost_usd is not None)
        if has_costs
        else None
    )
    total_requests = sum(t.request_count for t in trace_costs)
    avg_cost_per_trace = (
        total_cost / len(trace_costs)
        if total_cost is not None and len(trace_costs) > 0
        else None
    )
    return {
        "schemaVersion": "1.0",
        "totalTraces": len(trace_costs),
        "totalRequests": total_requests,
        "totalEstimatedCostUsd": total_cost,
        "avgCostPerTraceUsd": avg_cost_per_trace,
        "pricingRates": {
            "inputUsdPerMillion": rates.get("input_price"),
            "outputUsdPerMillion": rates.get("output_price"),
            "cachedInputUsdPerMillion": rates.get("cached_price"),
            "cacheWriteUsdPerMillion": rates.get("write_price"),
        },
        "traces": [
            {
                "traceId": t.trace_id,
                "requestCount": t.request_count,
                "inputTokens": t.input_tokens,
                "cachedInputTokens": t.cached_input_tokens,
                "uncachedInputTokens": t.uncached_input_tokens,
                "cacheWriteTokens": t.cache_write_tokens,
                "cacheHitRate": t.cache_hit_rate,
                "promptCacheReadRatio": t.prompt_cache_read_ratio,
                "outputTokens": t.output_tokens,
                "reasoningTokens": t.reasoning_tokens,
                "totalTokens": t.total_tokens,
                "estimatedCostUsd": t.estimated_cost_usd,
                "costPerRequestUsd": t.cost_per_request_usd,
            }
            for t in trace_costs
        ],
    }


def normalize_pricing(
    snapshot: dict[str, Any],
    usage: TokenUsage,
    enabled: bool,
    request_count: int | None = None,
    records: list[dict[str, Any]] | None = None,
) -> PricingResult:
    matches = snapshot.get("matchingEndpoints", [])
    if not enabled:
        return PricingResult(enabled=False, available=False, method="disabled")
    if not isinstance(matches, list) or not matches or not isinstance(matches[0], dict):
        return PricingResult(
            enabled=True,
            available=False,
            method="derived_from_pricing_snapshot",
            token_usage=usage,
            notes=["No matching provider endpoint price was present in the snapshot."],
        )
    rates = extract_pricing_rates(snapshot)
    cost = _calculate_cost(
        usage.input_tokens,
        usage.cached_input_tokens,
        usage.cache_write_tokens,
        usage.output_tokens,
        rates,
    )
    trace_costs = (
        calculate_trace_costs(snapshot, records, enabled=enabled)
        if records is not None
        else []
    )
    if request_count is None and records is not None:
        prof = profiling_records(records)
        request_count = len(prof) if prof else None

    notes = ["Price is derived, not a separate benchmark workload."]
    if len(matches) > 1:
        notes.append("Multiple provider endpoints matched; the first snapshot entry was used.")
    if usage.missing_input_records > 0:
        notes.append(
            f"{usage.missing_input_records} profiling record(s) missing input token telemetry; cost may be understated."
        )

    return PricingResult(
        enabled=True,
        available=cost is not None,
        method="derived_from_pricing_snapshot",
        input_usd_per_million=rates.get("input_price"),
        output_usd_per_million=rates.get("output_price"),
        cached_input_usd_per_million=rates.get("cached_price"),
        cache_write_usd_per_million=rates.get("write_price"),
        token_usage=usage,
        estimated_cost_usd=cost,
        cost_per_request_usd=cost / request_count if cost is not None and request_count else None,
        notes=notes,
        trace_costs=trace_costs,
    )
