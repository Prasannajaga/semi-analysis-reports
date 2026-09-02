"""Pricing normalization from the run-start OpenRouter snapshot."""

from __future__ import annotations

from typing import Any

from benchmark_tool.results import PricingResult, TokenUsage


def _price(pricing: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = pricing.get(name)
        try:
            return float(value) * 1_000_000 if value is not None else None
        except (TypeError, ValueError):
            continue
    return None


def normalize_pricing(
    snapshot: dict[str, Any],
    usage: TokenUsage,
    enabled: bool,
    request_count: int | None = None,
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
    endpoint = matches[0]
    raw_pricing = endpoint.get("pricing", {})
    if not isinstance(raw_pricing, dict):
        raw_pricing = {}
    input_price = _price(raw_pricing, "prompt", "input")
    output_price = _price(raw_pricing, "completion", "output")
    cached_price = _price(raw_pricing, "input_cache_read", "cached_prompt", "cache_read")
    write_price = _price(raw_pricing, "input_cache_write", "cache_write")
    cost = None
    if input_price is not None and output_price is not None:
        uncached = max(0, usage.input_tokens - usage.cached_input_tokens)
        cost = uncached * input_price / 1_000_000 + usage.output_tokens * output_price / 1_000_000
        if usage.cached_input_tokens:
            cost += usage.cached_input_tokens * (cached_price or input_price) / 1_000_000
        if usage.cache_write_tokens:
            cost += usage.cache_write_tokens * (write_price or input_price) / 1_000_000
    notes = ["Price is derived, not a separate benchmark workload."]
    if len(matches) > 1:
        notes.append("Multiple provider endpoints matched; the first snapshot entry was used.")
    return PricingResult(
        enabled=True,
        available=cost is not None,
        method="derived_from_pricing_snapshot",
        input_usd_per_million=input_price,
        output_usd_per_million=output_price,
        cached_input_usd_per_million=cached_price,
        cache_write_usd_per_million=write_price,
        token_usage=usage,
        estimated_cost_usd=cost,
        cost_per_request_usd=cost / request_count if cost is not None and request_count else None,
        notes=notes,
    )
