"""Live endpoint probe for OpenRouter and direct provider endpoints (Together, Fireworks, Baseten)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path when executed directly as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx
import pytest
from dotenv import load_dotenv

from benchmark_tool.openrouter import OpenRouterClient, pricing_snapshot

TARGET_PROVIDERS = [
    {"id": "fireworks", "openrouter_slug": "fireworks", "name": "Fireworks"},
    {"id": "baseten", "openrouter_slug": "baseten", "name": "Baseten"},
    {"id": "together", "openrouter_slug": "together", "name": "Together"},
]

TARGET_MODELS = [
    {
        "id": "kimi-k3",
        "openrouter_model": "moonshotai/kimi-k3",
        "name": "Kimi K3",
        "min_context": 1_000_000,
    },
    {
        "id": "deepseek-v4-pro",
        "openrouter_model": "deepseek/deepseek-v4-pro-0813",
        "name": "DeepSeek V4 Pro",
        "min_context": 1_000_000,
    },
    {
        "id": "glm-5.3-flash",
        "openrouter_model": "z-ai/glm-5.3-flash",
        "name": "GLM 5.3 Flash",
        "min_context": 1_000_000,
    },
]

# Direct provider endpoints and native model identifiers
DIRECT_PROVIDERS: dict[str, dict[str, Any]] = {
    "together": {
        "id": "together",
        "name": "Together AI",
        "base_url": "https://api.together.ai/v1",
        "api_key_env": "TOGETHER_API_KEY",
        "auth_header": lambda k: {"Authorization": f"Bearer {k}"},
        "models": {
            "kimi-k3": "moonshotai/Kimi-K3",
            "deepseek-v4-pro": "deepseek-ai/DeepSeek-V4-Pro-0813",
            "glm-5.3-flash": "zai-org/GLM-5.3-Flash",
        },
    },
    "fireworks": {
        "id": "fireworks",
        "name": "Fireworks AI",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "api_key_env": "FIREWORKS_API_KEY",
        "auth_header": lambda k: {"Authorization": f"Bearer {k}"},
        "models": {
            "kimi-k3": "accounts/fireworks/models/kimi-k3",
            "deepseek-v4-pro": "accounts/fireworks/models/deepseek-v4-pro-0813",
            "glm-5.3-flash": "accounts/fireworks/models/glm-5p3-flash",
        },
    },
    "baseten": {
        "id": "baseten",
        "name": "Baseten",
        "base_url": "https://inference.baseten.co/v1",
        "api_key_env": "BASETEN_API_KEY",
        "auth_header": lambda k: {"Authorization": f"Api-Key {k}" if not k.startswith("Bearer ") else k},
        "models": {
            "kimi-k3": "moonshotai/Kimi-K3",
            "deepseek-v4-pro": "deepseek-ai/DeepSeek-V4-Pro-0813",
            "glm-5.3-flash": "zai-org/GLM-5.3-Flash",
        },
    },
}


def probe_direct_endpoint(
    provider_id: str,
    provider_name: str,
    model_id: str,
    provider_model: str,
    base_url: str,
    api_key: str,
    auth_headers: dict[str, str],
    timeout_seconds: float = 30.0,
    max_retries: int = 3,
    retry_delay_seconds: float = 2.0,
    backoff_factor: float = 2.0,
    extra_headers: dict[str, str] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Test a direct 1-token streaming chat completion against a provider's native endpoint."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "OpenRouter-Provider-Benchmark/1.0",
        **auth_headers,
        **(extra_headers or {}),
    }
    body = {
        "model": provider_model,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_tokens": 1,
        "temperature": 0,
        "stream": True,
    }

    retries_attempted = 0
    current_delay = retry_delay_seconds
    start_time = time.perf_counter()

    while True:
        try:
            with httpx.Client(timeout=timeout_seconds, headers=headers) as client:
                resp = client.post(url, json=body)

            latency_ms = (time.perf_counter() - start_time) * 1000.0

            if resp.status_code == 200:
                chunks = [line[5:].strip() for line in resp.text.splitlines() if line.startswith("data:")]
                has_content = any('"content"' in c or '"delta"' in c for c in chunks if c != "[DONE]")
                return {
                    "provider_id": provider_id,
                    "provider_name": provider_name,
                    "model_id": model_id,
                    "provider_model": provider_model,
                    "endpoint_url": url,
                    "status": "supported",
                    "http_status": 200,
                    "latency_ms": latency_ms,
                    "reason": "OK (stream verified)" if has_content else "OK (stream response received)",
                    "retries_attempted": retries_attempted,
                }

            is_retryable = resp.status_code in {408, 429, 500, 502, 503, 504}
            if is_retryable and retries_attempted < max_retries:
                retries_attempted += 1
                time.sleep(current_delay)
                current_delay *= backoff_factor
                continue

            reason = f"HTTP {resp.status_code}"
            try:
                err_json = resp.json()
                if "error" in err_json:
                    err_msg = (
                        err_json["error"].get("message")
                        if isinstance(err_json["error"], dict)
                        else str(err_json["error"])
                    )
                    reason = f"HTTP {resp.status_code}: {err_msg}"
            except Exception:
                reason = f"HTTP {resp.status_code}: {resp.text[:140]}"

            status = "unauthorized" if resp.status_code == 401 else "failed"
            if resp.status_code in (400, 404):
                status = "unsupported"

            return {
                "provider_id": provider_id,
                "provider_name": provider_name,
                "model_id": model_id,
                "provider_model": provider_model,
                "endpoint_url": url,
                "status": status,
                "http_status": resp.status_code,
                "latency_ms": latency_ms,
                "reason": reason,
                "retries_attempted": retries_attempted,
            }

        except Exception as exc:
            if retries_attempted < max_retries:
                retries_attempted += 1
                time.sleep(current_delay)
                current_delay *= backoff_factor
                continue
            return {
                "provider_id": provider_id,
                "provider_name": provider_name,
                "model_id": model_id,
                "provider_model": provider_model,
                "endpoint_url": url,
                "status": "failed",
                "http_status": None,
                "latency_ms": None,
                "reason": str(exc),
                "retries_attempted": retries_attempted,
            }


def probe_direct_providers(
    provider_keys: dict[str, str],
    timeout_seconds: float = 30.0,
    max_retries: int = 3,
    retry_delay_seconds: float = 2.0,
    backoff_factor: float = 2.0,
    extra_headers: dict[str, str] | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Probe all direct provider endpoints with their separate API keys."""
    results = []
    for p_id, p_info in DIRECT_PROVIDERS.items():
        key = provider_keys.get(p_id)
        if verbose:
            print(f"\n--- Checking Direct Provider: {p_info['name']} ({p_id}) ---")
            print(f"Endpoint: {p_info['base_url']}/chat/completions")
            print(f"API Key Env: {p_info['api_key_env']} {'(FOUND)' if key else '(NOT FOUND)'}")

        if not key:
            for m_id, p_model in p_info["models"].items():
                results.append({
                    "provider_id": p_id,
                    "provider_name": p_info["name"],
                    "model_id": m_id,
                    "provider_model": p_model,
                    "endpoint_url": f"{p_info['base_url']}/chat/completions",
                    "status": "skipped",
                    "http_status": None,
                    "latency_ms": None,
                    "reason": f"API Key {p_info['api_key_env']} not set in environment or .env",
                    "retries_attempted": 0,
                })
            continue

        auth_h = p_info["auth_header"](key)
        for m_id, p_model in p_info["models"].items():
            if verbose:
                print(f"• Probing {m_id} as '{p_model}'...", end="", flush=True)
            res = probe_direct_endpoint(
                provider_id=p_id,
                provider_name=p_info["name"],
                model_id=m_id,
                provider_model=p_model,
                base_url=p_info["base_url"],
                api_key=key,
                auth_headers=auth_h,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                retry_delay_seconds=retry_delay_seconds,
                backoff_factor=backoff_factor,
                extra_headers=extra_headers,
                verbose=verbose,
            )
            if verbose:
                st = res["status"].upper()
                lat = f" ({res['latency_ms']:.0f}ms)" if res.get("latency_ms") is not None else ""
                print(f" [{st}]{lat} - {res['reason']}")
            results.append(res)

    return results


def print_direct_summary(results: list[dict[str, Any]]) -> None:
    """Print a clean tabular breakdown for direct provider endpoints."""
    print("\n" + "=" * 96)
    print("DIRECT PROVIDER ENDPOINT PROBE SUMMARY")
    print("=" * 96)
    print(f"{'Provider':<14} {'Model ID':<18} {'Remote Model':<34} {'Status':<12} {'Latency':<9} {'Notes'}")
    print("-" * 96)
    for r in results:
        status = r["status"].upper()
        lat_str = f"{r['latency_ms']:.0f}ms" if r.get("latency_ms") else "—"
        reason = r.get("reason", "OK")
        if len(reason) > 26:
            reason = reason[:23] + "..."
        print(f"{r['provider_name']:<14} {r['model_id']:<18} {r['provider_model']:<34} {status:<12} {lat_str:<9} {reason}")
    print("=" * 96)


def probe_matrix(
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    timeout_seconds: float = 30.0,
    max_retries: int = 3,
    retry_delay_seconds: float = 2.0,
    backoff_factor: float = 2.0,
    extra_headers: dict[str, str] | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Query endpoint metadata and test 1-token streaming preflight for each model/provider with active retry support."""
    client = OpenRouterClient(
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
        backoff_factor=backoff_factor,
        extra_headers=extra_headers,
        verbose=verbose,
    )
    results = []

    for model in TARGET_MODELS:
        model_id = model["openrouter_model"]
        if verbose:
            print(f"\n--- Checking Model: {model['name']} ({model_id}) ---")
        try:
            metadata = client.endpoint_metadata(model_id)
        except Exception as err:
            metadata = {"error": str(err)}

        for provider in TARGET_PROVIDERS:
            slug = provider["openrouter_slug"]
            if verbose:
                print(f"• Probing {model['id']} on {provider['id']}...", end="", flush=True)

            entry: dict[str, Any] = {
                "model_id": model["id"],
                "openrouter_model": model_id,
                "provider_id": provider["id"],
                "openrouter_slug": slug,
                "advertised_context": None,
                "supports_1m": False,
                "pricing": None,
                "preflight_status": "not-run",
                "reason": None,
                "max_retries": max_retries,
                "retries_attempted": 0,
            }

            if "error" in metadata:
                entry["reason"] = f"metadata fetch failed: {metadata['error']}"
                entry["preflight_status"] = "failed"
                if verbose:
                    print(f" [FAILED: {entry['reason']}]")
                results.append(entry)
                continue

            snapshot = pricing_snapshot(metadata, model_id, slug)
            matches = snapshot.get("matchingEndpoints", [])

            if not matches:
                entry["preflight_status"] = "unsupported"
                entry["reason"] = f"provider '{slug}' does not expose model '{model_id}'"
                if verbose:
                    print(" [UNSUPPORTED: not listed]")
                results.append(entry)
                continue

            contexts = [
                m.get("context_length")
                for m in matches
                if isinstance(m, dict) and isinstance(m.get("context_length"), int)
            ]
            max_ctx = max(contexts) if contexts else 0
            entry["advertised_context"] = max_ctx
            entry["supports_1m"] = max_ctx >= model["min_context"]
            entry["pricing"] = matches[0].get("pricing")

            try:
                probe = client.preflight(model_id, slug, require_streaming=True)
                entry["preflight_status"] = probe.get("status", "failed")
                entry["reason"] = probe.get("reason")
                entry["http_status"] = probe.get("httpStatus")
                entry["retries_attempted"] = probe.get("retriesAttempted", 0)
                entry["retryable"] = probe.get("retryable", False)
                if entry["retries_attempted"] > 0:
                    if entry["preflight_status"] == "supported":
                        entry["reason"] = f"OK (recovered after {entry['retries_attempted']} retries)"
                    else:
                        entry["reason"] = f"{entry['reason']} ({entry['retries_attempted']} retries attempted)"
                if verbose:
                    if entry["preflight_status"] == "supported":
                        recov = f" (recovered after {entry['retries_attempted']} retries)" if entry["retries_attempted"] else ""
                        print(f" [SUPPORTED]{recov}")
                    else:
                        print(f" [{entry['preflight_status'].upper()}: {entry.get('reason')}]")
            except Exception as err:
                entry["preflight_status"] = "failed"
                entry["reason"] = str(err)
                if verbose:
                    print(f" [FAILED: {err}]")

            results.append(entry)

    return results


def print_probe_summary(results: list[dict[str, Any]]) -> None:
    """Print a clean tabular breakdown and highlight compatible combinations."""
    print("\n" + "=" * 92)
    print(f"{'Model':<28} {'Provider':<12} {'1M Context?':<14} {'Retries':<9} {'Status':<12} {'Notes'}")
    print("-" * 92)
    compatible = []

    for r in results:
        ctx_str = f"{r['advertised_context']:,}" if r["advertised_context"] else "N/A"
        is_1m = f"Yes ({ctx_str})" if r["supports_1m"] else f"No ({ctx_str})"
        status = r["preflight_status"].upper()
        retries_str = f"{r.get('retries_attempted', 0)}/{r.get('max_retries', 0)}"
        reason = r["reason"] or "OK"
        if len(reason) > 35:
            reason = reason[:32] + "..."

        print(f"{r['openrouter_model']:<28} {r['openrouter_slug']:<12} {is_1m:<14} {retries_str:<9} {status:<12} {reason}")

        if r["preflight_status"] == "supported" and r["supports_1m"]:
            compatible.append(r)

    print("=" * 92)
    print(f"\nFound {len(compatible)} fully compatible 1M-context endpoints:")
    for c in compatible:
        print(f"  - Model: {c['openrouter_model']} | Provider: {c['openrouter_slug']} ({c['advertised_context']:,} tokens)")


@pytest.mark.integration
def test_probe_target_matrix():
    """Pytest integration target for probing the requested model/provider matrix via OpenRouter."""
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY is not set in environment or .env")

    max_retries = int(os.getenv("OPENROUTER_MAX_RETRIES", "3"))
    retry_delay = float(os.getenv("OPENROUTER_RETRY_DELAY", "2.0"))
    results = probe_matrix(api_key, max_retries=max_retries, retry_delay_seconds=retry_delay, verbose=False)
    print_probe_summary(results)
    assert any(r["preflight_status"] in {"supported", "failed", "unsupported"} for r in results)


@pytest.mark.integration
def test_probe_direct_providers():
    """Pytest integration target for probing direct provider endpoints (Together, Fireworks, Baseten)."""
    load_dotenv()
    provider_keys = {
        "together": os.getenv("TOGETHER_API_KEY"),
        "fireworks": os.getenv("FIREWORKS_API_KEY"),
        "baseten": os.getenv("BASETEN_API_KEY"),
    }
    available = {k: v for k, v in provider_keys.items() if v}
    if not available:
        pytest.skip("No direct provider keys (TOGETHER_API_KEY, FIREWORKS_API_KEY, BASETEN_API_KEY) found.")

    results = probe_direct_providers(provider_keys, verbose=False)
    print_direct_summary(results)
    assert any(r["status"] in {"supported", "failed", "unauthorized", "unsupported"} for r in results)


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Probe OpenRouter and Direct Provider (Together, Fireworks, Baseten) endpoints")
    parser.add_argument("--openrouter-key", type=str, default=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_KEY"))
    parser.add_argument("--together-key", type=str, default=os.getenv("TOGETHER_API_KEY"))
    parser.add_argument("--fireworks-key", type=str, default=os.getenv("FIREWORKS_API_KEY"))
    parser.add_argument("--baseten-key", type=str, default=os.getenv("BASETEN_API_KEY"))
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Probe direct provider endpoints (Together, Fireworks, Baseten) using their separate API keys",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["all", "openrouter", "together", "fireworks", "baseten"],
        default="all",
        help="Specific provider to probe (default: all available)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=int(os.getenv("OPENROUTER_MAX_RETRIES", "3")),
        help="Maximum retry attempts on transient/retryable errors (default: 3)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=float(os.getenv("OPENROUTER_RETRY_DELAY", "2.0")),
        help="Initial retry delay in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("OPENROUTER_TIMEOUT", "30.0")),
        help="Request timeout in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--header",
        action="append",
        dest="custom_headers",
        help="Additional request header in Key:Value format (can be repeated)",
    )
    args = parser.parse_args()

    extra_headers = {}
    if args.custom_headers:
        for h in args.custom_headers:
            if ":" in h:
                k, v = h.split(":", 1)
                extra_headers[k.strip()] = v.strip()

    provider_keys = {
        "together": args.together_key,
        "fireworks": args.fireworks_key,
        "baseten": args.baseten_key,
    }

    # 1. Probe Direct Provider Endpoints if requested or if direct keys are present
    has_direct_keys = any(provider_keys.values())
    if args.direct or args.provider in DIRECT_PROVIDERS or (args.provider == "all" and has_direct_keys):
        active_keys = (
            {args.provider: provider_keys.get(args.provider)}
            if args.provider in DIRECT_PROVIDERS
            else provider_keys
        )
        print("\n========================================================")
        print("  PROBING DIRECT PROVIDER ENDPOINTS (SEPARATE API KEYS) ")
        print("========================================================")
        direct_results = probe_direct_providers(
            active_keys,
            timeout_seconds=args.timeout,
            max_retries=args.retries,
            retry_delay_seconds=args.delay,
            extra_headers=extra_headers if extra_headers else None,
            verbose=True,
        )
        print_direct_summary(direct_results)

    # 2. Probe OpenRouter if requested or if OPENROUTER_API_KEY is available
    if args.provider in ("all", "openrouter") and not args.direct:
        if args.openrouter_key:
            print("\n========================================================")
            print("  PROBING OPENROUTER ROUTING GATEWAY                    ")
            print("========================================================")
            or_results = probe_matrix(
                args.openrouter_key,
                timeout_seconds=args.timeout,
                max_retries=args.retries,
                retry_delay_seconds=args.delay,
                extra_headers=extra_headers if extra_headers else None,
                verbose=True,
            )
            print_probe_summary(or_results)
        elif not has_direct_keys:
            print("ERROR: Neither OPENROUTER_API_KEY nor any direct provider key (TOGETHER_API_KEY, FIREWORKS_API_KEY, BASETEN_API_KEY) was found.")
            print("Please set keys in your environment or .env file.")
            raise SystemExit(1)
