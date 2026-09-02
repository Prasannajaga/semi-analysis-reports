"""OpenRouter metadata, pricing snapshots, and endpoint preflight."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from benchmark_tool.config import Gateway


def provider_routing(slug: str, gateway: Gateway | None = None) -> dict[str, Any]:
    routing = gateway.routing if gateway else None
    return {
        "only": [slug],
        "allow_fallbacks": routing.allow_fallbacks if routing else False,
        "require_parameters": routing.require_parameters if routing else True,
    }


def routed_request_body(model: str, provider_slug: str, **payload: Any) -> dict[str, Any]:
    """Build the common top-level body used by all runners and preflight."""

    return {"model": model, **payload, "provider": provider_routing(provider_slug)}


@dataclass(frozen=True)
class OpenRouterClient:
    base_url: str
    api_key: str
    timeout_seconds: float

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Metadata": "enabled",
        }

    def endpoint_metadata(self, model: str) -> dict[str, Any]:
        encoded = quote(model, safe="/")
        with httpx.Client(timeout=self.timeout_seconds, headers=self.headers) as client:
            response = client.get(f"{self.base_url.rstrip('/')}/models/{encoded}/endpoints")
            response.raise_for_status()
            value = response.json()
        if not isinstance(value, dict):
            raise ValueError("OpenRouter endpoint metadata was not a JSON object")
        return value

    def generation_metadata(self, generation_id: str) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout_seconds, headers=self.headers) as client:
            response = client.get(
                f"{self.base_url.rstrip('/')}/generation", params={"id": generation_id}
            )
            response.raise_for_status()
            value = response.json()
        if not isinstance(value, dict):
            raise ValueError("OpenRouter generation metadata was not a JSON object")
        return value

    def preflight(
        self,
        model: str,
        provider_slug: str,
        *,
        require_agentx: bool = False,
        require_streaming: bool = False,
        require_tools: bool = False,
        require_seed: bool = False,
    ) -> dict[str, Any]:
        streaming = require_agentx or require_streaming
        body = routed_request_body(
            model,
            provider_slug,
            messages=[{"role": "user", "content": "Reply with OK."}],
            max_tokens=1,
            temperature=0,
            stream=streaming,
        )
        if require_agentx:
            body["ignore_eos"] = True
        if require_seed:
            body["seed"] = 1
        if require_tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": "preflight_noop",
                        "description": "Compatibility probe",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
            body["tool_choice"] = "auto"
        result: dict[str, Any] = {
            "status": "failed",
            "checkedAt": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "model": model,
            "provider": provider_slug,
            "requestBody": body,
        }
        try:
            with httpx.Client(timeout=self.timeout_seconds, headers=self.headers) as client:
                response = client.post(f"{self.base_url.rstrip('/')}/chat/completions", json=body)
            result["httpStatus"] = response.status_code
            if response.is_error:
                detail = _error_details(response, self.api_key)
                result["status"] = "unsupported" if response.status_code in {400, 404, 422} else "failed"
                result["reason"] = _failure_reason(response.status_code, detail)
                result["failureCategory"] = _failure_category(response.status_code)
                result["retryable"] = response.status_code in {408, 409, 429, 502, 503, 504, 529}
                result["error"] = detail
                return result
            data: dict[str, Any] = {}
            if streaming:
                chunks = [line[5:].strip() for line in response.text.splitlines() if line.startswith("data:")]
                stream_error: dict[str, Any] | None = None
                for chunk in chunks:
                    if chunk == "[DONE]":
                        continue
                    try:
                        decoded = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(decoded, dict) and decoded.get("error"):
                        stream_error = _error_details_from_payload(
                            decoded,
                            response,
                            self.api_key,
                        )
                        break
                    if isinstance(decoded, dict):
                        data.update({key: value for key, value in decoded.items() if value is not None})
                if stream_error is not None:
                    error_code = _integer_code(stream_error.get("code")) or 502
                    result["status"] = "failed"
                    result["reason"] = _failure_reason(error_code, stream_error)
                    result["failureCategory"] = _failure_category(error_code)
                    result["retryable"] = error_code in {408, 409, 429, 502, 503, 504, 529}
                    result["error"] = stream_error
                    return result
                if not chunks or "[DONE]" not in chunks:
                    result["status"] = "unsupported"
                    result["reason"] = "endpoint did not return a valid streaming SSE response"
                    return result
                if not data:
                    result["status"] = "unsupported"
                    result["reason"] = "stream contained no valid JSON completion chunks"
                    return result
            else:
                decoded = response.json()
                if isinstance(decoded, dict) and decoded.get("error"):
                    detail = _error_details_from_payload(decoded, response, self.api_key)
                    error_code = _integer_code(detail.get("code")) or 502
                    result["status"] = "failed"
                    result["reason"] = _failure_reason(error_code, detail)
                    result["failureCategory"] = _failure_category(error_code)
                    result["retryable"] = error_code in {408, 409, 429, 502, 503, 504, 529}
                    result["error"] = detail
                    return result
                if not isinstance(decoded, dict) or not isinstance(decoded.get("choices"), list):
                    result["status"] = "unsupported"
                    result["reason"] = "endpoint response was not a chat completion object"
                    return result
                data = decoded
            generation_id = str(data.get("id") or response.headers.get("x-generation-id") or "")
            exposed = _exposed_provider(data, response.headers)
            if not exposed and generation_id:
                try:
                    route_metadata = self.generation_metadata(generation_id)
                    exposed = _exposed_provider(route_metadata, httpx.Headers())
                    route_data = route_metadata.get("data", route_metadata)
                    if isinstance(route_data, dict):
                        result["routeMetadata"] = {
                            key: route_data[key]
                            for key in (
                                "id",
                                "model",
                                "provider_name",
                                "streamed",
                                "total_cost",
                            )
                            if key in route_data
                        }
                except (httpx.HTTPError, ValueError, json.JSONDecodeError) as error:
                    result["routeLookupReason"] = f"{type(error).__name__}: {error}"
            result["exposedProvider"] = exposed
            if exposed and not _provider_matches(exposed, provider_slug):
                result["status"] = "unsupported"
                result["reason"] = f"OpenRouter exposed route {exposed!r}, expected {provider_slug!r}"
            else:
                result["status"] = "supported"
                result["routingVerified"] = bool(exposed)
                if not exposed:
                    result["reason"] = "routing accepted; response did not expose the selected provider"
            result["responseId"] = generation_id or None
            return result
        except httpx.TimeoutException as error:
            result["failureCategory"] = "timeout"
            result["retryable"] = True
            result["reason"] = (
                f"OpenRouter preflight timed out after {self.timeout_seconds:g}s: "
                f"{type(error).__name__}"
            )
            return result
        except httpx.ConnectError as error:
            result["failureCategory"] = "connection"
            result["retryable"] = True
            result["reason"] = f"Could not connect to OpenRouter: {type(error).__name__}: {error}"
            return result
        except (httpx.HTTPError, ValueError, json.JSONDecodeError) as error:
            result["failureCategory"] = "client"
            result["retryable"] = False
            result["reason"] = f"{type(error).__name__}: {error}"
            return result


def _error_details(response: httpx.Response, secret: str) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        payload = {"message": response.text[:2000] or response.reason_phrase}
    return _error_details_from_payload(payload, response, secret)


def _error_details_from_payload(
    payload: Any,
    response: httpx.Response,
    secret: str,
) -> dict[str, Any]:
    envelope = payload if isinstance(payload, dict) else {"message": str(payload)}
    raw_error = envelope.get("error", envelope)
    error = raw_error if isinstance(raw_error, dict) else {"message": str(raw_error)}
    code = _integer_code(error.get("code")) or response.status_code
    details: dict[str, Any] = {
        "code": code,
        "message": str(error.get("message") or response.reason_phrase or "request failed"),
    }
    metadata = error.get("metadata")
    if isinstance(metadata, dict) and metadata:
        details["providerMetadata"] = metadata
    router_metadata = envelope.get("openrouter_metadata")
    if isinstance(router_metadata, dict) and router_metadata:
        details["openrouterMetadata"] = router_metadata
    safe_headers = {
        name: response.headers[name]
        for name in (
            "retry-after",
            "x-generation-id",
            "x-ratelimit-limit",
            "x-ratelimit-remaining",
            "x-ratelimit-reset",
        )
        if response.headers.get(name)
    }
    if safe_headers:
        details["responseHeaders"] = safe_headers
    return _redact_value(details, secret)


def _integer_code(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _failure_category(status_code: int) -> str:
    if status_code == 401:
        return "authentication"
    if status_code == 402:
        return "credits"
    if status_code == 403:
        return "permission"
    if status_code == 404:
        return "routing"
    if status_code == 408:
        return "timeout"
    if status_code == 429:
        return "rate_limit"
    if status_code in {502, 503, 504, 529}:
        return "provider_availability"
    if 400 <= status_code < 500:
        return "request"
    if status_code >= 500:
        return "server"
    return "response"


def _failure_reason(status_code: int, details: dict[str, Any]) -> str:
    try:
        phrase = HTTPStatus(status_code).phrase
    except ValueError:
        phrase = "Request Failed"
    message = " ".join(str(details.get("message") or "request failed").split())
    reason = f"HTTP {status_code} {phrase}: {message}"
    metadata = details.get("providerMetadata")
    if isinstance(metadata, dict):
        provider = metadata.get("provider_name") or metadata.get("provider")
        raw = metadata.get("raw")
        if provider:
            reason += f" [provider: {provider}]"
        if raw and str(raw) != message:
            compact_raw = " ".join(str(raw).split())[:500]
            reason += f" [upstream: {compact_raw}]"
    if status_code == 429:
        reason += " The pinned route is rate-limited; retry later or check provider/account limits."
    elif status_code == 402:
        reason += " Check OpenRouter credits and API-key spending limits."
    elif status_code == 401:
        reason += " Check the configured OpenRouter API key."
    elif status_code in {502, 503, 504, 529}:
        reason += " The pinned provider is temporarily unavailable or overloaded."
    retry_after = details.get("responseHeaders", {}).get("retry-after")
    if retry_after:
        reason += f" Retry-After: {retry_after}."
    return reason


def _redact_value(value: Any, secret: str) -> Any:
    if isinstance(value, dict):
        return {key: _redact_value(item, secret) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_value(item, secret) for item in value]
    if isinstance(value, str) and secret:
        return value.replace(secret, "[REDACTED]")
    return value


def _exposed_provider(data: Any, headers: httpx.Headers) -> str | None:
    if isinstance(data, dict):
        if isinstance(data.get("data"), dict):
            nested = _exposed_provider(data["data"], headers)
            if nested:
                return nested
        for key in ("provider", "provider_name"):
            if isinstance(data.get(key), str):
                return data[key]
    for key in ("x-openrouter-provider", "x-provider"):
        if headers.get(key):
            return headers[key]
    return None


def _provider_matches(exposed: str, expected: str) -> bool:
    normalize = lambda value: "".join(character for character in value.lower() if character.isalnum())
    return normalize(exposed) == normalize(expected)


def pricing_snapshot(metadata: dict[str, Any], model: str, provider_slug: str) -> dict[str, Any]:
    data = metadata.get("data", metadata)
    endpoints = data.get("endpoints", []) if isinstance(data, dict) else []
    matches = []
    for endpoint in endpoints if isinstance(endpoints, list) else []:
        if not isinstance(endpoint, dict):
            continue
        exposed = str(endpoint.get("provider_name") or endpoint.get("provider") or endpoint.get("name") or "")
        if _provider_matches(exposed, provider_slug) or _provider_matches(str(endpoint.get("tag", "")), provider_slug):
            matches.append(endpoint)
    return {
        "schemaVersion": "1.0",
        "model": model,
        "provider": provider_slug,
        "matchingEndpoints": matches,
        "allEndpoints": endpoints,
    }
