from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import logging
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Any

import requests

DEFAULT_ENDPOINT_BASE = "https://api.agentrouter.org/v1"
DEFAULT_MODEL = "deepseek-r1-0528"
DEFAULT_MAX_OUTPUT_TOKENS = 8192
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120
DEFAULT_RETRY_DELAYS: tuple[int, ...] = (3, 6, 12, 24)
DEFAULT_INTER_REQUEST_DELAY_SECONDS = 1.5
DEFAULT_MAX_PROMPT_CHARS = 100_000
DEFAULT_CACHE_MAX_SIZE = 256

_LOG = logging.getLogger(__name__)


def _read_env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name, "")
        if value.strip():
            return value.strip()
    return default


def parse_int_env(*names: str, default: int, minimum: int | None = None) -> int:
    raw = _read_env(*names, default="")
    if not raw:
        value = default
    else:
        try:
            value = int(raw)
        except ValueError:
            _LOG.warning("Invalid integer for %s=%r. Using default=%d", "/".join(names), raw, default)
            value = default
    if minimum is not None:
        return max(minimum, value)
    return value


def parse_float_env(*names: str, default: float, minimum: float | None = None) -> float:
    raw = _read_env(*names, default="")
    if not raw:
        value = default
    else:
        try:
            value = float(raw)
        except ValueError:
            _LOG.warning("Invalid float for %s=%r. Using default=%s", "/".join(names), raw, default)
            value = default
    if minimum is not None:
        return max(minimum, value)
    return value


def parse_retry_delays_from_env(*names: str, default: tuple[int, ...]) -> tuple[float, ...]:
    raw = _read_env(*names, default="")
    if not raw:
        return tuple(float(item) for item in default)

    parsed: list[float] = []
    for token in raw.split(","):
        item = token.strip()
        if not item:
            continue
        try:
            value = float(item)
        except ValueError:
            _LOG.warning("Ignoring invalid retry delay token %r from %s", item, "/".join(names))
            continue
        if value > 0:
            parsed.append(value)

    if not parsed:
        return tuple(float(item) for item in default)
    return tuple(parsed)


def _dedupe_urls(urls: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for item in urls:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


class InvalidAPIKeyError(RuntimeError):
    """Raised when API key is missing or unauthorized (401)."""


class LLMRequestError(RuntimeError):
    """Base class for HTTP request failures."""


class ProviderQuotaExhaustedError(RuntimeError):
    """Raised when provider returns repeated 429 errors."""


class LLMNetworkError(LLMRequestError):
    """Raised on network failures after retries."""


class LLMTimeoutError(LLMNetworkError):
    """Raised on network timeout failures after retries."""


class LLMConnectionError(LLMNetworkError):
    """Raised on network connection failures after retries."""


class LLMHTTPError(LLMRequestError):
    """Raised when remote endpoint returns an HTTP error status."""


@dataclass(frozen=True)
class AgentRouterConfig:
    """Validated runtime configuration for AgentRouterClient.

    Environment variables (preferred first):
    - API key: API_KEY
    - Endpoint: LLM_ENDPOINT_BASE, OPENAI_ENDPOINT_BASE
    - Model: LLM_MODEL, OPENAI_MODEL, OPENAI_MODELS (first entry)
    - Prompt limit chars: LLM_MAX_PROMPT_CHARS, OPENAI_MAX_PROMPT_CHARS
      - Use 0 to disable prompt truncation.
    - Prompt truncation side: LLM_PROMPT_TRUNCATE_SIDE (tail|head)
    - Cache size: LLM_CACHE_MAX_SIZE, OPENAI_CACHE_MAX_SIZE
    - Timeouts/tokens/retries:
      - LLM_MAX_OUTPUT_TOKENS, OPENAI_MAX_TOKENS
      - LLM_REQUEST_TIMEOUT_SECONDS, OPENAI_REQUEST_TIMEOUT_SECONDS
      - LLM_RETRY_DELAYS
      - LLM_INTER_REQUEST_DELAY_SECONDS, OPENAI_INTER_REQUEST_DELAY_SECONDS
    """

    api_key: str
    endpoint_base: str
    model: str
    max_output_tokens: int
    request_timeout_seconds: int
    retry_delays: tuple[float, ...]
    inter_request_delay_seconds: float
    max_prompt_chars: int
    prompt_truncate_side: str
    cache_max_size: int

    @classmethod
    def from_env(cls) -> "AgentRouterConfig":
        api_key = os.environ.get("API_KEY", "").strip()

        endpoint_base = _read_env("LLM_ENDPOINT_BASE", "OPENAI_ENDPOINT_BASE", default=DEFAULT_ENDPOINT_BASE)
        model_csv = _read_env("OPENAI_MODELS", default="")
        model = (
            _read_env("LLM_MODEL", "OPENAI_MODEL", default="")
            or model_csv.split(",")[0].strip()
            or DEFAULT_MODEL
        )

        retry_delays = parse_retry_delays_from_env("LLM_RETRY_DELAYS", default=DEFAULT_RETRY_DELAYS)
        inter_delay = parse_float_env(
            "LLM_INTER_REQUEST_DELAY_SECONDS",
            "OPENAI_INTER_REQUEST_DELAY_SECONDS",
            default=DEFAULT_INTER_REQUEST_DELAY_SECONDS,
            minimum=0.0,
        )
        max_prompt_chars = parse_int_env(
            "LLM_MAX_PROMPT_CHARS",
            "OPENAI_MAX_PROMPT_CHARS",
            default=DEFAULT_MAX_PROMPT_CHARS,
            minimum=0,
        )
        prompt_truncate_side = _read_env("LLM_PROMPT_TRUNCATE_SIDE", default="tail").lower()
        if prompt_truncate_side not in {"tail", "head"}:
            _LOG.warning("Invalid LLM_PROMPT_TRUNCATE_SIDE=%r. Using 'tail'.", prompt_truncate_side)
            prompt_truncate_side = "tail"

        cache_max_size = parse_int_env(
            "LLM_CACHE_MAX_SIZE",
            "OPENAI_CACHE_MAX_SIZE",
            default=DEFAULT_CACHE_MAX_SIZE,
            minimum=0,
        )

        return cls(
            api_key=api_key,
            endpoint_base=endpoint_base,
            model=model,
            max_output_tokens=parse_int_env(
                "LLM_MAX_OUTPUT_TOKENS",
                "OPENAI_MAX_TOKENS",
                default=DEFAULT_MAX_OUTPUT_TOKENS,
                minimum=1,
            ),
            request_timeout_seconds=parse_int_env(
                "LLM_REQUEST_TIMEOUT_SECONDS",
                "OPENAI_REQUEST_TIMEOUT_SECONDS",
                default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
                minimum=1,
            ),
            retry_delays=tuple(max(1.0, float(item)) for item in retry_delays),
            inter_request_delay_seconds=inter_delay,
            max_prompt_chars=max_prompt_chars,
            prompt_truncate_side=prompt_truncate_side,
            cache_max_size=cache_max_size,
        )


class AgentRouterClient:
    """Reusable OpenAI-compatible client for Agent Router style endpoints."""

    def __init__(self, config: AgentRouterConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._last_request_time = 0.0
        self._cache_lock = threading.Lock()
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._cache_max_size = config.cache_max_size
        self._inflight: dict[str, threading.Event] = {}
        self._inflight_lock = threading.Lock()

        self._chat_endpoints = self._build_chat_completion_urls()
        self._model_endpoints = self._build_model_list_urls()
        self._preferred_chat_endpoint: str | None = None
        self._preferred_model_endpoint: str | None = None

        self._session = requests.Session()
        self._session.headers.update(self._build_headers())

    @classmethod
    def from_env(cls) -> "AgentRouterClient":
        return cls(AgentRouterConfig.from_env())

    @property
    def model(self) -> str:
        return self.config.model

    def _build_chat_completion_urls(self) -> list[str]:
        base = self.config.endpoint_base.rstrip("/")
        if base.endswith("/v1"):
            root = base[: -len("/v1")]
            return _dedupe_urls(
                [
                    f"{base}/chat/completions",
                    f"{root}/chat/completions",
                ]
            )
        return _dedupe_urls(
            [
                f"{base}/v1/chat/completions",
                f"{base}/chat/completions",
            ]
        )

    def _build_model_list_urls(self) -> list[str]:
        base = self.config.endpoint_base.rstrip("/")
        if base.endswith("/v1"):
            root = base[: -len("/v1")]
            return _dedupe_urls(
                [
                    f"{base}/models",
                    f"{root}/models",
                ]
            )
        return _dedupe_urls(
            [
                f"{base}/v1/models",
                f"{base}/models",
            ]
        )

    def chat_completion_urls(self) -> list[str]:
        return list(self._chat_endpoints)

    def model_list_urls(self) -> list[str]:
        return list(self._model_endpoints)

    def endpoint_for_chat(self) -> str:
        return self._preferred_chat_endpoint or self._chat_endpoints[0]

    def endpoint_for_models(self) -> str:
        return self._preferred_model_endpoint or self._model_endpoints[0]

    def _cache_key(self, model_name: str, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
        source = json.dumps(
            {
                "model": model_name,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha256(source.encode("utf-8")).hexdigest()

    def _wait_min_interval(self) -> None:
        elapsed = time.time() - self._last_request_time
        remaining = self.config.inter_request_delay_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is None:
                return None
            self._cache.move_to_end(key)
            return dict(cached)

    def _cache_set(self, key: str, value: dict[str, Any]) -> None:
        if self._cache_max_size <= 0:
            return
        with self._cache_lock:
            self._cache[key] = dict(value)
            self._cache.move_to_end(key)
            if len(self._cache) > self._cache_max_size:
                self._cache.popitem(last=False)

    def _wait_for_inflight(self, key: str) -> bool:
        with self._inflight_lock:
            existing = self._inflight.get(key)
            if existing is None:
                self._inflight[key] = threading.Event()
                return False
        existing.wait()
        return True

    def _finish_inflight(self, key: str) -> None:
        with self._inflight_lock:
            event = self._inflight.pop(key, None)
        if event is not None:
            event.set()

    def _ordered_endpoints(self, preferred: str | None, candidates: list[str]) -> list[str]:
        ordered: list[str] = []
        if preferred and preferred in candidates:
            ordered.append(preferred)
        for endpoint in candidates:
            if endpoint not in ordered:
                ordered.append(endpoint)
        return ordered

    def _request_json(
        self,
        method: str,
        endpoint_url: str,
        *,
        timeout: tuple[int, int] | int,
        json_payload: dict[str, Any] | None = None,
    ) -> requests.Response:
        try:
            response = self._session.request(method=method, url=endpoint_url, json=json_payload, timeout=timeout)
            return response
        except requests.Timeout as exc:
            raise LLMTimeoutError(f"Request timed out for endpoint={endpoint_url}") from exc
        except requests.ConnectionError as exc:
            raise LLMConnectionError(f"Connection failed for endpoint={endpoint_url}") from exc
        except requests.RequestException as exc:
            raise LLMNetworkError(f"Request failed for endpoint={endpoint_url}: {exc}") from exc

    def _optimized_prompt(self, text: str) -> str:
        clean_text = (text or "").strip()
        limit = self.config.max_prompt_chars
        if limit <= 0:
            return clean_text
        if len(clean_text) <= limit:
            return clean_text

        _LOG.warning(
            "Prompt truncated from %d to %d chars (side=%s). Set LLM_MAX_PROMPT_CHARS=0 to disable.",
            len(clean_text),
            limit,
            self.config.prompt_truncate_side,
        )
        if self.config.prompt_truncate_side == "head":
            return clean_text[:limit]
        return clean_text[-limit:]

    def _build_payload(
        self,
        model_name: str,
        optimized_system_prompt: str,
        optimized_user_prompt: str,
        temperature: float,
        effective_max_tokens: int,
        extra_params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": optimized_system_prompt},
                {"role": "user", "content": optimized_user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": effective_max_tokens,
        }
        if extra_params:
            for key, value in extra_params.items():
                if key in {"model", "messages"}:
                    _LOG.warning("Ignoring reserved payload key from extra params: %s", key)
                    continue
                payload[key] = value
        return payload

    def _resolve_chat_endpoint(
        self,
        payload: dict[str, Any],
        timeout: tuple[int, int],
    ) -> tuple[str, requests.Response]:
        errors: list[str] = []
        first_error: Exception | None = None
        for endpoint_url in self._ordered_endpoints(self._preferred_chat_endpoint, self._chat_endpoints):
            try:
                response = self._request_json("POST", endpoint_url, timeout=timeout, json_payload=payload)
            except LLMNetworkError as exc:
                errors.append(str(exc))
                if first_error is None:
                    first_error = exc
                continue
            if response.status_code in {404, 405}:
                continue
            self._preferred_chat_endpoint = endpoint_url
            return endpoint_url, response

        if errors:
            message = "Network failure across endpoint candidates: " + " | ".join(errors)
            root_cause = getattr(first_error, "__cause__", None) or first_error
            if isinstance(first_error, LLMTimeoutError):
                raise LLMTimeoutError(message) from root_cause
            if isinstance(first_error, LLMConnectionError):
                raise LLMConnectionError(message) from root_cause
            raise LLMNetworkError(message) from root_cause
        raise RuntimeError("No compatible chat/completions path found (tried /v1/chat/completions and /chat/completions)")

    def _resolve_model_endpoint(self, timeout: int) -> tuple[str, requests.Response]:
        errors: list[str] = []
        first_error: Exception | None = None
        for endpoint_url in self._ordered_endpoints(self._preferred_model_endpoint, self._model_endpoints):
            try:
                response = self._request_json("GET", endpoint_url, timeout=timeout)
            except LLMNetworkError as exc:
                errors.append(str(exc))
                if first_error is None:
                    first_error = exc
                continue
            if response.status_code in {404, 405}:
                continue
            self._preferred_model_endpoint = endpoint_url
            return endpoint_url, response

        if errors:
            message = "Could not reach endpoint: " + " | ".join(errors)
            root_cause = getattr(first_error, "__cause__", None) or first_error
            if isinstance(first_error, LLMTimeoutError):
                raise LLMTimeoutError(message) from root_cause
            if isinstance(first_error, LLMConnectionError):
                raise LLMConnectionError(message) from root_cause
            raise LLMNetworkError(message) from root_cause
        raise RuntimeError("No compatible models endpoint found (tried /v1/models and /models).")

    def chat_completion(
        self,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        timeout_seconds: int | None = None,
        max_tokens: int | None = None,
        use_cache: bool = True,
        extra_params: dict[str, Any] | None = None,
        **model_params: Any,
    ) -> dict[str, Any]:
        if not self.config.api_key:
            raise InvalidAPIKeyError("API_KEY is empty. Set API_KEY in .env")

        timeout = int(timeout_seconds or self.config.request_timeout_seconds)
        effective_max_tokens = int(max_tokens or self.config.max_output_tokens)
        optimized_user_prompt = self._optimized_prompt(user_prompt)
        optimized_system_prompt = (system_prompt or "").strip()

        key = self._cache_key(
            model_name=model_name,
            system_prompt=optimized_system_prompt,
            user_prompt=optimized_user_prompt,
            temperature=temperature,
            max_tokens=effective_max_tokens,
        )

        merged_extra_params = dict(extra_params or {})
        merged_extra_params.update(model_params)

        if use_cache:
            cached = self._cache_get(key)
            if isinstance(cached, dict):
                _LOG.debug("Cache hit for model=%s", model_name)
                return cached
            if self._wait_for_inflight(key):
                # Another thread completed the same key; return cached if present.
                cached = self._cache_get(key)
                if isinstance(cached, dict):
                    return cached

        payload = self._build_payload(
            model_name=model_name,
            optimized_system_prompt=optimized_system_prompt,
            optimized_user_prompt=optimized_user_prompt,
            temperature=temperature,
            effective_max_tokens=effective_max_tokens,
            extra_params=merged_extra_params,
        )

        try:
            with self._lock:
                for attempt, delay in enumerate(self.config.retry_delays, start=1):
                    self._wait_min_interval()
                    self._last_request_time = time.time()

                    endpoint_url, response = self._resolve_chat_endpoint(payload=payload, timeout=(10, max(timeout, 30)))

                    if response.status_code == 200:
                        try:
                            data = response.json()
                        except ValueError as exc:
                            raise LLMHTTPError("LLM response JSON decoding failed") from exc
                        if use_cache:
                            self._cache_set(key, data)
                        _LOG.info("LLM request succeeded: model=%s endpoint=%s", model_name, endpoint_url)
                        return data

                    if response.status_code == 401:
                        raise InvalidAPIKeyError("401 Unauthorized: invalid API key for configured endpoint")

                    if response.status_code == 429:
                        if attempt < len(self.config.retry_delays):
                            base_seconds = parse_int_env("LLM_429_BASE_DELAY_SECONDS", "OPENAI_429_BASE_DELAY_SECONDS", default=60, minimum=1)
                            max_seconds = parse_int_env("LLM_429_MAX_DELAY_SECONDS", "OPENAI_429_MAX_DELAY_SECONDS", default=900, minimum=1)
                            exp_seconds = max(1, int(base_seconds)) * (2 ** max(0, attempt - 1))
                            jitter = random.uniform(0.0, 10.0)
                            wait_seconds = min(float(max_seconds), float(exp_seconds) + jitter)
                            _LOG.warning(
                                "Rate limited (429) for model=%s attempt=%d/%d; retrying in %.2fs",
                                model_name,
                                attempt,
                                len(self.config.retry_delays),
                                wait_seconds,
                            )
                            time.sleep(wait_seconds)
                            continue
                        raise ProviderQuotaExhaustedError(f"PROVIDER_QUOTA_EXHAUSTED::{model_name}::429")

                    if attempt < len(self.config.retry_delays) and response.status_code in {500, 502, 503, 504}:
                        retry_sleep = max(0.5, float(delay))
                        _LOG.warning(
                            "Transient HTTP error status=%d for model=%s attempt=%d/%d; retrying in %.2fs",
                            response.status_code,
                            model_name,
                            attempt,
                            len(self.config.retry_delays),
                            retry_sleep,
                        )
                        time.sleep(retry_sleep)
                        continue

                    raise LLMHTTPError(
                        f"LLM request failed: status={response.status_code}, body={response.text}"
                    )
        finally:
            if use_cache:
                self._finish_inflight(key)

        raise LLMRequestError("LLM request failed unexpectedly")

    def check_health(self) -> tuple[bool, str]:
        if not self.config.api_key:
            return False, "API_KEY is empty. Set API_KEY."

        try:
            _endpoint, response = self._resolve_model_endpoint(timeout=10)
        except LLMNetworkError as exc:
            return False, str(exc)
        except RuntimeError as exc:
            return False, str(exc)

        if response.status_code == 401:
            return False, "Endpoint returned 401. Invalid API_KEY for configured endpoint."
        if response.status_code != 200:
            return False, f"Endpoint returned status {response.status_code}."

        probe_payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": "Respond with OK."},
                {"role": "user", "content": "health-check"},
            ],
            "temperature": 0.0,
            "max_tokens": 1,
        }
        try:
            endpoint_url, probe = self._resolve_chat_endpoint(payload=probe_payload, timeout=(10, 10))
        except LLMNetworkError as exc:
            return False, "Health probe failed: " + str(exc)
        except RuntimeError as exc:
            return False, "Health probe failed: " + str(exc)

        _LOG.info("Health probe used endpoint=%s model=%s", endpoint_url, self.config.model)

        if probe.status_code == 401:
            return False, "Health probe returned 401. Invalid API_KEY for configured endpoint."
        if probe.status_code != 200:
            return False, f"Health probe failed with status {probe.status_code}."
        return True, f"CONNECTION + MODEL PROBE VERIFIED ({self.config.model})"
