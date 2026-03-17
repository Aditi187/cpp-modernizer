from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence
import importlib
import os
import threading
import time

import requests

try:
    _langfuse_module = importlib.import_module("langfuse")
    Langfuse = getattr(_langfuse_module, "Langfuse", None)
except Exception:  # pragma: no cover - optional dependency at runtime
    Langfuse = None


# Fill this in directly only if you are not using environment variables.
# OPENROUTER_API_KEY from the environment takes precedence when set.
api_key: str = ""

DEFAULT_OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OPENROUTER_HTTP_REFERER = "http://localhost"
DEFAULT_OPENROUTER_X_TITLE = "Air-Gapped Codebase Modernization Engine"
DEFAULT_OPENROUTER_MODELS: tuple[str, ...] = (
    "anthropic/claude-3.5-sonnet",
    "openai/gpt-4o",
)
DEFAULT_RETRY_DELAYS: tuple[int, ...] = (1, 2, 4, 8, 16)
DEFAULT_MAX_TOKENS = 8192
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120
DEFAULT_SHORT_CODE_RESPONSE_THRESHOLD = 500
DEFAULT_HEALTH_PROBE_TIMEOUT_SECONDS = 10

CPP_MODERNIZATION_SYSTEM_PROMPT = (
    "You are a Master C++20 Developer. "
    "You have access to MCP tools. "
    "When modernizing C++ code, you MUST provide the ENTIRE file content in the "
    "`write_code` tool call. "
    "If asked for a whole-file rewrite, return the entire file. "
    "If asked for a single-function rewrite, return the entire function. "
    "Never truncate or use placeholders like // ...rest of code. "
    "Never summarize omitted sections. "
    "Never use JavaScript keywords like 'function'. "
    "Always use proper C++ syntax and preserve behavior exactly."
)

FULL_RESPONSE_REMINDER = (
    "CRITICAL OUTPUT REQUIREMENT: Return the complete requested code artifact in one response. "
    "Do not omit unchanged sections. Do not summarize. Do not use placeholders."
)


def _parse_int_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


SHORT_CODE_RESPONSE_THRESHOLD = max(
    50,
    _parse_int_env("OPENROUTER_SHORT_RESPONSE_THRESHOLD", DEFAULT_SHORT_CODE_RESPONSE_THRESHOLD),
)
ENABLE_HEALTH_PROBE = os.environ.get("OPENROUTER_HEALTH_PROBE", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _dedupe_models(models: Sequence[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for model in models:
        normalized = model.strip()
        if not normalized or normalized in seen:
            continue
        ordered.append(normalized)
        seen.add(normalized)
    return ordered


def _get_env_models() -> list[str]:
    configured = os.environ.get("OPENROUTER_MODELS", "").strip()
    if configured:
        candidates = [value for value in configured.split(",")]
    else:
        candidates = list(DEFAULT_OPENROUTER_MODELS)

    return _dedupe_models([*candidates, *DEFAULT_OPENROUTER_MODELS])


def _looks_like_model_unavailable(status_code: int, response_text: str) -> bool:
    lower_error = response_text.lower()
    return status_code in {400, 404} and "model" in lower_error and (
        "not found" in lower_error
        or "unavailable" in lower_error
        or "no endpoints" in lower_error
    )


def _expects_large_code_response(user_prompt: str) -> bool:
    lower_prompt = user_prompt.lower()
    triggers = (
        "entire file",
        "full file",
        "whole file",
        "full updated code",
        "single function",
        "```cpp",
        "write_code",
        "modernize",
    )
    return any(trigger in lower_prompt for trigger in triggers)


@dataclass(frozen=True)
class OpenRouterConfig:
    api_key: str
    endpoint: str
    http_referer: str
    x_title: str
    models: tuple[str, ...]
    max_tokens: int
    request_timeout_seconds: int
    retry_delays: tuple[int, ...]

    @classmethod
    def from_env(cls) -> "OpenRouterConfig":
        resolved_api_key = os.environ.get("OPENROUTER_API_KEY", "").strip() or api_key
        return cls(
            api_key=resolved_api_key,
            endpoint=os.environ.get("OPENROUTER_ENDPOINT", DEFAULT_OPENROUTER_ENDPOINT).strip()
            or DEFAULT_OPENROUTER_ENDPOINT,
            http_referer=os.environ.get(
                "OPENROUTER_HTTP_REFERER", DEFAULT_OPENROUTER_HTTP_REFERER
            ).strip()
            or DEFAULT_OPENROUTER_HTTP_REFERER,
            x_title=os.environ.get("OPENROUTER_X_TITLE", DEFAULT_OPENROUTER_X_TITLE).strip()
            or DEFAULT_OPENROUTER_X_TITLE,
            models=tuple(_get_env_models()),
            max_tokens=_parse_int_env("OPENROUTER_MAX_TOKENS", DEFAULT_MAX_TOKENS),
            request_timeout_seconds=_parse_int_env(
                "OPENROUTER_REQUEST_TIMEOUT_SECONDS", DEFAULT_REQUEST_TIMEOUT_SECONDS
            ),
            retry_delays=DEFAULT_RETRY_DELAYS,
        )

    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.http_referer,
            "X-Title": self.x_title,
        }


class LangfuseTracker:
    def __init__(self, log_fn: Callable[[str], None] | None = None) -> None:
        self._log_fn = log_fn
        self._local = threading.local()
        self._client: Any = None

        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
        host = os.environ.get("LANGFUSE_HOST", "").strip()

        if not public_key or not secret_key or not host:
            self._log("LangFuse is disabled. Set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST.")
            return

        if Langfuse is None:
            self._log("LangFuse SDK is not installed. Install package 'langfuse' to enable tracing.")
            return

        try:
            self._client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
            )
            self._log("LangFuse initialized.")
        except Exception as exc:
            self._client = None
            self._log(f"LangFuse init failed: {exc!r}")

    def _log(self, message: str) -> None:
        if self._log_fn is not None:
            self._log_fn(message)

    def enabled(self) -> bool:
        return self._client is not None

    def _set_active_trace(self, trace: Any) -> None:
        self._local.active_trace = trace

    def get_active_trace(self) -> Any:
        return getattr(self._local, "active_trace", None)

    def _current_trace_context(self) -> Any:
        trace = self.get_active_trace()
        if isinstance(trace, dict) and "trace_id" in trace:
            return {"trace_id": trace["trace_id"]}
        return None

    def create_trace(self, name: str = "CPP-Modernization", input_payload: Any = None) -> Any:
        if not self.enabled():
            return None
        try:
            # Legacy SDK path
            if hasattr(self._client, "trace"):
                trace = self._client.trace(name=name, input=input_payload)
                self._set_active_trace(trace)
                return trace

            # New SDK path: create a trace id and attach observations/events to it.
            if hasattr(self._client, "create_trace_id"):
                trace_id = self._client.create_trace_id()
                trace_obj = {"trace_id": trace_id, "name": name}
                self._set_active_trace(trace_obj)

                if hasattr(self._client, "create_event"):
                    self._client.create_event(
                        trace_context={"trace_id": trace_id},
                        name=name,
                        input=input_payload,
                        metadata={"status": "started"},
                    )
                return trace_obj

            self._log("LangFuse client does not expose a trace creation API.")
            return None
        except Exception as exc:
            self._log(f"Failed to create LangFuse trace: {exc!r}")
            return None

    def get_or_create_trace(self, name: str = "CPP-Modernization", input_payload: Any = None) -> Any:
        trace = self.get_active_trace()
        if trace is not None:
            return trace
        return self.create_trace(name=name, input_payload=input_payload)

    def start_span(self, name: str, input_payload: Any = None) -> Any:
        trace = self.get_or_create_trace()
        if trace is None:
            return None
        try:
            if hasattr(trace, "span"):
                return trace.span(name=name, input=input_payload)

            if hasattr(self._client, "start_observation"):
                return self._client.start_observation(
                    trace_context=self._current_trace_context(),
                    name=name,
                    as_type="span",
                    input=input_payload,
                )
        except Exception as exc:
            self._log(f"Failed to start LangFuse span '{name}': {exc!r}")
        return None

    def end_span(
        self,
        span: Any,
        output_payload: Any = None,
        level: str | None = None,
        input_payload: Any = None,
    ) -> None:
        if span is None:
            return
        try:
            # New SDK commonly expects output/level via update(), then plain end().
            if hasattr(span, "update"):
                update_kwargs: dict[str, Any] = {}
                if input_payload is not None:
                    update_kwargs["input"] = input_payload
                if output_payload is not None:
                    update_kwargs["output"] = output_payload
                if level:
                    update_kwargs["level"] = level
                    if level.upper() == "ERROR":
                        update_kwargs["status_message"] = "Error"
                if update_kwargs:
                    span.update(**update_kwargs)

            if hasattr(span, "end"):
                try:
                    # Legacy SDK path.
                    kwargs: dict[str, Any] = {}
                    if input_payload is not None:
                        kwargs["input"] = input_payload
                    if output_payload is not None:
                        kwargs["output"] = output_payload
                    if level:
                        kwargs["level"] = level
                    span.end(**kwargs)
                except TypeError:
                    # New SDK path.
                    span.end()
        except Exception as exc:
            self._log(f"Failed to end LangFuse span: {exc!r}")

    def capture_generation(
        self,
        model: str,
        prompt: str,
        response_text: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        total_tokens: int | None,
        system_prompt: str,
    ) -> None:
        """Capture one LLM generation with explicit input/output and usage details."""
        trace = self.get_or_create_trace()
        if trace is None:
            return

        metadata = {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        usage_details = {
            "prompt_tokens": int(prompt_tokens or 0),
            "completion_tokens": int(completion_tokens or 0),
            "total_tokens": int(total_tokens or 0),
        }
        generation_input = {"system_prompt": system_prompt, "user_prompt": prompt}

        try:
            if hasattr(trace, "generation"):
                generation = trace.generation(
                    name="OpenRouter-Completion",
                    model=model,
                    input=generation_input,
                    metadata=metadata,
                )
                if hasattr(generation, "end"):
                    generation.end(
                        output=response_text,
                        model=model,
                        usage_details=usage_details,
                        metadata=metadata,
                    )
                elif hasattr(generation, "update"):
                    generation.update(
                        output=response_text,
                        model=model,
                        usage_details=usage_details,
                        metadata=metadata,
                    )
                    if hasattr(generation, "end"):
                        generation.end()
                else:
                    # Fallback for SDK variants where generation(...) records immediately.
                    trace.generation(
                        name="OpenRouter-Completion",
                        model=model,
                        input=generation_input,
                        output=response_text,
                        metadata=metadata,
                    )
                return

            if hasattr(self._client, "start_observation"):
                generation = self._client.start_observation(
                    trace_context=self._current_trace_context(),
                    name="OpenRouter-Completion",
                    as_type="generation",
                    input=generation_input,
                    model=model,
                    usage_details=usage_details,
                    metadata=metadata,
                )
                if hasattr(generation, "update"):
                    generation.update(
                        output=response_text,
                        model=model,
                        usage_details=usage_details,
                        metadata=metadata,
                    )
                if hasattr(generation, "end"):
                    generation.end()
        except Exception as exc:
            self._log(f"Failed to capture LangFuse generation: {exc!r}")

    def mark_error(self, message: str, details: Any = None) -> None:
        trace = self.get_active_trace()
        if trace is None:
            return

        payload = {"error": message, "details": details}

        try:
            if hasattr(trace, "update"):
                trace.update(output=payload, metadata={"status": "Error"})
                return

            if hasattr(self._client, "start_observation"):
                error_span = self._client.start_observation(
                    trace_context=self._current_trace_context(),
                    name="Trace-Error",
                    as_type="span",
                    input=details,
                    level="ERROR",
                    status_message=message,
                    metadata={"status": "Error"},
                )
                if hasattr(error_span, "update"):
                    error_span.update(
                        output=payload,
                        level="ERROR",
                        status_message=message,
                        metadata={"status": "Error"},
                    )
                if hasattr(error_span, "end"):
                    error_span.end()
        except Exception as exc:
            self._log(f"Failed to mark LangFuse trace error: {exc!r}")

    def flush(self) -> None:
        if not self.enabled():
            return
        try:
            if hasattr(self._client, "flush"):
                self._client.flush()
        except Exception as exc:
            self._log(f"LangFuse flush failed: {exc!r}")


class OpenRouterBridge:
    """OpenRouter chat-completion bridge with generation-first Langfuse tracing."""

    def __init__(
        self,
        config: OpenRouterConfig | None = None,
        log_fn: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config or OpenRouterConfig.from_env()
        self._log_fn = log_fn
        self.tracker = LangfuseTracker(log_fn=log_fn)

    @classmethod
    def from_env(
        cls,
        log_fn: Callable[[str], None] | None = None,
    ) -> "OpenRouterBridge":
        """Create a bridge from environment-configured OpenRouter settings."""
        return cls(OpenRouterConfig.from_env(), log_fn=log_fn)

    def _log(self, message: str) -> None:
        if self._log_fn is not None:
            self._log_fn(message)

    def start_modernization_trace(self, input_payload: Any = None) -> Any:
        """Start or attach to a modernization trace."""
        return self.tracker.create_trace(name="CPP-Modernization", input_payload=input_payload)

    def start_span(self, name: str, input_payload: Any = None) -> Any:
        """Start a generic span for non-LLM operations."""
        return self.tracker.start_span(name=name, input_payload=input_payload)

    def end_span(
        self,
        span: Any,
        output_payload: Any = None,
        level: str | None = None,
        input_payload: Any = None,
    ) -> None:
        """End a generic span for non-LLM operations."""
        self.tracker.end_span(
            span=span,
            output_payload=output_payload,
            level=level,
            input_payload=input_payload,
        )

    def mark_trace_error(self, message: str, details: Any = None) -> None:
        """Attach an error marker to the active trace."""
        self.tracker.mark_error(message=message, details=details)

    def _build_payload(
        self,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> dict[str, object]:
        """Build an OpenRouter chat-completions payload."""
        return {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": temperature,
        }

    def _request_model_with_retries(
        self,
        *,
        model_name: str,
        payload: dict[str, object],
    ) -> dict[str, Any]:
        """Run one model request with retry/backoff for transient failures."""
        last_exc: Exception | None = None
        last_error_text = ""
        for attempt, delay in enumerate(self.config.retry_delays, start=1):
            try:
                response = requests.post(
                    self.config.endpoint,
                    headers=self.config.headers(),
                    json=payload,
                    timeout=self.config.request_timeout_seconds,
                )
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                if attempt < len(self.config.retry_delays):
                    self._log(
                        f"Network error contacting OpenRouter for model '{model_name}': {exc}. "
                        f"Retrying in {delay}s... (attempt {attempt}/{len(self.config.retry_delays)})"
                    )
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    f"OpenRouter network error for model '{model_name}' after retries: {exc!r}"
                ) from exc

            if response.status_code == 200:
                return response.json()

            last_error_text = response.text
            is_rate_limit = response.status_code == 429
            is_server_error = 500 <= response.status_code < 600
            if is_rate_limit or is_server_error:
                if attempt < len(self.config.retry_delays):
                    error_kind = "rate limited (429)" if is_rate_limit else f"server error {response.status_code}"
                    wait_seconds = 5 * attempt if is_rate_limit else delay
                    self._log(
                        f"OpenRouter {error_kind} for model '{model_name}'. "
                        f"Retrying in {wait_seconds}s... (attempt {attempt}/{len(self.config.retry_delays)})"
                    )
                    time.sleep(wait_seconds)
                    continue
                if is_rate_limit:
                    raise RuntimeError(f"PROVIDER_QUOTA_EXHAUSTED::{model_name}::{response.status_code}")
                raise RuntimeError(
                    f"OpenRouter server failure for model '{model_name}' after retries: "
                    f"{response.status_code} {response.text}"
                )

            if _looks_like_model_unavailable(response.status_code, response.text):
                raise RuntimeError(f"MODEL_UNAVAILABLE::{model_name}::{response.status_code}::{response.text}")

            raise RuntimeError(
                f"OpenRouter returned {response.status_code} for model '{model_name}': {response.text}"
            )

        raise RuntimeError(
            f"OpenRouter request failed for model '{model_name}'. "
            f"Last error: {last_exc!r}. Last response: {last_error_text}"
        )

    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        start_new_trace: bool = False,
    ) -> str:
        """Run a non-streaming chat completion with sequential model fallback.

        Tracing strategy: generation-first. No separate LLM span is created.
        """
        if not self.config.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is empty. Set it in your environment or .env file."
            )

        if start_new_trace:
            self.start_modernization_trace(
                input_payload={
                    "operation": "modernization",
                    "model_candidates": list(self.config.models),
                    "system_prompt": system_prompt,
                }
            )
        else:
            self.tracker.get_or_create_trace(name="CPP-Modernization")

        enforce_full_response = _expects_large_code_response(user_prompt)
        effective_user_prompt = user_prompt
        if enforce_full_response:
            effective_user_prompt = f"{user_prompt}\n\n{FULL_RESPONSE_REMINDER}"

        model_errors: list[str] = []

        for model_name in self.config.models:
            payload = self._build_payload(
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=effective_user_prompt,
                temperature=temperature,
            )
            try:
                data = self._request_model_with_retries(model_name=model_name, payload=payload)
                choice = (data.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                content = str(message.get("content") or "")
                finish_reason = str(choice.get("finish_reason") or "")

                usage = data.get("usage") or {}
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                total_tokens = usage.get("total_tokens")

                self.tracker.capture_generation(
                    model=model_name,
                    prompt=effective_user_prompt,
                    response_text=content,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    system_prompt=system_prompt,
                )

                if finish_reason == "length":
                    error_msg = (
                        f"{model_name}: finish_reason=length (response truncated by token limits). "
                        "Increase OPENROUTER_MAX_TOKENS."
                    )
                    model_errors.append(error_msg)
                    self._log(error_msg)
                    continue

                if enforce_full_response and len(content.strip()) < SHORT_CODE_RESPONSE_THRESHOLD:
                    error_msg = (
                        f"{model_name}: response below short threshold "
                        f"({len(content.strip())} < {SHORT_CODE_RESPONSE_THRESHOLD})."
                    )
                    model_errors.append(error_msg)
                    self._log(error_msg)
                    continue

                return content
            except RuntimeError as exc:
                message = str(exc)
                if message.startswith("PROVIDER_QUOTA_EXHAUSTED::"):
                    self.mark_trace_error(
                        message="PROVIDER_QUOTA_EXHAUSTED",
                        details={"model": model_name, "error": message},
                    )
                    raise RuntimeError("PROVIDER_QUOTA_EXHAUSTED") from exc

                model_errors.append(f"{model_name}: {message}")
                self._log(f"Model '{model_name}' failed: {message}. Trying next fallback model...")
                continue

        self.mark_trace_error(
            message="OPENROUTER_CALL_FAILED_ALL_MODELS",
            details={"models": list(self.config.models), "errors": model_errors},
        )
        raise RuntimeError(
            "OpenRouter call failed across all configured models. "
            f"Per-model errors: {model_errors}"
        )

    def check_health(self) -> tuple[bool, str]:
        """Verify API connectivity and optionally probe configured model inference."""
        if not self.config.api_key:
            return False, "OPENROUTER_API_KEY is empty. Set it in your environment or .env file."

        try:
            response = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers=self.config.headers(),
                timeout=10,
            )
        except requests.exceptions.RequestException as exc:
            return False, f"Could not reach OpenRouter: {exc}"

        if response.status_code == 200:
            if not ENABLE_HEALTH_PROBE:
                return True, "OPENROUTER CONNECTION VERIFIED"

            model_name = self.config.models[0] if self.config.models else ""
            if not model_name:
                return False, "OPENROUTER connected but no model is configured."

            probe_payload = self._build_payload(
                model_name=model_name,
                system_prompt="Respond with OK.",
                user_prompt="health-check",
                temperature=0.0,
            )
            probe_payload["max_tokens"] = 1
            try:
                probe_response = requests.post(
                    self.config.endpoint,
                    headers=self.config.headers(),
                    json=probe_payload,
                    timeout=DEFAULT_HEALTH_PROBE_TIMEOUT_SECONDS,
                )
            except requests.exceptions.RequestException as exc:
                return False, f"OpenRouter model probe failed: {exc}"

            if probe_response.status_code != 200:
                return (
                    False,
                    f"OpenRouter model probe failed for '{model_name}' "
                    f"with status {probe_response.status_code}.",
                )
            return True, "OPENROUTER CONNECTION + MODEL PROBE VERIFIED"

        return False, f"OpenRouter returned status {response.status_code}. Check your API key."
