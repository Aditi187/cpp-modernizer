from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence
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
SHORT_CODE_RESPONSE_THRESHOLD = 200

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

    def end_span(self, span: Any, output_payload: Any = None, level: str | None = None) -> None:
        if span is None:
            return
        try:
            # New SDK commonly expects output/level via update(), then plain end().
            if hasattr(span, "update"):
                update_kwargs: dict[str, Any] = {}
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
        trace = self.get_or_create_trace()
        if trace is None:
            return

        metadata = {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

        try:
            if hasattr(trace, "generation"):
                trace.generation(
                    name="OpenRouter-Completion",
                    model=model,
                    input={"system_prompt": system_prompt, "user_prompt": prompt},
                    output=response_text,
                    metadata=metadata,
                )
                return

            if hasattr(self._client, "start_observation"):
                generation = self._client.start_observation(
                    trace_context=self._current_trace_context(),
                    name="OpenRouter-Completion",
                    as_type="generation",
                    input={"system_prompt": system_prompt, "user_prompt": prompt},
                    model=model,
                    usage_details={
                        "prompt_tokens": int(prompt_tokens or 0),
                        "completion_tokens": int(completion_tokens or 0),
                        "total_tokens": int(total_tokens or 0),
                    },
                    metadata=metadata,
                )
                if hasattr(generation, "update"):
                    generation.update(
                        output=response_text,
                        model=model,
                        usage_details={
                            "prompt_tokens": int(prompt_tokens or 0),
                            "completion_tokens": int(completion_tokens or 0),
                            "total_tokens": int(total_tokens or 0),
                        },
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
        return cls(OpenRouterConfig.from_env(), log_fn=log_fn)

    def _log(self, message: str) -> None:
        if self._log_fn is not None:
            self._log_fn(message)

    def start_modernization_trace(self, input_payload: Any = None) -> Any:
        return self.tracker.create_trace(name="CPP-Modernization", input_payload=input_payload)

    def start_span(self, name: str, input_payload: Any = None) -> Any:
        return self.tracker.start_span(name=name, input_payload=input_payload)

    def end_span(self, span: Any, output_payload: Any = None, level: str | None = None) -> None:
        self.tracker.end_span(span=span, output_payload=output_payload, level=level)

    def mark_trace_error(self, message: str, details: Any = None) -> None:
        self.tracker.mark_error(message=message, details=details)

    def _build_payload(
        self,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> dict[str, object]:
        return {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": temperature,
        }

    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        start_new_trace: bool = False,
    ) -> str:
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
        prompt_variants = [user_prompt]
        if enforce_full_response:
            prompt_variants.append(f"{user_prompt}\n\n{FULL_RESPONSE_REMINDER}")

        last_exc: Exception | None = None
        last_error_text = ""

        for model_name in self.config.models:
            for prompt_index, prompt_variant in enumerate(prompt_variants, start=1):
                payload = self._build_payload(
                    model_name=model_name,
                    system_prompt=system_prompt,
                    user_prompt=prompt_variant,
                    temperature=temperature,
                )

                for attempt, delay in enumerate(self.config.retry_delays, start=1):
                    try:
                        response = requests.post(
                            self.config.endpoint,
                            headers=self.config.headers(),
                            json=payload,
                            timeout=self.config.request_timeout_seconds,
                        )
                        if response.status_code == 200:
                            data = response.json()
                            choice = data["choices"][0]
                            content = choice["message"]["content"]
                            finish_reason = choice.get("finish_reason")

                            usage = data.get("usage") or {}
                            prompt_tokens = usage.get("prompt_tokens")
                            completion_tokens = usage.get("completion_tokens")
                            total_tokens = usage.get("total_tokens")

                            self.tracker.capture_generation(
                                model=model_name,
                                prompt=prompt_variant,
                                response_text=content,
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens,
                                total_tokens=total_tokens,
                                system_prompt=system_prompt,
                            )

                            if finish_reason == "length":
                                last_error_text = (
                                    f"Model '{model_name}' stopped because of token limits. "
                                    "Increase OPENROUTER_MAX_TOKENS."
                                )
                                break

                            if (
                                enforce_full_response
                                and len(content.strip()) < SHORT_CODE_RESPONSE_THRESHOLD
                                and prompt_index < len(prompt_variants)
                            ):
                                self._log(
                                    f"Model '{model_name}' returned only {len(content.strip())} characters. "
                                    "Retrying with full-response reminder..."
                                )
                                break

                            return content

                        if response.status_code == 429:
                            self._log(
                                f"OpenRouter rate limited (429) for model '{model_name}'. "
                                f"Retrying in {delay}s... (attempt {attempt}/{len(self.config.retry_delays)})"
                            )
                        elif 500 <= response.status_code < 600:
                            self._log(
                                f"OpenRouter server error {response.status_code} for model '{model_name}'. "
                                f"Retrying in {delay}s... (attempt {attempt}/{len(self.config.retry_delays)})"
                            )
                        else:
                            last_error_text = response.text
                            if _looks_like_model_unavailable(response.status_code, response.text):
                                self._log(
                                    f"Model '{model_name}' unavailable. Trying next fallback model..."
                                )
                                break
                            raise RuntimeError(
                                f"OpenRouter returned {response.status_code} for model '{model_name}': {response.text}"
                            )
                    except RuntimeError:
                        raise
                    except requests.exceptions.RequestException as exc:
                        last_exc = exc
                        self._log(
                            f"Network error contacting OpenRouter for model '{model_name}': {exc}. "
                            f"Retrying in {delay}s... (attempt {attempt}/{len(self.config.retry_delays)})"
                        )

                    if attempt < len(self.config.retry_delays):
                        time.sleep(delay)

        raise RuntimeError(
            "OpenRouter call failed across all configured models. "
            f"Last error: {last_exc!r}. Last response body: {last_error_text}"
        )

    def check_health(self) -> tuple[bool, str]:
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
            return True, "OPENROUTER CONNECTION VERIFIED"

        return False, f"OpenRouter returned status {response.status_code}. Check your API key."
