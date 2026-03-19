from __future__ import annotations

import atexit
import importlib
import logging
import os
import re
from importlib import metadata as importlib_metadata
from typing import Any


def _trace_context(trace: Any) -> dict[str, str] | None:
    if isinstance(trace, dict):
        trace_id = str(trace.get("trace_id") or "").strip()
        if trace_id:
            return {"trace_id": trace_id}
    return None


def _invoke_with_keyword_fallback(method: Any, kwargs: dict[str, Any]) -> None:
    """Invoke SDK methods while tolerating keyword-shape differences across versions."""
    attempt_kwargs = dict(kwargs)
    max_attempts = max(2, len(attempt_kwargs) + 2)
    for _ in range(max_attempts):
        try:
            method(**attempt_kwargs)
            return
        except TypeError as exc:
            message = str(exc)

            if "output" in attempt_kwargs and "output" in message and "output_data" not in attempt_kwargs:
                output_value = attempt_kwargs.pop("output")
                attempt_kwargs["output_data"] = output_value
                continue

            match = re.search(r"unexpected keyword argument '([^']+)'", message)
            if match:
                bad_key = match.group(1)
                if bad_key in attempt_kwargs:
                    attempt_kwargs.pop(bad_key)
                    continue

            if attempt_kwargs:
                attempt_kwargs = {}
                continue
            raise


try:
    _langfuse_module = importlib.import_module("langfuse")
    Langfuse = getattr(_langfuse_module, "Langfuse", None)
except ImportError:  # pragma: no cover - optional dependency at runtime
    Langfuse = None
except Exception as exc:  # pragma: no cover - broken installation should be visible
    logging.getLogger(__name__).exception("Unexpected failure importing langfuse: %r", exc)
    raise


_LOG = logging.getLogger(__name__)
_DEFAULT_TRACE_NAME = "CPP-Modernization"
_MIN_LANGFUSE_VERSION = (2, 0, 0)


def _parse_version_tuple(raw: str) -> tuple[int, int, int]:
    parts = [token for token in (raw or "").split(".") if token.isdigit()]
    while len(parts) < 3:
        parts.append("0")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _langfuse_version_supported() -> bool:
    try:
        raw_version = importlib_metadata.version("langfuse")
    except Exception:
        return False
    return _parse_version_tuple(raw_version) >= _MIN_LANGFUSE_VERSION


def parse_int_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name, "").strip()
    if not raw_value:
        return default
    try:
        return int(raw_value)
    except ValueError:
        _LOG.warning("Invalid integer for %s=%r. Using default=%d", name, raw_value, default)
        return default


def expects_large_code_response(user_prompt: str) -> bool:
    """Heuristic for detecting prompts likely to request large code responses.

    Limitations:
    - Keyword heuristics can produce false positives/negatives.
    - This is not semantic intent classification.

    Configure triggers with `LLM_LARGE_RESPONSE_TRIGGERS` as comma-separated
    values. Empty/invalid config falls back to defaults.
    """
    lower_prompt = user_prompt.lower()
    raw_triggers = os.environ.get("LLM_LARGE_RESPONSE_TRIGGERS", "").strip()
    if raw_triggers:
        parsed = [item.strip().lower() for item in raw_triggers.split(",") if item.strip()]
        triggers = tuple(parsed)
    else:
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


class LangfuseTracker:
    """Thin wrapper around Langfuse client with explicit trace ownership.

    This wrapper targets Langfuse >= 2.0.0. If unavailable or misconfigured,
    tracing is disabled gracefully.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        default_trace_name: str | None = None,
    ) -> None:
        self._logger = logger or _LOG
        self._default_trace_name = (default_trace_name or os.environ.get("LANGFUSE_TRACE_NAME", "").strip() or _DEFAULT_TRACE_NAME)
        self._client: Any = None
        self._flush_registered = False

        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
        host = os.environ.get("LANGFUSE_HOST", "").strip()

        if not public_key or not secret_key or not host:
            self._logger.info("Langfuse disabled: missing LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY/LANGFUSE_HOST.")
            return

        if Langfuse is None:
            self._logger.info("Langfuse SDK not installed; tracing disabled.")
            return

        if not _langfuse_version_supported():
            self._logger.warning("Langfuse version is below %s.%s.%s; tracing disabled.", *_MIN_LANGFUSE_VERSION)
            return

        try:
            self._client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
            )
            self._logger.info("Langfuse initialized.")
            atexit.register(self.flush)
            self._flush_registered = True
        except Exception as exc:
            self._client = None
            self._logger.exception("Langfuse init failed: %r", exc)

    def enabled(self) -> bool:
        return self._client is not None

    def create_trace(self, name: str | None = None, input_payload: Any = None) -> Any:
        if not self.enabled():
            return None
        try:
            trace_name = name or self._default_trace_name
            if hasattr(self._client, "trace"):
                return self._client.trace(name=trace_name, input=input_payload)

            if hasattr(self._client, "create_trace_id"):
                trace_id = str(self._client.create_trace_id())
                trace_ref = {"trace_id": trace_id, "name": trace_name}
                if hasattr(self._client, "create_event"):
                    self._client.create_event(
                        trace_context={"trace_id": trace_id},
                        name=trace_name,
                        input=input_payload,
                        metadata={"status": "started"},
                    )
                return trace_ref

            self._logger.warning("Langfuse client has no supported trace creation method.")
            return None
        except Exception as exc:
            self._logger.exception("Failed to create Langfuse trace: %r", exc)
            return None

    def start_span(self, trace: Any, name: str, input_payload: Any = None) -> Any:
        if trace is None:
            return None
        try:
            if hasattr(trace, "span"):
                return trace.span(name=name, input=input_payload)

            context = _trace_context(trace)
            if context is not None and hasattr(self._client, "start_observation"):
                return self._client.start_observation(
                    trace_context=context,
                    name=name,
                    as_type="span",
                    input=input_payload,
                )

            self._logger.warning("Could not start span '%s': unsupported trace/client shape.", name)
            return None
        except Exception as exc:
            self._logger.exception("Failed to start Langfuse span '%s': %r", name, exc)
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
        kwargs: dict[str, Any] = {}
        if input_payload is not None:
            kwargs["input"] = input_payload
        if output_payload is not None:
            kwargs["output"] = output_payload
        if level:
            kwargs["level"] = level
        try:
            if hasattr(span, "end"):
                span.end(**kwargs)
                return
            self._logger.warning("Span object has no end() method; skipping finalize.")
        except Exception as exc:
            self._logger.exception("Failed to end Langfuse span: %r", exc)

    def create_generation(
        self,
        trace: Any,
        name: str,
        model: str,
        input_data: Any,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        if trace is None:
            return None

        try:
            if hasattr(trace, "generation"):
                return trace.generation(
                    name=name,
                    model=model,
                    input=input_data,
                    metadata=metadata or {},
                )

            context = _trace_context(trace)
            if context is not None and hasattr(self._client, "start_observation"):
                return self._client.start_observation(
                    trace_context=context,
                    name=name,
                    as_type="generation",
                    model=model,
                    input=input_data,
                    metadata=metadata or {},
                )

            self._logger.warning("Could not create generation '%s': unsupported trace/client shape.", name)
            return None
        except Exception as exc:
            self._logger.exception("Failed to create Langfuse generation: %r", exc)
        return None

    def finalize_generation(
        self,
        generation: Any,
        *,
        output: Any = None,
        model: str | None = None,
        usage_details: dict[str, int] | None = None,
        metadata: dict[str, Any] | None = None,
        level: str | None = None,
        status_message: str | None = None,
    ) -> None:
        if generation is None:
            return

        kwargs: dict[str, Any] = {}
        if output is not None:
            kwargs["output"] = output
        if model is not None:
            kwargs["model"] = model
        if usage_details is not None:
            kwargs["usage_details"] = usage_details
        if metadata is not None:
            kwargs["metadata"] = metadata
        if level is not None:
            kwargs["level"] = level
        if status_message is not None:
            kwargs["status_message"] = status_message

        try:
            if hasattr(generation, "end"):
                _invoke_with_keyword_fallback(generation.end, kwargs)
                return
            if hasattr(generation, "update"):
                _invoke_with_keyword_fallback(generation.update, kwargs)
                return
            self._logger.warning("Generation object has no end()/update(); skipping finalize.")
        except Exception as exc:
            self._logger.exception("Failed to finalize Langfuse generation: %r", exc)

    def mark_error(self, trace: Any, message: str, details: Any = None) -> None:
        if trace is None:
            return

        payload = {"error": message, "details": details}

        try:
            if hasattr(trace, "update"):
                trace.update(output=payload, metadata={"status": "Error"})
                return
            self._logger.warning("Trace object does not support update(); cannot mark error for message=%s", message)
        except Exception as exc:
            self._logger.exception("Failed to mark Langfuse trace error: %r", exc)

    def flush(self) -> None:
        if not self.enabled():
            return
        try:
            self._client.flush()
        except Exception as exc:
            self._logger.exception("Langfuse flush failed: %r", exc)
