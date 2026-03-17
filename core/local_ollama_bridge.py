from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable

import requests

from core.openrouter_bridge import LangfuseTracker


CPP_MODERNIZATION_SYSTEM_PROMPT = (
    "You are a senior C++ modernization engineer.\n"
    "Rewrite legacy C++ code to modern C++23.\n\n"
    "Rules:\n"
    "- Do not return unchanged code\n"
    "- Replace printf with std::print\n"
    "- Replace raw pointers with smart pointers\n"
    "- Prefer ranges over index loops\n"
    "- Use std::span instead of raw buffers\n"
    "- Use std::optional or std::expected\n"
    "- Output ONLY valid C++ code\n"
)


@dataclass(frozen=True)
class LocalOllamaConfig:
    base_url: str
    model: str
    timeout_seconds: int

    @classmethod
    def from_env(cls) -> "LocalOllamaConfig":
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").strip() or "http://localhost:11434"
        model = os.environ.get("OLLAMA_MODEL", "deepseek-coder:6.7b").strip() or "deepseek-coder:6.7b"
        raw_timeout = os.environ.get("OLLAMA_TIMEOUT_SECONDS", "300").strip() or "300"
        try:
            timeout_seconds = max(30, int(raw_timeout))
        except ValueError:
            timeout_seconds = 300
        return cls(base_url=base_url.rstrip("/"), model=model, timeout_seconds=timeout_seconds)


class LocalOllamaBridge:
    """Bridge for local Ollama chat completions with optional Langfuse tracing."""

    def __init__(self, config: LocalOllamaConfig | None = None, log_fn: Callable[[str], None] | None = None) -> None:
        self.config = config or LocalOllamaConfig.from_env()
        self._log_fn = log_fn
        self.tracker = LangfuseTracker(log_fn=log_fn)

    @classmethod
    def from_env(cls, log_fn: Callable[[str], None] | None = None) -> "LocalOllamaBridge":
        return cls(LocalOllamaConfig.from_env(), log_fn=log_fn)

    def _log(self, message: str) -> None:
        if self._log_fn is not None:
            self._log_fn(message)

    def start_modernization_trace(self, input_payload: Any = None) -> Any:
        """Start or attach to a modernization trace."""
        return self.tracker.create_trace(name="CPP-Modernization", input_payload=input_payload)

    def start_span(self, name: str, input_payload: Any = None) -> Any:
        """Start a generic span for non-LLM operations."""
        return self.tracker.start_span(name=name, input_payload=input_payload)

    def end_span(self, span: Any, output_payload: Any = None, level: str | None = None) -> None:
        """End a generic span for non-LLM operations."""
        self.tracker.end_span(span=span, output_payload=output_payload, level=level)

    def mark_trace_error(self, message: str, details: Any = None) -> None:
        """Attach an error to the active trace."""
        self.tracker.mark_error(message=message, details=details)

    @staticmethod
    def _prompt_snippet(text: str, max_chars: int = 200) -> str:
        cleaned = " ".join((text or "").split())
        return cleaned[:max_chars]

    @staticmethod
    def _extract_chat_text(data: dict[str, Any]) -> str:
        message = data.get("message")
        if isinstance(message, dict):
            content = str(message.get("content") or "").strip()
            if content:
                return content
        return str(data.get("response") or "").strip()

    @staticmethod
    def _extract_token_counts(data: dict[str, Any]) -> tuple[int, int, int]:
        prompt_tokens = int(data.get("prompt_eval_count") or 0)
        completion_tokens = int(data.get("eval_count") or 0)
        total_tokens = prompt_tokens + completion_tokens
        return prompt_tokens, completion_tokens, total_tokens

    def _post_chat_once(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        timeout_seconds: int,
    ) -> tuple[str, int, int, int]:
        """Issue one non-streaming Ollama chat request and extract text/tokens."""
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            "stream": False,
            "options": {
                "temperature": float(temperature),
            },
        }

        url = f"{self.config.base_url}/api/chat"
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=(10, timeout_seconds),
            )
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"OLLAMA_TIMEOUT model={self.config.model} endpoint={url} timeout={timeout_seconds}s"
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"OLLAMA_CONNECTION_ERROR model={self.config.model} endpoint={url} error={exc}"
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(
                f"OLLAMA_REQUEST_EXCEPTION model={self.config.model} endpoint={url} error={exc}"
            ) from exc

        if response.status_code != 200:
            raise RuntimeError(f"OLLAMA_HTTP_{response.status_code}: {response.text}")

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"OLLAMA_INVALID_JSON model={self.config.model} status={response.status_code}"
            ) from exc

        text = self._extract_chat_text(data)
        if not text:
            raise RuntimeError("OLLAMA_EMPTY_RESPONSE")

        prompt_tokens, completion_tokens, total_tokens = self._extract_token_counts(data)
        return text, prompt_tokens, completion_tokens, total_tokens

    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        start_new_trace: bool = False,
        timeout_seconds: int | None = None,
    ) -> str:
        """Run one non-streaming Ollama chat completion.

        Uses the Ollama `/api/chat` endpoint for model-compatible chat templates
        and captures a single Langfuse generation with input/output/token usage.
        """
        if start_new_trace:
            self.start_modernization_trace(
                input_payload={
                    "operation": "modernization",
                    "provider": "ollama",
                    "model": self.config.model,
                }
            )
        effective_timeout = int(timeout_seconds or self.config.timeout_seconds)
        retry_delays_seconds = (0.4, 1.0, 2.0)
        attempts = len(retry_delays_seconds)

        for attempt_idx, delay in enumerate(retry_delays_seconds, start=1):
            try:
                text, prompt_tokens, completion_tokens, total_tokens = self._post_chat_once(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    timeout_seconds=effective_timeout,
                )
                self.tracker.capture_generation(
                    model=self.config.model,
                    prompt=user_prompt,
                    response_text=text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    system_prompt=system_prompt,
                )
                return text
            except requests.RequestException as exc:
                is_last_attempt = attempt_idx == attempts
                if is_last_attempt:
                    self.mark_trace_error(
                        message="OLLAMA_REQUEST_FAILED",
                        details={
                            "model": self.config.model,
                            "attempt": attempt_idx,
                            "error": str(exc),
                            "prompt_snippet": self._prompt_snippet(user_prompt),
                        },
                    )
                    raise RuntimeError(
                        "OLLAMA_REQUEST_FAILED"
                        f" model={self.config.model}"
                        f" prompt='{self._prompt_snippet(user_prompt)}'"
                        f" error={exc}"
                    ) from exc
            except RuntimeError as exc:
                is_last_attempt = attempt_idx == attempts
                if is_last_attempt:
                    self.mark_trace_error(
                        message="OLLAMA_CHAT_FAILED",
                        details={
                            "model": self.config.model,
                            "attempt": attempt_idx,
                            "error": str(exc),
                            "prompt_snippet": self._prompt_snippet(user_prompt),
                        },
                    )
                    raise RuntimeError(
                        "OLLAMA_CHAT_FAILED"
                        f" model={self.config.model}"
                        f" prompt='{self._prompt_snippet(user_prompt)}'"
                        f" error={exc}"
                    ) from exc

            if delay > 0:
                time.sleep(delay)

        raise RuntimeError("OLLAMA_CHAT_FAILED: retries exhausted")

    def check_health(self) -> tuple[bool, str]:
        """Check that Ollama is reachable and the configured model is available."""
        try:
            response = requests.get(
                f"{self.config.base_url}/api/tags",
                timeout=(5, 15),
            )
            if response.status_code != 200:
                return False, f"OLLAMA HEALTH CHECK FAILED ({response.status_code})"
            data = response.json()
            models = data.get("models") or []
            model_names = {str(item.get("name") or "") for item in models if isinstance(item, dict)}
            requested_model = self.config.model.strip()
            requested_base = requested_model.split(":", 1)[0]
            model_bases = {name.split(":", 1)[0] for name in model_names if name}
            if requested_model not in model_names and requested_base not in model_bases:
                return False, (
                    f"OLLAMA CONNECTED, BUT MODEL '{self.config.model}' NOT FOUND. "
                    f"Run: ollama pull {self.config.model}"
                )
            return True, "OLLAMA CONNECTION VERIFIED"
        except requests.RequestException as exc:
            return False, (
                "OLLAMA UNREACHABLE. Ensure Ollama is running locally and reachable at "
                f"{self.config.base_url}. Error: {exc}"
            )
