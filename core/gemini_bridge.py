from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence
import os
import time
import re

import requests

from core.openrouter_bridge import (
    FULL_RESPONSE_REMINDER,
    LangfuseTracker,
    SHORT_CODE_RESPONSE_THRESHOLD,
    _expects_large_code_response,
    _parse_int_env,
)
from core.differential_tester import compile_cpp_source


# Fill this in directly only if you are not using environment variables.
# GEMINI_API_KEY from the environment takes precedence when set.
api_key: str = ""

DEFAULT_GEMINI_ENDPOINT_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_GEMINI_MODELS: tuple[str, ...] = (
    "gemini-2.5-flash",
)
DEFAULT_RETRY_DELAYS: tuple[int, ...] = (1, 2, 4, 8, 16)
DEFAULT_MAX_OUTPUT_TOKENS = 8192
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120

CPP_MODERNIZATION_SYSTEM_PROMPT = (
    "You are a C++23 expert. Output valid C++23 code only. "
    "Use std::expected for errors. Do not truncate. "
    "If the snippet is too long, provide a logical stopping point and request a continuation."
)

_CODE_FENCE_RE = re.compile(r"```(?:\w*)\n(.*?)```", re.DOTALL)


def _extract_code_text(text: str) -> str:
    match = _CODE_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    cleaned = re.sub(r"```(?:[^\n]*)\n?", "", text)
    return cleaned.strip()


def _passes_truncation_guard(candidate_text: str) -> bool:
    candidate = _extract_code_text(candidate_text)
    if not candidate.strip().endswith("}"):
        return False
    compile_result = compile_cpp_source(candidate, enable_sanitizers=False)
    return bool(compile_result.get("success"))


def _dedupe_models(models: Sequence[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for model in models:
        normalized = model.strip()
        if normalized.lower().startswith("models/"):
            normalized = normalized.split("/", 1)[1]
        if not normalized or normalized in seen:
            continue
        ordered.append(normalized)
        seen.add(normalized)
    return ordered


def _get_env_models() -> list[str]:
    configured = (
        os.environ.get("GEMINI_MODELS", "").strip()
        or os.environ.get("OPENROUTER_MODELS", "").strip()
    )
    if configured:
        candidates = [value for value in configured.split(",")]
    else:
        candidates = list(DEFAULT_GEMINI_MODELS)

    gemini_candidates = [value for value in candidates if value.strip().lower().startswith("gemini")]
    if not gemini_candidates:
        gemini_candidates = list(DEFAULT_GEMINI_MODELS)

    models = _dedupe_models([*gemini_candidates, *DEFAULT_GEMINI_MODELS])
    flash_primary = "gemini-2.5-flash"
    if flash_primary in models:
        models = [flash_primary, *[m for m in models if m != flash_primary]]
    return models


def _looks_like_model_unavailable(status_code: int, response_text: str) -> bool:
    lower_error = response_text.lower()
    return status_code in {400, 404} and "model" in lower_error and (
        "not found" in lower_error
        or "unsupported" in lower_error
        or "unavailable" in lower_error
    )


@dataclass(frozen=True)
class GeminiConfig:
    api_key: str
    endpoint_base: str
    models: tuple[str, ...]
    max_output_tokens: int
    request_timeout_seconds: int
    retry_delays: tuple[int, ...]

    @classmethod
    def from_env(cls) -> "GeminiConfig":
        resolved_api_key = (
            os.environ.get("GEMINI_API_KEY", "").strip()
            or os.environ.get("OPENROUTER_API_KEY", "").strip()
            or api_key
        )
        return cls(
            api_key=resolved_api_key,
            endpoint_base=(
                os.environ.get("GEMINI_ENDPOINT_BASE", DEFAULT_GEMINI_ENDPOINT_BASE).strip()
                or DEFAULT_GEMINI_ENDPOINT_BASE
            ),
            models=tuple(_get_env_models()),
            max_output_tokens=(
                _parse_int_env("GEMINI_MAX_OUTPUT_TOKENS", 0)
                or _parse_int_env("OPENROUTER_MAX_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS)
            ),
            request_timeout_seconds=(
                _parse_int_env("GEMINI_REQUEST_TIMEOUT_SECONDS", 0)
                or _parse_int_env("OPENROUTER_REQUEST_TIMEOUT_SECONDS", DEFAULT_REQUEST_TIMEOUT_SECONDS)
            ),
            retry_delays=DEFAULT_RETRY_DELAYS,
        )


class GeminiBridge:
    def __init__(
        self,
        config: GeminiConfig | None = None,
        log_fn: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config or GeminiConfig.from_env()
        self._log_fn = log_fn
        self.tracker = LangfuseTracker(log_fn=log_fn)

    @classmethod
    def from_env(
        cls,
        log_fn: Callable[[str], None] | None = None,
    ) -> "GeminiBridge":
        return cls(GeminiConfig.from_env(), log_fn=log_fn)

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

    def _endpoint_for_model(self, model_name: str) -> str:
        return f"{self.config.endpoint_base}/models/{model_name}:generateContent?key={self.config.api_key}"

    def _build_payload(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> dict[str, object]:
        return {
            "systemInstruction": {
                "parts": [{"text": system_prompt}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}],
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": self.config.max_output_tokens,
            },
        }

    def _extract_text(self, data: dict[str, Any]) -> tuple[str, str, int | None, int | None, int | None]:
        candidates = data.get("candidates") or []
        candidate = candidates[0] if candidates else {}
        content = candidate.get("content") or {}
        parts = content.get("parts") or []
        text = "".join(str(part.get("text") or "") for part in parts if isinstance(part, dict))
        finish_reason = str(candidate.get("finishReason") or "")

        usage = data.get("usageMetadata") or {}
        prompt_tokens = usage.get("promptTokenCount")
        completion_tokens = usage.get("candidatesTokenCount")
        total_tokens = usage.get("totalTokenCount")
        return text, finish_reason, prompt_tokens, completion_tokens, total_tokens

    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        start_new_trace: bool = False,
    ) -> str:
        if not self.config.api_key:
            raise ValueError(
                "GEMINI_API_KEY is empty. Set it in your environment or .env file."
            )

        if start_new_trace:
            self.start_modernization_trace(
                input_payload={
                    "operation": "modernization",
                    "model_candidates": list(self.config.models),
                    "system_prompt": system_prompt,
                    "provider": "gemini",
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
                    system_prompt=system_prompt,
                    user_prompt=prompt_variant,
                    temperature=temperature,
                )

                for attempt, delay in enumerate(self.config.retry_delays, start=1):
                    try:
                        response = requests.post(
                            self._endpoint_for_model(model_name),
                            headers={"Content-Type": "application/json"},
                            json=payload,
                            timeout=self.config.request_timeout_seconds,
                        )
                        if response.status_code == 200:
                            data = response.json()
                            content, finish_reason, prompt_tokens, completion_tokens, total_tokens = self._extract_text(data)

                            self.tracker.capture_generation(
                                model=model_name,
                                prompt=prompt_variant,
                                response_text=content,
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens,
                                total_tokens=total_tokens,
                                system_prompt=system_prompt,
                            )

                            if finish_reason == "MAX_TOKENS":
                                last_error_text = (
                                    f"Model '{model_name}' stopped because of token limits. "
                                    "Increase GEMINI_MAX_OUTPUT_TOKENS."
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

                            if enforce_full_response and not _passes_truncation_guard(content):
                                last_error_text = (
                                    f"Model '{model_name}' response failed truncation guard "
                                    "(missing final brace or C++23 compile failure)."
                                )
                                self._log(last_error_text)
                                break

                            return content

                        if response.status_code == 429:
                            self._log(
                                f"Gemini rate limited (429) for model '{model_name}'. "
                                f"Retrying in {delay}s... (attempt {attempt}/{len(self.config.retry_delays)})"
                            )
                        elif 500 <= response.status_code < 600:
                            self._log(
                                f"Gemini server error {response.status_code} for model '{model_name}'. "
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
                                f"Gemini returned {response.status_code} for model '{model_name}': {response.text}"
                            )
                    except RuntimeError:
                        raise
                    except requests.exceptions.RequestException as exc:
                        last_exc = exc
                        self._log(
                            f"Network error contacting Gemini for model '{model_name}': {exc}. "
                            f"Retrying in {delay}s... (attempt {attempt}/{len(self.config.retry_delays)})"
                        )

                    if attempt < len(self.config.retry_delays):
                        time.sleep(delay)

        raise RuntimeError(
            "Gemini call failed across all configured models. "
            f"Last error: {last_exc!r}. Last response body: {last_error_text}"
        )

    def check_health(self) -> tuple[bool, str]:
        if not self.config.api_key:
            return False, "GEMINI_API_KEY is empty. Set it in your environment or .env file."

        try:
            response = requests.get(
                f"{self.config.endpoint_base}/models?key={self.config.api_key}",
                timeout=10,
            )
        except requests.exceptions.RequestException as exc:
            return False, f"Could not reach Gemini: {exc}"

        if response.status_code == 200:
            return True, "GEMINI CONNECTION VERIFIED"

        return False, f"Gemini returned status {response.status_code}. Check your API key."