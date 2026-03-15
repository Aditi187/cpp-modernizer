from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence
import os
import time
import re
import threading

import requests

from core.openrouter_bridge import (
    FULL_RESPONSE_REMINDER,
    LangfuseTracker,
    SHORT_CODE_RESPONSE_THRESHOLD,
    _expects_large_code_response,
    _parse_int_env,
)


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
LARGE_PROMPT_WARNING_CHARS = 120_000
MIN_LARGE_CODE_RESPONSE_CHARS = 100
MAX_CONTINUATION_PASSES = 3

LAST_REQUEST_TIME = 0.0
MIN_REQUEST_INTERVAL = 3.0
_RATE_LIMIT_LOCK = threading.Lock()

CPP_MODERNIZATION_SYSTEM_PROMPT = (
    "You are a C++23 expert. Output valid C++23 code only. "
    "Use std::expected for errors. Do not truncate. "
    "If the snippet is too long, provide a logical stopping point and request a continuation."
)

_CODE_FENCE_RE = re.compile(r"```(?:\w*)\n(.*?)```", re.DOTALL)
_CPP_FENCE_RE = re.compile(r"```(?:cpp|c\+\+|cc|cxx|hpp|hxx|h)?\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)


def _strip_assistant_prefixes(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^\s*(assistant|model|ai)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*(here is|here's|below is)\b.*?\n", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _find_likely_cpp_start(text: str) -> int:
    markers = [
        "#include",
        "using namespace",
        "namespace ",
        "template<",
        "template <",
        "class ",
        "struct ",
        "typedef ",
        "int main(",
        "void ",
        "auto ",
    ]
    starts = [text.find(marker) for marker in markers if text.find(marker) != -1]
    return min(starts) if starts else 0


def _extract_code_candidates(text: str) -> list[str]:
    normalized = _strip_assistant_prefixes(text)
    candidates = [match.group(1).strip() for match in _CPP_FENCE_RE.finditer(normalized) if match.group(1).strip()]

    if not candidates:
        generic_fences = [match.group(1).strip() for match in _CODE_FENCE_RE.finditer(normalized) if match.group(1).strip()]
        candidates.extend(generic_fences)

    if not candidates:
        unfenced = re.sub(r"```(?:[^\n]*)\n?", "", normalized).strip()
        start = _find_likely_cpp_start(unfenced)
        unfenced = unfenced[start:].strip()
        if unfenced:
            candidates.append(unfenced)

    return candidates


def _score_cpp_candidate(code: str) -> tuple[int, int]:
    text = code.strip()
    if not text:
        return (0, 0)

    signal_score = 0
    cpp_signals = (
        "#include",
        "std::",
        "namespace ",
        "class ",
        "struct ",
        "template<",
        "template <",
        ";",
        "{",
        "}",
    )
    for signal in cpp_signals:
        if signal in text:
            signal_score += 1
    return (signal_score, len(text))


def _extract_code_text(text: str) -> str:
    candidates = _extract_code_candidates(text)
    if not candidates:
        return ""
    return max(candidates, key=_score_cpp_candidate).strip()


def _strip_strings_and_comments(code: str) -> str:
    result: list[str] = []
    i = 0
    in_single = False
    in_double = False
    in_line_comment = False
    in_block_comment = False
    escape = False

    while i < len(code):
        ch = code[i]
        nxt = code[i + 1] if i + 1 < len(code) else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
                result.append(ch)
            i += 1
            continue

        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue

        if in_single:
            if not escape and ch == "'":
                in_single = False
            escape = (ch == "\\" and not escape)
            i += 1
            continue

        if in_double:
            if not escape and ch == '"':
                in_double = False
            escape = (ch == "\\" and not escape)
            i += 1
            continue

        escape = False
        if ch == "/" and nxt == "/":
            in_line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            i += 2
            continue
        if ch == "'":
            in_single = True
            i += 1
            continue
        if ch == '"':
            in_double = True
            i += 1
            continue

        result.append(ch)
        i += 1

    return "".join(result)


def _has_balanced_braces(code: str) -> bool:
    text = _strip_strings_and_comments(code)
    depth = 0
    for ch in text:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _passes_truncation_guard(candidate_text: str, expects_large: bool) -> bool:
    candidate = _extract_code_text(candidate_text)
    candidate_stripped = candidate.strip()
    if not candidate_stripped:
        return False
    if expects_large and len(candidate_stripped) < MIN_LARGE_CODE_RESPONSE_CHARS:
        return False
    if not _has_balanced_braces(candidate_stripped):
        return False
    return candidate_stripped.endswith("}") or candidate_stripped.endswith("};")


def _clean_cpp_response_text(raw_text: str) -> str:
    extracted = _extract_code_text(raw_text)
    extracted = re.sub(r"```(?:[^\n]*)\n?", "", extracted)
    extracted = _strip_assistant_prefixes(extracted)
    return extracted.strip()


def _enforce_global_rate_limit() -> None:
    global LAST_REQUEST_TIME
    with _RATE_LIMIT_LOCK:
        now = time.time()
        elapsed = now - LAST_REQUEST_TIME
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        LAST_REQUEST_TIME = time.time()


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

    def _post_completion(
        self,
        model_name: str,
        payload: dict[str, object],
        attempt: int,
        max_attempts: int,
        purpose: str,
    ) -> requests.Response:
        _enforce_global_rate_limit()
        return requests.post(
            self._endpoint_for_model(model_name),
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=self.config.request_timeout_seconds,
        )

    def _stream_completion(
        self,
        model_name: str,
        payload: dict[str, object],
    ) -> str:
        """Placeholder for future streaming support; synchronous mode remains default."""
        _ = (model_name, payload)
        raise NotImplementedError("Streaming completion is not enabled yet.")

    def _request_with_retries(
        self,
        model_name: str,
        payload: dict[str, object],
        purpose: str,
    ) -> tuple[dict[str, Any], str, int | None, int | None, int | None]:
        last_network_exc: Exception | None = None
        last_error_body = ""
        max_attempts = len(self.config.retry_delays)

        for attempt_index, delay in enumerate(self.config.retry_delays, start=1):
            try:
                response = self._post_completion(
                    model_name=model_name,
                    payload=payload,
                    attempt=attempt_index,
                    max_attempts=max_attempts,
                    purpose=purpose,
                )
            except requests.exceptions.RequestException as exc:
                last_network_exc = exc
                if attempt_index < max_attempts:
                    self._log(
                        f"Gemini network error for model '{model_name}' during {purpose}: {exc}. "
                        f"Retrying in {delay}s... (attempt {attempt_index}/{max_attempts})"
                    )
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    f"Gemini network failure for model '{model_name}' during {purpose}: {exc!r}"
                ) from exc

            if response.status_code == 200:
                data = response.json()
                text, finish_reason, prompt_tokens, completion_tokens, total_tokens = self._extract_text(data)
                self._log(
                    f"Gemini response received for model '{model_name}' ({purpose}): "
                    f"chars={len(text.strip())}, finish_reason={finish_reason or 'UNKNOWN'}, "
                    f"tokens(prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens})"
                )
                return data, finish_reason, prompt_tokens, completion_tokens, total_tokens

            is_rate_limit = response.status_code == 429
            is_server_error = 500 <= response.status_code < 600
            if is_rate_limit or is_server_error:
                if attempt_index < max_attempts:
                    error_kind = "rate limited (429)" if is_rate_limit else f"server error {response.status_code}"
                    wait_seconds = 5 * attempt_index if is_rate_limit else delay
                    self._log(
                        f"Gemini {error_kind} for model '{model_name}' during {purpose}. "
                        f"Retrying in {wait_seconds}s... (attempt {attempt_index}/{max_attempts})"
                    )
                    time.sleep(wait_seconds)
                    continue

                raise RuntimeError(
                    f"Gemini request failed after retries for model '{model_name}' during {purpose}. "
                    f"Status {response.status_code}: {response.text}"
                )

            last_error_body = response.text
            if _looks_like_model_unavailable(response.status_code, response.text):
                raise RuntimeError(
                    f"MODEL_UNAVAILABLE::{response.status_code}::{response.text}"
                )

            raise RuntimeError(
                f"Gemini returned {response.status_code} for model '{model_name}': {response.text}"
            )

        raise RuntimeError(
            f"Gemini request failed for model '{model_name}' during {purpose}. "
            f"Last network error: {last_network_exc!r}. Last response body: {last_error_body}"
        )

    def _complete_with_continuations(
        self,
        model_name: str,
        system_prompt: str,
        base_user_prompt: str,
        first_text: str,
        first_finish_reason: str,
        temperature: float,
    ) -> str:
        combined_text = first_text
        finish_reason = first_finish_reason

        for continuation_index in range(1, MAX_CONTINUATION_PASSES + 1):
            if finish_reason != "MAX_TOKENS":
                break

            previous_tail = combined_text[-4000:]
            continuation_prompt = (
                f"{base_user_prompt}\n\n"
                "The previous response reached token limits. "
                "Continue from the previous code without repeating earlier lines. "
                "Return only C++ code.\n\n"
                "Previous partial output tail:\n"
                f"```cpp\n{previous_tail}\n```"
            )
            payload = self._build_payload(
                system_prompt=system_prompt,
                user_prompt=continuation_prompt,
                temperature=temperature,
            )
            data, finish_reason, _, _, _ = self._request_with_retries(
                model_name=model_name,
                payload=payload,
                purpose=f"continuation-{continuation_index}",
            )
            continuation_text, _, _, _, _ = self._extract_text(data)
            cleaned_part = _clean_cpp_response_text(continuation_text)
            if not cleaned_part:
                break
            combined_text = (combined_text.rstrip() + "\n" + cleaned_part.lstrip()).strip()

        return combined_text

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

        prompt_len = len(user_prompt)
        if prompt_len > LARGE_PROMPT_WARNING_CHARS:
            self._log(
                f"Warning: very large prompt ({prompt_len} chars). Reliability may degrade."
            )

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

                self._log(
                    f"Gemini request starting: model='{model_name}', prompt_variant={prompt_index}/{len(prompt_variants)}, "
                    f"prompt_chars={len(prompt_variant)}"
                )

                for attempt, delay in enumerate(self.config.retry_delays, start=1):
                    try:
                        data, finish_reason, prompt_tokens, completion_tokens, total_tokens = self._request_with_retries(
                            model_name=model_name,
                            payload=payload,
                            purpose=f"main-attempt-{attempt}",
                        )
                        content, extracted_finish_reason, _, _, _ = self._extract_text(data)
                        finish_reason = extracted_finish_reason or finish_reason

                        combined_content = self._complete_with_continuations(
                            model_name=model_name,
                            system_prompt=system_prompt,
                            base_user_prompt=prompt_variant,
                            first_text=content,
                            first_finish_reason=finish_reason,
                            temperature=temperature,
                        )
                        cleaned_content = _clean_cpp_response_text(combined_content)

                        self.tracker.capture_generation(
                            model=model_name,
                            prompt=prompt_variant,
                            response_text=cleaned_content,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=total_tokens,
                            system_prompt=system_prompt,
                        )

                        if enforce_full_response and len(cleaned_content) < MIN_LARGE_CODE_RESPONSE_CHARS:
                            last_error_text = (
                                f"Model '{model_name}' response too short ({len(cleaned_content)} chars) "
                                "for a large-code request."
                            )
                            self._log(
                                f"{last_error_text} Retrying... (attempt {attempt}/{len(self.config.retry_delays)})"
                            )
                            if attempt < len(self.config.retry_delays):
                                time.sleep(delay)
                                continue
                            break

                        if (
                            enforce_full_response
                            and len(cleaned_content) < SHORT_CODE_RESPONSE_THRESHOLD
                            and prompt_index < len(prompt_variants)
                        ):
                            self._log(
                                f"Model '{model_name}' returned only {len(cleaned_content)} characters. "
                                "Retrying with full-response reminder..."
                            )
                            break

                        if enforce_full_response and not _passes_truncation_guard(cleaned_content, True):
                            last_error_text = (
                                f"Model '{model_name}' response failed lightweight truncation guard "
                                "(size/brace/end-check)."
                            )
                            self._log(last_error_text)
                            if attempt < len(self.config.retry_delays):
                                time.sleep(delay)
                                continue
                            break

                        self._log(
                            f"Gemini completion accepted: model='{model_name}', chars={len(cleaned_content)}, "
                            f"attempt={attempt}, prompt_variant={prompt_index}"
                        )
                        return cleaned_content

                    except RuntimeError as exc:
                        message = str(exc)
                        if message.startswith("MODEL_UNAVAILABLE::"):
                            self._log(
                                f"Model '{model_name}' unavailable or quota-blocked. Trying next fallback model..."
                            )
                            last_error_text = message
                            break
                        last_exc = exc
                        raise

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