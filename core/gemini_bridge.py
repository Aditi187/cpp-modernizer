from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence
import hashlib
import json
import os
import random
import time
import re
import threading

import requests

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency at runtime
    load_dotenv = None

from core.openrouter_bridge import (
    LangfuseTracker,
    _expects_large_code_response,
    _parse_int_env,
)


# Fill this in directly only if you are not using environment variables.
# GEMINI_API_KEY from the environment takes precedence when set.
api_key: str = ""

DEFAULT_GEMINI_ENDPOINT_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_OPENAI_ENDPOINT_BASE = "https://api.openai.com/v1"
DEFAULT_GEMINI_MODELS: tuple[str, ...] = (
    "gemini-2.0-flash",
)
DEFAULT_OPENAI_MODELS: tuple[str, ...] = (
    "gpt-4o-mini",
)
DEFAULT_RETRY_DELAYS: tuple[int, ...] = (3, 6, 12, 24)
DEFAULT_MAX_OUTPUT_TOKENS = 8192
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120
LARGE_PROMPT_WARNING_CHARS = 120_000
MAX_PROMPT_CHARS = 100_000
DEFAULT_PROMPT_COMPRESSION_THRESHOLD_CHARS = 50_000
DEFAULT_MIN_LARGE_CODE_RESPONSE_CHARS = 200
MAX_CONTINUATION_PASSES = 3
_GEMINI_CACHE_FILENAME = ".gemini_cache.json"
DEFAULT_SHORT_CODE_RESPONSE_THRESHOLD = 500
DEFAULT_HEALTH_PROBE_TIMEOUT_SECONDS = 10
DEFAULT_GEMINI_CACHE_VERSION = "v1"

LAST_REQUEST_TIME = 0.0
MIN_REQUEST_INTERVAL = float(os.environ.get("LLM_MIN_REQUEST_INTERVAL_SECONDS", "6.0") or "6.0")
_RATE_LIMIT_LOCK = threading.Lock()
PROMPT_COMPRESSION_ENABLED = os.environ.get("GEMINI_ENABLE_PROMPT_COMPRESSION", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
PROMPT_COMPRESSION_THRESHOLD_CHARS = max(
    1_000,
    _parse_int_env("GEMINI_PROMPT_COMPRESSION_THRESHOLD_CHARS", DEFAULT_PROMPT_COMPRESSION_THRESHOLD_CHARS),
)
MIN_LARGE_CODE_RESPONSE_CHARS = max(
    10,
    _parse_int_env("GEMINI_MIN_LARGE_CODE_RESPONSE_CHARS", DEFAULT_MIN_LARGE_CODE_RESPONSE_CHARS),
)
SHORT_CODE_RESPONSE_THRESHOLD = max(
    50,
    _parse_int_env("GEMINI_SHORT_RESPONSE_THRESHOLD", DEFAULT_SHORT_CODE_RESPONSE_THRESHOLD),
)
TRUNCATION_GUARD_ENABLED = os.environ.get("GEMINI_ENABLE_TRUNCATION_GUARD", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
ENABLE_HEALTH_PROBE = os.environ.get("GEMINI_HEALTH_PROBE", "1").strip().lower() in {"1", "true", "yes", "on"}

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

_CODE_FENCE_RE = re.compile(r"```(?:\w*)\n(.*?)```", re.DOTALL)
_CPP_FENCE_RE = re.compile(r"```(?:cpp|c\+\+|cc|cxx|hpp|hxx|h)?\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)


def _strip_assistant_prefixes(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^\s*sure[^\n]*\n", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*(assistant|model|ai)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*(here is|here's|below is)\b.*?\n", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _compress_prompt(prompt_text: str) -> str:
    """Lightweight prompt compression for very large contexts."""
    if len(prompt_text) <= PROMPT_COMPRESSION_THRESHOLD_CHARS:
        return prompt_text
    # Drop C/C++ comments first to preserve code signal under size pressure.
    without_block_comments = re.sub(r"/\*.*?\*/", "", prompt_text, flags=re.DOTALL)
    without_comments = re.sub(r"//[^\n]*", "", without_block_comments)
    # Collapse extra blank lines but keep one newline for structure.
    collapsed = re.sub(r"\n{3,}", "\n\n", without_comments)
    return collapsed


def _maybe_compress_prompt(prompt_text: str) -> tuple[str, bool]:
    if not PROMPT_COMPRESSION_ENABLED:
        return prompt_text, False
    if len(prompt_text) <= PROMPT_COMPRESSION_THRESHOLD_CHARS:
        return prompt_text, False
    compressed = _compress_prompt(prompt_text)
    return compressed, compressed != prompt_text


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
    while True:
        with _RATE_LIMIT_LOCK:
            now = time.time()
            sleep_for = max(0.0, LAST_REQUEST_TIME + MIN_REQUEST_INTERVAL - now)
            if sleep_for <= 0:
                LAST_REQUEST_TIME = now
                return

        # Sleep outside the lock to reduce contention for concurrent callers.
        time.sleep(sleep_for + random.uniform(0.05, 0.2))


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
    configured = os.environ.get("GEMINI_MODELS", "").strip()
    if configured:
        candidates = [value for value in configured.split(",")]
    else:
        candidates = list(DEFAULT_GEMINI_MODELS)

    gemini_candidates = [value for value in candidates if value.strip().lower().startswith("gemini")]
    if not gemini_candidates:
        gemini_candidates = list(DEFAULT_GEMINI_MODELS)

    models = _dedupe_models([*gemini_candidates, *DEFAULT_GEMINI_MODELS])
    primary_model = os.environ.get("GEMINI_PRIMARY_MODEL", "gemini-2.0-flash").strip()
    if primary_model and primary_model in models:
        models = [primary_model, *[m for m in models if m != primary_model]]
    return models


def _get_openai_env_models() -> list[str]:
    configured = (
        os.environ.get("OPENAI_MODELS", "").strip()
        or os.environ.get("CHATGPT_MODELS", "").strip()
        or os.environ.get("OPENAI_MODEL", "").strip()
    )
    if configured:
        candidates = [value for value in configured.split(",")]
    else:
        candidates = list(DEFAULT_OPENAI_MODELS)
    models = _dedupe_models([*candidates, *DEFAULT_OPENAI_MODELS])
    return models or list(DEFAULT_OPENAI_MODELS)


def _get_retry_delays_from_env() -> tuple[int, ...]:
    configured = os.environ.get("LLM_RETRY_DELAYS", "").strip()
    if not configured:
        return DEFAULT_RETRY_DELAYS
    parsed: list[int] = []
    for item in configured.split(","):
        token = item.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError:
            continue
        if value > 0:
            parsed.append(value)
    return tuple(parsed) if parsed else DEFAULT_RETRY_DELAYS


def _load_env_if_present() -> None:
    if load_dotenv is None:
        return
    cwd = os.getcwd()
    candidates = [
        os.path.join(cwd, ".env"),
        os.path.join(os.path.dirname(cwd), ".env"),
    ]
    for env_path in candidates:
        if os.path.isfile(env_path):
            load_dotenv(dotenv_path=env_path, override=False)


def _looks_like_model_unavailable(status_code: int, response_text: str) -> bool:
    lower_error = response_text.lower()
    return status_code in {400, 404} and "model" in lower_error and (
        "not found" in lower_error
        or "unsupported" in lower_error
        or "unavailable" in lower_error
    )


@dataclass(frozen=True)
class GeminiConfig:
    provider: str
    api_key: str
    endpoint_base: str
    models: tuple[str, ...]
    max_output_tokens: int
    request_timeout_seconds: int
    retry_delays: tuple[int, ...]

    @classmethod
    def from_env(cls) -> "GeminiConfig":
        _load_env_if_present()
        gemini_key = (
            os.environ.get("GEMINI_API_KEY", "").strip()
            or os.environ.get("GOOGLE_API_KEY", "").strip()
        )
        openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        openai_key = (
            os.environ.get("CHATGPT_API_KEY", "").strip()
            or os.environ.get("OPENAI_API_KEY", "").strip()
        )

        if gemini_key or openrouter_key:
            provider = "gemini"
            resolved_api_key = gemini_key or openrouter_key
            endpoint_base = (
                os.environ.get("GEMINI_ENDPOINT_BASE", DEFAULT_GEMINI_ENDPOINT_BASE).strip()
                or DEFAULT_GEMINI_ENDPOINT_BASE
            )
            models = tuple(_get_env_models())
        elif openai_key:
            provider = "openai"
            resolved_api_key = openai_key
            endpoint_base = (
                os.environ.get("OPENAI_ENDPOINT_BASE", DEFAULT_OPENAI_ENDPOINT_BASE).strip()
                or DEFAULT_OPENAI_ENDPOINT_BASE
            )
            models = tuple(_get_openai_env_models())
        else:
            provider = "gemini"
            resolved_api_key = api_key
            endpoint_base = (
                os.environ.get("GEMINI_ENDPOINT_BASE", DEFAULT_GEMINI_ENDPOINT_BASE).strip()
                or DEFAULT_GEMINI_ENDPOINT_BASE
            )
            models = tuple(_get_env_models())

        return cls(
            provider=provider,
            api_key=resolved_api_key,
            endpoint_base=endpoint_base,
            models=models,
            max_output_tokens=(
                _parse_int_env("GEMINI_MAX_OUTPUT_TOKENS", 0)
                or _parse_int_env("OPENROUTER_MAX_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS)
            ),
            request_timeout_seconds=(
                _parse_int_env("GEMINI_REQUEST_TIMEOUT_SECONDS", 0)
                or _parse_int_env("OPENROUTER_REQUEST_TIMEOUT_SECONDS", DEFAULT_REQUEST_TIMEOUT_SECONDS)
            ),
            retry_delays=_get_retry_delays_from_env(),
        )


class GeminiBridge:
    """Bridge for Gemini/OpenAI-compatible chat completion with Langfuse tracing."""

    def __init__(
        self,
        config: GeminiConfig | None = None,
        log_fn: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config or GeminiConfig.from_env()
        self._log_fn = log_fn
        self.tracker = LangfuseTracker(log_fn=log_fn)
        self._cache_enabled = os.environ.get("GEMINI_ENABLE_CACHE", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._cache_version = os.environ.get("GEMINI_CACHE_VERSION", DEFAULT_GEMINI_CACHE_VERSION).strip() or DEFAULT_GEMINI_CACHE_VERSION
        self._cache_path = os.path.join(os.getcwd(), _GEMINI_CACHE_FILENAME)
        self._response_cache: dict[str, str] = self._load_cache()

    @classmethod
    def from_env(
        cls,
        log_fn: Callable[[str], None] | None = None,
    ) -> "GeminiBridge":
        """Create a bridge using environment-based configuration."""
        return cls(GeminiConfig.from_env(), log_fn=log_fn)

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

    def _endpoint_for_model(self, model_name: str) -> str:
        if self.config.provider == "openai":
            return f"{self.config.endpoint_base}/chat/completions"
        return f"{self.config.endpoint_base}/models/{model_name}:generateContent?key={self.config.api_key}"

    def _build_payload(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
    ) -> dict[str, object]:
        if self.config.provider == "openai":
            return {
                "model": self.config.models[0],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": self.config.max_output_tokens,
            }
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

    def _cache_key(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        digest_source = json.dumps(
            {
                "cache_version": self._cache_version,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": temperature,
                "models": list(self.config.models),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
        return hashlib.sha256(digest_source.encode("utf-8")).hexdigest()

    def _load_cache(self) -> dict[str, str]:
        if not self._cache_enabled:
            return {}
        if not os.path.isfile(self._cache_path):
            return {}
        try:
            with open(self._cache_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return {
                    str(key): str(value)
                    for key, value in data.items()
                    if isinstance(key, str) and isinstance(value, str)
                }
        except Exception:
            return {}
        return {}

    def _save_cache(self) -> None:
        if not self._cache_enabled:
            return
        try:
            with open(self._cache_path, "w", encoding="utf-8") as fh:
                json.dump(self._response_cache, fh, ensure_ascii=True, indent=2)
        except Exception:
            pass

    def _extract_text(self, data: dict[str, Any]) -> tuple[str, str, int | None, int | None, int | None]:
        if self.config.provider == "openai":
            choices = data.get("choices") or []
            choice = choices[0] if choices else {}
            message = choice.get("message") or {}
            text = str(message.get("content") or "")
            finish_reason = str(choice.get("finish_reason") or "")
            usage = data.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")
            return text, finish_reason, prompt_tokens, completion_tokens, total_tokens

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
        self._log(
            f"Sending request to {self.config.provider} model={model_name}, "
            f"prompt_chars={len(str(payload))}"
        )
        headers = {"Content-Type": "application/json"}
        if self.config.provider == "openai":
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return requests.post(
            self._endpoint_for_model(model_name),
            headers=headers,
            json=payload,
            timeout=(10, max(self.config.request_timeout_seconds, 120)),
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
                        f"{self.config.provider.upper()} network error for model '{model_name}' during {purpose}: {exc}. "
                        f"Retrying in {delay}s... (attempt {attempt_index}/{max_attempts})"
                    )
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    f"{self.config.provider.upper()} network failure for model '{model_name}' during {purpose}: {exc!r}"
                ) from exc

            if response.status_code == 200:
                data = response.json()
                text, finish_reason, prompt_tokens, completion_tokens, total_tokens = self._extract_text(data)
                self._log(
                    f"{self.config.provider.upper()} response received for model '{model_name}' ({purpose}): "
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
                        f"{self.config.provider.upper()} {error_kind} for model '{model_name}' during {purpose}. "
                        f"Retrying in {wait_seconds}s... (attempt {attempt_index}/{max_attempts})"
                    )
                    time.sleep(wait_seconds)
                    continue

                if is_rate_limit:
                    raise RuntimeError(
                        "PROVIDER_QUOTA_EXHAUSTED::"
                        f"{self.config.provider}::{model_name}::{response.status_code}"
                    )
                raise RuntimeError(
                    f"{self.config.provider.upper()} request failed after retries for model '{model_name}' during {purpose}. "
                    f"Status {response.status_code}: {response.text}"
                )

            last_error_body = response.text
            if _looks_like_model_unavailable(response.status_code, response.text):
                raise RuntimeError(
                    f"MODEL_UNAVAILABLE::{response.status_code}::{response.text}"
                )

            raise RuntimeError(
                f"{self.config.provider.upper()} returned {response.status_code} for model '{model_name}': {response.text}"
            )

        raise RuntimeError(
            f"{self.config.provider.upper()} request failed for model '{model_name}' during {purpose}. "
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
            if finish_reason not in {"MAX_TOKENS", "length", "max_tokens"}:
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
            if self.config.provider == "openai":
                payload["model"] = model_name
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
        temperature: float = 0.2,
        start_new_trace: bool = False,
    ) -> str:
        """Request a completion from configured models and return cleaned C++ output.

        Tracing strategy: this LLM call path records a single Langfuse generation
        with real input/output and token usage (no separate LLM span is created).
        """
        if not self.config.api_key:
            raise ValueError(
                "Provider API key is empty. Set GEMINI_API_KEY or GOOGLE_API_KEY, OPENROUTER_API_KEY, CHATGPT_API_KEY, or OPENAI_API_KEY."
            )

        if start_new_trace:
            self.start_modernization_trace(
                input_payload={
                    "operation": "modernization",
                    "model_candidates": list(self.config.models),
                    "system_prompt": system_prompt,
                    "provider": self.config.provider,
                }
            )
        else:
            self.tracker.get_or_create_trace(name="CPP-Modernization")

        prompt_len = len(user_prompt)
        if prompt_len > LARGE_PROMPT_WARNING_CHARS:
            self._log(
                f"Warning: very large prompt ({prompt_len} chars). Reliability may degrade."
            )

        prompt_for_request, compression_applied = _maybe_compress_prompt(user_prompt)
        if compression_applied:
            self._log(
                f"Prompt compression applied: {len(user_prompt)} -> {len(prompt_for_request)} chars. "
                "Comments/extra blank lines were reduced."
            )
        if len(prompt_for_request) > MAX_PROMPT_CHARS:
            self._log(
                f"Prompt exceeds {MAX_PROMPT_CHARS} chars after compression; truncating to tail."
            )
            prompt_for_request = prompt_for_request[-MAX_PROMPT_CHARS:]

        cache_key = self._cache_key(system_prompt, prompt_for_request, temperature)
        cached = self._response_cache.get(cache_key) if self._cache_enabled else None
        if cached is not None:
            self._log("Gemini cache hit; returning cached completion.")
            return cached

        enforce_full_response = _expects_large_code_response(prompt_for_request)
        last_exc: Exception | None = None
        last_error_text = ""
        model_errors: list[str] = []

        for model_name in self.config.models:
            payload = self._build_payload(
                system_prompt=system_prompt,
                user_prompt=prompt_for_request,
                temperature=temperature,
            )
            if self.config.provider == "openai":
                payload["model"] = model_name

            self._log(
                f"{self.config.provider.upper()} request starting: model='{model_name}', "
                f"prompt_chars={len(prompt_for_request)}"
            )

            try:
                data, finish_reason, prompt_tokens, completion_tokens, total_tokens = self._request_with_retries(
                    model_name=model_name,
                    payload=payload,
                    purpose="main-attempt",
                )
                content, extracted_finish_reason, _, _, _ = self._extract_text(data)
                finish_reason = extracted_finish_reason or finish_reason

                combined_content = self._complete_with_continuations(
                    model_name=model_name,
                    system_prompt=system_prompt,
                    base_user_prompt=prompt_for_request,
                    first_text=content,
                    first_finish_reason=finish_reason,
                    temperature=temperature,
                )
                cleaned_content = _clean_cpp_response_text(combined_content)

                self.tracker.capture_generation(
                    model=model_name,
                    prompt=prompt_for_request,
                    response_text=cleaned_content,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    system_prompt=system_prompt,
                )

                if len(cleaned_content.strip()) < 10:
                    raise RuntimeError("Model returned empty or invalid code")
                if cleaned_content.strip() == prompt_for_request.strip():
                    raise RuntimeError("Model echoed prompt without modernization")

                if enforce_full_response and len(cleaned_content) < MIN_LARGE_CODE_RESPONSE_CHARS:
                    raise RuntimeError(
                        f"Model '{model_name}' response too short ({len(cleaned_content)} chars) "
                        "for a large-code request."
                    )

                if enforce_full_response and len(cleaned_content) < SHORT_CODE_RESPONSE_THRESHOLD:
                    raise RuntimeError(
                        f"Model '{model_name}' response below short-response threshold "
                        f"({len(cleaned_content)} < {SHORT_CODE_RESPONSE_THRESHOLD})."
                    )

                if enforce_full_response and TRUNCATION_GUARD_ENABLED and not _passes_truncation_guard(cleaned_content, True):
                    raise RuntimeError(
                        f"Model '{model_name}' response failed lightweight truncation guard "
                        "(size/brace/end-check)."
                    )

                self._log(
                    f"{self.config.provider.upper()} completion accepted: model='{model_name}', "
                    f"chars={len(cleaned_content)}"
                )
                if self._cache_enabled:
                    self._response_cache[cache_key] = cleaned_content
                    self._save_cache()
                return cleaned_content

            except RuntimeError as exc:
                message = str(exc)
                lowered = message.lower()

                if message.startswith("PROVIDER_QUOTA_EXHAUSTED::") or "quota" in lowered or "429" in lowered:
                    self.mark_trace_error(
                        message="PROVIDER_QUOTA_EXHAUSTED",
                        details={"provider": self.config.provider, "model": model_name, "error": message},
                    )
                    raise RuntimeError("PROVIDER_QUOTA_EXHAUSTED") from exc

                if message.startswith("MODEL_UNAVAILABLE::"):
                    model_error = f"{model_name}: unavailable ({message})"
                    model_errors.append(model_error)
                    self._log(
                        f"Model '{model_name}' unavailable or blocked. Trying next fallback model..."
                    )
                    last_error_text = message
                    continue

                model_error = f"{model_name}: {message}"
                model_errors.append(model_error)
                last_exc = exc
                last_error_text = message
                self._log(
                    f"Model '{model_name}' failed with runtime error: {message}. Trying next fallback model..."
                )
                continue

        self.mark_trace_error(
            message="GEMINI_CALL_FAILED_ALL_MODELS",
            details={
                "provider": self.config.provider,
                "models": list(self.config.models),
                "errors": model_errors,
            },
        )
        raise RuntimeError(
            "Gemini call failed across all configured models. "
            f"Last error: {last_exc!r}. Last response body: {last_error_text}. "
            f"Per-model errors: {model_errors}"
        )

    def check_health(self) -> tuple[bool, str]:
        """Validate provider connectivity and optionally run a tiny completion probe."""
        if not self.config.api_key:
            return False, "API key is empty. Set GEMINI_API_KEY or GOOGLE_API_KEY, OPENROUTER_API_KEY, CHATGPT_API_KEY, or OPENAI_API_KEY."

        try:
            if self.config.provider == "openai":
                response = requests.get(
                    f"{self.config.endpoint_base}/models",
                    headers={"Authorization": f"Bearer {self.config.api_key}"},
                    timeout=10,
                )
            else:
                response = requests.get(
                    f"{self.config.endpoint_base}/models?key={self.config.api_key}",
                    timeout=10,
                )
        except requests.exceptions.RequestException as exc:
            return False, f"Could not reach {self.config.provider}: {exc}"

        if response.status_code == 200:
            if not ENABLE_HEALTH_PROBE:
                return True, f"{self.config.provider.upper()} CONNECTION VERIFIED"

            # Optional shallow inference probe to catch model-level permission/availability issues.
            model_name = self.config.models[0] if self.config.models else ""
            if not model_name:
                return False, f"{self.config.provider.upper()} connected but no models configured."

            try:
                probe_payload = self._build_payload(
                    system_prompt="Respond with OK.",
                    user_prompt="health-check",
                    temperature=0.0,
                )
                if self.config.provider == "openai":
                    probe_payload["model"] = model_name
                    probe_payload["max_tokens"] = 1
                else:
                    raw_generation_cfg = probe_payload.get("generationConfig")
                    if isinstance(raw_generation_cfg, dict):
                        generation_cfg: dict[str, object] = dict(raw_generation_cfg)
                    else:
                        generation_cfg = {}
                    generation_cfg["maxOutputTokens"] = 1
                    probe_payload["generationConfig"] = generation_cfg

                probe_response = requests.post(
                    self._endpoint_for_model(model_name),
                    headers=(
                        {"Content-Type": "application/json", "Authorization": f"Bearer {self.config.api_key}"}
                        if self.config.provider == "openai"
                        else {"Content-Type": "application/json"}
                    ),
                    json=probe_payload,
                    timeout=DEFAULT_HEALTH_PROBE_TIMEOUT_SECONDS,
                )
                if probe_response.status_code != 200:
                    return (
                        False,
                        f"{self.config.provider.upper()} model probe failed for '{model_name}' "
                        f"with status {probe_response.status_code}.",
                    )
            except requests.exceptions.RequestException as exc:
                return False, f"{self.config.provider.upper()} model probe failed: {exc}"

            return True, f"{self.config.provider.upper()} CONNECTION + MODEL PROBE VERIFIED"

        return False, f"{self.config.provider} returned status {response.status_code}. Check your API key."