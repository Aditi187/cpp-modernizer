from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence
import hashlib
import json
import os
import re
import threading
import time

from core.llm_shared import LangfuseTracker, expects_large_code_response, parse_int_env
from core.llm_client import (
    AgentRouterClient,
    InvalidAPIKeyError,
    LLMNetworkError,
    ProviderQuotaExhaustedError as ClientProviderQuotaExhaustedError,
)
from core.retry_policy import RateLimiter, parse_retry_delays_from_env
from core.usage_tracker import (
    is_near_monthly_cost_limit,
    monthly_cost_limit_usd,
    record_token_usage,
    reorder_models_by_cost,
)

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency at runtime
    load_dotenv = None


api_key: str = ""

DEFAULT_OPENAI_ENDPOINT_BASE = "https://api.agentrouter.org/v1"
DEFAULT_OPENAI_MODELS: tuple[str, ...] = ("gpt-5.3-codex-xhigh",)
DEFAULT_RETRY_DELAYS: tuple[int, ...] = (3, 6, 12, 24)
DEFAULT_MAX_OUTPUT_TOKENS = 8192
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120
DEFAULT_HEALTH_PROBE_TIMEOUT_SECONDS = 10
DEFAULT_SHORT_CODE_RESPONSE_THRESHOLD = 500
DEFAULT_OPENAI_CACHE_VERSION = "v1"
DEFAULT_CACHE_TTL_SECONDS = 7 * 86400
MAX_PROMPT_CHARS = 100_000
DEFAULT_REQUESTS_PER_MINUTE = 40
DEFAULT_INTER_REQUEST_DELAY_SECONDS = 1.5

CPP_MODERNIZATION_SYSTEM_PROMPT = (
    "You are a C++ code modernization engine.\n\n"
    "Your task is to modernize the given C++ code strictly to the C++17 standard.\n\n"
    "STRICT RULES (MUST FOLLOW):\n"
    "1) Target standard: C++17 only. Do not use C++20/C++23 features.\n"
    "2) Forbidden: concepts, std::ranges, coroutines, std::format, std::span, modules.\n"
    "3) Allowed: auto, range-based for loops, smart pointers, std::optional/std::variant, structured bindings, C++17 constexpr, std::filesystem, improved STL usage.\n"
    "4) Preservation: preserve behavior; keep required includes.\n"
    "4.1) You MAY refactor function signatures and ownership-related interfaces when it improves safety and modern C++17 usage; update in-file call sites accordingly.\n"
    "4.2) Replace manual memory management with RAII containers/types when possible (malloc/free/new/delete -> std::unique_ptr, std::string, std::vector, stack objects).\n"
    "4.3) Keep C-style pointer ownership only when externally constrained by ABI visible in the file.\n"
    "5) Quality: compile under C++17; prioritize deep modernization (RAII, smart ownership, string safety) over minimal diffs.\n"
    "6) If C++20/23 is needed, replace with equivalent C++17-compatible approach.\n\n"
    "Output rules: return ONLY final C++ code; no explanations; no markdown fences.\n"
)

_CODE_FENCE_RE = re.compile(r"```(?:\w*)\n(.*?)```", re.DOTALL)
_CPP_FENCE_RE = re.compile(r"```(?:cpp|c\+\+|cc|cxx|hpp|hxx|h)?\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)


class ProviderQuotaExhaustedError(RuntimeError):
    """Raised when provider quota/rate-limit is exhausted after retries."""


class ModelUnavailableError(RuntimeError):
    """Raised when a configured model is unavailable for requests."""


@dataclass(frozen=True)
class OpenAIConfig:
    provider: str
    api_key: str
    endpoint_base: str
    models: tuple[str, ...]
    max_output_tokens: int
    request_timeout_seconds: int
    retry_delays: tuple[int, ...]

    @classmethod
    def from_env(cls) -> "OpenAIConfig":
        _load_env_if_present()
        resolved_api_key = os.environ.get("API_KEY", "").strip() or api_key
        endpoint_base = (
            os.environ.get("OPENAI_ENDPOINT_BASE", DEFAULT_OPENAI_ENDPOINT_BASE).strip()
            or DEFAULT_OPENAI_ENDPOINT_BASE
        )
        models = tuple(_get_openai_env_models())

        if is_near_monthly_cost_limit():
            models = tuple(reorder_models_by_cost(list(models)))

        return cls(
            provider="openai",
            api_key=resolved_api_key,
            endpoint_base=endpoint_base,
            models=models,
            max_output_tokens=parse_int_env("OPENAI_MAX_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS),
            request_timeout_seconds=parse_int_env("OPENAI_REQUEST_TIMEOUT_SECONDS", DEFAULT_REQUEST_TIMEOUT_SECONDS),
            retry_delays=_get_retry_delays_from_env(),
        )


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
    parsed = parse_retry_delays_from_env("LLM_RETRY_DELAYS", DEFAULT_RETRY_DELAYS)
    return tuple(int(max(1.0, value)) for value in parsed)


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


def _strip_assistant_prefixes(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^\s*sure[^\n]*\n", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*(assistant|model|ai)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*(here is|here's|below is)\b.*?\n", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _extract_code_text(text: str) -> str:
    normalized = _strip_assistant_prefixes(text)
    cpp_match = _CPP_FENCE_RE.search(normalized)
    if cpp_match:
        return cpp_match.group(1).strip()
    fence_match = _CODE_FENCE_RE.search(normalized)
    if fence_match:
        return fence_match.group(1).strip()
    return normalized.strip()


def _clean_cpp_response_text(text: str) -> str:
    return _extract_code_text(text)


class OpenAIBridge:
    """Bridge for OpenAI-compatible chat completion with Langfuse tracing."""

    def __init__(
        self,
        config: OpenAIConfig | None = None,
        log_fn: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config or OpenAIConfig.from_env()
        self._llm_client = AgentRouterClient.from_env()
        self._log_fn = log_fn
        self.tracker = LangfuseTracker()
        self._active_trace: Any = None
        self._success_delay_seconds = self._read_float_env(
            "OPENAI_INTER_REQUEST_DELAY_SECONDS",
            DEFAULT_INTER_REQUEST_DELAY_SECONDS,
            minimum=0.0,
        )
        requests_per_minute = self._read_float_env(
            "OPENAI_MAX_CALLS_PER_MINUTE",
            DEFAULT_REQUESTS_PER_MINUTE,
            minimum=1.0,
        )
        self._rate_limiter = RateLimiter(requests_per_minute)
        _disable_cache = os.environ.get("DISABLE_CACHE", "0").strip().lower() in {"1", "true", "yes", "on"}
        _use_cache_global = os.environ.get("USE_CACHE", "1").strip().lower() not in {"0", "false", "no", "off"}
        _openai_cache_enabled = os.environ.get("OPENAI_ENABLE_CACHE", "1").strip().lower() in {"1", "true", "yes", "on"}
        self._cache_enabled = (not _disable_cache) and _use_cache_global and _openai_cache_enabled
        self._cache_version = (
            os.environ.get("CACHE_VERSION", "")
            or os.environ.get("OPENAI_CACHE_VERSION", DEFAULT_OPENAI_CACHE_VERSION)
        ).strip() or DEFAULT_OPENAI_CACHE_VERSION
        try:
            _raw_ttl = os.environ.get("CACHE_TTL_SECONDS", str(DEFAULT_CACHE_TTL_SECONDS)).strip()
            self._cache_ttl_seconds: int = max(0, int(_raw_ttl)) if _raw_ttl else DEFAULT_CACHE_TTL_SECONDS
        except ValueError:
            self._cache_ttl_seconds = DEFAULT_CACHE_TTL_SECONDS
        self._cache_path = os.path.join(os.getcwd(), ".openai_cache.json")
        self._cache_lock = threading.Lock()
        self._response_cache: dict[str, dict] = self._load_cache()

    @staticmethod
    def _read_float_env(name: str, default: float, minimum: float = 0.0) -> float:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return max(minimum, float(default))
        try:
            parsed = float(raw)
        except ValueError:
            return max(minimum, float(default))
        return max(minimum, parsed)

    @classmethod
    def from_env(cls, log_fn: Callable[[str], None] | None = None) -> "OpenAIBridge":
        return cls(OpenAIConfig.from_env(), log_fn=log_fn)

    def _log(self, message: str) -> None:
        if self._log_fn is not None:
            self._log_fn(message)

    def start_modernization_trace(self, input_payload: Any = None) -> Any:
        trace = self.tracker.create_trace(name="CPP-Modernization", input_payload=input_payload)
        self._active_trace = trace
        return trace

    def start_span(self, name: str, input_payload: Any = None) -> Any:
        trace = self._active_trace or self.start_modernization_trace()
        return self.tracker.start_span(trace=trace, name=name, input_payload=input_payload)

    def end_span(self, span: Any, output_payload: Any = None, level: str | None = None, input_payload: Any = None) -> None:
        self.tracker.end_span(
            span=span,
            output_payload=output_payload,
            level=level,
            input_payload=input_payload,
        )

    def mark_trace_error(self, message: str, details: Any = None) -> None:
        self.tracker.mark_error(trace=self._active_trace, message=message, details=details)

    def _endpoint_for_model(self, _model_name: str) -> str:
        return f"{self.config.endpoint_base}/chat/completions"

    def _build_payload(self, model_name: str, system_prompt: str, user_prompt: str, temperature: float) -> dict[str, object]:
        return {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": self.config.max_output_tokens,
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

    def _load_cache(self) -> dict[str, dict]:
        if not self._cache_enabled:
            return {}
        if not os.path.isfile(self._cache_path):
            return {}
        try:
            with open(self._cache_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        now = time.time()
        result: dict[str, dict] = {}
        for key, entry in data.items():
            if not isinstance(key, str):
                continue
            if isinstance(entry, str):
                entry = {"v": entry, "ts": 0.0}
            if not isinstance(entry, dict):
                continue
            cached_value = entry.get("v")
            if not isinstance(cached_value, str) or not cached_value:
                continue
            if self._cache_ttl_seconds > 0:
                try:
                    entry_ts = float(entry.get("ts") or 0.0)
                except (TypeError, ValueError):
                    entry_ts = 0.0
                if entry_ts > 0 and now - entry_ts > self._cache_ttl_seconds:
                    continue
            result[key] = {"v": cached_value, "ts": float(entry.get("ts") or 0.0)}
        return result

    def _save_cache(self) -> None:
        if not self._cache_enabled:
            return
        try:
            with self._cache_lock:
                snapshot = dict(self._response_cache)
            with open(self._cache_path, "w", encoding="utf-8") as fh:
                json.dump(snapshot, fh, ensure_ascii=True, indent=2)
        except Exception:
            pass

    def _extract_text(self, data: dict[str, Any]) -> tuple[str, str, int | None, int | None, int | None]:
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

    def _request_with_retries(
        self,
        model_name: str,
        payload: dict[str, object],
        purpose: str,
        timeout_seconds: int,
    ) -> tuple[dict[str, Any], str, int | None, int | None, int | None]:
        messages = payload.get("messages")
        system_prompt = ""
        user_prompt = ""
        if isinstance(messages, list):
            if len(messages) > 0 and isinstance(messages[0], dict):
                system_prompt = str(messages[0].get("content") or "")
            if len(messages) > 1 and isinstance(messages[1], dict):
                user_prompt = str(messages[1].get("content") or "")
        raw_temperature = payload.get("temperature", 0.2)
        try:
            temperature = float(str(raw_temperature))
        except (TypeError, ValueError):
            temperature = 0.2
        raw_max_tokens = payload.get("max_tokens", self.config.max_output_tokens)
        try:
            max_tokens = int(str(raw_max_tokens))
        except (TypeError, ValueError):
            max_tokens = self.config.max_output_tokens

        try:
            self._rate_limiter.wait()
            data = self._llm_client.chat_completion(
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                max_tokens=max_tokens,
                use_cache=self._cache_enabled,
            )
            text, finish_reason, prompt_tokens, completion_tokens, total_tokens = self._extract_text(data)
            self._log(
                f"OPENAI response received: model={model_name}, chars={len(text.strip())}, finish_reason={finish_reason or 'UNKNOWN'}"
            )
            return data, finish_reason, prompt_tokens, completion_tokens, total_tokens
        except ClientProviderQuotaExhaustedError as exc:
            raise ProviderQuotaExhaustedError(str(exc)) from exc
        except InvalidAPIKeyError as exc:
            raise RuntimeError(f"OPENAI returned status 401. Check your API key. Details: {exc}") from exc
        except LLMNetworkError as exc:
            raise RuntimeError(f"OPENAI network failure during {purpose}: {exc!r}") from exc

    def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        start_new_trace: bool = False,
        timeout_seconds: int | None = None,
    ) -> str:
        if not self.config.api_key:
            raise ValueError("OPENAI API key is empty. Set API_KEY.")

        if start_new_trace:
            self.start_modernization_trace(
                input_payload={
                    "operation": "modernization",
                    "model_candidates": list(self.config.models),
                    "provider": self.config.provider,
                }
            )
        elif self._active_trace is None:
            self.start_modernization_trace(
                input_payload={
                    "operation": "modernization",
                    "model_candidates": list(self.config.models),
                    "provider": self.config.provider,
                }
            )

        prompt_for_request = user_prompt[-MAX_PROMPT_CHARS:]
        cache_key = self._cache_key(system_prompt, prompt_for_request, temperature)
        if self._cache_enabled:
            with self._cache_lock:
                cached_entry = self._response_cache.get(cache_key)
            if isinstance(cached_entry, dict) and cached_entry.get("v"):
                return str(cached_entry["v"])

        enforce_full_response = expects_large_code_response(prompt_for_request)
        effective_timeout = int(timeout_seconds or self.config.request_timeout_seconds)

        model_errors: list[str] = []
        trace = self._active_trace
        for model_name in self.config.models:
            generation = self.tracker.create_generation(
                trace=trace,
                name=self.config.provider,
                model=model_name,
                input_data=user_prompt,
                metadata={"provider": self.config.provider, "system_prompt": system_prompt},
            )
            payload = self._build_payload(model_name, system_prompt, prompt_for_request, temperature)
            try:
                data, finish_reason, prompt_tokens, completion_tokens, total_tokens = self._request_with_retries(
                    model_name=model_name,
                    payload=payload,
                    purpose="main-attempt",
                    timeout_seconds=effective_timeout,
                )
                content, _, _, _, _ = self._extract_text(data)
                cleaned_content = _clean_cpp_response_text(content)
                if not cleaned_content.strip():
                    raise RuntimeError("Model returned empty response")
                if enforce_full_response and len(cleaned_content) < DEFAULT_SHORT_CODE_RESPONSE_THRESHOLD:
                    raise RuntimeError("Model returned short response for large-code request")

                self.tracker.finalize_generation(
                    generation,
                    output=cleaned_content,
                    model=model_name,
                    usage_details={
                        "prompt_tokens": int(prompt_tokens or 0),
                        "completion_tokens": int(completion_tokens or 0),
                        "total_tokens": int(total_tokens or 0),
                    },
                    metadata={"finish_reason": finish_reason},
                )

                if self._cache_enabled:
                    with self._cache_lock:
                        self._response_cache[cache_key] = {"v": cleaned_content, "ts": time.time()}
                    self._save_cache()

                call_cost, month_cost = record_token_usage(
                    provider=self.config.provider,
                    model_name=model_name,
                    prompt_tokens=int(prompt_tokens or 0),
                    completion_tokens=int(completion_tokens or 0),
                )
                limit = monthly_cost_limit_usd()
                if limit > 0 and month_cost >= (limit * 0.8):
                    self._log(
                        f"Warning: monthly LLM cost nearing limit (${month_cost:.4f}/${limit:.4f}); latest call cost ${call_cost:.4f}."
                    )
                if self._success_delay_seconds > 0:
                    time.sleep(self._success_delay_seconds)
                return cleaned_content

            except ProviderQuotaExhaustedError as exc:
                self.tracker.finalize_generation(generation, level="ERROR", status_message=str(exc))
                self.mark_trace_error(
                    message="PROVIDER_QUOTA_EXHAUSTED",
                    details={"provider": self.config.provider, "model": model_name, "error": str(exc)},
                )
                raise RuntimeError("PROVIDER_QUOTA_EXHAUSTED") from exc
            except ModelUnavailableError as exc:
                self.tracker.finalize_generation(generation, level="ERROR", status_message=str(exc))
                model_errors.append(f"{model_name}: unavailable ({exc})")
                continue
            except RuntimeError as exc:
                self.tracker.finalize_generation(generation, level="ERROR", status_message=str(exc))
                model_errors.append(f"{model_name}: {exc}")
                continue

        self.mark_trace_error(
            message="OPENAI_CALL_FAILED_ALL_MODELS",
            details={"provider": self.config.provider, "models": list(self.config.models), "errors": model_errors},
        )
        raise RuntimeError(f"OpenAI call failed across all configured models: {model_errors}")

    def check_health(self) -> tuple[bool, str]:
        return self._llm_client.check_health()
