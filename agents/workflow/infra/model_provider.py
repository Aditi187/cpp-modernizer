from __future__ import annotations

"""
Multi-model LLM bridge for the Air-Gapped C++ Modernization Engine.

Role routing (from .env):
    analyze  → DeepSeek-V3   (deep reasoning, thinking mode)
    modernize → Llama-3.3-70B (code rewriting)
    fixer    → Llama-3.3-70B  (small compiler-error fixes)

Falls back to RuleModernizer if LLM is unavailable / returns invalid code.
"""
import logging
import os
import re
import time
import random
from typing import Optional, Tuple

from openai import OpenAI, RateLimitError as OpenAI_RateLimitError

from core.rule_modernizer import RuleModernizer
from agents.workflow.context import WorkflowContext

logger = logging.getLogger(__name__)

from agents.workflow.infra.code_utils import extract_code, _CODE_FENCE_RE

class ProviderError(Exception): pass
class RateLimitError(ProviderError): pass
class ProviderQuotaExhaustedError(ProviderError): pass
class ModelUnavailableError(ProviderError): pass
class ContextExhaustedError(ProviderError): pass


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()

def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key) or default)
    except ValueError:
        return default

def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key) or default)
    except ValueError:
        return default


# Keys that look real but are placeholders — treat as missing
_PLACEHOLDER_KEYS = {
    "your_api_key_here",
    "nvapi-xxxx",
    "nvapi-xxx",
    "sk-...",
    "sk-placeholder",
    "dummy",
    "test",
    "changeme",
    "",
}


class _RoleConfig:
    """Holds the model/endpoint/key/params for one role."""

    def __init__(self, prefix: str, fallback_key: str, fallback_url: str, fallback_model: str):
        raw_key = _env(f"{prefix}_API_KEY") or _env(fallback_key)
        # Treat placeholder strings exactly like a missing key
        self.api_key   = raw_key if raw_key not in _PLACEHOLDER_KEYS else ""
        self.base_url  = _env(f"{prefix}_ENDPOINT_BASE") or fallback_url
        self.model     = _env(f"{prefix}_MODEL")     or fallback_model
        self.temp      = _env_float(f"{prefix}_TEMPERATURE", 0.1)
        self.top_p     = _env_float(f"{prefix}_TOP_P", 0.85)
        self.max_tokens = _env_int(f"{prefix}_MAX_TOKENS", 4096)
        self.thinking  = _env(f"{prefix}_ENABLE_THINKING") in ("1", "true", "yes")

    def client(self) -> OpenAI:
        import httpx
        timeout_val = _env_float("LLM_TIMEOUT", 120.0)
        return OpenAI(
            api_key=self.api_key, 
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_val, connect=10.0)
        )


_FALLBACK_URL   = "http://localhost:11434/v1"
_FALLBACK_KEY   = _env("API_KEY") or _env("OPENAI_API_KEY")

def _get_role_config(role: str) -> _RoleConfig:
    fallback_model = _env("OPENAI_MODELS", "qwen2.5-coder:7b")
    configs = {
        "analyzer":  _RoleConfig("ANALYZER",   "API_KEY", _FALLBACK_URL, _env("ANALYZER_MODEL", fallback_model)),
        "modernizer": _RoleConfig("MODERNIZER", "API_KEY", _FALLBACK_URL, _env("MODERNIZER_MODEL", fallback_model)),
        "fixer":     _RoleConfig("FIXER",      "API_KEY", _FALLBACK_URL, _env("FIXER_MODEL", fallback_model)),
        "planner":   _RoleConfig("PLANNER",    "API_KEY", _FALLBACK_URL, _env("PLANNER_MODEL", fallback_model)),
    }
    return configs.get(role, _RoleConfig("OPENAI", "API_KEY", _FALLBACK_URL, fallback_model))


def _with_retry(fn, max_attempts: int = 3, base_wait: float = 5.0):
    last_err = None
    # Optional inter-call courtesy delay (default: 0). Set LLM_SUCCESS_DELAY=5
    # in .env only if you're hitting sustained rate limits across many files.
    success_delay = _env_float("LLM_SUCCESS_DELAY", 0.0)
    
    import httpx
    import openai
    
    for attempt in range(max_attempts):
        try:
            result = fn()
            if success_delay > 0:
                time.sleep(success_delay)
            return result
        except OpenAI_RateLimitError as e:
            last_err = e
            # Exponential backoff capped at 120s
            wait = min((2 ** attempt) * base_wait, 120.0) + random.uniform(1.0, 5.0)
            logger.warning("Rate-limited (attempt %d/%d). Waiting %.1fs...", attempt + 1, max_attempts, wait)
            time.sleep(wait)
        except (httpx.TimeoutException, httpx.ConnectError, openai.APIConnectionError, openai.APITimeoutError) as e:
            last_err = e
            if attempt == 0:
                logger.warning("Connection/Timeout error: %s. Quick retry in 2s...", e)
                time.sleep(2.0)
            else:
                raise
        except Exception:
            raise
    logger.error(f"Rate limit persisted after {max_attempts} attempts: {last_err}")
    logger.error(f"[ERROR] ModelProvider: Rate limit persisted after {max_attempts} attempts.")
    raise RateLimitError(f"Rate limit persisted after {max_attempts} attempts: {last_err}")


def _call_llm(role: str, system: str, user: str, context: Optional[WorkflowContext] = None, bypass_cache: bool = False) -> Optional[str]:
    cfg = _get_role_config(role)

    if not cfg.api_key:
        logger.warning("No API key configured for role=%s (checked env vars: %s_API_KEY and API_KEY). Falling back to RuleModernizer.", role, role.upper())
        logger.warning(f"[WARNING] ModelProvider: No API key for role={role} — LLM unavailable, using rule-based fallback.")
        return None

    # Caching support
    cache_key = None
    if context is not None and not bypass_cache:
        cache_key = f"{role}|{cfg.model}|{system}|{user}"
        cached = context.get_cached_llm_response(cache_key)
        if cached:
            logger.info(f"[CACHE] LLM cache hit for role={role}")
            if hasattr(context, "llm_calls_succeeded"):
                context.llm_calls_succeeded += 1
            return cached

    def _do_call():
        client = cfg.client()
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]

        # Use configured max_tokens
        max_tokens = cfg.max_tokens

        kwargs = dict(
            model=cfg.model,
            messages=messages,
            temperature=cfg.temp,
            top_p=cfg.top_p,
            max_tokens=max_tokens,
        )

        if cfg.thinking and "deepseek" in str(cfg.model).lower():
            try:
                resp = client.chat.completions.create(**kwargs, extra_body={"thinking": {"type": "enabled", "budget_tokens": 2048}})
            except TypeError:
                resp = client.chat.completions.create(**kwargs)
        else:
            resp = client.chat.completions.create(**kwargs)

        content = resp.choices[0].message.content or ""
        tokens = getattr(resp.usage, "total_tokens", 0) or 0
        return content, tokens

    try:
        raw, tokens = _with_retry(_do_call)
        logger.debug("LLM [%s/%s] returned %d chars, %d tokens.", role, cfg.model, len(raw), tokens)
        if context is not None:
            context.add_tokens(tokens)
            if hasattr(context, "llm_calls_succeeded"):
                context.llm_calls_succeeded += 1
            if cache_key:
                context.cache_llm_response(cache_key, raw)
        logger.info(f"[WORKING] ModelProvider: LLM call successful for role={role} (model={cfg.model}, tokens={tokens})")
        return raw
    except Exception as e:
        logger.error("LLM call failed for role=%s: %s", role, e)
        logger.error(f"[ERROR] ModelProvider: LLM call failed for role={role}: {e}")
        return None




def _is_valid_cpp(code: str) -> bool:
    s = code.strip()
    if not s or len(s) < 20:
        return False
    return ("{" in s or ";" in s) and s != "NO_CHANGE"


# ---------------------------------------------------------------------------
# Public ModelClient — drop-in replacement for the old stub
# ---------------------------------------------------------------------------

class ModelClient:
    """Routes LLM calls by role; falls back to RuleModernizer for code roles."""

    def __init__(self, context: WorkflowContext):
        self.context = context
        self._rules  = RuleModernizer()
        self._use_llm = getattr(context.config, "use_llm", True)
        if self._use_llm:
            cfg = _get_role_config("modernizer")
            if not cfg.api_key:
                logger.warning(
                    "⚠️  No valid API key found for the 'modernizer' role. "
                    "LLM is DISABLED — output will be rule-based only (NULL→nullptr, "
                    "headers, typedef→using, etc.). "
                    "Set MODERNIZER_API_KEY (or API_KEY) to a real key in your .env file."
                )

    # ------------------------------------------------------------------
    def call(self, system_prompt: str, user_prompt: str, role: str = "modernizer", bypass_cache: bool = False) -> Optional[str]:
        logger.info("ModelClient.call  role=%-12s  llm=%s", role, self._use_llm)
        logger.info(f"[WORKING] ModelClient: Requesting {role} (LLM enabled: {self._use_llm})")

        # --- LLM path with exponential backoff and caching ---
        if self._use_llm:
            try:
                raw = _call_llm(role, system_prompt, user_prompt, context=self.context, bypass_cache=bypass_cache)
                if raw:
                    code = extract_code(raw) if role in ("modernizer", "fixer") else raw
                    if role not in ("modernizer", "fixer") or _is_valid_cpp(code):
                        return code
                    else:
                        logger.warning("LLM output invalid, falling back to rules.")
            except RateLimitError as e:
                logger.warning(f"429 Rate Limit hit ultimately: {e}")
            except Exception as e:
                logger.warning(f"LLM call failed: {e}")

        # --- Rule-based fallback (code roles only) ---
        if role in ("modernizer", "fixer"):
            matches = _CODE_FENCE_RE.findall(user_prompt)
            src = matches[-1] if matches else user_prompt
            modernized = self._rules.modernize_text(src)
            logger.info("Fell back to RuleModernizer for role=%s.", role)
            return modernized

        return None

    # ------------------------------------------------------------------
    def check_health(self) -> Tuple[bool, str]:
        parts = []
        for role in ["analyzer", "modernizer", "fixer", "planner"]:
            cfg = _get_role_config(role)
            parts.append(f"{role}={cfg.model}")
        return True, "Multi-model bridge: " + " | ".join(parts)
