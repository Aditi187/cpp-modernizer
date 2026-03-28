from __future__ import annotations
from core.rule_modernizer import RuleModernizer
from agents.workflow.context import WorkflowContext

"""
Multi-model LLM bridge for the Air-Gapped C++ Modernization Engine.

Role routing (from .env):
    analyze  → DeepSeek-V3   (deep reasoning, thinking mode)
    modernize → Llama-3.3-70B (code rewriting)
    fixer    → Qwen           (small compiler-error fixes)

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

_CODE_FENCE_RE = re.compile(
    r"```(?:cpp|c\+\+|cxx|cc|hpp|h)?\s*\n?(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


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


class _RoleConfig:
    """Holds the model/endpoint/key/params for one role."""

    def __init__(self, prefix: str, fallback_key: str, fallback_url: str, fallback_model: str):
        self.api_key   = _env(f"{prefix}_API_KEY")   or _env(fallback_key)
        self.base_url  = _env(f"{prefix}_ENDPOINT_BASE") or fallback_url
        self.model     = _env(f"{prefix}_MODEL")     or fallback_model
        # Force deterministic output
        self.temp      = 0.0
        self.top_p     = 1.0
        self.max_tokens = _env_int(f"{prefix}_MAX_TOKENS", 8192)
        self.thinking  = _env(f"{prefix}_ENABLE_THINKING") in ("1", "true", "yes")

    def client(self) -> OpenAI:
        return OpenAI(api_key=self.api_key, base_url=self.base_url)


_FALLBACK_URL   = "https://integrate.api.nvidia.com/v1"
_FALLBACK_KEY   = _env("API_KEY") or _env("OPENAI_API_KEY")

_ROLE_CONFIGS = {
    # Phase 4: Simplified role mapping
    "analyzer":  _RoleConfig("ANALYZER",   "API_KEY", _FALLBACK_URL, "deepseek-ai/deepseek-v3"),
    "modernizer": _RoleConfig("MODERNIZER", "API_KEY", _FALLBACK_URL, "meta/llama-3.3-70b-instruct"),
    "fixer":     _RoleConfig("MODERNIZER", "API_KEY", _FALLBACK_URL, "meta/llama-3.3-70b-instruct"),
    "planner":   _RoleConfig("ANALYZER",   "API_KEY", _FALLBACK_URL, "deepseek-ai/deepseek-v3"),
}
_DEFAULT_CONFIG = _RoleConfig("OPENAI", "API_KEY", _FALLBACK_URL, _env("OPENAI_MODELS", "meta/llama-3.3-70b-instruct"))


def _with_retry(fn, max_attempts: int = 15, base_wait: float = 30.0):
    last_err = None
    min_delay = 60.0  # Extreme throttle: 1 minute between successes
    for attempt in range(max_attempts):
        try:
            result = fn()
            time.sleep(min_delay)
            return result
        except OpenAI_RateLimitError as e:
            last_err = e
            # Exponential backoff: 30s, 60s, 120s, 240s...
            wait = (2 ** attempt) * base_wait + random.uniform(1.0, 5.0)
            logger.warning("Rate-limited (attempt %d/%d). Waiting %.1fs…", attempt + 1, max_attempts, wait)
            time.sleep(wait)
        except Exception:
            raise
    logger.error(f"Rate limit persisted after {max_attempts} attempts: {last_err}")
    print(f"[ERROR] ModelProvider: Rate limit persisted after {max_attempts} attempts.")
    raise RateLimitError(f"Rate limit persisted after {max_attempts} attempts: {last_err}")


def _call_llm(role: str, system: str, user: str, context: Optional[WorkflowContext] = None) -> Optional[str]:
    cfg = _ROLE_CONFIGS.get(role, _DEFAULT_CONFIG)

    if not cfg.api_key:
        logger.warning("No API key for role=%s; skipping LLM call.", role)
        return None

    # Caching support
    cache_key = None
    if context is not None:
        cache_key = f"{role}|{cfg.model}|{system}|{user}"
        cached = context.get_cached_llm_response(cache_key)
        if cached:
            logger.info(f"[CACHE] LLM cache hit for role={role}")
            return cached

    def _do_call():
        client = cfg.client()
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]

        # Max token safety
        max_tokens = min(cfg.max_tokens, 4096)

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
        return resp.choices[0].message.content or ""

    try:
        raw = _with_retry(_do_call)
        logger.debug("LLM [%s/%s] returned %d chars.", role, cfg.model, len(raw))
        if context is not None and cache_key:
            context.cache_llm_response(cache_key, raw)
        print(f"[WORKING] ModelProvider: LLM call successful for role={role} (model={cfg.model})")
        return raw
    except Exception as e:
        logger.error("LLM call failed for role=%s: %s", role, e)
        print(f"[ERROR] ModelProvider: LLM call failed for role={role}: {e}")
        return None


def _extract_code(text: str) -> str:
    """
    Extract the best C++ code block from LLM output.
    Preference order:
      1. Block containing 'main()'
      2. Block with the most #include and class/struct/function keywords
      3. Longest block by non-whitespace length
    If no code fences, return the whole text stripped.
    """
    if not text:
        return ""
    blocks = list(_CODE_FENCE_RE.finditer(text))
    if not blocks:
        return text.strip()
    # 1. Prefer block with main()
    for m in blocks:
        code = m.group(1)
        if 'main(' in code:
            return code.strip()
    # 2. Prefer block with most includes and class/struct/function
    def score_block(code):
        includes = code.count('#include')
        classes = code.count('class ') + code.count('struct ')
        functions = code.count('(')  # crude, but helps
        return includes * 3 + classes * 2 + functions
    best = max(blocks, key=lambda m: (score_block(m.group(1)), len(m.group(1).strip())))
    return best.group(1).strip()


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

    # ------------------------------------------------------------------
    def call(self, system_prompt: str, user_prompt: str, role: str = "modernizer") -> Optional[str]:
        logger.info("ModelClient.call  role=%-12s  llm=%s", role, self._use_llm)
        print(f"[WORKING] ModelClient: Requesting {role} (LLM enabled: {self._use_llm})")

        # --- LLM path with exponential backoff and caching ---
        if self._use_llm:
            max_attempts = 5
            for attempt in range(max_attempts):
                try:
                    raw = _call_llm(role, system_prompt, user_prompt, context=self.context)
                    if raw:
                        code = _extract_code(raw) if role in ("modernizer", "fixer") else raw
                        if role not in ("modernizer", "fixer") or _is_valid_cpp(code):
                            return code
                        else:
                            logger.warning("LLM output invalid, falling back to rules.")
                    break
                except RateLimitError as e:
                    wait = (2 ** attempt)
                    logger.warning(f"429 Rate Limit hit. Backing off for {wait}s (attempt {attempt+1}/{max_attempts})")
                    time.sleep(wait)
                except Exception as e:
                    logger.warning(f"LLM call failed: {e}")
                    break

        # --- Rule-based fallback (code roles only) ---
        if role in ("modernizer", "fixer"):
            m = _CODE_FENCE_RE.search(user_prompt)
            src = m.group(1) if m else user_prompt
            modernized = self._rules.modernize_text(src)
            logger.info("Fell back to RuleModernizer for role=%s.", role)
            return modernized

        return None

    # ------------------------------------------------------------------
    def check_health(self) -> Tuple[bool, str]:
        parts = []
        for role, cfg in _ROLE_CONFIGS.items():
            parts.append(f"{role}={cfg.model}")
        return True, "Multi-model bridge: " + " | ".join(parts)
