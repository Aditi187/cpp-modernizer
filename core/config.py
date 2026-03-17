"""Centralized configuration for the Air-Gapped Modernization Engine.

All environment-variable reads that govern runtime behaviour live here so
consumers import constants rather than scattering ``os.environ.get`` calls
throughout the codebase.

Usage::

    from core.config import cfg
    print(cfg.ollama_model)
    if cfg.use_cache:
        ...

The ``cfg`` singleton is evaluated once at import time.  Call
``Config.reload()`` in tests to force a fresh read.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _bool_env(name: str, default: bool) -> bool:
    val = os.environ.get(name, "").strip().lower()
    if not val:
        return default
    return val in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int, minimum: int = 0) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


@dataclass(frozen=True)
class Config:
    # ------------------------------------------------------------------ #
    # LLM provider selection                                               #
    # ------------------------------------------------------------------ #
    workflow_model_provider: str = field(
        default_factory=lambda: os.environ.get("WORKFLOW_MODEL_PROVIDER", "").strip().lower()
    )

    # ------------------------------------------------------------------ #
    # Ollama                                                               #
    # ------------------------------------------------------------------ #
    ollama_base_url: str = field(
        default_factory=lambda: os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").strip()
    )
    ollama_model: str = field(
        default_factory=lambda: os.environ.get("OLLAMA_MODEL", "deepseek-coder:6.7b").strip()
    )
    ollama_timeout_seconds: int = field(
        default_factory=lambda: _int_env("OLLAMA_TIMEOUT_SECONDS", 300, 30)
    )
    ollama_num_parallel: int = field(
        default_factory=lambda: _int_env("OLLAMA_NUM_PARALLEL", 1, 1)
    )
    ollama_keep_alive: int = field(
        default_factory=lambda: _int_env("OLLAMA_KEEP_ALIVE", 0, 0)
    )

    # ------------------------------------------------------------------ #
    # Gemini / Google                                                      #
    # ------------------------------------------------------------------ #
    gemini_api_key: str = field(
        default_factory=lambda: (
            os.environ.get("GEMINI_API_KEY", "")
            or os.environ.get("GOOGLE_API_KEY", "")
        ).strip()
    )
    gemini_models: list[str] = field(
        default_factory=lambda: [
            m.strip()
            for m in os.environ.get("GEMINI_MODELS", "gemini-2.0-flash").split(",")
            if m.strip()
        ]
    )

    # ------------------------------------------------------------------ #
    # OpenRouter                                                           #
    # ------------------------------------------------------------------ #
    openrouter_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", "").strip()
    )

    # ------------------------------------------------------------------ #
    # Cache                                                                #
    # ------------------------------------------------------------------ #
    use_cache: bool = field(
        default_factory=lambda: _bool_env("USE_CACHE", True)
    )
    cache_version: str = field(
        default_factory=lambda: os.environ.get("CACHE_VERSION", "v1").strip() or "v1"
    )
    cache_ttl_seconds: int = field(
        # Default 7 days; 0 means no TTL.
        default_factory=lambda: _int_env("CACHE_TTL_SECONDS", 7 * 86400, 0)
    )

    # ------------------------------------------------------------------ #
    # Langfuse tracing                                                     #
    # ------------------------------------------------------------------ #
    langfuse_public_key: str = field(
        default_factory=lambda: os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
    )
    langfuse_secret_key: str = field(
        default_factory=lambda: os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
    )
    langfuse_host: str = field(
        default_factory=lambda: os.environ.get("LANGFUSE_HOST", "").strip()
    )

    # ------------------------------------------------------------------ #
    # Workflow                                                             #
    # ------------------------------------------------------------------ #
    strict_cpp23_mode: bool = field(
        default_factory=lambda: (
            _bool_env("WORKFLOW_STRICT_MODE", False)
            or _bool_env("CPP23_STRICT_MODE", False)
        )
    )
    strict_cpp23_target_percent: int = field(
        default_factory=lambda: _int_env("CPP23_STRICT_TARGET_PERCENT", 70, 0)
    )

    # ------------------------------------------------------------------ #
    # Logging                                                              #
    # ------------------------------------------------------------------ #
    log_level: str = field(
        default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO").strip().upper()
    )
    log_file: str = field(
        default_factory=lambda: os.environ.get("LOG_FILE", "modernization.log").strip()
    )

    @classmethod
    def reload(cls) -> "Config":
        """Re-read all environment variables and return a fresh Config.

        Useful in tests that set env vars after import time.
        """
        global cfg
        cfg = cls()
        return cfg


# Module-level singleton — evaluated once at import time.
cfg = Config()
