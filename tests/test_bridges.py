"""Unit tests for LLM bridge configuration and cache helpers."""

from __future__ import annotations

import json
import os
import tempfile
import time

import pytest


# ---------------------------------------------------------------------------
# GeminiBridge cache tests
# ---------------------------------------------------------------------------

class TestGeminiBridgeCache:
    """Tests that cover USE_CACHE, CACHE_VERSION, and CACHE_TTL_SECONDS."""

    def _make_bridge(self, env_overrides: dict | None = None):
        """Create a GeminiBridge with a temp cache path and optional env overrides."""
        from core.gemini_bridge import GeminiBridge, GeminiConfig

        overrides = env_overrides or {}
        original = {k: os.environ.get(k) for k in overrides}
        for k, v in overrides.items():
            os.environ[k] = v
        try:
            bridge = GeminiBridge.from_env()
            bridge._cache_path = os.path.join(tempfile.mkdtemp(), "test_cache.json")
        finally:
            for k, original_v in original.items():
                if original_v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = original_v
        return bridge

    def test_use_cache_false_disables_cache(self, monkeypatch):
        monkeypatch.setenv("USE_CACHE", "false")
        monkeypatch.setenv("GEMINI_ENABLE_CACHE", "1")
        from core.gemini_bridge import GeminiBridge
        bridge = GeminiBridge.from_env()
        assert bridge._cache_enabled is False

    def test_use_cache_true_and_gemini_flag_enabled(self, monkeypatch):
        monkeypatch.setenv("USE_CACHE", "1")
        monkeypatch.setenv("GEMINI_ENABLE_CACHE", "1")
        from core.gemini_bridge import GeminiBridge
        bridge = GeminiBridge.from_env()
        assert bridge._cache_enabled is True

    def test_gemini_flag_false_overrides_use_cache(self, monkeypatch):
        monkeypatch.setenv("USE_CACHE", "1")
        monkeypatch.setenv("GEMINI_ENABLE_CACHE", "false")
        from core.gemini_bridge import GeminiBridge
        bridge = GeminiBridge.from_env()
        assert bridge._cache_enabled is False

    def test_cache_version_from_use_cache_version_env(self, monkeypatch):
        monkeypatch.setenv("CACHE_VERSION", "v2")
        from core.gemini_bridge import GeminiBridge
        bridge = GeminiBridge.from_env()
        assert bridge._cache_version == "v2"

    def test_cache_ttl_respected_on_load(self, monkeypatch, tmp_path):
        """Entries older than CACHE_TTL_SECONDS should be filtered out on load."""
        monkeypatch.setenv("USE_CACHE", "1")
        monkeypatch.setenv("GEMINI_ENABLE_CACHE", "1")
        monkeypatch.setenv("CACHE_TTL_SECONDS", "60")

        from core.gemini_bridge import GeminiBridge
        bridge = GeminiBridge.from_env()
        cache_path = tmp_path / "cache.json"
        bridge._cache_path = str(cache_path)

        # Write an entry that is 120 seconds old — past the 60 s TTL.
        old_ts = time.time() - 120
        stale_data = {"deadbeef": {"v": "old value", "ts": old_ts}}
        cache_path.write_text(json.dumps(stale_data), encoding="utf-8")

        loaded = bridge._load_cache()
        assert "deadbeef" not in loaded

    def test_cache_ttl_zero_means_no_expiry(self, monkeypatch, tmp_path):
        monkeypatch.setenv("USE_CACHE", "1")
        monkeypatch.setenv("GEMINI_ENABLE_CACHE", "1")
        monkeypatch.setenv("CACHE_TTL_SECONDS", "0")

        from core.gemini_bridge import GeminiBridge
        bridge = GeminiBridge.from_env()
        cache_path = tmp_path / "cache.json"
        bridge._cache_path = str(cache_path)

        old_ts = time.time() - 100_000  # expired by any non-zero TTL
        data = {"abc": {"v": "keep me", "ts": old_ts}}
        cache_path.write_text(json.dumps(data), encoding="utf-8")

        loaded = bridge._load_cache()
        assert "abc" in loaded

    def test_cache_migrates_legacy_string_entries(self, monkeypatch, tmp_path):
        """Legacy cache entries that are plain strings should be loaded without error."""
        monkeypatch.setenv("USE_CACHE", "1")
        monkeypatch.setenv("GEMINI_ENABLE_CACHE", "1")
        monkeypatch.setenv("CACHE_TTL_SECONDS", "0")

        from core.gemini_bridge import GeminiBridge
        bridge = GeminiBridge.from_env()
        cache_path = tmp_path / "cache.json"
        bridge._cache_path = str(cache_path)

        legacy = {"mykey": "plain string value"}
        cache_path.write_text(json.dumps(legacy), encoding="utf-8")

        loaded = bridge._load_cache()
        assert "mykey" in loaded
        assert loaded["mykey"]["v"] == "plain string value"


# ---------------------------------------------------------------------------
# core/config.py tests
# ---------------------------------------------------------------------------

class TestCoreConfig:
    def test_defaults_are_reasonable(self):
        from core.config import Config
        cfg = Config()
        assert cfg.cache_version == "v1" or isinstance(cfg.cache_version, str)
        assert isinstance(cfg.use_cache, bool)
        assert isinstance(cfg.cache_ttl_seconds, int)
        assert cfg.cache_ttl_seconds >= 0

    def test_reload_picks_up_env_change(self, monkeypatch):
        monkeypatch.setenv("CACHE_VERSION", "v99")
        from core.config import Config
        fresh = Config.reload()
        assert fresh.cache_version == "v99"

    def test_use_cache_false_from_env(self, monkeypatch):
        monkeypatch.setenv("USE_CACHE", "false")
        from core.config import Config
        cfg = Config()
        assert cfg.use_cache is False


# ---------------------------------------------------------------------------
# core/logger.py tests
# ---------------------------------------------------------------------------

class TestCoreLogger:
    def test_get_logger_returns_logger(self):
        from core.logger import get_logger
        import logging
        lgr = get_logger("test.module")
        assert isinstance(lgr, logging.Logger)

    def test_logger_name(self):
        from core.logger import get_logger
        lgr = get_logger("my.component")
        assert lgr.name == "my.component"

    def test_configure_called_only_once(self):
        import core.logger as mod
        # Call twice — should be idempotent (no duplicate handlers added).
        mod._configure()
        import logging
        before = len(logging.getLogger().handlers)
        mod._configure()
        after = len(logging.getLogger().handlers)
        assert after == before
