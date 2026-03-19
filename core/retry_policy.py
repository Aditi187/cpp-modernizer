from __future__ import annotations

import os
import random
import threading
import time


def parse_retry_delays_from_env(env_name: str, default_delays: tuple[float, ...]) -> tuple[float, ...]:
    configured = os.environ.get(env_name, "").strip()
    if not configured:
        return tuple(float(item) for item in default_delays)

    parsed: list[float] = []
    for token in configured.split(","):
        item = token.strip()
        if not item:
            continue
        try:
            value = float(item)
        except ValueError:
            continue
        if value > 0:
            parsed.append(value)
    if not parsed:
        return tuple(float(item) for item in default_delays)
    return tuple(parsed)


def with_jitter(base_delay_seconds: float) -> float:
    jitter_ratio_raw = os.environ.get("LLM_RETRY_JITTER_RATIO", "0.2").strip()
    try:
        jitter_ratio = float(jitter_ratio_raw)
    except ValueError:
        jitter_ratio = 0.2
    jitter_ratio = min(1.0, max(0.0, jitter_ratio))

    if base_delay_seconds <= 0:
        return 0.0
    spread = base_delay_seconds * jitter_ratio
    return max(0.0, base_delay_seconds + random.uniform(-spread, spread))


class RateLimiter:
    """Simple token-bucket style limiter for request pacing."""

    def __init__(self, max_calls_per_minute: float) -> None:
        safe_rate = max(1.0, float(max_calls_per_minute or 1.0))
        self.interval_seconds = 60.0 / safe_rate
        self._last_call_time = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call_time
            if elapsed < self.interval_seconds:
                time.sleep(self.interval_seconds - elapsed)
            self._last_call_time = time.time()
