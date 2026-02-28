"""Multi-key API key manager with round-robin rotation and rate limiting.

Supports splitting keys into named pools (e.g. "redforge" for attacker/judge,
"target" for HarmBench target) so rate limits don't interfere.
"""

import os
import time
import threading
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class KeyState:
    key: str
    requests_this_minute: int = 0
    requests_today: int = 0
    minute_reset: float = field(default_factory=time.time)
    day_reset: float = field(default_factory=time.time)
    errors: int = 0


class KeyManager:
    """Round-robin key rotation with per-key rate tracking."""

    def __init__(
        self,
        keys: list[str],
        requests_per_minute: int = 30,
        requests_per_day: int = 14400,
    ):
        # Filter out empty/whitespace-only keys
        valid_keys = [k for k in keys if k and k.strip()]
        if not valid_keys:
            raise ValueError("At least one non-empty API key is required")
        self.states = [KeyState(key=k) for k in valid_keys]
        self.rpm_limit = requests_per_minute
        self.rpd_limit = requests_per_day
        self._index = 0
        self._lock = threading.Lock()

    @property
    def num_keys(self) -> int:
        return len(self.states)

    def get_key(self) -> str:
        """Get the next available key, rotating round-robin."""
        with self._lock:
            now = time.time()
            for _ in range(len(self.states)):
                state = self.states[self._index]
                self._index = (self._index + 1) % len(self.states)

                # Reset minute window
                if now - state.minute_reset > 60:
                    state.requests_this_minute = 0
                    state.minute_reset = now

                # Reset day window
                if now - state.day_reset > 86400:
                    state.requests_today = 0
                    state.day_reset = now

                # Skip if over limits
                if state.requests_this_minute >= self.rpm_limit:
                    continue
                if state.requests_today >= self.rpd_limit:
                    continue

                state.requests_this_minute += 1
                state.requests_today += 1
                return state.key

            # All keys at limit — wait for the minute window to reset
            logger.warning("All keys at rate limit, waiting 30s for window reset...")
            time.sleep(30)
            # Reset all minute counters after waiting
            now = time.time()
            for state in self.states:
                state.requests_this_minute = 0
                state.minute_reset = now
            first = self.states[0]
            first.requests_this_minute += 1
            first.requests_today += 1
            return first.key

    def report_rate_limit(self, key: str):
        """Mark a key as rate-limited for the current minute."""
        with self._lock:
            for state in self.states:
                if state.key == key:
                    state.requests_this_minute = self.rpm_limit
                    state.errors += 1
                    break

    def get_usage_summary(self) -> list[dict]:
        """Return usage stats per key (masked)."""
        return [
            {
                "key_suffix": s.key[-6:],
                "rpm_used": s.requests_this_minute,
                "rpd_used": s.requests_today,
                "errors": s.errors,
            }
            for s in self.states
        ]


class KeyPool:
    """Manages named pools of keys for different purposes.

    Example:
        pool = KeyPool()
        pool.create_pool("redforge", keys[:3], rpm=30)
        pool.create_pool("target", keys[3:], rpm=30)
        key = pool.get_key("redforge")
    """

    def __init__(self):
        self.pools: dict[str, KeyManager] = {}

    def create_pool(
        self,
        name: str,
        keys: list[str],
        rpm: int = 30,
        rpd: int = 14400,
    ):
        self.pools[name] = KeyManager(keys, rpm, rpd)
        logger.info(f"Key pool '{name}': {len(keys)} keys, {rpm} rpm, {rpd} rpd")

    def get_key(self, pool_name: str) -> str:
        if pool_name not in self.pools:
            raise ValueError(f"Unknown key pool: {pool_name}. Available: {list(self.pools.keys())}")
        return self.pools[pool_name].get_key()

    def report_rate_limit(self, pool_name: str, key: str):
        if pool_name in self.pools:
            self.pools[pool_name].report_rate_limit(key)

    def get_summary(self) -> dict:
        return {
            name: {
                "num_keys": mgr.num_keys,
                "usage": mgr.get_usage_summary(),
            }
            for name, mgr in self.pools.items()
        }


def load_groq_keys_from_env() -> list[str]:
    """Load Groq API keys from environment variables.

    Supports two formats:
      GROQ_API_KEYS=key1,key2,key3       (comma-separated)
      GROQ_API_KEY_1=key1                 (numbered)
      GROQ_API_KEY_2=key2
    """
    # Try comma-separated first
    combined = os.getenv("GROQ_API_KEYS", "")
    if combined:
        keys = [k.strip() for k in combined.split(",") if k.strip()]
        if keys:
            return keys

    # Try numbered keys
    keys = []
    for i in range(1, 11):  # Support up to 10 keys
        key = os.getenv(f"GROQ_API_KEY_{i}", "")
        if key:
            keys.append(key)

    # Fallback: single key
    single = os.getenv("GROQ_API_KEY", "")
    if single and not keys:
        keys = [single]

    return keys


def build_key_pool(
    keys: list[str],
    redforge_count: int | None = None,
    rpm: int = 30,
    rpd: int = 14400,
) -> KeyPool:
    """Build a KeyPool with 'redforge' and 'target' pools.

    Splits keys: first `redforge_count` go to "redforge", rest to "target".
    If only 1 key, it's shared across both pools.
    """
    pool = KeyPool()

    if len(keys) == 0:
        raise ValueError("No API keys provided")

    if len(keys) == 1:
        # Single key shared for both
        pool.create_pool("redforge", keys, rpm, rpd)
        pool.create_pool("target", keys, rpm, rpd)
    else:
        if redforge_count is None:
            # Auto-split: ~60% for redforge, ~40% for target
            redforge_count = max(1, (len(keys) * 3) // 5)

        redforge_keys = keys[:redforge_count]
        target_keys = keys[redforge_count:]
        if not target_keys:
            target_keys = keys[-1:]  # At least 1 for target

        pool.create_pool("redforge", redforge_keys, rpm, rpd)
        pool.create_pool("target", target_keys, rpm, rpd)

    return pool
