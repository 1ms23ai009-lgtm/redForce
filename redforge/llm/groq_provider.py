"""Groq LLM provider using OpenAI-compatible API.

Groq provides fast inference for open-source models at no cost (free tier).
Since Groq's API is OpenAI-compatible, we use the openai SDK directly.
"""

import time
import json
import logging
from openai import OpenAI

from redforge.llm.key_manager import KeyPool

logger = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Model recommendations per role
# NOTE: llama-3.3-70b-versatile has strict rate limits on Groq free tier.
# For large runs (100+ behaviors), use 8b-instant for attacker/judge too.
GROQ_MODELS = {
    "attacker": "llama-3.1-8b-instant",         # Fast, high rate limit
    "judge": "llama-3.3-70b-versatile",         # Accurate classification
    "target": "llama-3.1-8b-instant",           # Fast — the model being attacked
    "digest": "llama-3.1-8b-instant",           # Fast summaries
    "graph": "llama-3.1-8b-instant",            # Graph extraction
}

# Available free-tier Groq models
AVAILABLE_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]


class GroqProvider:
    """LLM provider that routes through Groq with multi-key rotation."""

    def __init__(self, key_pool: KeyPool, pool_name: str = "redforge"):
        self.key_pool = key_pool
        self.pool_name = pool_name
        self._clients: dict[str, OpenAI] = {}
        self.total_tokens_used = 0
        self.total_requests = 0

    def _get_client(self, api_key: str) -> OpenAI:
        if api_key not in self._clients:
            self._clients[api_key] = OpenAI(
                api_key=api_key,
                base_url=GROQ_BASE_URL,
            )
        return self._clients[api_key]

    def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        role: str = "attacker",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        retries: int = 3,
    ) -> dict:
        """Send a chat completion request to Groq.

        Args:
            messages: OpenAI-format message list
            model: Specific model name, or None to use role default
            role: One of "attacker", "judge", "target", "digest", "graph"
            temperature: Sampling temperature
            max_tokens: Max response tokens
            retries: Number of retry attempts on failure

        Returns:
            dict with "content", "model", "tokens", "finish_reason"
        """
        if model is None:
            model = GROQ_MODELS.get(role, GROQ_MODELS["attacker"])

        for attempt in range(retries):
            api_key = self.key_pool.get_key(self.pool_name)
            client = self._get_client(api_key)

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                self.total_requests += 1
                tokens = {
                    "input": response.usage.prompt_tokens if response.usage else 0,
                    "output": response.usage.completion_tokens if response.usage else 0,
                    "total": response.usage.total_tokens if response.usage else 0,
                }
                self.total_tokens_used += tokens["total"]

                return {
                    "content": response.choices[0].message.content or "",
                    "model": response.model,
                    "tokens": tokens,
                    "finish_reason": response.choices[0].finish_reason,
                }

            except Exception as e:
                error_str = str(e)
                is_rate_limit = "rate_limit" in error_str.lower() or "429" in error_str

                if is_rate_limit:
                    self.key_pool.report_rate_limit(self.pool_name, api_key)
                    # Longer backoff for rate limits: 30s, 60s, 90s
                    wait = 30 * (attempt + 1)
                    logger.warning(
                        f"Rate limit hit (attempt {attempt + 1}/{retries}, "
                        f"pool={self.pool_name}), waiting {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    logger.warning(
                        f"Groq API error (attempt {attempt + 1}/{retries}, "
                        f"pool={self.pool_name}): {error_str[:100]}"
                    )
                    if attempt < retries - 1:
                        time.sleep(2 ** attempt)

        # All retries exhausted
        return {
            "content": "",
            "model": model,
            "tokens": {"input": 0, "output": 0, "total": 0},
            "finish_reason": "error",
        }

    def get_stats(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens_used,
            "pool": self.pool_name,
        }
