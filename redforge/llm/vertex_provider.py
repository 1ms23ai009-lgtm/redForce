"""Vertex AI LLM provider using google-genai SDK.

Uses Application Default Credentials (ADC) for authentication.
Mirrors the GroqProvider interface for drop-in swapping.

Authenticate by either:
  1. Setting GOOGLE_APPLICATION_CREDENTIALS env var to a service account JSON
  2. Running `gcloud auth application-default login`
"""

import time
import logging

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# Model recommendations per role for Vertex AI
VERTEX_MODELS = {
    "attacker": "gemini-2.0-flash",
    "judge": "gemini-2.0-flash",
    "target": "gemini-2.0-flash",
    "digest": "gemini-2.0-flash",
    "graph": "gemini-2.0-flash",
}


class VertexAIProvider:
    """LLM provider that routes through Google Cloud Vertex AI."""

    def __init__(self, project: str, location: str = "us-central1"):
        self.project = project
        self.location = location
        self._client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
        )
        self.total_tokens_used = 0
        self.total_requests = 0

    def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        role: str = "attacker",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        retries: int = 3,
    ) -> dict:
        """Send a chat completion request to Vertex AI.

        Args:
            messages: OpenAI-format message list [{"role": ..., "content": ...}]
            model: Specific model name, or None to use role default
            role: One of "attacker", "judge", "target", "digest", "graph"
            temperature: Sampling temperature
            max_tokens: Max response tokens
            retries: Number of retry attempts on failure

        Returns:
            dict with "content", "model", "tokens", "finish_reason"
        """
        if model is None:
            model = VERTEX_MODELS.get(role, VERTEX_MODELS["attacker"])

        # Convert OpenAI-format messages to google-genai format
        system_instruction = None
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=msg["content"])],
                    )
                )
            elif msg["role"] == "assistant":
                contents.append(
                    types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=msg["content"])],
                    )
                )

        # Disable safety filters for authorized red-teaming use
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_instruction,
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="OFF",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="OFF",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="OFF",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="OFF",
                ),
            ],
        )

        for attempt in range(retries):
            try:
                response = self._client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )

                self.total_requests += 1

                # Extract token counts from usage metadata
                usage = response.usage_metadata
                tokens = {
                    "input": usage.prompt_token_count if usage else 0,
                    "output": usage.candidates_token_count if usage else 0,
                    "total": usage.total_token_count if usage else 0,
                }
                self.total_tokens_used += tokens["total"]

                content = ""
                if response.candidates and response.candidates[0].content.parts:
                    content = response.candidates[0].content.parts[0].text

                finish_reason = "stop"
                if response.candidates:
                    fr = response.candidates[0].finish_reason
                    if fr:
                        finish_reason = str(fr).lower()

                return {
                    "content": content,
                    "model": model,
                    "tokens": tokens,
                    "finish_reason": finish_reason,
                }

            except Exception as e:
                error_str = str(e)
                is_rate_limit = (
                    "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
                )

                if is_rate_limit:
                    wait = 30 * (attempt + 1)
                    logger.warning(
                        f"Vertex AI rate limit (attempt {attempt + 1}/{retries}), "
                        f"waiting {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    logger.warning(
                        f"Vertex AI error (attempt {attempt + 1}/{retries}): "
                        f"{error_str[:100]}"
                    )
                    if attempt < retries - 1:
                        time.sleep(2 ** attempt)

        # All retries exhausted
        return {
            "content": "",
            "model": model or "",
            "tokens": {"input": 0, "output": 0, "total": 0},
            "finish_reason": "error",
        }

    def get_stats(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens_used,
            "project": self.project,
            "location": self.location,
        }
