"""Target LLM connection abstraction layer.

Supports multiple target types: OpenAI, Anthropic, generic OpenAI-compatible,
LangChain agent endpoints, and custom HTTP endpoints.
"""

import os
import time
import json
from typing import Optional

import httpx
import tiktoken


class TargetConnector:
    """Abstraction layer for communicating with target LLM systems."""

    def __init__(self, config: dict):
        """Initialize the target connector.

        Args:
            config: Target configuration dict containing:
                - endpoint_url: API endpoint URL
                - model_name: Model identifier
                - api_key_env_var: Environment variable name for API key
                - target_type: One of "openai", "anthropic", "openai_compatible",
                               "langchain", "custom_http"
                - tool_schemas: Optional list of tool schemas
                - memory_system: Optional memory system identifier
                - agent_topology: Optional agent topology dict
                - timeout: Optional request timeout in seconds
        """
        self.config = config
        self.endpoint_url = config["endpoint_url"]
        self.model_name = config.get("model_name", "unknown")
        self.target_type = config.get("target_type", "openai_compatible")
        self.tool_schemas = config.get("tool_schemas", [])
        self.memory_system = config.get("memory_system")
        self.agent_topology = config.get("agent_topology")
        self.timeout = config.get("timeout", 60)

        api_key_var = config.get("api_key_env_var", "TARGET_API_KEY")
        self.api_key = os.getenv(api_key_var, "")

        self._client = httpx.Client(timeout=self.timeout)
        self._total_tokens = 0
        self._total_cost = 0.0

        try:
            self._tokenizer = tiktoken.encoding_for_model(self.model_name)
        except Exception:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def send_message(
        self,
        message: str,
        conversation_history: Optional[list] = None,
        tool_choice: Optional[str] = None,
    ) -> dict:
        """Send a message to the target LLM.

        Args:
            message: The user message to send
            conversation_history: Optional prior conversation turns
            tool_choice: Optional tool selection hint

        Returns:
            Dict with: response, tokens_used, cost, latency
        """
        if conversation_history is None:
            conversation_history = []

        start_time = time.time()

        if self.target_type in ("openai", "openai_compatible"):
            result = self._send_openai(message, conversation_history, tool_choice)
        elif self.target_type == "anthropic":
            result = self._send_anthropic(message, conversation_history)
        elif self.target_type == "langchain":
            result = self._send_langchain(message, conversation_history)
        elif self.target_type == "custom_http":
            result = self._send_custom_http(message, conversation_history)
        else:
            raise ValueError(f"Unsupported target type: {self.target_type}")

        latency = time.time() - start_time
        result["latency"] = latency

        self._total_tokens += result.get("tokens_used", 0)
        self._total_cost += result.get("cost", 0.0)

        return result

    def _send_openai(
        self, message: str, history: list, tool_choice: Optional[str]
    ) -> dict:
        """Send message via OpenAI-compatible API."""
        messages = list(history) + [{"role": "user", "content": message}]

        payload = {
            "model": self.model_name,
            "messages": messages,
        }

        if self.tool_schemas and tool_choice:
            payload["tools"] = self.tool_schemas
            payload["tool_choice"] = tool_choice

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = self._client.post(
            self.endpoint_url, json=payload, headers=headers
        )
        response.raise_for_status()
        data = response.json()

        response_text = data["choices"][0]["message"]["content"] or ""
        tokens_used = data.get("usage", {}).get("total_tokens", 0)
        cost = self._estimate_cost(tokens_used)

        return {
            "response": response_text,
            "tokens_used": tokens_used,
            "cost": cost,
            "raw_response": data,
        }

    def _send_anthropic(self, message: str, history: list) -> dict:
        """Send message via Anthropic API."""
        messages = list(history) + [{"role": "user", "content": message}]

        payload = {
            "model": self.model_name,
            "max_tokens": 4096,
            "messages": messages,
        }

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        response = self._client.post(
            self.endpoint_url, json=payload, headers=headers
        )
        response.raise_for_status()
        data = response.json()

        response_text = data["content"][0]["text"] if data.get("content") else ""
        input_tokens = data.get("usage", {}).get("input_tokens", 0)
        output_tokens = data.get("usage", {}).get("output_tokens", 0)
        tokens_used = input_tokens + output_tokens
        cost = self._estimate_cost(tokens_used)

        return {
            "response": response_text,
            "tokens_used": tokens_used,
            "cost": cost,
            "raw_response": data,
        }

    def _send_langchain(self, message: str, history: list) -> dict:
        """Send message to a LangChain agent endpoint."""
        payload = {
            "input": message,
            "chat_history": history,
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = self._client.post(
            self.endpoint_url, json=payload, headers=headers
        )
        response.raise_for_status()
        data = response.json()

        response_text = data.get("output", data.get("response", str(data)))
        tokens_used = len(self._tokenizer.encode(message + response_text))
        cost = self._estimate_cost(tokens_used)

        return {
            "response": response_text,
            "tokens_used": tokens_used,
            "cost": cost,
            "raw_response": data,
        }

    def _send_custom_http(self, message: str, history: list) -> dict:
        """Send message to a custom HTTP endpoint."""
        payload = {
            "message": message,
            "history": history,
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = self._client.post(
            self.endpoint_url, json=payload, headers=headers
        )
        response.raise_for_status()
        data = response.json()

        response_text = str(data.get("response", data.get("output", data)))
        tokens_used = len(self._tokenizer.encode(message + response_text))
        cost = self._estimate_cost(tokens_used)

        return {
            "response": response_text,
            "tokens_used": tokens_used,
            "cost": cost,
            "raw_response": data,
        }

    def invoke_tool(self, tool_name: str, parameters: dict) -> dict:
        """Invoke a tool on the target system.

        This sends a message that requests tool use on the target.
        """
        tool_request = (
            f"Please use the {tool_name} tool with these parameters: "
            f"{json.dumps(parameters)}"
        )
        return self.send_message(tool_request)

    def read_memory(self, query: str) -> list[dict]:
        """Read from target's memory system if available."""
        if not self.memory_system:
            return []
        result = self.send_message(f"What do you remember about: {query}")
        return [{"query": query, "response": result["response"]}]

    def write_memory(self, content: str) -> bool:
        """Write to target's memory system if available."""
        if not self.memory_system:
            return False
        result = self.send_message(content)
        return bool(result.get("response"))

    def get_tool_schemas(self) -> list[dict]:
        """Return known tool schemas for the target."""
        return self.tool_schemas

    def _estimate_cost(self, tokens: int) -> float:
        """Estimate cost based on token count and model."""
        cost_per_1k = {
            "gpt-4o": 0.005,
            "gpt-4o-mini": 0.00015,
            "gpt-4": 0.03,
            "gpt-3.5-turbo": 0.0005,
            "claude-3-5-sonnet-20241022": 0.003,
            "claude-3-opus-20240229": 0.015,
            "claude-3-haiku-20240307": 0.00025,
        }
        rate = cost_per_1k.get(self.model_name, 0.001)
        return (tokens / 1000) * rate

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def total_cost(self) -> float:
        return self._total_cost

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __del__(self):
        try:
            self._client.close()
        except Exception:
            pass
