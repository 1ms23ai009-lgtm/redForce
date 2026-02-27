"""LLM providers for REDFORGE.

Factory functions for creating provider-appropriate LLM instances.
"""

from typing import Optional


def create_langchain_llm(
    provider: str,
    model_name: str,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    vertex_project: Optional[str] = None,
    vertex_location: str = "us-central1",
):
    """Create a LangChain chat model for the specified provider.

    Args:
        provider: "vertex", "openai", or "anthropic" (auto-detected from model name)
        model_name: Model name (e.g., "gemini-2.0-flash", "gpt-4o")
        api_key: API key (for OpenAI/Anthropic; ignored for Vertex)
        temperature: Sampling temperature
        vertex_project: GCP project ID (required for Vertex)
        vertex_location: GCP region (default: us-central1)

    Returns:
        A LangChain BaseChatModel instance
    """
    if provider == "vertex":
        from langchain_google_vertexai import ChatVertexAI
        return ChatVertexAI(
            model_name=model_name,
            project=vertex_project,
            location=vertex_location,
            temperature=temperature,
        )
    elif "claude" in model_name:
        from langchain_anthropic import ChatAnthropic
        kwargs = {"model": model_name, "temperature": temperature}
        if api_key:
            kwargs["api_key"] = api_key
        return ChatAnthropic(**kwargs)
    else:
        from langchain_openai import ChatOpenAI
        kwargs = {"model": model_name, "temperature": temperature}
        if api_key:
            kwargs["api_key"] = api_key
        return ChatOpenAI(**kwargs)
