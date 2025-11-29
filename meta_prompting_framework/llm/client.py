"""
Meta-Prompting Framework v2 - LLM Client Adapter

Provides compatibility layer between v1 LLM clients and v2 modules.
"""

from typing import List, Dict, Any, Optional
from meta_prompting_engine.llm_clients.base import BaseLLMClient, Message, LLMResponse


class LLMClientAdapter:
    """
    Adapter that wraps v1 BaseLLMClient for v2 module compatibility.

    Allows v2 modules to use v1 LLM clients seamlessly.
    Handles message format conversion and response extraction.
    """

    def __init__(self, v1_client: BaseLLMClient):
        """
        Initialize adapter with a v1 client.

        Args:
            v1_client: Instance of BaseLLMClient from v1 framework
        """
        self.client = v1_client

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> LLMResponse:
        """
        Generate completion using dict-style messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            LLMResponse from v1 client
        """
        # Convert dict messages to Message objects
        message_objects = [
            Message(role=msg["role"], content=msg["content"])
            for msg in messages
        ]

        # Call v1 client
        return self.client.complete(
            messages=message_objects,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding using v1 client.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        return self.client.embed(text)

    @property
    def call_history(self) -> List[Dict[str, Any]]:
        """Access v1 client's call history."""
        return self.client.call_history


def create_v2_client(
    provider: str = "claude",
    api_key: Optional[str] = None,
    model: Optional[str] = None
) -> LLMClientAdapter:
    """
    Factory function to create v2-compatible LLM client.

    Args:
        provider: LLM provider ("claude", future: "openai", etc.)
        api_key: API key (uses env var if None)
        model: Model name (uses default if None)

    Returns:
        LLMClientAdapter wrapping the appropriate v1 client
    """
    if provider == "claude":
        from meta_prompting_engine.llm_clients.claude import ClaudeClient

        # ClaudeClient handles env var if api_key is None
        if api_key:
            v1_client = ClaudeClient(api_key=api_key, model=model)
        else:
            v1_client = ClaudeClient(model=model)

        return LLMClientAdapter(v1_client)
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: 'claude'")


# Convenience class for direct use
class ClaudeClientV2(LLMClientAdapter):
    """
    Claude client for v2 framework.

    Convenience wrapper that creates and wraps a v1 ClaudeClient.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        from meta_prompting_engine.llm_clients.claude import ClaudeClient

        v1_client = ClaudeClient(api_key=api_key, model=model) if api_key else ClaudeClient(model=model)
        super().__init__(v1_client)
