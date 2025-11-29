"""
Meta-Prompting Framework v2 - LLM Integration

Client adapters for LLM integration with v2 modules.
"""

from .client import (
    LLMClientAdapter,
    create_v2_client,
    ClaudeClientV2,
)

__all__ = [
    "LLMClientAdapter",
    "create_v2_client",
    "ClaudeClientV2",
]
