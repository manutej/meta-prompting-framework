"""
Meta-Prompting Framework - Prompts Module

Typed prompts with compositional guarantees.
"""

from .signature import (
    Field,
    InputField,
    OutputField,
    Signature,
    ChainOfThoughtSignature,
    RAGSignature,
    CodeGenerationSignature,
    MathSignature,
    DebugSignature,
)

from .module import (
    Module,
    Predict,
    ChainOfThought,
    ReAct,
    SequentialModule,
    RMPModule,
)

__all__ = [
    # Signature system
    "Field",
    "InputField",
    "OutputField",
    "Signature",
    "ChainOfThoughtSignature",
    "RAGSignature",
    "CodeGenerationSignature",
    "MathSignature",
    "DebugSignature",
    # Module system
    "Module",
    "Predict",
    "ChainOfThought",
    "ReAct",
    "SequentialModule",
    "RMPModule",
]
