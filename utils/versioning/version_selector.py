"""
Meta-Prompting Framework - Version Selector

Provides a unified interface to select between v1 and v2 frameworks.
"""

from typing import Optional, Any
import warnings


def create_engine(
    version: str = "v1",
    skill: Optional[str] = None,
    llm_client: Optional[Any] = None,
    **kwargs
) -> Any:
    """
    Create a meta-prompting engine with version selection.

    Args:
        version: Framework version ("v1" or "v2")
        skill: Skill/role for v1 engine
        llm_client: LLM client instance
        **kwargs: Additional arguments passed to the engine

    Returns:
        Engine instance (v1 or v2)

    Examples:
        # Use v1 (stable)
        engine = create_engine(version="v1", skill="math_expert", llm_client=client)
        result = engine.execute_with_meta_prompting(task="Solve x^2 + 5x + 6 = 0")

        # Use v2 (when Phase 2 is ready)
        engine = create_engine(version="v2", signature=MathSignature, llm_client=client)
        result = engine(question="What is 2+2?")
    """
    if version == "v1":
        return _create_v1_engine(skill=skill, llm_client=llm_client, **kwargs)
    elif version == "v2":
        return _create_v2_engine(llm_client=llm_client, **kwargs)
    else:
        raise ValueError(f"Unknown version: {version}. Use 'v1' or 'v2'")


def _create_v1_engine(
    skill: Optional[str] = None,
    llm_client: Optional[Any] = None,
    **kwargs
):
    """Create v1 (meta_prompting_engine) instance."""
    try:
        from meta_prompting_engine.core import MetaPromptingEngine

        if llm_client is None:
            from meta_prompting_engine.llm_clients.claude import ClaudeClient
            llm_client = ClaudeClient()

        if skill is None:
            skill = "general_expert"

        return MetaPromptingEngine(skill=skill, llm_client=llm_client, **kwargs)

    except ImportError as e:
        raise ImportError(
            "v1 (meta_prompting_engine) not available. "
            "Make sure the package is installed."
        ) from e


def _create_v2_engine(llm_client: Optional[Any] = None, **kwargs):
    """Create v2 (meta_prompting_framework) instance."""

    try:
        # Phase 2 is now complete! Import modules
        from meta_prompting_framework.prompts.module import ChainOfThought, Predict
        from meta_prompting_framework.prompts.signature import ChainOfThoughtSignature
        from meta_prompting_framework.llm import create_v2_client

        # Create LLM client if not provided
        if llm_client is None:
            llm_client = create_v2_client("claude")

        # Get module type from kwargs, default to ChainOfThought
        module_type = kwargs.pop('module_type', 'chain_of_thought')
        signature_type = kwargs.pop('signature', ChainOfThoughtSignature)

        if module_type == 'chain_of_thought':
            return ChainOfThought(signature_type, llm_client=llm_client, **kwargs)
        elif module_type == 'predict':
            return Predict(signature_type, llm_client=llm_client, **kwargs)
        else:
            # Default to ChainOfThought
            return ChainOfThought(signature_type, llm_client=llm_client, **kwargs)

    except ImportError as e:
        raise ImportError(
            "v2 (meta_prompting_framework) modules not available. "
            "Make sure Phase 2 components are installed."
        ) from e


def get_version_info() -> dict:
    """
    Get information about available framework versions.

    Returns:
        Dictionary with version info
    """
    info = {
        "v1": {
            "name": "Meta-Prompting Engine",
            "status": "stable",
            "location": "meta_prompting_engine/",
            "available": False
        },
        "v2": {
            "name": "Categorical Meta-Prompting Framework",
            "status": "development",
            "phase": "Phase 1-2 complete (categorical + modules)",
            "location": "meta_prompting_framework/",
            "available": False
        }
    }

    # Check v1 availability
    try:
        from meta_prompting_engine.core import MetaPromptingEngine
        info["v1"]["available"] = True
    except ImportError:
        pass

    # Check v2 availability
    try:
        from meta_prompting_framework.categorical import RMPMonad
        from meta_prompting_framework.prompts import ChainOfThought
        info["v2"]["available"] = True
    except ImportError:
        pass

    return info


def print_version_info():
    """Print version information to console."""
    info = get_version_info()

    print("=" * 60)
    print("Meta-Prompting Framework - Version Information")
    print("=" * 60)
    print()

    for version, details in info.items():
        status_icon = "✅" if details["available"] else "❌"
        print(f"{version.upper()}: {details['name']}")
        print(f"  Status: {details['status']}")
        print(f"  Available: {status_icon}")
        if "phase" in details:
            print(f"  Progress: {details['phase']}")
        print(f"  Location: {details['location']}")
        print()

    print("See docs/VERSION_GUIDE.md for detailed comparison and usage instructions.")
    print()


if __name__ == "__main__":
    print_version_info()
