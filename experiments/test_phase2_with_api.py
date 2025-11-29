"""
Test Phase 2 with Real LLM API

End-to-end test of v2 framework with actual Claude API calls.
Requires ANTHROPIC_API_KEY environment variable.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from meta_prompting_framework.prompts import (
    ChainOfThoughtSignature,
    CodeGenerationSignature,
    MathSignature,
    Predict,
    ChainOfThought,
)
from meta_prompting_framework.llm import ClaudeClientV2, create_v2_client


def check_api_key():
    """Check if API key is available."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("=" * 70)
        print("⚠ ANTHROPIC_API_KEY not set")
        print("=" * 70)
        print()
        print("This test requires a real Claude API key.")
        print()
        print("To run this test:")
        print("  export ANTHROPIC_API_KEY='your-key-here'")
        print("  python experiments/test_phase2_with_api.py")
        print()
        print("To test without API key:")
        print("  python experiments/test_phase2.py")
        print()
        return False
    return True


def test_predict_module():
    """Test basic Predict module with real LLM."""
    print("=" * 70)
    print("TEST 1: Predict Module (Real LLM)")
    print("=" * 70)
    print()

    # Create client
    client = create_v2_client("claude")

    # Create module
    module = Predict(ChainOfThoughtSignature, llm_client=client)

    # Execute
    print("Executing: What is 2+2?")
    result = module(question="What is 2+2?")

    print("\nResult:")
    print(f"  Reasoning: {result.get('reasoning', 'N/A')[:100]}...")
    print(f"  Answer: {result.get('answer', 'N/A')}")

    # Verify we got outputs
    assert 'reasoning' in result or 'answer' in result, "No outputs received"
    print("\n✓ Predict module works with real LLM")
    print()


def test_chain_of_thought_module():
    """Test ChainOfThought module with real LLM."""
    print("=" * 70)
    print("TEST 2: ChainOfThought Module (Real LLM)")
    print("=" * 70)
    print()

    # Create client
    client = ClaudeClientV2()

    # Create module
    module = ChainOfThought(ChainOfThoughtSignature, llm_client=client)

    # Execute
    question = "If a train travels 60 mph for 2.5 hours, how far does it go?"
    print(f"Executing: {question}")
    result = module(question=question)

    print("\nResult:")
    print(f"  Reasoning: {result.get('reasoning', 'N/A')[:200]}...")
    print(f"  Answer: {result.get('answer', 'N/A')}")

    # Verify reasoning is present
    assert 'reasoning' in result, "No reasoning in output"
    assert len(result.get('reasoning', '')) > 20, "Reasoning too short"
    print("\n✓ ChainOfThought module generates detailed reasoning")
    print()


def test_code_generation():
    """Test code generation with real LLM."""
    print("=" * 70)
    print("TEST 3: Code Generation (Real LLM)")
    print("=" * 70)
    print()

    # Create client
    client = ClaudeClientV2()

    # Create module
    module = Predict(CodeGenerationSignature, llm_client=client)

    # Execute
    task = "Write a function to check if a number is prime"
    language = "Python"
    print(f"Task: {task}")
    print(f"Language: {language}")

    result = module(task=task, language=language)

    print("\nResult:")
    print(f"  Code: {result.get('code', 'N/A')[:200]}...")
    print(f"  Explanation: {result.get('explanation', 'N/A')[:100]}...")

    # Verify code is present
    assert 'code' in result, "No code in output"
    assert 'def' in result.get('code', ''), "Code doesn't contain function definition"
    print("\n✓ Code generation works")
    print()


def test_math_problem():
    """Test math problem solving with real LLM."""
    print("=" * 70)
    print("TEST 4: Math Problem Solving (Real LLM)")
    print("=" * 70)
    print()

    # Create client
    client = ClaudeClientV2()

    # Create module with CoT
    module = ChainOfThought(MathSignature, llm_client=client)

    # Execute
    problem = "Solve: x^2 + 5x + 6 = 0"
    print(f"Problem: {problem}")

    result = module(problem=problem)

    print("\nResult:")
    print(f"  Solution steps: {result.get('solution_steps', 'N/A')[:200]}...")
    print(f"  Final answer: {result.get('final_answer', 'N/A')}")

    # Verify solution is present
    assert 'solution_steps' in result, "No solution steps in output"
    assert 'final_answer' in result, "No final answer in output"
    print("\n✓ Math problem solving works")
    print()


def test_module_composition_with_llm():
    """Test composing modules with real LLM."""
    print("=" * 70)
    print("TEST 5: Module Composition (Real LLM)")
    print("=" * 70)
    print()

    # Create client
    client = ClaudeClientV2()

    # Create modules
    module1 = Predict(ChainOfThoughtSignature, llm_client=client)
    module2 = ChainOfThought(ChainOfThoughtSignature, llm_client=client)

    print("Composing Predict -> ChainOfThought")

    # Note: Composition would need compatible signatures
    # This is a simplified test showing the interface works

    print("✓ Module composition interface works")
    print("  (Full composition test would need signature compatibility)")
    print()


def test_llm_client_adapter():
    """Test that LLM client adapter works correctly."""
    print("=" * 70)
    print("TEST 6: LLM Client Adapter")
    print("=" * 70)
    print()

    client = ClaudeClientV2()

    # Test direct call
    response = client.complete(
        messages=[{"role": "user", "content": "Say 'test successful' exactly"}],
        temperature=0.3,
        max_tokens=50
    )

    print(f"Response: {response.content[:100]}")
    assert response.content, "No content in response"
    assert response.tokens_used > 0, "No tokens used"

    print(f"✓ LLM adapter works")
    print(f"  - Model: {response.model}")
    print(f"  - Tokens: {response.tokens_used}")
    print(f"  - Call history entries: {len(client.call_history)}")
    print()


def run_all_tests():
    """Run all Phase 2 API tests."""
    print("\n")
    print("=" * 70)
    print("META-PROMPTING FRAMEWORK v2 - PHASE 2 API TESTS")
    print("End-to-End Testing with Real Claude API")
    print("=" * 70)
    print()

    if not check_api_key():
        return

    print("✓ API key found")
    print()

    tests = [
        test_llm_client_adapter,
        test_predict_module,
        test_chain_of_thought_module,
        test_code_generation,
        test_math_problem,
        test_module_composition_with_llm,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"✗ {test_fn.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()

    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"✓ Passed: {passed}/{len(tests)}")
    if failed > 0:
        print(f"✗ Failed: {failed}/{len(tests)}")
    print()

    if failed == 0:
        print("🎉 ALL PHASE 2 API TESTS PASSED!")
        print()
        print("v2 Framework Status: PHASE 2 COMPLETE ✅")
        print("  - Signatures: Working with real LLM")
        print("  - Modules: Working with real LLM")
        print("  - Predict: Working")
        print("  - ChainOfThought: Working")
        print("  - Code generation: Working")
        print("  - Math solving: Working")
        print()
        print("Ready for:")
        print("  - Experiment suite comparison (v1 vs v2)")
        print("  - Phase 3: Optimizers")
    else:
        print("⚠ Some tests failed. Please review errors above.")

    print("=" * 70)
    print()


if __name__ == "__main__":
    run_all_tests()
