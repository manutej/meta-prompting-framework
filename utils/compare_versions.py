#!/usr/bin/env python3
"""
Version Comparison Utility

Compares v1 (meta_prompting_engine) and v2 (meta_prompting_framework) performance.
"""

import sys
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
import argparse

# v1 imports
try:
    from meta_prompting_engine.core import MetaPromptingEngine
    from meta_prompting_engine.llm_clients.claude import ClaudeClient
    V1_AVAILABLE = True
except ImportError:
    V1_AVAILABLE = False
    print("Warning: v1 (meta_prompting_engine) not available")

# v2 imports
try:
    from meta_prompting_framework.categorical import RMPMonad
    from meta_prompting_framework.prompts import ChainOfThought, ChainOfThoughtSignature
    from meta_prompting_framework.llm import create_v2_client
    V2_AVAILABLE = True
except ImportError:
    V2_AVAILABLE = False
    print("Warning: v2 (meta_prompting_framework) not available")


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    version: str
    output: str
    quality_score: float
    iterations: int
    total_tokens: int
    execution_time: float
    error: Optional[str] = None


class VersionComparator:
    """Compare v1 and v2 meta-prompting frameworks."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize comparator.

        Args:
            api_key: Anthropic API key (optional, will use env var if not provided)
        """
        self.api_key = api_key
        self.results = []

    def run_v1(
        self,
        task: str,
        skill: str = "general_expert",
        max_iterations: int = 3,
        quality_threshold: float = 0.9
    ) -> BenchmarkResult:
        """
        Run task with v1 (meta_prompting_engine).

        Args:
            task: Task to execute
            skill: Skill/role for the engine
            max_iterations: Maximum recursive iterations
            quality_threshold: Early stopping threshold

        Returns:
            Benchmark result
        """
        if not V1_AVAILABLE:
            return BenchmarkResult(
                version="v1",
                output="",
                quality_score=0.0,
                iterations=0,
                total_tokens=0,
                execution_time=0.0,
                error="v1 not available"
            )

        try:
            start_time = time.time()

            # Initialize v1 engine
            client = ClaudeClient(api_key=self.api_key)
            engine = MetaPromptingEngine(skill=skill, llm_client=client)

            # Execute
            result = engine.execute_with_meta_prompting(
                task=task,
                max_iterations=max_iterations,
                quality_threshold=quality_threshold,
                verbose=False
            )

            execution_time = time.time() - start_time

            return BenchmarkResult(
                version="v1",
                output=result.output,
                quality_score=result.quality_score,
                iterations=result.iterations,
                total_tokens=result.total_tokens,
                execution_time=execution_time
            )

        except Exception as e:
            return BenchmarkResult(
                version="v1",
                output="",
                quality_score=0.0,
                iterations=0,
                total_tokens=0,
                execution_time=0.0,
                error=str(e)
            )

    def run_v2(
        self,
        task: str,
        max_iterations: int = 3,
        quality_threshold: float = 0.9
    ) -> BenchmarkResult:
        """
        Run task with v2 (meta_prompting_framework).

        Uses Phase 2 modules (ChainOfThought with real LLM calls).

        Args:
            task: Task to execute
            max_iterations: Maximum recursive iterations
            quality_threshold: Early stopping threshold

        Returns:
            Benchmark result
        """
        if not V2_AVAILABLE:
            return BenchmarkResult(
                version="v2",
                output="",
                quality_score=0.0,
                iterations=0,
                total_tokens=0,
                execution_time=0.0,
                error="v2 not available"
            )

        try:
            start_time = time.time()

            # Create v2 client and module
            client = create_v2_client("claude", api_key=self.api_key)
            module = ChainOfThought(ChainOfThoughtSignature, llm_client=client)

            # Execute - v2 doesn't have recursive meta-prompting loop yet
            # For now, just run once with ChainOfThought
            result = module(question=task)

            execution_time = time.time() - start_time

            # Extract outputs
            output_text = result.get('answer', '') or result.get('reasoning', '')

            # Get token count from client history
            total_tokens = sum(
                call['tokens']
                for call in client.call_history
            ) if hasattr(client, 'call_history') else 0

            return BenchmarkResult(
                version="v2",
                output=output_text,
                quality_score=0.8,  # TODO: Add quality assessment when Phase 3 ready
                iterations=1,  # Single iteration for now
                total_tokens=total_tokens,
                execution_time=execution_time
            )

        except Exception as e:
            return BenchmarkResult(
                version="v2",
                output="",
                quality_score=0.0,
                iterations=0,
                total_tokens=0,
                execution_time=0.0,
                error=str(e)
            )

    def compare(
        self,
        task: str,
        max_iterations: int = 3,
        quality_threshold: float = 0.9
    ) -> Dict[str, BenchmarkResult]:
        """
        Run task on both v1 and v2, compare results.

        Args:
            task: Task to execute
            max_iterations: Maximum recursive iterations
            quality_threshold: Early stopping threshold

        Returns:
            Dictionary mapping version to results
        """
        print(f"\n{'='*60}")
        print(f"Comparing v1 vs v2 on task:")
        print(f"  {task}")
        print(f"{'='*60}\n")

        results = {}

        # Run v1
        print("Running v1 (meta_prompting_engine)...")
        v1_result = self.run_v1(task, max_iterations=max_iterations, quality_threshold=quality_threshold)
        results["v1"] = v1_result
        print(f"  ✓ Completed in {v1_result.execution_time:.2f}s")

        # Run v2
        print("\nRunning v2 (meta_prompting_framework)...")
        v2_result = self.run_v2(task, max_iterations=max_iterations, quality_threshold=quality_threshold)
        results["v2"] = v2_result
        print(f"  ✓ Completed in {v2_result.execution_time:.2f}s")

        self.results.append(results)
        return results

    def print_comparison(self, results: Dict[str, BenchmarkResult]):
        """
        Pretty print comparison results.

        Args:
            results: Comparison results from compare()
        """
        print(f"\n{'='*60}")
        print("COMPARISON RESULTS")
        print(f"{'='*60}\n")

        for version, result in results.items():
            print(f"{version.upper()}:")
            print(f"  Quality Score: {result.quality_score:.2f}")
            print(f"  Iterations: {result.iterations}")
            print(f"  Tokens Used: {result.total_tokens:,}")
            print(f"  Execution Time: {result.execution_time:.2f}s")
            if result.error:
                print(f"  Error: {result.error}")
            print(f"  Output Preview: {result.output[:200]}...")
            print()

        # Comparison
        if results["v1"].error is None and results["v2"].error is None:
            print("WINNER:")
            if results["v1"].quality_score > results["v2"].quality_score:
                print("  🏆 v1 (higher quality)")
            elif results["v2"].quality_score > results["v1"].quality_score:
                print("  🏆 v2 (higher quality)")
            else:
                print("  🤝 Tie (equal quality)")

            if results["v1"].execution_time < results["v2"].execution_time:
                print("  ⚡ v1 (faster)")
            elif results["v2"].execution_time < results["v1"].execution_time:
                print("  ⚡ v2 (faster)")
            else:
                print("  ⚡ Tie (equal speed)")

            if results["v1"].total_tokens < results["v2"].total_tokens:
                print("  💰 v1 (lower cost)")
            elif results["v2"].total_tokens < results["v1"].total_tokens:
                print("  💰 v2 (lower cost)")
            else:
                print("  💰 Tie (equal cost)")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compare v1 and v2 meta-prompting frameworks"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task to execute"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Maximum recursive iterations (default: 3)"
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.9,
        help="Early stopping quality threshold (default: 0.9)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Anthropic API key (optional, uses ANTHROPIC_API_KEY env var if not provided)"
    )

    args = parser.parse_args()

    # Create comparator
    comparator = VersionComparator(api_key=args.api_key)

    # Run comparison
    results = comparator.compare(
        task=args.task,
        max_iterations=args.iterations,
        quality_threshold=args.quality_threshold
    )

    # Print results
    comparator.print_comparison(results)


if __name__ == "__main__":
    main()
