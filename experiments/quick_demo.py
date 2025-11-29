"""
Quick Demo: Run 3 lightweight experiments to demonstrate the comparison system.

This is a fast demo that doesn't require API calls for v2 (since Phase 2 isn't ready).
Shows the structure and output format.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.run_suite import ExperimentRunner, Experiment


# 3 lightweight experiments for quick demonstration
DEMO_EXPERIMENTS = [
    Experiment(
        id=1,
        name="Simple Code Task",
        category="Code Generation",
        task="Write a Python function that checks if a string is a palindrome. Include docstring.",
        description="Quick code generation test",
        expected_quality_min=0.7
    ),

    Experiment(
        id=2,
        name="Quick Math Problem",
        category="Problem Solving",
        task="Calculate: If you save $100 per month at 5% annual interest compounded monthly, how much will you have after 1 year?",
        description="Simple calculation test",
        expected_quality_min=0.7
    ),

    Experiment(
        id=3,
        name="Code Review",
        category="Code Optimization",
        task="Review this code and suggest improvements:\n\nx = []\nfor i in range(10):\n    x.append(i * i)",
        description="Quick refactoring test",
        expected_quality_min=0.65
    ),
]


def main():
    """Run quick demo."""
    print("\n" + "="*70)
    print("QUICK DEMO: v1 vs v2 Comparison (3 experiments)")
    print("="*70)
    print("\nThis demo runs 3 lightweight experiments to show the system.")
    print("For full suite (10 experiments), run: python -m experiments.run_suite")
    print()

    # Check if API key is available
    import os
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  WARNING: ANTHROPIC_API_KEY not set")
        print("   v1 tests will fail without API key")
        print("   v2 tests will run (Phase 1 demo mode)\n")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting. Set API key and try again.")
            return

    # Run experiments
    runner = ExperimentRunner()
    runner.run_suite(experiments=DEMO_EXPERIMENTS)

    # Generate reports
    runner.generate_report(output_file="demo_results.json")
    runner.generate_markdown_report(output_file="DEMO_REPORT.md")

    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - demo_results.json (raw data)")
    print("  - DEMO_REPORT.md (human-readable report)")
    print("\nTo run full suite:")
    print("  python -m experiments.run_suite")
    print()


if __name__ == "__main__":
    main()
