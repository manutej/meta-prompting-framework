"""
Comprehensive Experiment Suite: v1 vs v2 Comparison

Tests both frameworks on 10 practical tasks representing real-world use cases.
"""

import json
import time
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.compare_versions import VersionComparator, BenchmarkResult


@dataclass
class Experiment:
    """Definition of a single experiment."""
    id: int
    name: str
    category: str
    task: str
    description: str
    expected_quality_min: float = 0.6


# Define 10 practical experiments
EXPERIMENTS: List[Experiment] = [
    Experiment(
        id=1,
        name="Binary Search Implementation",
        category="Code Generation",
        task="Write a binary search function in Python with comprehensive error handling, type hints, and docstrings. Include edge cases for empty arrays and single elements.",
        description="Tests ability to generate production-ready algorithms with proper documentation",
        expected_quality_min=0.7
    ),

    Experiment(
        id=2,
        name="Mathematical Problem Solving",
        category="Problem Solving",
        task="Solve this problem step-by-step: A train travels from City A to City B at 60 mph and returns at 40 mph. What is the average speed for the entire journey? Explain why it's not simply 50 mph.",
        description="Tests mathematical reasoning and explanation capabilities",
        expected_quality_min=0.75
    ),

    Experiment(
        id=3,
        name="Code Refactoring",
        category="Code Optimization",
        task="Refactor this code to be more Pythonic and efficient:\n\nresult = []\nfor i in range(len(items)):\n    if items[i] % 2 == 0:\n        result.append(items[i] * 2)\nreturn result",
        description="Tests ability to improve code quality and apply best practices",
        expected_quality_min=0.7
    ),

    Experiment(
        id=4,
        name="System Design Explanation",
        category="Technical Explanation",
        task="Explain how a distributed cache (like Redis) maintains consistency across multiple nodes. Include CAP theorem implications and common strategies like leader election or eventual consistency.",
        description="Tests ability to explain complex distributed systems concepts",
        expected_quality_min=0.65
    ),

    Experiment(
        id=5,
        name="Bug Diagnosis",
        category="Debugging",
        task="This code has a subtle bug:\n\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\nIdentify the performance issue and provide an optimized solution with memoization.",
        description="Tests debugging and optimization skills",
        expected_quality_min=0.7
    ),

    Experiment(
        id=6,
        name="API Design",
        category="Software Design",
        task="Design a RESTful API for a task management system. Include endpoints for CRUD operations on tasks, user authentication, task assignment, and filtering by status/priority. Provide endpoint specs with HTTP methods, request/response formats.",
        description="Tests architectural and API design capabilities",
        expected_quality_min=0.65
    ),

    Experiment(
        id=7,
        name="Data Structure Selection",
        category="Problem Solving",
        task="I need to store user sessions with the following requirements: fast lookup by session ID, automatic expiration after 30 minutes, and ability to get all active sessions. Which data structure(s) should I use and why? Provide implementation guidance.",
        description="Tests ability to choose appropriate data structures for requirements",
        expected_quality_min=0.7
    ),

    Experiment(
        id=8,
        name="SQL Query Optimization",
        category="Database Optimization",
        task="This SQL query is slow:\n\nSELECT u.name, COUNT(o.id) \nFROM users u, orders o \nWHERE u.id = o.user_id AND o.created_at > '2024-01-01'\nGROUP BY u.name\n\nIdentify issues and provide an optimized version with proper JOINs, indexes, and explain the improvements.",
        description="Tests database query optimization knowledge",
        expected_quality_min=0.65
    ),

    Experiment(
        id=9,
        name="Test Case Generation",
        category="Testing",
        task="Generate comprehensive test cases for a function that validates email addresses. Include positive cases, negative cases, edge cases, and security considerations (like injection attempts).",
        description="Tests ability to think through testing scenarios comprehensively",
        expected_quality_min=0.7
    ),

    Experiment(
        id=10,
        name="Technical Documentation",
        category="Documentation",
        task="Write a README section explaining how to use a rate limiter library. Include installation, basic usage, advanced configuration (burst limits, sliding windows), and common pitfalls to avoid.",
        description="Tests technical writing and documentation skills",
        expected_quality_min=0.65
    ),
]


class ExperimentRunner:
    """Runs experiment suite and generates reports."""

    def __init__(self, api_key: str = None):
        self.comparator = VersionComparator(api_key=api_key)
        self.results: List[Dict[str, Any]] = []

    def run_experiment(self, experiment: Experiment) -> Dict[str, Any]:
        """
        Run a single experiment.

        Args:
            experiment: Experiment to run

        Returns:
            Dictionary with experiment results
        """
        print(f"\n{'='*70}")
        print(f"Experiment {experiment.id}: {experiment.name}")
        print(f"Category: {experiment.category}")
        print(f"{'='*70}")
        print(f"\nTask: {experiment.task[:200]}...")
        print()

        # Run comparison
        comparison = self.comparator.compare(
            task=experiment.task,
            max_iterations=3,
            quality_threshold=0.9
        )

        # Package results
        result = {
            "experiment": asdict(experiment),
            "v1_result": asdict(comparison["v1"]),
            "v2_result": asdict(comparison["v2"]),
            "timestamp": time.time()
        }

        # Calculate winner
        v1 = comparison["v1"]
        v2 = comparison["v2"]

        winner = {}
        if v1.error is None and v2.error is None:
            winner["quality"] = "v1" if v1.quality_score > v2.quality_score else "v2" if v2.quality_score > v1.quality_score else "tie"
            winner["speed"] = "v1" if v1.execution_time < v2.execution_time else "v2" if v2.execution_time < v1.execution_time else "tie"
            winner["cost"] = "v1" if v1.total_tokens < v2.total_tokens else "v2" if v2.total_tokens < v1.total_tokens else "tie"
        else:
            winner["quality"] = "v1" if v1.error is None else "v2" if v2.error is None else "both_failed"
            winner["speed"] = "n/a"
            winner["cost"] = "n/a"

        result["winner"] = winner

        # Print summary
        print(f"\n{'='*70}")
        print("RESULTS:")
        print(f"  v1 Quality: {v1.quality_score:.2f} | v2 Quality: {v2.quality_score:.2f} → Winner: {winner['quality']}")
        print(f"  v1 Speed: {v1.execution_time:.2f}s | v2 Speed: {v2.execution_time:.2f}s → Winner: {winner['speed']}")
        print(f"  v1 Tokens: {v1.total_tokens:,} | v2 Tokens: {v2.total_tokens:,} → Winner: {winner['cost']}")
        print(f"{'='*70}\n")

        return result

    def run_suite(self, experiments: List[Experiment] = None) -> List[Dict[str, Any]]:
        """
        Run full experiment suite.

        Args:
            experiments: List of experiments (defaults to all)

        Returns:
            List of experiment results
        """
        if experiments is None:
            experiments = EXPERIMENTS

        print(f"\n{'#'*70}")
        print(f"# EXPERIMENT SUITE: v1 vs v2 Comparison")
        print(f"# Total Experiments: {len(experiments)}")
        print(f"{'#'*70}\n")

        results = []
        for exp in experiments:
            try:
                result = self.run_experiment(exp)
                results.append(result)
                self.results.append(result)
            except Exception as e:
                print(f"❌ Experiment {exp.id} failed: {e}")
                results.append({
                    "experiment": asdict(exp),
                    "error": str(e),
                    "timestamp": time.time()
                })

        return results

    def generate_report(self, output_file: str = "experiment_results.json"):
        """
        Generate comprehensive report.

        Args:
            output_file: Path to output JSON file
        """
        # Save raw results
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n{'#'*70}")
        print(f"# FINAL REPORT")
        print(f"{'#'*70}\n")

        # Calculate statistics
        v1_wins_quality = 0
        v2_wins_quality = 0
        ties_quality = 0

        v1_wins_speed = 0
        v2_wins_speed = 0
        ties_speed = 0

        v1_wins_cost = 0
        v2_wins_cost = 0
        ties_cost = 0

        v1_avg_quality = 0
        v2_avg_quality = 0
        v1_avg_time = 0
        v2_avg_time = 0
        v1_total_tokens = 0
        v2_total_tokens = 0

        successful_experiments = 0

        for result in self.results:
            if "winner" in result:
                winner = result["winner"]

                # Quality
                if winner["quality"] == "v1":
                    v1_wins_quality += 1
                elif winner["quality"] == "v2":
                    v2_wins_quality += 1
                elif winner["quality"] == "tie":
                    ties_quality += 1

                # Speed
                if winner["speed"] == "v1":
                    v1_wins_speed += 1
                elif winner["speed"] == "v2":
                    v2_wins_speed += 1
                elif winner["speed"] == "tie":
                    ties_speed += 1

                # Cost
                if winner["cost"] == "v1":
                    v1_wins_cost += 1
                elif winner["cost"] == "v2":
                    v2_wins_cost += 1
                elif winner["cost"] == "tie":
                    ties_cost += 1

                # Averages
                if result["v1_result"]["error"] is None:
                    v1_avg_quality += result["v1_result"]["quality_score"]
                    v1_avg_time += result["v1_result"]["execution_time"]
                    v1_total_tokens += result["v1_result"]["total_tokens"]

                if result["v2_result"]["error"] is None:
                    v2_avg_quality += result["v2_result"]["quality_score"]
                    v2_avg_time += result["v2_result"]["execution_time"]
                    v2_total_tokens += result["v2_result"]["total_tokens"]

                successful_experiments += 1

        if successful_experiments > 0:
            v1_avg_quality /= successful_experiments
            v2_avg_quality /= successful_experiments
            v1_avg_time /= successful_experiments
            v2_avg_time /= successful_experiments

        # Print summary
        print(f"Total Experiments: {len(self.results)}")
        print(f"Successful: {successful_experiments}")
        print(f"\n{'='*70}")
        print("QUALITY COMPARISON:")
        print(f"  v1 Wins: {v1_wins_quality} | v2 Wins: {v2_wins_quality} | Ties: {ties_quality}")
        print(f"  v1 Avg Quality: {v1_avg_quality:.2f}")
        print(f"  v2 Avg Quality: {v2_avg_quality:.2f}")

        print(f"\n{'='*70}")
        print("SPEED COMPARISON:")
        print(f"  v1 Wins: {v1_wins_speed} | v2 Wins: {v2_wins_speed} | Ties: {ties_speed}")
        print(f"  v1 Avg Time: {v1_avg_time:.2f}s")
        print(f"  v2 Avg Time: {v2_avg_time:.2f}s")

        print(f"\n{'='*70}")
        print("COST COMPARISON:")
        print(f"  v1 Wins: {v1_wins_cost} | v2 Wins: {v2_wins_cost} | Ties: {ties_cost}")
        print(f"  v1 Total Tokens: {v1_total_tokens:,}")
        print(f"  v2 Total Tokens: {v2_total_tokens:,}")

        print(f"\n{'='*70}")
        print(f"OVERALL WINNER:")

        # Determine overall winner
        v1_total_wins = v1_wins_quality + v1_wins_speed + v1_wins_cost
        v2_total_wins = v2_wins_quality + v2_wins_speed + v2_wins_cost

        if v1_total_wins > v2_total_wins:
            print(f"  🏆 v1 (meta_prompting_engine)")
            print(f"     {v1_total_wins} total category wins vs {v2_total_wins}")
        elif v2_total_wins > v1_total_wins:
            print(f"  🏆 v2 (meta_prompting_framework)")
            print(f"     {v2_total_wins} total category wins vs {v1_total_wins}")
        else:
            print(f"  🤝 TIE")
            print(f"     Both: {v1_total_wins} category wins")

        print(f"\n{'='*70}")
        print(f"\nDetailed results saved to: {output_file}")
        print()

    def generate_markdown_report(self, output_file: str = "EXPERIMENT_REPORT.md"):
        """Generate a markdown report for documentation."""

        with open(output_file, 'w') as f:
            f.write("# Meta-Prompting Framework: v1 vs v2 Experimental Comparison\n\n")
            f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Experiments:** {len(self.results)}\n\n")

            f.write("---\n\n")
            f.write("## Experiment Suite\n\n")

            # Summary table
            f.write("| # | Name | Category | v1 Quality | v2 Quality | Winner |\n")
            f.write("|---|------|----------|------------|------------|--------|\n")

            for result in self.results:
                if "winner" in result:
                    exp = result["experiment"]
                    v1 = result["v1_result"]
                    v2 = result["v2_result"]
                    winner = result["winner"]["quality"]

                    winner_icon = "🏆 v1" if winner == "v1" else "🏆 v2" if winner == "v2" else "🤝 Tie"

                    f.write(f"| {exp['id']} | {exp['name']} | {exp['category']} | ")
                    f.write(f"{v1['quality_score']:.2f} | {v2['quality_score']:.2f} | {winner_icon} |\n")

            f.write("\n---\n\n")
            f.write("## Detailed Results\n\n")

            for result in self.results:
                if "winner" in result:
                    exp = result["experiment"]
                    v1 = result["v1_result"]
                    v2 = result["v2_result"]
                    winner = result["winner"]

                    f.write(f"### Experiment {exp['id']}: {exp['name']}\n\n")
                    f.write(f"**Category:** {exp['category']}\n\n")
                    f.write(f"**Task:** {exp['task']}\n\n")

                    f.write("**Results:**\n\n")
                    f.write("| Metric | v1 | v2 | Winner |\n")
                    f.write("|--------|----|----|--------|\n")
                    f.write(f"| Quality | {v1['quality_score']:.2f} | {v2['quality_score']:.2f} | {winner['quality']} |\n")
                    f.write(f"| Speed | {v1['execution_time']:.2f}s | {v2['execution_time']:.2f}s | {winner['speed']} |\n")
                    f.write(f"| Tokens | {v1['total_tokens']:,} | {v2['total_tokens']:,} | {winner['cost']} |\n")
                    f.write(f"| Iterations | {v1['iterations']} | {v2['iterations']} | - |\n")

                    f.write("\n---\n\n")

        print(f"Markdown report saved to: {output_file}")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run experiment suite comparing v1 and v2")
    parser.add_argument("--api-key", type=str, help="Anthropic API key")
    parser.add_argument("--experiments", type=str, help="Comma-separated experiment IDs (e.g., '1,2,3')")
    parser.add_argument("--output", type=str, default="experiment_results.json", help="Output file")
    parser.add_argument("--markdown", type=str, default="EXPERIMENT_REPORT.md", help="Markdown report file")

    args = parser.parse_args()

    # Select experiments
    experiments = EXPERIMENTS
    if args.experiments:
        ids = [int(x.strip()) for x in args.experiments.split(",")]
        experiments = [exp for exp in EXPERIMENTS if exp.id in ids]

    # Run experiments
    runner = ExperimentRunner(api_key=args.api_key)
    runner.run_suite(experiments)

    # Generate reports
    runner.generate_report(output_file=args.output)
    runner.generate_markdown_report(output_file=args.markdown)


if __name__ == "__main__":
    main()
