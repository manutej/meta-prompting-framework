#!/usr/bin/env python3
"""
Comprehensive Experiments: v1 vs v2 on Non-Trivial Domains

Tests both frameworks on challenging, diverse problem spaces that require
real reasoning, multi-step thinking, and domain expertise.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.compare_versions import VersionComparator


# Non-trivial experiment definitions
EXPERIMENTS = [
    {
        "id": 1,
        "name": "Advanced Mathematical Reasoning",
        "domain": "Mathematics",
        "difficulty": "Hard",
        "task": """Solve this system of equations and explain your reasoning:

x^2 + y^2 = 25
x + y = 7

Find all real solutions (x, y) and verify your answer.""",
        "requires": ["algebraic manipulation", "equation solving", "verification"]
    },

    {
        "id": 2,
        "name": "Algorithm Design with Complexity Analysis",
        "domain": "Computer Science",
        "difficulty": "Hard",
        "task": """Design an efficient algorithm to find the longest palindromic substring in a given string.

Requirements:
1. Provide the algorithm in pseudocode
2. Analyze time and space complexity
3. Explain why your approach is optimal
4. Give an example walkthrough with the string "babad"
5. Discuss any tradeoffs in your design""",
        "requires": ["algorithm design", "complexity analysis", "optimization thinking"]
    },

    {
        "id": 3,
        "name": "Distributed System Architecture",
        "domain": "System Design",
        "difficulty": "Expert",
        "task": """Design a distributed caching system for a high-traffic e-commerce platform that serves 10 million users.

Address:
1. Cache invalidation strategy
2. Consistency vs availability tradeoffs (CAP theorem)
3. Sharding/partitioning approach
4. Failure handling and recovery
5. Monitoring and observability
6. Estimated capacity requirements

Explain your architectural decisions and tradeoffs.""",
        "requires": ["system design", "distributed systems", "tradeoff analysis"]
    },

    {
        "id": 4,
        "name": "Code Optimization with Tradeoffs",
        "domain": "Software Engineering",
        "difficulty": "Medium-Hard",
        "task": """Optimize this Python function and explain your decisions:

```python
def find_duplicates(arr):
    duplicates = []
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] == arr[j] and arr[i] not in duplicates:
                duplicates.append(arr[i])
    return duplicates
```

Provide:
1. The optimized version with better time complexity
2. Explanation of the optimization
3. Time/space complexity comparison (before and after)
4. Any tradeoffs made
5. When you would/wouldn't use this optimization""",
        "requires": ["code analysis", "optimization", "complexity analysis"]
    },

    {
        "id": 5,
        "name": "Multi-Step Logic Puzzle",
        "domain": "Logic & Reasoning",
        "difficulty": "Hard",
        "task": """Solve this logic puzzle:

Five people (Alice, Bob, Carol, David, Eve) live in five adjacent houses (1-5 from left to right).
- Alice lives in a red house
- The green house is immediately to the right of the white house
- Bob lives in house 3
- Carol doesn't live in the blue house
- The person in house 1 has a dog
- Eve lives in the yellow house
- The person in the red house lives immediately to the left of the person with a cat
- David has a bird
- The person in the green house has a fish

Determine:
1. Who lives in which house number
2. What color is each house
3. What pet does each person have

Show your step-by-step reasoning.""",
        "requires": ["logical deduction", "constraint satisfaction", "systematic reasoning"]
    },

    {
        "id": 6,
        "name": "Machine Learning System Design",
        "domain": "Machine Learning",
        "difficulty": "Expert",
        "task": """Design a recommendation system for a video streaming platform (like Netflix).

Address:
1. Choice of recommendation algorithm (collaborative filtering, content-based, hybrid)
2. Cold start problem solutions
3. Real-time vs batch prediction strategy
4. Feature engineering approach
5. Evaluation metrics
6. A/B testing strategy
7. Handling scalability (millions of users, thousands of videos)

Justify your technical decisions.""",
        "requires": ["ML system design", "algorithm selection", "scalability planning"]
    },

    {
        "id": 7,
        "name": "Database Query Optimization",
        "domain": "Database Systems",
        "difficulty": "Medium-Hard",
        "task": """Given this slow SQL query on a large e-commerce database:

```sql
SELECT o.order_id, u.username, p.product_name, o.order_date
FROM orders o
JOIN users u ON o.user_id = u.user_id
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
WHERE o.order_date >= '2024-01-01'
AND p.category = 'Electronics'
AND o.total_amount > 1000
ORDER BY o.order_date DESC;
```

Tables have millions of rows. Optimize this query by:
1. Suggesting appropriate indexes
2. Rewriting the query if beneficial
3. Explaining the query execution plan improvements
4. Estimating performance gains
5. Discussing any tradeoffs (e.g., write performance, storage)""",
        "requires": ["SQL optimization", "indexing strategy", "performance analysis"]
    },

    {
        "id": 8,
        "name": "Cryptographic Protocol Design",
        "domain": "Security",
        "difficulty": "Expert",
        "task": """Design a secure authentication protocol for a mobile banking app.

Requirements:
1. Protect against common attacks (MITM, replay, phishing)
2. Support both biometric and PIN authentication
3. Handle network failures gracefully
4. Minimize user friction
5. Comply with security best practices

Provide:
1. Protocol flow diagram (in text)
2. Cryptographic primitives used (and why)
3. Threat model and mitigations
4. Session management strategy
5. Recovery mechanisms for lost credentials""",
        "requires": ["security design", "protocol design", "threat modeling"]
    },

    {
        "id": 9,
        "name": "Compiler Optimization Technique",
        "domain": "Compilers",
        "difficulty": "Hard",
        "task": """Explain how a compiler would optimize this code:

```c
int sum = 0;
for (int i = 0; i < 1000; i++) {
    sum += i;
}
return sum;
```

Describe:
1. Specific optimization techniques applicable (e.g., loop unrolling, strength reduction, constant folding)
2. The optimized version the compiler would generate
3. Why these optimizations are valid (correctness preservation)
4. Any scenarios where these optimizations might not apply
5. Expected performance improvement

Be technical and precise.""",
        "requires": ["compiler theory", "optimization techniques", "program analysis"]
    },

    {
        "id": 10,
        "name": "Complex Debugging Scenario",
        "domain": "Software Engineering",
        "difficulty": "Hard",
        "task": """A production web service is experiencing intermittent 500 errors (about 2% of requests).

Symptoms:
- Errors are random, not tied to specific endpoints
- Happens more during high traffic (but not consistently)
- Database queries are fast
- No errors in application logs
- Memory usage is normal
- CPU usage spikes briefly when errors occur

Debugging approach:
1. What would you investigate first and why?
2. What metrics/logs would you examine?
3. What are the likely root causes (rank by probability)?
4. How would you reproduce the issue?
5. What monitoring would you add to catch this earlier?
6. Describe your step-by-step debugging strategy""",
        "requires": ["debugging", "system analysis", "production troubleshooting"]
    }
]


def check_api_key():
    """Check if API key is available."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("=" * 80)
        print("⚠ ANTHROPIC_API_KEY not set")
        print("=" * 80)
        print()
        print("This experiment suite requires a real Claude API key.")
        print()
        print("To run experiments:")
        print("  export ANTHROPIC_API_KEY='your-key-here'")
        print("  python experiments/run_comprehensive_experiments.py")
        print()
        return False
    return True


def run_all_experiments(max_iterations=3, quality_threshold=0.9, save_results=True):
    """Run all experiments and save results."""

    if not check_api_key():
        return

    print("\n")
    print("=" * 80)
    print("COMPREHENSIVE EXPERIMENTS: v1 vs v2")
    print("Non-Trivial Domains & Example Spaces")
    print("=" * 80)
    print()
    print(f"Total Experiments: {len(EXPERIMENTS)}")
    print(f"Max Iterations: {max_iterations}")
    print(f"Quality Threshold: {quality_threshold}")
    print()

    # Create comparator
    comparator = VersionComparator()

    all_results = []

    for exp in EXPERIMENTS:
        print("\n" + "=" * 80)
        print(f"EXPERIMENT {exp['id']}/{len(EXPERIMENTS)}: {exp['name']}")
        print("=" * 80)
        print(f"Domain: {exp['domain']}")
        print(f"Difficulty: {exp['difficulty']}")
        print(f"Requires: {', '.join(exp['requires'])}")
        print()

        # Run comparison
        results = comparator.compare(
            task=exp['task'],
            max_iterations=max_iterations,
            quality_threshold=quality_threshold
        )

        # Print comparison
        comparator.print_comparison(results)

        # Store results
        exp_result = {
            "experiment": exp,
            "timestamp": datetime.now().isoformat(),
            "v1": {
                "output": results["v1"].output,
                "quality": results["v1"].quality_score,
                "iterations": results["v1"].iterations,
                "tokens": results["v1"].total_tokens,
                "time": results["v1"].execution_time,
                "error": results["v1"].error
            },
            "v2": {
                "output": results["v2"].output,
                "quality": results["v2"].quality_score,
                "iterations": results["v2"].iterations,
                "tokens": results["v2"].total_tokens,
                "time": results["v2"].execution_time,
                "error": results["v2"].error
            }
        }
        all_results.append(exp_result)

        print()
        input("Press Enter to continue to next experiment...")

    # Save results
    if save_results:
        results_file = f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results_path = Path(__file__).parent / results_file

        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        print()
        print("=" * 80)
        print(f"Results saved to: {results_file}")
        print("=" * 80)

    # Generate summary
    generate_summary(all_results)

    return all_results


def generate_summary(results):
    """Generate summary statistics from experiment results."""
    print("\n")
    print("=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print()

    v1_wins = 0
    v2_wins = 0
    ties = 0

    v1_total_time = 0
    v2_total_time = 0
    v1_total_tokens = 0
    v2_total_tokens = 0

    for result in results:
        v1 = result["v1"]
        v2 = result["v2"]

        # Quality comparison
        if v1["error"] is None and v2["error"] is None:
            if v1["quality"] > v2["quality"]:
                v1_wins += 1
            elif v2["quality"] > v1["quality"]:
                v2_wins += 1
            else:
                ties += 1

        # Accumulate metrics
        v1_total_time += v1["time"]
        v2_total_time += v2["time"]
        v1_total_tokens += v1["tokens"]
        v2_total_tokens += v2["tokens"]

    total_experiments = len(results)

    print(f"Total Experiments: {total_experiments}")
    print()
    print("Quality Wins:")
    print(f"  v1: {v1_wins} ({v1_wins/total_experiments*100:.1f}%)")
    print(f"  v2: {v2_wins} ({v2_wins/total_experiments*100:.1f}%)")
    print(f"  Ties: {ties} ({ties/total_experiments*100:.1f}%)")
    print()
    print("Performance:")
    print(f"  v1 avg time: {v1_total_time/total_experiments:.2f}s")
    print(f"  v2 avg time: {v2_total_time/total_experiments:.2f}s")
    print()
    print("Token Usage:")
    print(f"  v1 total: {v1_total_tokens:,}")
    print(f"  v2 total: {v2_total_tokens:,}")
    print(f"  v1 avg: {v1_total_tokens/total_experiments:,.0f}")
    print(f"  v2 avg: {v2_total_tokens/total_experiments:,.0f}")
    print()

    # Domain breakdown
    domain_stats = {}
    for result in results:
        domain = result["experiment"]["domain"]
        if domain not in domain_stats:
            domain_stats[domain] = {"v1": 0, "v2": 0, "tie": 0}

        v1 = result["v1"]
        v2 = result["v2"]

        if v1["error"] is None and v2["error"] is None:
            if v1["quality"] > v2["quality"]:
                domain_stats[domain]["v1"] += 1
            elif v2["quality"] > v1["quality"]:
                domain_stats[domain]["v2"] += 1
            else:
                domain_stats[domain]["tie"] += 1

    print("Performance by Domain:")
    for domain, stats in sorted(domain_stats.items()):
        print(f"  {domain}:")
        print(f"    v1: {stats['v1']}, v2: {stats['v2']}, ties: {stats['tie']}")
    print()


def run_quick_sample(num_experiments=3):
    """Run a quick sample of experiments."""
    print("\n")
    print("=" * 80)
    print("QUICK SAMPLE: Running first 3 experiments")
    print("=" * 80)
    print()

    # Use only first N experiments
    sample_experiments = EXPERIMENTS[:num_experiments]

    # Temporarily replace global
    global EXPERIMENTS
    original = EXPERIMENTS
    EXPERIMENTS = sample_experiments

    results = run_all_experiments(save_results=False)

    EXPERIMENTS = original
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run comprehensive v1 vs v2 experiments on non-trivial domains"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick sample (first 3 experiments only)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Maximum iterations per experiment (default: 3)"
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.9,
        help="Quality threshold for early stopping (default: 0.9)"
    )
    parser.add_argument(
        "--experiment",
        type=int,
        help="Run specific experiment by ID (1-10)"
    )

    args = parser.parse_args()

    if args.quick:
        run_quick_sample()
    elif args.experiment:
        exp = EXPERIMENTS[args.experiment - 1]
        EXPERIMENTS = [exp]
        run_all_experiments(
            max_iterations=args.iterations,
            quality_threshold=args.quality_threshold
        )
    else:
        run_all_experiments(
            max_iterations=args.iterations,
            quality_threshold=args.quality_threshold
        )
