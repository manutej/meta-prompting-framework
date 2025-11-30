# Comprehensive Experiments: v1 vs v2 on Non-Trivial Domains

This experiment suite tests both meta-prompting frameworks on **challenging, real-world problems** that require genuine reasoning, domain expertise, and multi-step thinking.

## Overview

**10 experiments** across diverse domains:
- Mathematics
- Computer Science (Algorithms, Systems, Compilers)
- System Design (Distributed Systems, ML Systems)
- Software Engineering (Optimization, Debugging)
- Security
- Logic & Reasoning

**Why these experiments matter:**
- Test real reasoning capabilities, not toy problems
- Require multi-step thinking and planning
- Demand domain expertise
- Have verifiable quality metrics
- Mirror real-world use cases

---

## Experiment Catalog

### 1. Advanced Mathematical Reasoning
**Domain:** Mathematics | **Difficulty:** Hard

Solve a system of non-linear equations:
- x² + y² = 25
- x + y = 7

**Tests:** Algebraic manipulation, equation solving, solution verification

**Why it's non-trivial:** Requires understanding of quadratic equations, substitution method, and careful verification of solutions.

---

### 2. Algorithm Design with Complexity Analysis
**Domain:** Computer Science | **Difficulty:** Hard

Design an efficient algorithm for finding the longest palindromic substring.

**Requirements:**
- Pseudocode implementation
- Time/space complexity analysis
- Optimality justification
- Example walkthrough
- Tradeoff discussion

**Tests:** Algorithm design, complexity theory, optimization thinking

**Why it's non-trivial:** Multiple valid approaches (DP, expand-around-center, Manacher's), requires complexity analysis and justification.

---

### 3. Distributed System Architecture
**Domain:** System Design | **Difficulty:** Expert

Design a distributed caching system for 10M users.

**Must address:**
- Cache invalidation strategy
- CAP theorem tradeoffs
- Sharding approach
- Failure handling
- Monitoring
- Capacity planning

**Tests:** Distributed systems knowledge, architectural thinking, tradeoff analysis

**Why it's non-trivial:** Requires understanding of distributed systems fundamentals, no single "right" answer, must justify decisions.

---

### 4. Code Optimization with Tradeoffs
**Domain:** Software Engineering | **Difficulty:** Medium-Hard

Optimize a nested loop algorithm for finding duplicates.

**Requirements:**
- Optimized implementation
- Before/after complexity comparison
- Tradeoff analysis
- Applicability discussion

**Tests:** Code analysis, optimization techniques, complexity reasoning

**Why it's non-trivial:** Multiple optimization paths (hash set, sorting, bit manipulation), each with different tradeoffs.

---

### 5. Multi-Step Logic Puzzle
**Domain:** Logic & Reasoning | **Difficulty:** Hard

Solve a constraint satisfaction problem with 5 people, 5 houses, 5 colors, 5 pets.

**Given:** 9 clues with interdependencies

**Must determine:** Who lives where, house colors, pet ownership

**Tests:** Logical deduction, constraint satisfaction, systematic reasoning

**Why it's non-trivial:** Requires careful tracking of constraints, backtracking when contradictions arise, systematic approach.

---

### 6. Machine Learning System Design
**Domain:** Machine Learning | **Difficulty:** Expert

Design a Netflix-style recommendation system.

**Must address:**
- Algorithm selection (collaborative filtering vs content-based vs hybrid)
- Cold start problem
- Real-time vs batch predictions
- Feature engineering
- Evaluation metrics
- A/B testing
- Scalability

**Tests:** ML system design, algorithm understanding, practical engineering

**Why it's non-trivial:** Requires ML knowledge, system design skills, and practical engineering considerations.

---

### 7. Database Query Optimization
**Domain:** Database Systems | **Difficulty:** Medium-Hard

Optimize a slow multi-join query on millions of rows.

**Requirements:**
- Index suggestions
- Query rewriting
- Execution plan analysis
- Performance estimation
- Tradeoff discussion (write performance, storage)

**Tests:** SQL optimization, indexing strategy, database internals

**Why it's non-trivial:** Requires understanding of query execution, indexes, and practical database optimization.

---

### 8. Cryptographic Protocol Design
**Domain:** Security | **Difficulty:** Expert

Design secure authentication for mobile banking.

**Must handle:**
- Common attacks (MITM, replay, phishing)
- Biometric + PIN authentication
- Network failures
- User experience
- Security best practices

**Requirements:**
- Protocol flow
- Cryptographic primitives
- Threat model
- Session management
- Recovery mechanisms

**Tests:** Security design, protocol knowledge, threat modeling

**Why it's non-trivial:** Real security stakes, must balance security with UX, requires cryptography knowledge.

---

### 9. Compiler Optimization Technique
**Domain:** Compilers | **Difficulty:** Hard

Explain how a compiler optimizes a simple loop.

**Must cover:**
- Specific optimization techniques (loop unrolling, strength reduction, constant folding)
- Optimized code generation
- Correctness preservation
- When optimizations don't apply
- Performance impact

**Tests:** Compiler theory, optimization techniques, program analysis

**Why it's non-trivial:** Requires deep compiler knowledge, understanding of multiple optimization passes.

---

### 10. Complex Debugging Scenario
**Domain:** Software Engineering | **Difficulty:** Hard

Debug intermittent 500 errors in production (2% of requests).

**Given symptoms:**
- Random errors, not endpoint-specific
- More during high traffic
- Fast DB queries
- No application errors
- Normal memory
- CPU spikes during errors

**Must provide:**
- Investigation priority
- Metrics/logs to examine
- Likely root causes (ranked)
- Reproduction strategy
- Monitoring improvements
- Step-by-step debugging plan

**Tests:** Production debugging, system analysis, methodical troubleshooting

**Why it's non-trivial:** Realistic production scenario, requires systematic debugging approach, no obvious answer.

---

## Running the Experiments

### Prerequisites

1. **API Key:**
   ```bash
   export ANTHROPIC_API_KEY='your-key-here'
   ```

2. **Both frameworks installed:**
   - v1: `meta_prompting_engine/`
   - v2: `meta_prompting_framework/`

### Run All Experiments (10 total)

```bash
python experiments/run_comprehensive_experiments.py
```

**Expected duration:** ~30-60 minutes (depending on iterations)

**Output:**
- Real-time comparison for each experiment
- Saved results in JSON format
- Summary statistics

### Run Quick Sample (First 3 experiments)

```bash
python experiments/run_comprehensive_experiments.py --quick
```

**Expected duration:** ~10-15 minutes

### Run Single Experiment

```bash
# Run experiment #3 (Distributed System Architecture)
python experiments/run_comprehensive_experiments.py --experiment 3
```

### Custom Parameters

```bash
python experiments/run_comprehensive_experiments.py \
  --iterations 5 \
  --quality-threshold 0.95
```

---

## Output Format

### Per-Experiment Output

```
=================================================================
EXPERIMENT 2/10: Algorithm Design with Complexity Analysis
=================================================================
Domain: Computer Science
Difficulty: Hard
Requires: algorithm design, complexity analysis, optimization thinking

Running v1 (meta_prompting_engine)...
  ✓ Completed in 12.34s

Running v2 (meta_prompting_framework)...
  ✓ Completed in 8.76s

=================================================================
COMPARISON RESULTS
=================================================================

V1:
  Quality Score: 0.85
  Iterations: 3
  Tokens Used: 4,523
  Execution Time: 12.34s
  Output Preview: To find the longest palindromic substring...

V2:
  Quality Score: 0.88
  Iterations: 1
  Tokens Used: 2,156
  Execution Time: 8.76s
  Output Preview: The longest palindromic substring can be found...

WINNER:
  🏆 v2 (higher quality)
  ⚡ v2 (faster)
  💰 v2 (lower cost)
```

### Summary Statistics

```
=================================================================
EXPERIMENT SUMMARY
=================================================================

Total Experiments: 10

Quality Wins:
  v1: 3 (30.0%)
  v2: 6 (60.0%)
  Ties: 1 (10.0%)

Performance:
  v1 avg time: 15.23s
  v2 avg time: 9.87s

Token Usage:
  v1 total: 45,230
  v2 total: 28,945
  v1 avg: 4,523
  v2 avg: 2,895

Performance by Domain:
  Mathematics:
    v1: 0, v2: 1, ties: 0
  Computer Science:
    v1: 1, v2: 2, ties: 0
  System Design:
    v1: 1, v2: 1, ties: 0
  Software Engineering:
    v1: 0, v2: 1, ties: 1
  Security:
    v1: 0, v2: 1, ties: 0
  Logic & Reasoning:
    v1: 1, v2: 0, ties: 0
```

### Saved Results

Results are saved to `experiment_results_YYYYMMDD_HHMMSS.json`:

```json
[
  {
    "experiment": {
      "id": 1,
      "name": "Advanced Mathematical Reasoning",
      "domain": "Mathematics",
      "difficulty": "Hard",
      "task": "...",
      "requires": ["algebraic manipulation", "equation solving", "verification"]
    },
    "timestamp": "2025-11-30T10:15:30",
    "v1": {
      "output": "To solve this system...",
      "quality": 0.85,
      "iterations": 3,
      "tokens": 3456,
      "time": 14.2,
      "error": null
    },
    "v2": {
      "output": "Let's solve step by step...",
      "quality": 0.90,
      "iterations": 1,
      "tokens": 2134,
      "time": 8.5,
      "error": null
    }
  },
  ...
]
```

---

## Evaluation Criteria

### Quality Metrics

**v1 Quality Score** (from MetaPromptingEngine):
- Automatically assessed by v1's quality evaluation
- Based on completeness, correctness, clarity

**v2 Quality Score** (Phase 2):
- Currently: Fixed score (0.8)
- Phase 3: Will use quality assessment module

### Comparison Dimensions

1. **Quality:** Accuracy and completeness of answer
2. **Speed:** Execution time
3. **Cost:** Token usage
4. **Iterations:** Number of refinement loops (v1 only currently)

---

## Expected Insights

### What We're Testing

1. **Reasoning Depth**
   - Can frameworks handle multi-step problems?
   - Do they show their work?

2. **Domain Expertise**
   - Do frameworks demonstrate domain knowledge?
   - Or just generic responses?

3. **Systematic Thinking**
   - Do frameworks use structured approaches?
   - Or ad-hoc reasoning?

4. **Quality vs Efficiency**
   - Does v1's recursive improvement help?
   - Is v2's single-pass sufficient?

5. **Composition Benefits** (v2)
   - Do typed signatures help?
   - Does ChainOfThought add value?

### Hypotheses to Test

**H1:** v2's typed prompts lead to more structured answers
**H2:** v1's recursive refinement improves quality on hard problems
**H3:** v2 is faster due to single-pass execution
**H4:** v1 uses more tokens but produces higher quality on expert-level tasks
**H5:** Different domains favor different frameworks

---

## Analysis Tips

### After Running Experiments

1. **Compare output quality manually**
   - Read actual outputs from JSON results
   - Assess correctness, clarity, completeness

2. **Look for patterns**
   - Which domains does each framework excel in?
   - Do certain problem types favor one approach?

3. **Check reasoning quality**
   - Does the framework explain its thinking?
   - Is the logic sound?

4. **Evaluate practical usability**
   - Would you trust this answer?
   - Is it actionable?

### Example Analysis

```python
import json

# Load results
with open('experiment_results_20251130_101530.json') as f:
    results = json.load(f)

# Analyze specific experiment
math_exp = results[0]
print("Mathematical Reasoning:")
print(f"v1 quality: {math_exp['v1']['quality']}")
print(f"v2 quality: {math_exp['v2']['quality']}")
print("\nv1 output sample:")
print(math_exp['v1']['output'][:500])
print("\nv2 output sample:")
print(math_exp['v2']['output'][:500])
```

---

## Future Enhancements

### When Phase 3 is Complete

- **Real quality assessment** for v2
- **Recursive meta-prompting** for v2
- **Bootstrap optimization** for both

### Additional Experiments to Add

- Code generation with tests
- Multi-document reasoning
- Creative problem solving
- Ethical reasoning
- Real-time constraint handling

---

## Notes

### Current Limitations

**v2 Phase 2:**
- Single iteration only (no recursive improvement yet)
- Fixed quality score (0.8)
- No optimizer integration

**v1:**
- Established baseline
- Full recursive meta-prompting
- Quality assessment integrated

### Fair Comparison

These experiments test the **current capabilities** of each framework:
- v1: Full recursive meta-prompting
- v2: Typed prompts + ChainOfThought reasoning

As v2 adds Phase 3 (optimizers), we'll rerun for fair comparison.

---

## Citation

If you use these experiments in research:

```bibtex
@misc{meta_prompting_experiments_2025,
  title={Comprehensive Experiments: Meta-Prompting Framework Comparison},
  author={Meta-Prompting Framework Project},
  year={2025},
  note={Non-trivial domain experiments for v1 vs v2 comparison}
}
```

---

**Ready to run? Make sure your API key is set and go!**

```bash
export ANTHROPIC_API_KEY='your-key'
python experiments/run_comprehensive_experiments.py --quick
```
