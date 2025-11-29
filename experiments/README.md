# Experiment Suite: v1 vs v2 Comparison

Comprehensive benchmarking of **v1 (meta_prompting_engine)** vs **v2 (meta_prompting_framework)** across 10 practical use cases.

---

## 📋 Experiments Overview

### 10 Real-World Test Cases

| # | Name | Category | What It Tests |
|---|------|----------|---------------|
| 1 | Binary Search Implementation | Code Generation | Algorithm implementation with docs |
| 2 | Mathematical Problem Solving | Problem Solving | Step-by-step reasoning |
| 3 | Code Refactoring | Code Optimization | Best practices and efficiency |
| 4 | System Design Explanation | Technical Explanation | Complex distributed systems |
| 5 | Bug Diagnosis | Debugging | Performance issue identification |
| 6 | API Design | Software Design | RESTful API architecture |
| 7 | Data Structure Selection | Problem Solving | Choosing right data structures |
| 8 | SQL Query Optimization | Database Optimization | Query performance tuning |
| 9 | Test Case Generation | Testing | Comprehensive test scenarios |
| 10 | Technical Documentation | Documentation | Clear technical writing |

---

## 🚀 Quick Start

### Run All 10 Experiments

```bash
python -m experiments.run_suite
```

**Note:** This will make **~60 LLM API calls** (10 experiments × 2 versions × ~3 iterations each).
Estimated cost: ~$0.50-1.00 depending on output length.
Estimated time: ~15-20 minutes.

### Run Specific Experiments

```bash
# Run just experiments 1, 2, and 5
python -m experiments.run_suite --experiments "1,2,5"

# Run with custom API key
python -m experiments.run_suite --api-key "your-key-here"

# Custom output files
python -m experiments.run_suite \
  --output my_results.json \
  --markdown MY_REPORT.md
```

---

## 📊 Output

### JSON Results

Raw data saved to `experiment_results.json`:

```json
{
  "experiment": {
    "id": 1,
    "name": "Binary Search Implementation",
    "category": "Code Generation",
    "task": "..."
  },
  "v1_result": {
    "version": "v1",
    "quality_score": 0.85,
    "iterations": 2,
    "total_tokens": 4200,
    "execution_time": 45.3
  },
  "v2_result": {
    "version": "v2",
    "quality_score": 0.72,
    "iterations": 1,
    "total_tokens": 0,
    "execution_time": 0.5,
    "error": "Phase 2 not yet implemented"
  },
  "winner": {
    "quality": "v1",
    "speed": "v2",
    "cost": "v2"
  }
}
```

### Markdown Report

Human-readable report saved to `EXPERIMENT_REPORT.md`:

- Summary table with all experiments
- Detailed breakdown of each experiment
- Quality/Speed/Cost comparison
- Winner determination

### Console Output

Real-time progress with summary:

```
======================================================================
Experiment 1: Binary Search Implementation
Category: Code Generation
======================================================================

Running v1 (meta_prompting_engine)...
  ✓ Completed in 45.30s

Running v2 (meta_prompting_framework)...
  ✓ Completed in 0.50s

======================================================================
RESULTS:
  v1 Quality: 0.85 | v2 Quality: 0.00 → Winner: v1
  v1 Speed: 45.30s | v2 Speed: 0.50s → Winner: v2
  v1 Tokens: 4,200 | v2 Tokens: 0 → Winner: v2
======================================================================
```

---

## 📈 Comparison Metrics

Each experiment measures:

### 1. Quality Score (0.0 - 1.0)
- How well the output solves the task
- Measured by LLM-based assessment
- Higher is better

### 2. Execution Time (seconds)
- Wall-clock time from start to finish
- Includes all LLM API calls and processing
- Lower is better

### 3. Token Usage (count)
- Total tokens consumed (input + output)
- Direct proxy for API cost
- Lower is better

### 4. Iterations (count)
- Number of recursive improvement cycles
- Shows convergence behavior
- Optimal varies by task

---

## 🎯 Expected Results (Current State)

### v1 (meta_prompting_engine)

**Strengths:**
- ✅ High quality scores (production-ready)
- ✅ Actual recursive improvement
- ✅ Real LLM integration
- ✅ Proven on real tasks

**Weaknesses:**
- ⚠️ Slower (multiple LLM calls)
- ⚠️ Higher token cost
- ⚠️ No type safety or categorical guarantees

### v2 (meta_prompting_framework)

**Current State (Phase 1):**
- ✅ Fast (no LLM calls yet)
- ✅ Zero cost (no API usage)
- ✅ Categorical laws verified
- ⚠️ **Phase 2 not yet implemented** - no actual prompting yet

**Expected After Phase 2-5:**
- ✅ Type-safe composition
- ✅ Constraint-based generation
- ✅ True RMP monad (meta-prompt self-improvement)
- ✅ Quality as compositional property

---

## 🔮 Future Comparisons

Once v2 Phase 2+ is complete, expect to see:

### Quality Improvements
- Better quality through true RMP (meta-prompts improve themselves)
- Compositional quality tracking
- Constraint satisfaction guarantees

### Efficiency Gains
- Potentially fewer iterations (smarter improvement)
- Better prompt engineering via type system
- Reduced token usage through optimization

### New Capabilities
- Type-safe prompt composition
- Constraint-based validation
- Tool/agent composition via polynomial functors

---

## 📝 Adding Your Own Experiments

Edit `run_suite.py` and add to the `EXPERIMENTS` list:

```python
Experiment(
    id=11,
    name="Your Experiment Name",
    category="Your Category",
    task="Detailed task description for the LLM...",
    description="What this experiment tests",
    expected_quality_min=0.7  # Minimum acceptable quality
)
```

Then run:
```bash
python -m experiments.run_suite --experiments "11"
```

---

## 🐛 Troubleshooting

### "v1 not available" error

Make sure `meta_prompting_engine` is in your path:
```bash
pip install -r requirements.txt
```

### "API key not found" error

Set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY="your-key-here"
# Or pass via --api-key flag
```

### "v2 Phase 2 not yet implemented" warning

This is expected! v2 only has Phase 1 (categorical foundations) complete.
The experiments show v2's structure, but actual prompting requires Phase 2-5.

---

## 📊 Interpreting Results

### Quality Winner
- **v1**: Better at actual task completion (has real LLM integration)
- **v2**: Currently placeholder (Phase 2 needed for real comparison)

### Speed Winner
- **v1**: Slower (multiple LLM API calls)
- **v2**: Faster now (no LLM calls), but will slow down in Phase 2+

### Cost Winner
- **v1**: Higher cost (real token usage)
- **v2**: Zero cost now (no API calls), will have cost in Phase 2+

**The real comparison begins when v2 Phase 2 is complete!**

---

## 🎓 Learning from Results

Use the experiment results to:

1. **Validate v1 production readiness** - See real quality scores
2. **Identify improvement areas** - Which categories need better prompting?
3. **Compare strategies** - Does complexity routing help?
4. **Guide v2 development** - What features are most important?
5. **Benchmark progress** - Re-run as v2 develops

---

## 📚 Related Documentation

- [VERSION_GUIDE.md](../VERSION_GUIDE.md) - Detailed version comparison
- [GAP_ANALYSIS.md](../docs/GAP_ANALYSIS.md) - What v2 adds
- [PHASE1_IMPLEMENTATION_SUMMARY.md](../docs/PHASE1_IMPLEMENTATION_SUMMARY.md) - v2 progress

---

## 🚀 Next Steps

1. **Run the suite** to establish v1 baseline
2. **Review results** to understand current capabilities
3. **Wait for Phase 2** to see real v2 performance
4. **Re-run comparison** as v2 develops
5. **Make informed decision** about migration

---

**Ready to run? Let's go!**

```bash
python -m experiments.run_suite
```
