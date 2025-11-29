# Experiment Suite Overview

**Purpose:** Comprehensive benchmarking system for comparing v1 and v2 meta-prompting frameworks across 10 practical use cases.

---

## 🎯 What This Suite Does

The experiment suite provides:

1. **Standardized benchmarks** - 10 real-world tasks covering common use cases
2. **Automated comparison** - Side-by-side v1 vs v2 performance testing
3. **Multiple metrics** - Quality, speed, cost, and iteration count
4. **Reproducible results** - JSON and Markdown reports for sharing
5. **Extensible framework** - Easy to add your own experiments

---

## 📊 The 10 Experiments

### Code Generation (3 experiments)
1. **Binary Search Implementation** - Algorithm with error handling
3. **Code Refactoring** - Pythonic improvements
9. **Test Case Generation** - Comprehensive test scenarios

### Problem Solving (2 experiments)
2. **Mathematical Problem Solving** - Step-by-step reasoning
7. **Data Structure Selection** - Choosing right structures

### Technical Explanation (1 experiment)
4. **System Design Explanation** - Distributed systems concepts

### Debugging & Optimization (2 experiments)
5. **Bug Diagnosis** - Performance issue identification
8. **SQL Query Optimization** - Database performance

### Design & Documentation (2 experiments)
6. **API Design** - RESTful architecture
10. **Technical Documentation** - Clear documentation writing

---

## 🚀 Quick Start

### 1. Run Full Suite (10 experiments)

```bash
python -m experiments.run_suite
```

**Warning:** Makes ~60 LLM API calls, costs ~$0.50-1.00, takes ~15-20 minutes.

### 2. Run Quick Demo (3 experiments)

```bash
python experiments/quick_demo.py
```

Faster demonstration with lightweight tasks.

### 3. Run Specific Experiments

```bash
python -m experiments.run_suite --experiments "1,2,5"
```

### 4. Test v2 Structure Only

```bash
python experiments/test_v2_structure.py
```

Shows v2's categorical abstractions working (no API calls needed).

---

## 📈 What Gets Measured

### Quality Score (0.0 - 1.0)
- **How:** LLM-based assessment of output quality
- **What:** Does the output solve the task effectively?
- **Goal:** Higher is better

### Execution Time (seconds)
- **How:** Wall-clock time from start to finish
- **What:** Total time including all processing and API calls
- **Goal:** Lower is better (faster)

### Token Usage (count)
- **How:** Sum of all input + output tokens
- **What:** Direct proxy for API cost
- **Goal:** Lower is better (cheaper)

### Iterations (count)
- **How:** Number of recursive improvement cycles
- **What:** How many times the prompt was refined
- **Goal:** Optimal varies (fewer iterations can be good or bad)

---

## 📁 Generated Outputs

### experiment_results.json

Raw data for all experiments:

```json
{
  "experiment": {...},
  "v1_result": {
    "quality_score": 0.85,
    "execution_time": 45.3,
    "total_tokens": 4200,
    "iterations": 2
  },
  "v2_result": {...},
  "winner": {
    "quality": "v1",
    "speed": "v2",
    "cost": "v2"
  }
}
```

### EXPERIMENT_REPORT.md

Human-readable markdown report:
- Summary table
- Detailed breakdown per experiment
- Overall winner determination
- Performance statistics

### Console Output

Real-time progress and summary:
```
======================================================================
Experiment 1: Binary Search Implementation
======================================================================
Running v1... ✓ Completed in 45.30s
Running v2... ✓ Completed in 0.50s

RESULTS:
  v1 Quality: 0.85 | v2 Quality: 0.00 → Winner: v1
  v1 Speed: 45.30s | v2 Speed: 0.50s → Winner: v2
  ...
```

---

## 🔬 Current State Expectations

### v1 (meta_prompting_engine) - Production Ready

**Expected Results:**
- ✅ High quality scores (0.7-0.9 range)
- ✅ Real recursive improvement visible
- ✅ 2-3 iterations typical
- ✅ ~3,000-5,000 tokens per experiment
- ⏱️ 30-60 seconds per experiment

**Strengths:**
- Proven production quality
- Real LLM integration
- Context extraction working
- Complexity routing effective

**Weaknesses:**
- Higher latency (multiple API calls)
- Higher cost (more tokens)
- No type safety
- No categorical guarantees

### v2 (meta_prompting_framework) - Phase 1 Only

**Expected Results (Current):**
- ⚠️ Quality score: 0.0 (Phase 2 not ready)
- ⚡ Speed: <1 second (no LLM calls)
- 💰 Cost: 0 tokens (no API usage)
- ℹ️ Error: "Phase 2 not yet implemented"

**Why v2 Shows 0.0 Quality:**
Phase 1 only implemented categorical foundations (functors, monads, enriched categories).
It doesn't have the Signature/Module system yet to actually generate prompts.

**What Works in v2:**
- ✅ RMP monad structure
- ✅ Quality-enriched categories
- ✅ All categorical laws verified
- ✅ Ready for Phase 2 integration

**What's Missing (Phase 2-5):**
- ❌ Signatures (typed prompts)
- ❌ Modules (ChainOfThought, ReAct)
- ❌ Real LLM integration
- ❌ Optimizers

### After Phase 2+ Complete

**Expected v2 Results:**
- 🎯 Quality: Match or exceed v1 (true RMP)
- ⚡ Speed: Potentially faster (smarter iterations)
- 💰 Cost: Potentially lower (optimized prompting)
- ✨ Plus: Type safety, constraints, categorical guarantees

---

## 🎓 Interpreting Results

### Quality Winner

**If v1 wins:**
- Expected now (v1 has real prompting, v2 doesn't yet)
- Shows v1 is production-ready
- Validates current framework

**If v2 wins (after Phase 2):**
- True RMP monad is working
- Meta-prompts are self-improving
- Categorical approach is paying off

**If tie:**
- Both produce similar quality
- Choose based on other factors (speed, cost, features)

### Speed Winner

**v1 slower = Expected**
- Multiple LLM API calls take time
- Each iteration has network latency
- Context extraction adds processing

**v2 faster (now) = Misleading**
- Currently no LLM calls (Phase 2 not ready)
- Will slow down when real prompting added
- Final speed depends on iteration efficiency

**v2 faster (after Phase 2) = Good Sign**
- Smarter iterations (fewer needed)
- Better prompt engineering
- Categorical optimization working

### Cost Winner

**v1 higher cost = Expected**
- More iterations = more tokens
- Context extraction prompts add cost
- Each improvement cycle costs tokens

**v2 lower cost (now) = Misleading**
- Zero cost because no API calls yet
- Will have cost when Phase 2 ready

**v2 lower cost (after Phase 2) = Excellent**
- Fewer iterations needed
- Better prompts from the start
- Type system reduces waste

---

## 🔄 Iteration Analysis

### Healthy Iteration Patterns

**v1 (typical):**
- Iteration 1: Initial attempt (quality ~0.5-0.6)
- Iteration 2: Improved with context (quality ~0.7-0.8)
- Iteration 3: Refinement if needed (quality ~0.8-0.9)
- Early stop if threshold met

**v2 (expected after Phase 2):**
- Fewer iterations due to better initial prompts
- Each iteration more effective (true RMP)
- Quality increases faster per iteration

### Warning Signs

**Too many iterations (>5):**
- Task might be too complex
- Prompt strategy not working
- Quality plateau reached

**Quality decreasing:**
- Should never happen (monotonicity)
- Bug if this occurs
- Check implementation

**No improvement:**
- Quality stuck across iterations
- Need different strategy
- Task might need constraints

---

## 📊 Statistical Analysis

The suite calculates:

### Aggregate Metrics
- Average quality per version
- Average execution time per version
- Total token usage per version
- Win rate per category (quality/speed/cost)

### Category Breakdown
- Performance by task type
- Strengths/weaknesses per category
- Optimal strategies per use case

### Overall Winner
- Total category wins
- Weighted scoring
- Recommendation based on use case

---

## 🔧 Customization

### Add Your Own Experiment

Edit `run_suite.py`:

```python
Experiment(
    id=11,
    name="Custom Task",
    category="Your Category",
    task="Detailed task description...",
    description="What this tests",
    expected_quality_min=0.7
)
```

### Change Iteration Limits

```bash
python -m experiments.run_suite --iterations 5
```

### Change Quality Threshold

```bash
python -m experiments.run_suite --quality-threshold 0.85
```

### Custom Output Files

```bash
python -m experiments.run_suite \
  --output my_results.json \
  --markdown MY_REPORT.md
```

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'meta_prompting_engine'"

Install dependencies:
```bash
pip install -r requirements.txt
```

### "API key not found"

Set your Anthropic API key:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### "v2 Phase 2 not yet implemented"

This is expected! v2 only has Phase 1 complete.
- Current: Categorical foundations only
- Coming: Phase 2 (Signatures, Modules)
- Then: Real v1 vs v2 comparison possible

### High API costs

Reduce experiments:
```bash
python -m experiments.run_suite --experiments "1,2,3"
```

Or use quick demo:
```bash
python experiments/quick_demo.py
```

---

## 📅 Roadmap

### Now (Phase 1 Complete)
- ✅ Run v1 benchmarks (establish baseline)
- ✅ Test v2 structure (verify abstractions)
- ✅ Understand current capabilities

### Phase 2 (Signatures & Modules)
- Implement type-safe prompt system
- Add ChainOfThought, ReAct modules
- Real LLM integration with categorical guarantees
- **First real v1 vs v2 comparison possible**

### Phase 3 (Optimizers)
- RMP optimizer using monad
- Bootstrap few-shot
- See optimization improvements

### Phase 4 (Benchmarks)
- GSM8K, MATH, HotPotQA
- Compare against DSPy, LMQL
- Publish results

### Phase 5 (Production)
- Async/concurrent execution
- Caching layer
- Observability
- Performance optimization
- **v2 production-ready**

---

## 🎯 Success Criteria

### Baseline Established (Now)
- ✅ v1 baseline quality scores recorded
- ✅ v1 performance characteristics known
- ✅ v2 structure verified working

### Phase 2 Success
- v2 quality matches v1 on most tasks
- Categorical guarantees enforced
- Type safety working

### Phase 3 Success
- v2 quality exceeds v1 via RMP optimizer
- Fewer iterations needed
- Better prompt engineering

### Phase 4 Success
- v2 competitive with DSPy, LMQL
- Unique categorical features demonstrated
- Research contribution validated

### Phase 5 Success
- v2 production-ready
- Clear migration path from v1
- Performance optimized
- Community adoption

---

## 📚 Related Documentation

- [experiments/README.md](README.md) - Usage instructions
- [VERSION_GUIDE.md](../VERSION_GUIDE.md) - v1 vs v2 comparison
- [GAP_ANALYSIS.md](../docs/GAP_ANALYSIS.md) - Detailed gap analysis
- [PHASE1_IMPLEMENTATION_SUMMARY.md](../docs/PHASE1_IMPLEMENTATION_SUMMARY.md) - Phase 1 results

---

## 💡 Tips for Best Results

1. **Start with quick demo** - Understand output format first
2. **Run specific experiments** - Focus on your use case
3. **Track costs** - Monitor token usage
4. **Compare over time** - Re-run as v2 develops
5. **Share results** - Help improve both frameworks

---

**Ready to benchmark? Start here:**

```bash
# Quick demo (3 experiments, ~5 minutes)
python experiments/quick_demo.py

# Full suite (10 experiments, ~20 minutes, ~$1)
python -m experiments.run_suite

# Test v2 structure (instant, free)
python experiments/test_v2_structure.py
```
