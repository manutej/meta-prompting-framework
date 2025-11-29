# Meta-Prompting Framework Version Guide

This repository contains **two frameworks** that can be used independently or together:

---

## Framework v1: Production-Ready Meta-Prompting Engine

**Location:** `meta_prompting_engine/`

**Status:** ✅ Stable, production-ready

**Use when:**
- You need proven, battle-tested meta-prompting
- You want recursive output improvement (2-3 iterations typical)
- You need complexity-based routing (simple/medium/complex strategies)
- You want context extraction and quality assessment

**Features:**
- ✅ Recursive meta-prompting with real LLM API calls
- ✅ Complexity analysis (4-factor scoring)
- ✅ 3-strategy routing (direct/multi-approach/autonomous)
- ✅ 7-phase context extraction
- ✅ Quality assessment with early stopping
- ✅ Token tracking and metrics

**API:**
```python
from meta_prompting_engine.core import MetaPromptingEngine
from meta_prompting_engine.llm_clients.claude import ClaudeClient

client = ClaudeClient()
engine = MetaPromptingEngine(skill="math_expert", llm_client=client)

result = engine.execute_with_meta_prompting(
    task="Solve x^2 + 5x + 6 = 0",
    max_iterations=3,
    quality_threshold=0.9
)

print(f"Output: {result.output}")
print(f"Quality: {result.quality_score}")
print(f"Iterations: {result.iterations}")
```

**Test Results:**
- Palindrome checker: 0.72 quality, 2 iterations, 4,316 tokens
- Find maximum: 0.78 quality, 2 iterations, 3,998 tokens

---

## Framework v2: Categorical Meta-Prompting Framework (Advanced)

**Location:** `meta_prompting_framework/`

**Status:** 🚧 Phase 1 complete (categorical foundations), Phase 2-5 in progress

**Use when:**
- You need rigorous mathematical guarantees (functor/monad laws)
- You want quality as a compositional property (enriched categories)
- You need true recursive meta-prompt self-improvement (RMP monad)
- You want tool/agent composition with polynomial functors
- You're doing research or need cutting-edge capabilities

**Features (Phase 1 - Complete):**
- ✅ Functors with verified laws (identity + composition)
- ✅ RMP Monad with quality monotonicity
- ✅ Natural transformations between strategies
- ✅ Quality-enriched categories (compositional quality)
- ✅ Polynomial functors for tool composition

**Features (Phase 2-5 - Planned):**
- 🚧 Type-safe signatures (DSPy-like)
- 🚧 Composable modules (ChainOfThought, ReAct, etc.)
- 🚧 Constraint-based generation (LMQL-like)
- 🚧 RMP Optimizer (meta-prompt self-improvement)
- 🚧 Benchmarks (GSM8K, MATH, HotPotQA)

**API (Current - Categorical Abstractions):**
```python
from meta_prompting_framework.categorical import (
    RMPMonad,
    QualityEnrichedPrompts,
    PolynomialFunctor
)

# RMP Monad for recursive improvement
rmp = RMPMonad.unit("basic prompt")
improved = rmp.flat_map(improve_function)

# Quality-enriched prompts
prompts = QualityEnrichedPrompts()
prompts.add_prompt_refinement("basic", "improved", quality=0.8)
final_quality = prompts.compose("basic", "improved", "optimized")

# Polynomial functors for tools
tool = ToolInterface("database", state_machine={...})
composed = wire_tools(tool1.as_polynomial(), tool2.as_polynomial())
```

**API (Future - Phase 2+):**
```python
from meta_prompting_framework.prompts import Signature, InputField, OutputField
from meta_prompting_framework.prompts.module import ChainOfThought
from meta_prompting_framework.optimizers.rmp import RMPOptimizer

class MathSignature(Signature):
    """Solve math problems."""
    question = InputField(str)
    answer = OutputField(str)

module = ChainOfThought(MathSignature, llm_client=client)
optimizer = RMPOptimizer(llm_client=client)
optimized = optimizer.compile(module, trainset=examples)

result = optimized(question="What is 25% of 80?")
```

---

## Version Comparison

| Feature | v1 (Engine) | v2 (Framework) |
|---------|-------------|----------------|
| **Production Ready** | ✅ Yes | ⚠️ Phase 1 only |
| **API Stability** | ✅ Stable | 🚧 Evolving |
| **Recursive Meta-Prompting** | ✅ Output improvement | ✅ Meta-prompt improvement |
| **Categorical Foundations** | ❌ No | ✅ Yes (verified laws) |
| **Type Safety** | ❌ No | 🚧 Coming in Phase 2 |
| **Quality Tracking** | ✅ Post-hoc | ✅ Compositional |
| **Tool Composition** | ❌ No | ✅ Polynomial functors |
| **Constraints** | ❌ No | 🚧 Coming in Phase 2 |
| **Benchmarks** | ⚠️ 2 toy tasks | 🚧 Coming in Phase 4 |

---

## Usage Recommendations

### For Production Use (Now)

**Use v1 (meta_prompting_engine):**
```python
from meta_prompting_engine.core import MetaPromptingEngine

engine = MetaPromptingEngine(skill="expert", llm_client=client)
result = engine.execute_with_meta_prompting(task="your task")
```

✅ Stable, tested, proven
✅ Simple API
✅ Good for most use cases

### For Research/Experimentation

**Use v2 (meta_prompting_framework):**
```python
from meta_prompting_framework.categorical import RMPMonad, QualityEnrichedPrompts

# Explore categorical abstractions
rmp = RMPMonad.unit("prompt")
prompts = QualityEnrichedPrompts()
```

✅ Cutting-edge research
✅ Mathematical guarantees
⚠️ API still evolving (Phases 2-5)

### For Hybrid Approach

**Use v1 for production, v2 for specific features:**
```python
# Use v1 for main task
from meta_prompting_engine.core import MetaPromptingEngine

# Use v2 for quality tracking
from meta_prompting_framework.categorical import QualityEnrichedPrompts

# Execute with v1
engine = MetaPromptingEngine(skill="expert", llm_client=client)
result = engine.execute_with_meta_prompting(task="task")

# Track quality with v2
quality_tracker = QualityEnrichedPrompts()
quality_tracker.add_prompt_refinement(
    original="iteration 0",
    refined="iteration 1",
    quality_improvement=result.quality_score
)
```

---

## Accessing Specific Versions

### Via Git Tags

```bash
# View all versions
git tag

# Checkout v1.0.0 (original framework)
git checkout v1.0.0

# Return to latest
git checkout main
```

### Via Python Import

```python
# v1: Original engine
from meta_prompting_engine.core import MetaPromptingEngine

# v2: Categorical framework
from meta_prompting_framework.categorical import RMPMonad
```

### Via Version Selector (Coming Soon)

```python
# Proposed API for future
from meta_prompting import create_engine

# Use v1
engine_v1 = create_engine(version="v1", skill="expert")

# Use v2
engine_v2 = create_engine(version="v2", signature=MathSignature)
```

---

## Migration Path

### Phase 1 (Now)
- v1 and v2 coexist independently
- Use v1 for production
- Experiment with v2 categorical abstractions

### Phase 2-3 (Weeks 3-6)
- v2 gains Signatures, Modules, Optimizers
- Begin migrating specific use cases to v2
- Continue using v1 for stable workloads

### Phase 4 (Weeks 7-8)
- v2 benchmarked against v1
- Performance comparison available
- Decision point: migrate or keep hybrid

### Phase 5+ (Weeks 9+)
- v2 becomes production-ready
- v1 maintained for backward compatibility
- Version selector for easy switching

---

## Comparison/Benchmark Utilities

See `utils/compare_versions.py` for:
- Side-by-side performance comparison
- Quality metric comparison
- Token usage comparison
- Execution time comparison

Usage:
```bash
python -m utils.compare_versions \
  --task "Solve x^2 + 5x + 6 = 0" \
  --iterations 3
```

---

## When to Upgrade from v1 to v2

**Stick with v1 if:**
- Your use case is working well
- You need maximum stability
- You don't need type safety or constraints
- Simple recursive meta-prompting is sufficient

**Upgrade to v2 when:**
- Phase 2+ is complete (Signatures, Modules ready)
- You need compositional type safety
- You want constraint-based generation
- You need true RMP (meta-prompt self-improvement)
- You're composing tools via MCP
- You need mathematical guarantees

---

## Support and Compatibility

### Backward Compatibility

✅ v2 does not break v1
✅ v1 continues to work independently
✅ Can use both in same project

### Long-Term Support

- **v1:** Maintained for backward compatibility, bug fixes only
- **v2:** Active development, new features, research contributions

### Questions?

- v1 Issues: See existing README and tests
- v2 Progress: See `docs/PHASE1_IMPLEMENTATION_SUMMARY.md`
- Comparison: See `utils/compare_versions.py` (coming soon)
