# Meta-Prompting Framework - Directory Structure

**Organized for easy navigation and future merges**

---

## Root Level

```
meta-prompting-framework/
├── README.md                          # Main project overview
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variable template
└── demo_meta_prompting.py            # Quick demo script (v1)
```

---

## Documentation (`docs/`)

**All framework documentation and analysis**

```
docs/
├── VERSION_GUIDE.md                   # v1 vs v2 comparison guide
├── GAP_ANALYSIS.md                    # Comparison with research frameworks
├── ADVANCED_FRAMEWORK_DESIGN.md       # Complete architectural specification
└── PHASE1_IMPLEMENTATION_SUMMARY.md   # Phase 1 completion report
```

**Purpose:** Centralized documentation for design, analysis, and guides

---

## v1: Production Framework (`meta_prompting_engine/`)

**Stable, production-ready meta-prompting engine**

```
meta_prompting_engine/
├── core.py                            # Main MetaPromptingEngine
├── complexity.py                      # Complexity analysis & routing
├── extraction.py                      # Context extraction (7-phase)
└── llm_clients/
    ├── base.py                        # Abstract LLM interface
    └── claude.py                      # Claude/Anthropic client
```

**Purpose:** Production-ready recursive meta-prompting with real LLM integration

---

## v2: Advanced Framework (`meta_prompting_framework/`)

**Categorical meta-prompting framework (Phase 1 complete)**

```
meta_prompting_framework/
├── categorical/                       # Phase 1: Categorical foundations ✅
│   ├── functor.py                    # Functors with law verification
│   ├── monad.py                      # RMP monad + quality monotonicity
│   ├── natural_transformation.py     # Strategy transformations
│   ├── enriched.py                   # Quality-enriched categories
│   └── polynomial.py                 # Tool composition via polynomial functors
│
├── prompts/                           # Phase 2: Prompt system 🚧
│   └── modules/                      # (Signatures, Modules, Constraints)
│
├── optimizers/                        # Phase 3: Optimizers 🚧
│                                      # (RMP optimizer, Bootstrap)
│
├── applications/                      # Phase 4: Benchmarks 🚧
│   └── benchmarks/                   # (GSM8K, MATH, HotPotQA)
│
└── utils/                             # Phase 5: Production features 🚧
                                       # (Async, caching, observability)
```

**Purpose:** Advanced categorical framework with verified mathematical foundations

---

## Experiments (`experiments/`)

**Benchmarking and comparison suite**

```
experiments/
├── README.md                          # Usage guide
├── EXPERIMENT_SUITE_OVERVIEW.md       # Comprehensive documentation
├── run_suite.py                       # Main runner (10 experiments)
├── quick_demo.py                      # Fast demo (3 experiments)
├── test_v2_structure.py              # v2 categorical tests (no API needed)
└── run_without_api.py                # API-free test suite
```

**Purpose:** Compare v1 vs v2 across 10 practical use cases

---

## Utilities (`utils/`)

**Reusable utilities and tools**

```
utils/
├── compare_versions.py                # v1 vs v2 benchmark tool
└── versioning/
    ├── version_selector.py           # Unified version interface
    └── __init__.py
```

**Purpose:** Version management and comparison tools

---

## Tests (`tests/`)

**Test suite for v1 framework**

```
tests/
└── test_core_engine.py               # Integration tests for v1
```

**Purpose:** Validate v1 production framework

---

## Theory (`theory/`)

**Theoretical foundations and specifications**

```
theory/
├── META-META-PROMPTING-FRAMEWORK.md   # Meta² framework specification
└── META-CUBED-PROMPT-FRAMEWORK.md     # Meta³ categorical proof
```

**Purpose:** Mathematical foundations and categorical theory

---

## Examples (`examples/`)

**Framework instantiation examples**

```
examples/
├── js-categorical-templates/          # JavaScript Kan extensions
├── categorical-fp-framework/          # 10-level FP framework
├── rust-fp-framework/                 # Rust functional programming
├── ai-agent-composability/            # Agent composition patterns
└── luxor-marketplace-frameworks/      # Complete marketplace architecture
```

**Purpose:** Real-world applications of the frameworks

---

## Skills (`skills/`)

**Claude Code integration skills**

```
skills/
├── analyze-complexity/
├── extract-context/
├── meta-prompt-iterate/
└── assess-quality/
```

**Purpose:** Claude Code CLI integration

---

## Navigation Quick Reference

| What You Want | Where to Look |
|---------------|---------------|
| **Getting started** | `README.md` |
| **v1 vs v2 comparison** | `docs/VERSION_GUIDE.md` |
| **Gap analysis** | `docs/GAP_ANALYSIS.md` |
| **Architecture design** | `docs/ADVANCED_FRAMEWORK_DESIGN.md` |
| **Run experiments** | `experiments/README.md` |
| **v1 implementation** | `meta_prompting_engine/` |
| **v2 categorical code** | `meta_prompting_framework/categorical/` |
| **Version selector** | `utils/versioning/` |
| **Theoretical foundations** | `theory/` |
| **Example applications** | `examples/` |

---

## For Future Merges

**Clean structure for PRs:**

1. **Documentation changes** → `docs/`
2. **v1 changes** → `meta_prompting_engine/`
3. **v2 changes** → `meta_prompting_framework/`
4. **Experiments** → `experiments/`
5. **Utilities** → `utils/`

**No scattered files at root level** - everything is organized into logical subdirectories.

---

## Testing Without API Key

```bash
# Test v2 categorical structure
python experiments/test_v2_structure.py

# Run API-free test suite
python experiments/run_without_api.py

# Check version information
python -m utils.versioning.version_selector
```

## Testing With API Key

```bash
# Set API key
export ANTHROPIC_API_KEY="your-key-here"

# Run v1 demo
python demo_meta_prompting.py

# Run experiment suite
python experiments/quick_demo.py          # 3 experiments
python -m experiments.run_suite           # All 10 experiments
```

---

**Last Updated:** November 2025
**Structure Version:** 1.0
