# CLAUDE.md - Meta-Prompting Framework Configuration

## Project Overview

This is a **Meta-Prompting Framework** that combines:
- Recursive prompt improvement via real LLM integration
- Category theory foundations (Kan extensions, toposes, ∞-categories)
- Intuitionistic logic for constructive reasoning
- First-principles thinking (Musk) and contrarian epistemology (Thiel)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run meta-prompting engine
python -m meta_prompting_engine.core

# Run tests
pytest tests/
```

## Core Principles

### 1. Constructive Proof Requirement
This framework operates under **intuitionistic logic** principles:
- Never claim existence without constructing a witness
- Reject proof-by-contradiction for existence claims
- Every assertion must have computational content

### 2. First Principles Thinking
Decompose problems to verified axioms, then construct solutions from ground truth.

### 3. Contrarian Epistemology
Consensus is not proof. Construct your own understanding.

---

## .claude/ Directory Structure

```
.claude/
├── skills/
│   ├── intuitionistic-logic.md    # Core logic principles
│   └── constructive-proof.md      # Proof construction methods
├── agents/
│   ├── witness-constructor.md     # Builds existence witnesses
│   └── logic-analyzer.md          # Analyzes argument validity
└── commands/
    ├── witness.md                 # /witness <claim>
    ├── analyze-logic.md           # /analyze-logic <argument>
    ├── first-principles.md        # /first-principles <problem>
    └── contrarian.md              # /contrarian <consensus>
```

---

## Skills

### Intuitionistic Logic Skill
**Location:** `.claude/skills/intuitionistic-logic.md`

Apply intuitionistic/constructive logic to problem-solving:
- BHK interpretation (proofs as constructions)
- Rejected axioms: LEM, DNE
- Accepted: Modus Ponens, Contraposition, Ex Falso
- Practical type-system applications

### Constructive Proof Skill
**Location:** `.claude/skills/constructive-proof.md`

Generate proofs with explicit witnesses:
- Curry-Howard correspondence
- Direct construction strategies
- Anti-patterns (proof by contradiction for existence)
- Decidability analysis

---

## Agents

### Witness Constructor
**Location:** `.claude/agents/witness-constructor.md`

Constructs concrete witnesses for existence claims:
```
INPUT:  "There exists X such that P(X)"
OUTPUT: Specific X₀ with verification that P(X₀)
```

Operates in three modes:
- **Musk Mode:** Build prototypes as proofs
- **Thiel Mode:** Find secrets that unlock construction
- **Ramanujan Mode:** Divine intuition, rigorous verification

### Logic Analyzer
**Location:** `.claude/agents/logic-analyzer.md`

Analyzes arguments for logical validity:
- Classical vs Intuitionistic evaluation
- Inference rule identification
- Non-constructive step flagging
- Remediation suggestions

---

## Slash Commands

### `/witness <claim>`
Construct a witness for an existence claim.

```
/witness prime number greater than 10000

WITNESS: 10007
VERIFICATION: Not divisible by primes up to √10007
```

### `/analyze-logic <argument>`
Full logical analysis with classical/intuitionistic comparison.

```
/analyze-logic Either it works or it doesn't

⚠️ Uses LEM - valid classically, not constructively
```

### `/first-principles <problem>`
Musk-style deconstruction to axioms and reconstruction.

```
/first-principles Rockets are expensive

AXIOM: Materials = 2% of cost
CONSTRUCTION: Vertical integration + reusability
WITNESS: Falcon 9
```

### `/contrarian <consensus>`
Thiel-style search for non-obvious truths.

```
/contrarian Everyone needs a college degree

SECRET: For builders, college is often negative-value
WITNESS: Thiel Fellows, tech founders without degrees
```

---

## Key Documentation

| Document | Description |
|----------|-------------|
| `theory/INTUITIONISTIC-LOGIC-TECH-LEADERS.md` | Deep exploration of intuitionistic logic through Musk/Trump/Thiel |
| `theory/META-META-PROMPTING-FRAMEWORK.md` | Theoretical foundations |
| `skills/category-master/SKILL.md` | PhD-level category theory |
| `skills/discopy-categorical-computing/SKILL.md` | String diagrams and quantum |
| `meta-prompts/v2/META_PROMPTS.md` | Production meta-prompts (82-92% quality) |

---

## Architecture

```
meta_prompting_engine/
├── core.py              # Recursive improvement loop
├── complexity.py        # Task complexity analysis
├── extraction.py        # 7-phase context extraction
└── llm_clients/
    ├── base.py          # Abstract LLM interface
    └── claude.py        # Claude Sonnet 4.5 integration
```

## The Builder's Creed

```
I do not claim something exists until I construct it.
I do not claim I know something until I derive it.
I do not accept proof-by-contradiction for existence.
I build witnesses, not assertions.

The witness IS the proof.
The company IS the theorem.
The rocket landing IS the QED.
```

---

## Integration Points

### Category Theory
- Kan extensions for monad/comonad construction
- Topos theory for intuitionistic semantics
- Curry-Howard-Lambek correspondence

### Meta-Prompting
- 6 production meta-prompts with validated quality
- Recursive improvement via real Claude API calls
- Complexity-based routing

### Practical Frameworks
- Go, Rust, JavaScript, F*, Wolfram implementations
- Luxor Marketplace enterprise patterns
- Verification frameworks (F* with formal proofs)

---

## Contributing

1. All proofs must be constructive (provide witnesses)
2. No classical shortcuts (LEM, DNE) without explicit justification
3. Code is proof; types are theorems
4. First principles over convention

---

## References

- Brouwer, Heyting, Kolmogorov (BHK Interpretation)
- Martin-Löf (Intuitionistic Type Theory)
- Lambek & Scott (Categorical Logic)
- Thiel (Zero to One)
- Vance (Elon Musk biography)
