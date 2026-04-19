# Intuitionistic Logic Framework

**A Claude Code plugin for constructive logic, first-principles thinking, and epistemic analysis**

[![License](https://img.shields.io/badge/license-MIT-blue)]()
[![Version](https://img.shields.io/badge/version-0.1.0-green)]()
[![Plugin](https://img.shields.io/badge/Claude%20Code-plugin-blueviolet)]()

> *"The witness IS the proof. Build, don't argue."*

---

## What Is This?

A **Claude Code plugin** that brings intuitionistic logic, constructive reasoning, and first-principles thinking directly into your terminal.

### Six New Commands

| Command | Purpose |
|---------|---------|
| `/ilf:witness <claim>` | Construct a concrete witness for an existence claim |
| `/ilf:first-principles <problem>` | Musk-style deconstruction to verified axioms |
| `/ilf:contrarian <consensus>` | Thiel-style search for non-obvious truths |
| `/ilf:analyze-logic <argument>` | Classical vs intuitionistic validity analysis |
| `/ilf:prove <theorem>` | Generate constructive proofs via Curry-Howard |
| `/ilf:check-type <term>` | Type-level verification |

### Three Specialist Agents

- **witness-constructor**: Builds explicit witnesses for existence claims
- **logic-analyzer**: Evaluates arguments under both classical and intuitionistic logic
- **proof-verifier**: Validates constructive proofs for computational content

---

## Installation

### Option 1: Local Development (Recommended for Testing)

```bash
# Clone the repo
git clone https://github.com/manutej/intuitionistic-logic-framework.git

# Use with Claude Code
claude --plugin-dir ./intuitionistic-logic-framework
```

### Option 2: Plugin Install (Once Published)

```bash
claude plugin install intuitionistic-logic-framework
```

---

## Quick Examples

### Construct a Witness

```bash
/ilf:witness prime number greater than 1,000,000

# Output:
# CLAIM: ∃n. n > 1,000,000 ∧ isPrime(n)
# WITNESS: 1,000,003
# VERIFICATION: ✓ Not divisible by primes ≤ √1,000,003
```

### First-Principles Analysis

```bash
/ilf:first-principles Rockets are too expensive

# Output:
# CONVENTIONAL WISDOM: "Aerospace is inherently expensive"
# HIDDEN ASSUMPTIONS: Expendability, cost-plus markup, material costs
# ACTUAL AXIOMS: Materials = 2% of cost, physics allows reuse
# CONSTRUCTION: SpaceX model with vertical integration
# WITNESS: Falcon 9 Block 5 - 20+ reuses
```

### Contrarian Analysis

```bash
/ilf:contrarian College is required for success

# Output:
# CONSENSUS: Correlation degrees ↔ income
# CRITIQUE: Survivorship bias, not causation
# SECRET: For builders, college is often negative-value
# WITNESSES: Thiel Fellows, Jobs, Gates, Zuckerberg
```

### Logic Analysis

```bash
/ilf:analyze-logic Either the program halts or doesn't, so halting is decidable

# Output:
# ⚠️ Uses LEM on undecidable proposition
# Classical: VALID (but misleading)
# Intuitionistic: INVALID
# Missing: Halting oracle (Turing 1936 proves impossible)
```

### Generate Proof

```bash
/ilf:prove For all naturals n, n + 0 = n

# Output:
# PROOF STRATEGY: Induction on n
# Base: 0 + 0 = 0 by definition
# Step: S k + 0 = S (k + 0) = S k by IH
# AGDA: plus-zero (suc n) = cong suc (plus-zero n)
# QED ∎
```

### Type Check

```bash
/ilf:check-type head : Vec A (S n) → A

# Output:
# ✓ TYPE CHECKS
# Dependent type encodes non-empty vector
# No runtime error possible - type IS the proof
```

---

## Plugin Structure

```
intuitionistic-logic-framework/
├── .claude-plugin/
│   └── plugin.json              # Plugin manifest
├── skills/                       # Slash commands (namespaced /ilf:*)
│   ├── witness/SKILL.md
│   ├── first-principles/SKILL.md
│   ├── contrarian/SKILL.md
│   ├── analyze-logic/SKILL.md
│   ├── prove/SKILL.md
│   └── check-type/SKILL.md
├── agents/                       # Specialist agents
│   ├── witness-constructor.md
│   ├── logic-analyzer.md
│   └── proof-verifier.md
├── theory/                       # Deep theoretical docs
├── examples/                     # Code examples
├── README.md
├── CHANGELOG.md
└── LICENSE
```

---

## The Three Paradigms

### 🚀 Musk Mode: First Principles
> "Don't tell me it's possible. Show me the prototype."

Deconstruct to physics, rebuild from axioms. The construction IS the proof.

### 🎓 Thiel Mode: Contrarian Epistemology
> "What important truth do few people agree with you on?"

Consensus is not proof. Find the non-obvious truth with witnesses.

### 🧮 Ramanujan Mode: Divine Intuition + Rigor
> "The formula came from the goddess. But I verify it nonetheless."

Inspiration generates hypotheses. Construction verifies them.

---

## Core Philosophy

### Intuitionistic vs Classical Logic

| Classical | Intuitionistic |
|-----------|----------------|
| `P ∨ ¬P` always true (LEM) | Only if decidable |
| `¬¬P → P` (DNE) | Only weaker `¬¬P` |
| Proof by contradiction works | Only for negations |
| Existence can be abstract | Need concrete witness |

### The BHK Interpretation

Every proof must be a **construction**:

| Proposition | Required Proof |
|-------------|---------------|
| `P ∧ Q` | Pair `(proof_P, proof_Q)` |
| `P ∨ Q` | Tagged `Left(proof_P)` or `Right(proof_Q)` |
| `P → Q` | Function `proof_P → proof_Q` |
| `∃x.P(x)` | Pair `(witness_x, proof_P(x))` |

### Curry-Howard Correspondence

```
Proposition = Type
Proof = Program
True = Inhabited type
False = Empty type

Code IS proof. Types ARE theorems.
```

---

## Why Use This?

### For Software Engineers
- Design type-safe APIs that prevent bugs at compile time
- Encode invariants in types
- Replace runtime errors with type errors

### For Founders/Builders
- Evaluate ideas by first principles, not analogy
- Find contrarian opportunities others miss
- Build witnesses (prototypes) instead of arguments

### For Mathematicians
- Generate constructive proofs with computational content
- Verify proofs via type checking
- Bridge intuition and rigor

### For Decision Makers
- Demand evidence (witnesses), not assertions
- Identify non-constructive reasoning in arguments
- Apply logical rigor to business decisions

---

## The Builder's Creed

```
I do not claim something exists until I construct it.
I do not claim I know something until I derive it.
I do not accept proof-by-contradiction for existence.
I build witnesses, not assertions.

The witness IS the proof.
The company IS the theorem.
The rocket landing IS the QED.

Code is proof. Types are propositions. Programs are mathematics.
```

---

## Status

**Version:** 0.1.0
**Status:** Initial release
**Testing:** Needs real-world validation

### What Works
- ✅ Plugin structure follows Claude Code spec
- ✅ 6 skills defined with proper frontmatter
- ✅ 3 specialist agents defined
- ✅ Clear documentation

### What Needs Verification
- ⚠️ Installation and activation in real Claude Code sessions
- ⚠️ Namespace behavior (`/ilf:*` prefix)
- ⚠️ Agent auto-invocation
- ⚠️ Edge cases and error handling

### Honest Assessment
This is a **first release following the official Claude Code plugin spec**. It should work based on the documented architecture, but hasn't been battle-tested yet. Please file issues if you find problems.

---

## Contributing

Contributions welcome! The framework follows intuitionistic principles - **all contributions should provide witnesses** (concrete examples, test cases, working code).

1. Fork the repository
2. Create a feature branch
3. Add your skill/agent/example with a witness
4. Submit a PR with the witness as proof of improvement

---

## License

MIT - see [LICENSE](LICENSE)

---

## References

### Theory
- Brouwer, L.E.J. (1912). "Intuitionism and Formalism"
- Heyting, A. (1930). "Die formalen Regeln der intuitionistischen Logik"
- Martin-Löf, P. (1984). "Intuitionistic Type Theory"
- Lambek & Scott (1986). "Introduction to Higher Order Categorical Logic"

### Inspiration
- Peter Thiel. *Zero to One*
- Ashlee Vance. *Elon Musk*
- G.H. Hardy. *A Mathematician's Apology*

### Claude Code Docs
- [Plugin Documentation](https://code.claude.com/docs/en/plugins.md)
- [Plugin Reference](https://code.claude.com/docs/en/plugins-reference.md)

---

*"The best way to predict the future is to construct it."*
