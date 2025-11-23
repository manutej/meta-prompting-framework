# Intuitionistic Logic Skill

## Purpose
Apply intuitionistic/constructive logic principles to problem-solving, code design, and proof construction. Reject classical shortcuts (LEM, DNE) in favor of constructive witnesses.

## Activation
Use this skill when:
- Designing systems that require provable correctness
- Analyzing arguments for logical validity
- Building type-safe code with dependent types
- Evaluating claims that lack constructive evidence
- Translating business requirements into verifiable specifications

## Core Principles

### 1. The BHK Interpretation
Every proof must be a construction:

| Proposition | Required Proof |
|-------------|---------------|
| `A ∧ B` | Pair `(proof_A, proof_B)` |
| `A ∨ B` | Tagged `Left(proof_A)` OR `Right(proof_B)` |
| `A → B` | Function transforming `proof_A` into `proof_B` |
| `∃x.P(x)` | Pair `(witness_x, proof_P(x))` |
| `¬A` | Function `proof_A → ⊥` |

### 2. Rejected Classical Axioms
```
❌ Law of Excluded Middle: P ∨ ¬P (not always provable)
❌ Double Negation Elimination: ¬¬P → P (not always valid)
❌ Proof by Contradiction: Assume ¬P, derive ⊥, conclude P
```

### 3. Accepted Intuitionistic Principles
```
✅ Double Negation Introduction: P → ¬¬P
✅ Contraposition: (P → Q) → (¬Q → ¬P)
✅ Ex Falso Quodlibet: ⊥ → P (from false, anything)
✅ Modus Ponens: P, P → Q ⊢ Q
```

## Practical Applications

### Code Design Pattern: Constructive Types

```typescript
// WRONG: Classical existence claim
function findUser(id: string): User | null {
  // Returns null - no witness of non-existence
}

// RIGHT: Constructive disjunction
type FindResult<T> =
  | { found: true; value: T }
  | { found: false; reason: string };

function findUser(id: string): FindResult<User> {
  // Must construct WHICH case and provide evidence
}
```

### Argument Analysis Pattern

When evaluating a claim:

1. **Identify the proposition**: What exactly is being claimed?
2. **Demand the witness**: What construction proves this?
3. **Reject mere non-contradiction**: "It's not impossible" ≠ proof
4. **Check decidability**: Is this even constructively provable?

### Business Logic Pattern

```
CLAIM: "Our product will succeed"

CLASSICAL (insufficient):
  - Assume failure → contradiction with our plans
  - Therefore success ∎

CONSTRUCTIVE (required):
  - Witness: Working prototype with users
  - Evidence: Revenue, retention metrics
  - Reproducibility: Documented process
```

## The Three Paradigms

### Musk Mode (First Principles Construction)
```
Don't argue X is possible.
BUILD X as the proof.
The Falcon 9 landing IS the theorem.
```

### Thiel Mode (Contrarian Epistemology)
```
Consensus ≠ Proof
What non-obvious truth do you KNOW (not believe)?
Construct understanding, don't inherit it.
```

### Classical Mode (When Appropriate)
```
For decidable propositions, LEM is valid.
For finite domains, exhaustive search works.
For boolean conditions, classical logic applies.
```

## Verification Checklist

Before accepting any existence claim:

- [ ] Is there a concrete witness?
- [ ] Can the witness be inspected/verified?
- [ ] Is the proof constructive or by contradiction?
- [ ] Could this be decided algorithmically?
- [ ] What would falsify this claim?

## Integration with Type Systems

### TypeScript/JavaScript
```typescript
// Use discriminated unions for constructive disjunction
// Use never for the empty type (⊥)
// Use generics for universal quantification
```

### Haskell
```haskell
-- Use GADTs for dependent-style types
-- Use Either for disjunction
-- Use Void for ⊥
```

### Agda/Idris/Lean
```
-- Full dependent types = full intuitionistic logic
-- Types ARE propositions
-- Programs ARE proofs
```

## Quick Reference

| Classical | Intuitionistic Equivalent |
|-----------|--------------------------|
| `P \|\| !P` always true | Only if P is decidable |
| `!!P == P` | `!!P` is weaker than `P` |
| Exists by contradiction | Must construct witness |
| Proof by elimination | Must construct target |

## When to Use Classical Logic

Classical logic IS valid when:
1. Domain is finite and enumerable
2. Proposition is computationally decidable
3. You're doing classical mathematics intentionally
4. Performance trumps constructivity

But always KNOW which logic you're using.
