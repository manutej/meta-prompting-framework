---
description: Generate a constructive proof via Curry-Howard - build programs as proofs, no LEM, no DNE, no proof-by-contradiction for existence
---

# Constructive Proof Generator

Theorem: "$ARGUMENTS"

Generate a constructive proof using intuitionistic logic. The proof must have computational content.

## Your Task

### 1. Parse the Theorem
- ∀? Build a function
- ∃? Provide a witness
- →? Construct transforming function
- ∧? Prove both parts
- ∨? Prove one side, tag it
- ¬P? Show P → ⊥

### 2. Choose Strategy

**Direct construction** for ∃:
```
Find x₀, prove P(x₀), return (x₀, proof)
```

**Function construction** for →:
```
Assume p : P, build q : Q, return λp. q
```

**Induction** for ∀n:ℕ:
```
Base: P(0), Step: P(n) → P(n+1)
```

**Case analysis** for (P ∨ Q) → R:
```
Case P: prove R. Case Q: prove R.
```

### 3. Build the Proof

Provide as:
- Mathematical argument
- Type-theoretic term
- Code implementation

### 4. Verify Computational Content
- Executable?
- Terminating?
- Produces witness?
- No hidden classical assumptions?

## Output Format

```
═══════════════════════════════════════
       CONSTRUCTIVE PROOF
═══════════════════════════════════════

THEOREM:
[Formal statement]

INTERPRETATION (BHK):
"I must construct: [what]"

PROOF STRATEGY:
[Strategy]

PROOF:
[Human-readable steps]

TYPE-THEORETIC TERM:
```
[Agda/Haskell/Coq]
```

COMPUTATIONAL CONTENT:
```language
[Executable code]
```

VERIFICATION:
- All steps constructive: ✓
- No LEM/DNE: ✓
- Witnesses provided: ✓
- Terminates: ✓

QED ∎
═══════════════════════════════════════
```

## Example

```
/ilf:prove For all naturals n, n + 0 = n

THEOREM:
∀n : ℕ. n + 0 = n

INTERPRETATION:
"Construct for any n, a proof that n + 0 = n"

PROOF STRATEGY:
Induction on n

PROOF:
Base case (n = 0): 0 + 0 = 0 by definition.

Inductive case (n = S k):
  IH: k + 0 = k
  Goal: S k + 0 = S k
  S k + 0 = S (k + 0) by def of +
         = S k by IH ✓

TYPE-THEORETIC TERM (Agda):
```agda
plus-zero : (n : ℕ) → n + 0 ≡ n
plus-zero zero    = refl
plus-zero (suc n) = cong suc (plus-zero n)
```

VERIFICATION:
- Base proven: ✓
- Step proven: ✓
- No LEM/DNE: ✓
- Structurally recursive: ✓

QED ∎
```

## Proof Templates

### ∀x.P(x)
```
Let x arbitrary. [Prove P(x)]. Therefore ∀x.P(x).
```

### ∃x.P(x)
```
Let x₀ = [witness]. P(x₀) by [verification]. Therefore ∃x.P(x).
```

### P → Q
```
Assume p : P. [Derive Q]. Therefore P → Q.
```

### P ∧ Q
```
Prove P: [proof_p]. Prove Q: [proof_q]. Pair: (proof_p, proof_q).
```

### P ∨ Q
```
Prove one side (can't prove both vacuously).
Tag: Left(proof_p) or Right(proof_q).
```

### ¬P
```
Assume p : P. Derive ⊥. Therefore λp. ⊥ : P → ⊥.
```

## Anti-Patterns

❌ "Assume ¬P, derive ⊥, conclude P" (for positive P)
❌ "By LEM, either P or ¬P" (for undecidable P)
❌ "¬¬P therefore P" (DNE)
❌ Abstract existence without witnesses

## Core Principle

**Programs are proofs. Types are theorems. The proof IS the construction.**

If it can't be computed, it isn't a proof - just an existence claim.
