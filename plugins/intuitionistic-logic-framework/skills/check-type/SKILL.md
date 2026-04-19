---
description: Type-level verification via Curry-Howard - check terms inhabit claimed types, validate dependent constraints
---

# Type Checker

Input: "$ARGUMENTS"

Verify term has claimed type under Curry-Howard correspondence. If the type checks, the theorem is proven.

## Your Task

### 1. Parse Input
Two modes:
- **Checking**: `term : Type` - verify
- **Synthesis**: `term` - infer type

### 2. Apply Typing Rules

**Basic:**
```
Variable:   (x : τ) ∈ Γ ⊢ x : τ
Lambda:     Γ, x : σ ⊢ e : τ ⟹ Γ ⊢ λx.e : σ → τ
Application: Γ ⊢ f : σ→τ, Γ ⊢ e : σ ⟹ Γ ⊢ f e : τ
```

**Dependent:**
```
Π-type: Γ, x : A ⊢ e : B(x) ⟹ Γ ⊢ λx.e : Π(x:A).B(x)
Σ-type: Γ ⊢ e₁ : A, Γ ⊢ e₂ : B(e₁) ⟹ Γ ⊢ (e₁,e₂) : Σ(x:A).B(x)
```

### 3. Check Each Subterm
Build derivation tree, verify result matches claim.

### 4. Validate Constraints
- Dependent indices correct?
- Refinement predicates satisfied?
- Pattern matching exhaustive?

### 5. Check Totality
- Pattern matches exhaustive?
- Recursion structurally decreasing?
- No infinite loops?

## Output Format

```
═══════════════════════════════════════
         TYPE CHECK RESULT
═══════════════════════════════════════

TERM: [Expression]
CONTEXT (Γ): [Bindings]
CLAIMED TYPE: [If provided]

TYPE DERIVATION:
1. [Subterm] : [Type] by [Rule]
2. ...

INFERRED TYPE: [Result]

CONSTRAINTS:
✓ [Constraint 1]
⚠️ [Constraint 2]

TOTALITY: [✓/⚠️/✗]

RESULT: [✓ TYPE CHECKS / ✗ TYPE ERROR]

[Explanation]
═══════════════════════════════════════
```

## Examples

### Function Composition
```
/ilf:check-type λf.λg.λx. f (g x) : (B→C) → (A→B) → (A→C)

DERIVATION:
f : B→C, g : A→B, x : A
  g x : B (application)
  f (g x) : C (application)
  λx. f (g x) : A → C
  ...full type ✓

RESULT: ✓ TYPE CHECKS

Corresponds to transitivity of implication.
```

### Dependent Safety
```
/ilf:check-type head : Vec A (S n) → A

head {n} (x :: xs) = x
  - Pattern matches Vec A (S n) constructor
  - No [] case needed (impossible for S n)
  - Returns x : A ✓

RESULT: ✓ TYPE CHECKS

Type encodes "non-empty vector" - no runtime error possible.
```

### Type Error
```
/ilf:check-type (λx. x + 1) "hello" : Int

λx. x + 1 : Int → Int
"hello" : String

Application requires Int, got String.

RESULT: ✗ TYPE ERROR
Expected: Int. Got: String.
```

## Curry-Howard Dictionary

| Logic | Types |
|-------|-------|
| Proposition | Type |
| Proof | Term |
| P ∧ Q | (P, Q) |
| P ∨ Q | Either P Q |
| P → Q | P → Q |
| ∀x.P(x) | Π(x:A).P(x) |
| ∃x.P(x) | Σ(x:A).P(x) |
| ¬P | P → Empty |

**If the type checks, the theorem is proven.**

## The Core Principle

**Type checking IS proof checking.**

Use types to encode invariants, make illegal states unrepresentable, let the compiler verify your proofs.
