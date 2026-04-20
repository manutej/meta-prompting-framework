---
description: Analyze arguments for classical vs intuitionistic validity - flag non-constructive steps, identify missing witnesses, suggest remediation
---

# Logical Argument Analysis

Argument: "$ARGUMENTS"

Evaluate logical validity under BOTH classical AND intuitionistic logic. Flag non-constructive steps (LEM, DNE, RAA for existence).

## Your Task

### 1. Parse Structure
```
P1: [Premise 1]
P2: [Premise 2]
C: [Conclusion]
```

### 2. Trace Every Inference

**Intuitionistically valid:**
- Modus Ponens, And-Intro/Elim, Or-Intro (with tag), Impl-Intro
- Universal Instantiation, Existential Intro (with witness)
- Double Negation Intro, Contraposition, Ex Falso

**Classical-only (FLAG):**
- ❌ Law of Excluded Middle: ⊢ P ∨ ¬P
- ❌ Double Negation Elim: ¬¬P ⊢ P
- ❌ RAA for positive existence

**Note:** RAA IS valid for proving NEGATIONS (¬P).

### 3. Evaluate Validity
- Classical validity (with LEM/DNE)
- Intuitionistic validity (without)
- Missing witnesses for ∃ claims

### 4. Identify Gaps
For each non-constructive step explain why.

### 5. Provide Remediation
How to make constructive.

## Output Format

```
═══════════════════════════════════════
         LOGICAL ANALYSIS
═══════════════════════════════════════

STRUCTURE:
P1: [Formal]
P2: [Formal]
C: [Formal]

INFERENCE TRACE:
1. [Step] by [Rule] ✓
2. [Step] by [Rule] ⚠️ CLASSICAL ONLY
3. [Step] by [Rule] ✓

NON-CONSTRUCTIVE FLAGS:
⚠️ Step 2 uses LEM on undecidable proposition
⚠️ Missing witness for existence

VERDICT:
• Classical:      [VALID/INVALID]
• Intuitionistic: [VALID/INVALID]
• Missing Witnesses: [list]

REMEDIATION:
1. Replace [step] with [constructive alternative]
2. Provide witness for [claim]
═══════════════════════════════════════
```

## Example

```
/ilf:analyze-logic Either program halts or doesn't, so halting is decidable

STRUCTURE:
P1: ∀p. Halts(p) ∨ ¬Halts(p)    (LEM)
P2: Disjunction implies decidability
C: ∀p. Decidable(Halts(p))

INFERENCE TRACE:
1. P1 uses LEM ⚠️ CLASSICAL ONLY
2. P2 ⚠️ INVALID - disjunction ≠ decidability
3. C conflates truth with computability

VERDICT:
• Classical:      VALID (but with caveat)
• Intuitionistic: INVALID (LEM unavailable)
• Halting is undecidable (Turing 1936)

REMEDIATION:
1. Accept halting is undecidable
2. Restrict to decidable subsets (total, primitive recursive)
3. Use bounded halting (within N steps)
```

## Common Fallacies to Flag

- "Either X or not-X, so we can decide X" - LEM ≠ decidability
- "Not impossible, so possible" - ¬¬ ≠ constructive possibility
- "Assume failure → contradiction → must succeed" - proves ¬¬success only
- "Must exist because nonexistence absurd" - no witness

## Quick Format
```
⚡ LOGIC CHECK:
• Classically: [✓/✗]
• Intuitionistically: [✓/✗]
• Main issue: [brief]
• Fix: [brief]
```

## The Core Principle

**Classical validity ≠ Intuitionistic validity.**

Both are valid logics answering different questions:
- Classical: Truth-functionally consistent?
- Intuitionistic: Actually computable/constructible?

For software, correctness proofs, evidence-based decisions - demand constructive proof.
