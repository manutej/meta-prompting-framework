# /analyze-logic - Logical Argument Analysis

Analyze an argument for both classical and intuitionistic validity.

---

## Input
$ARGUMENTS - The argument or proof to analyze

---

## Instructions

You are the Logic Analyzer. Given the argument "$ARGUMENTS":

1. **Parse Structure**
   - Extract premises (P1, P2, ...)
   - Identify conclusion (C)
   - Formalize in logical notation

2. **Trace Inferences**
   - List each inference step
   - Identify the rule used (Modus Ponens, LEM, DNE, RAA, etc.)
   - Flag any non-constructive rules

3. **Evaluate Validity**
   - Classical validity (with LEM/DNE)
   - Intuitionistic validity (without LEM/DNE)

4. **Identify Missing Witnesses**
   - For any existence claim, is a witness provided?
   - For any disjunction, is the disjunct specified?

5. **Output Format**
```
═══════════════════════════════════════
         LOGICAL ANALYSIS
═══════════════════════════════════════

STRUCTURE:
P1: [Premise 1]
P2: [Premise 2]
...
C: [Conclusion]

INFERENCE TRACE:
1. [Step] by [Rule]
2. [Step] by [Rule]
...

FLAGS:
⚠️ [Non-constructive step] - [Issue]
...

VERDICT:
• Classical:       [VALID/INVALID]
• Intuitionistic:  [VALID/INVALID]
• Missing Witnesses: [List or "None"]

REMEDIATION: [How to make constructive, if applicable]
═══════════════════════════════════════
```

## Example Usage

```
/analyze-logic Either the code has a bug or it doesn't. We tested and found no bugs. Therefore the code is correct.

═══════════════════════════════════════
         LOGICAL ANALYSIS
═══════════════════════════════════════

STRUCTURE:
P1: Bug ∨ ¬Bug (LEM)
P2: ¬FoundBug (testing result)
C: ¬Bug (correctness)

INFERENCE TRACE:
1. Bug ∨ ¬Bug by LEM [⚠️ non-constructive]
2. ¬FoundBug by Observation
3. ¬Bug by... [INVALID INFERENCE]

FLAGS:
⚠️ P1 uses Law of Excluded Middle
⚠️ Step 3 conflates "not found" with "doesn't exist"

VERDICT:
• Classical:       INVALID (¬FoundBug ≠ ¬Bug)
• Intuitionistic:  INVALID (same + LEM issue)
• Missing Witnesses: Proof of exhaustive search

REMEDIATION:
To prove ¬Bug constructively, need:
- Formal verification covering ALL paths, or
- Proof that test coverage is complete
═══════════════════════════════════════
```
