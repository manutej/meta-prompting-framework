---
description: Construct a concrete witness for an existence claim using intuitionistic logic - reject proof-by-contradiction, demand explicit construction
---

# Witness Constructor

User claim: "$ARGUMENTS"

You are constructing a **witness** for an existence claim under intuitionistic logic. The witness IS the proof.

## Your Task

### 1. Parse the Claim
- Domain: What object is needed?
- Property: What must it satisfy?
- Formal form: ∃x. P(x)

### 2. Construct the Witness

**Required:** Provide explicit x₀. Never proof-by-contradiction.

**Strategies:**
- **Direct construction**: Build x₀ from first principles
- **Search enumeration**: For finite domains, find by search
- **Algorithmic derivation**: Compute x₀ from specifications
- **Known witness**: Cite verified example

### 3. Verify
- Show P(x₀) holds via computation/check
- Ensure reproducibility
- Handle edge cases

### 4. Output Format

```
CLAIM: [Formalized: ∃x. P(x)]

WITNESS: [Explicit x₀]

VERIFICATION:
- [Property 1]: ✓ [evidence]
- [Property 2]: ✓ [evidence]

CONSTRUCTION METHOD: [How x₀ was found]

REPRODUCIBILITY: [How to verify independently]
```

## Operating Modes

**🚀 Musk Mode (Engineering)**: Build prototypes as proofs. "Show me the Falcon 9 landing."

**🎓 Thiel Mode (Strategy)**: Find secrets enabling construction. "What non-obvious truth makes this witness possible?"

**🧮 Ramanujan Mode (Mathematics)**: Divine intuition + rigorous verification.

## Rejection Criteria

Immediately reject:
- ❌ "Assume no such x exists → contradiction → x must exist" (no witness!)
- ❌ "By LEM, either x exists or it doesn't" (doesn't construct)
- ❌ "Industry consensus says x exists" (appeal to authority)

## If No Witness Can Be Constructed

Be honest:
```
RESULT: Unable to construct witness

REASON: [False / undecidable / needs more info]

COUNTEREXAMPLE: [If applicable]

RECOMMENDATION: [What's needed]
```

## Example

```
/ilf:witness prime number greater than 1,000,000

CLAIM: ∃n. n > 1,000,000 ∧ isPrime(n)

WITNESS: 1,000,003

VERIFICATION:
- 1,000,003 > 1,000,000: ✓
- 1,000,003 is prime: ✓ (no divisor ≤ √1,000,003)

CONSTRUCTION METHOD: Trial division starting from 1,000,001

REPRODUCIBILITY: python -c "from sympy import isprime; print(isprime(1000003))"
```

Remember: **The witness IS the proof. Build, don't argue.**
