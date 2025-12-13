# /witness - Construct Existence Witness

Construct a concrete witness for an existence claim using intuitionistic logic principles.

---

## Input
$ARGUMENTS - The existence claim to prove (e.g., "prime > 1000", "sorting algorithm O(n log n)")

---

## Instructions

You are invoking the Witness Constructor. Given the existence claim "$ARGUMENTS":

1. **Parse the Claim**
   - Identify the domain (what kind of object?)
   - Identify the property (what must it satisfy?)
   - Formalize as ∃x.P(x)

2. **Construct Witness**
   - Find or build a specific x₀
   - Do NOT use proof by contradiction
   - The witness must be explicit and verifiable

3. **Verify**
   - Demonstrate P(x₀) holds
   - Show the verification is reproducible

4. **Output Format**
```
CLAIM: [Formalized proposition]

WITNESS: [Explicit construction]

VERIFICATION:
- [Property 1]: ✓ [evidence]
- [Property 2]: ✓ [evidence]
...

CONSTRUCTION METHOD: [How the witness was found/built]
```

## Example Usage

```
/witness prime number greater than 10000

CLAIM: ∃n. n > 10000 ∧ isPrime(n)

WITNESS: 10007

VERIFICATION:
- 10007 > 10000: ✓ (10007 - 10000 = 7)
- isPrime(10007): ✓ (not divisible by 2,3,5,7,11,...,97)

CONSTRUCTION METHOD: Checked odd numbers starting from 10001
```

If no witness can be constructed, explain why (proposition may be false, undecidable, or need more information).
