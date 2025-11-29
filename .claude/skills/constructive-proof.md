# Constructive Proof Skill

## Purpose
Generate constructive proofs that provide explicit witnesses rather than existence-by-contradiction arguments. Every proof produces a computable object.

## Activation
Use when:
- Proving existence claims (must construct witness)
- Designing algorithms from specifications
- Verifying program correctness
- Translating mathematical proofs to code

## Core Method: Proof-as-Program

### The Curry-Howard Correspondence

```
╔════════════════════╦════════════════════╗
║ Logic              ║ Programming        ║
╠════════════════════╬════════════════════╣
║ Proposition        ║ Type               ║
║ Proof              ║ Term/Program       ║
║ A → B              ║ Function A → B     ║
║ A ∧ B              ║ Tuple (A, B)       ║
║ A ∨ B              ║ Either A B         ║
║ ∀x.P(x)            ║ (x: A) → P(x)      ║
║ ∃x.P(x)            ║ Σ(x: A). P(x)      ║
║ ⊥ (False)          ║ Void/Never         ║
║ ⊤ (True)           ║ Unit/()            ║
╚════════════════════╩════════════════════╝
```

## Proof Strategies

### 1. Direct Construction
For `∃x.P(x)`: Find specific `x₀` and prove `P(x₀)`

```typescript
// Prove: ∃n. n > 100 ∧ isPrime(n)
function proveExistsLargePrime(): { witness: number; proof: PrimeProof } {
  const witness = 101;  // Explicit construction
  const proof = verifyPrime(101);  // Explicit verification
  return { witness, proof };
}
```

### 2. Function Construction
For `A → B`: Build a function transforming any proof of A into proof of B

```typescript
// Prove: isEven(n) → isEven(n + 2)
function evenPlusTwo(evenProof: EvenProof<N>): EvenProof<N + 2> {
  // The function IS the proof
  return extendEven(evenProof);
}
```

### 3. Case Analysis
For `A ∨ B → C`: Handle both cases constructively

```typescript
// Prove: (A ∨ B) → C
function fromDisjunction<A, B, C>(
  disjunction: Either<A, B>,
  handleA: (a: A) => C,
  handleB: (b: B) => C
): C {
  switch (disjunction.tag) {
    case 'left': return handleA(disjunction.value);
    case 'right': return handleB(disjunction.value);
  }
}
```

### 4. Induction
For properties over recursive structures: Base case + inductive step

```typescript
// Prove: ∀n. sum(1..n) = n*(n+1)/2
function sumFormula(n: Nat): Proof<Sum(1,n) = n*(n+1)/2> {
  if (n === 0) {
    return baseCase();  // sum(1..0) = 0 = 0*1/2 ✓
  } else {
    const ih = sumFormula(n - 1);  // Inductive hypothesis
    return inductiveStep(ih, n);    // Extend to n
  }
}
```

## Anti-Patterns to Avoid

### ❌ Proof by Contradiction (for existence)
```
// INVALID in intuitionistic logic:
"Assume no prime > 100 exists"
"Derive contradiction"
"Therefore prime > 100 exists"
// WHERE IS IT? No witness provided!
```

### ❌ Excluded Middle Assumption
```typescript
// INVALID: Cannot implement without knowing P
function excludedMiddle<P>(): Either<P, Not<P>> {
  // ??? No general implementation possible
}
```

### ❌ Double Negation Elimination
```typescript
// INVALID: Cannot extract P from ¬¬P
function dne<P>(nnp: (f: (p: P) => never) => never): P {
  // ??? No way to construct P
}
```

## Valid Intuitionistic Theorems

### Double Negation Introduction ✓
```typescript
function dni<P>(p: P): (f: (p: P) => never) => never {
  return (f) => f(p);
}
```

### Contraposition ✓
```typescript
function contrapose<P, Q>(
  pq: (p: P) => Q
): (nq: (q: Q) => never) => (p: P) => never {
  return (nq) => (p) => nq(pq(p));
}
```

### Ex Falso Quodlibet ✓
```typescript
function exFalso<A>(falsity: never): A {
  return falsity;  // never has no inhabitants, so this is vacuously valid
}
```

## Decidability Analysis

Before attempting a proof, classify the proposition:

| Type | Constructively Provable? |
|------|-------------------------|
| Decidable (finite check) | Yes, with witness |
| Semi-decidable (halting) | If true, can find witness |
| Undecidable | May need classical axioms |

```typescript
// Decidable: equality on finite types
function decideNatEq(a: Nat, b: Nat): Either<Equal<a,b>, NotEqual<a,b>> {
  // CAN implement - finite comparison
}

// NOT decidable in general: halting problem
function decideHalts(program: Program): Either<Halts, NotHalts> {
  // CANNOT implement - undecidable
}
```

## Output Format

When providing constructive proofs:

```
THEOREM: [Statement]

WITNESS: [Explicit construction]

PROOF:
1. [Step with justification]
2. [Step with justification]
...
n. [QED with constructed object]

VERIFICATION: [How to check the proof]
```

## Integration Examples

### Proving List Non-Empty
```typescript
type NonEmpty<T> = { head: T; tail: T[] };

function proveNonEmpty<T>(list: T[]): Either<NonEmpty<T>, Empty> {
  if (list.length > 0) {
    return left({ head: list[0], tail: list.slice(1) });  // Witness!
  } else {
    return right({ proof: "length is 0" });  // Witness of emptiness!
  }
}
```

### Proving Algorithm Correctness
```typescript
// Specification: sorting produces ordered output
type Sorted<T> = { data: T[]; proof: IsOrdered<T[]> };

function sort<T>(input: T[]): Sorted<T> {
  const result = quicksort(input);
  const proof = verifySorted(result);  // Constructive verification
  return { data: result, proof };
}
```
