# Intuitionistic Logic: A Ramanujan-Depth Exploration

## Through the Lenses of Elon Musk, Donald Trump, and Peter Thiel

> *"An equation has no meaning to me unless it expresses a thought of God."* — Srinivasa Ramanujan

---

## Prologue: Why These Three Minds?

**Elon Musk** represents *first-principles reasoning* — deconstructing problems to foundational axioms and rebuilding from ground truth. This mirrors intuitionistic logic's demand for *constructive proof*.

**Donald Trump** represents *binary decisionism* — the classical logic of "you're either with me or against me." His contrast with intuitionistic thinking illuminates what we *lose* and *gain* by rejecting the law of excluded middle.

**Peter Thiel** represents *contrarian epistemology* — questioning consensus reality and asking "What important truth do few people agree with you on?" This parallels intuitionistic logic's rejection of proof-by-contradiction as sufficient for existence.

---

## Part I: The Foundation — What Is Intuitionistic Logic?

### 1.1 The Classical vs. Intuitionistic Divide

**Classical Logic** (Boolean, Aristotelian):
```
For any proposition P: P ∨ ¬P (Law of Excluded Middle - LEM)
For any proposition P: ¬¬P → P (Double Negation Elimination - DNE)
```

**Intuitionistic Logic** (Brouwer, Heyting, Kolmogorov):
```
P ∨ ¬P is NOT universally valid
¬¬P → P is NOT universally valid
To prove ∃x.P(x), you MUST construct a witness x₀ and prove P(x₀)
```

### 1.2 The BHK Interpretation (Brouwer-Heyting-Kolmogorov)

In intuitionistic logic, a proof IS the meaning:

| Proposition | What Counts as a Proof |
|-------------|----------------------|
| `A ∧ B` | A pair `(proof of A, proof of B)` |
| `A ∨ B` | Either `(left, proof of A)` or `(right, proof of B)` |
| `A → B` | A function transforming any proof of A into a proof of B |
| `∃x.P(x)` | A pair `(witness x₀, proof of P(x₀))` |
| `∀x.P(x)` | A function giving a proof of P(x) for any x |
| `¬A` | A function transforming any proof of A into a proof of ⊥ (absurdity) |

---

## Part II: The Musk Paradigm — First Principles Construction

### 2.1 Musk's Epistemology Maps to Constructivism

> *"I think it's important to reason from first principles rather than by analogy."* — Elon Musk

**Classical reasoning (proof by analogy/contradiction):**
```
"Rockets are expensive because rockets have always been expensive."
"Assume cheap rockets exist → contradiction with industry consensus → cheap rockets don't exist."
```

**Intuitionistic reasoning (constructive proof):**
```
"What are rockets made of? Aluminum, titanium, copper, carbon fiber."
"What do these materials cost on the commodity market? ~2% of the rocket price."
"Here is SpaceX — a constructed witness that cheap rockets exist."
```

### 2.2 The Falcon 9 as a Constructive Proof

In classical logic, you could "prove" reusable rockets are possible by:
```
Assume reusable rockets are impossible.
Derive some contradiction.
Therefore, reusable rockets are possible. ∎
```

**But this gives you NOTHING.** No rocket. No landing. No witness.

Musk's intuitionistic approach:
```haskell
-- Type-theoretic proof of reusable rockets
data ReusableRocket = Falcon9 {
  stage1 :: ReturnableStage,
  landingLegs :: DeployableLegs,
  gridFins :: SteerableFins,
  propulsiveLanding :: LandingAlgorithm
}

-- The WITNESS is the proof
proof_reusable_rockets_exist :: Exists ReusableRocket
proof_reusable_rockets_exist = (Falcon9 {...}, certification_data)
```

### 2.3 Tesla and the Witness Problem

**Classical "proof" of EV viability (circa 2005):**
```
Major automakers say EVs aren't viable.
Assume EVs are viable → contradicts expert consensus → EVs not viable.
```

**Musk's constructive proof:**
```python
class TeslaRoadster:
    """Constructive witness that high-performance EVs exist"""
    def __init__(self):
        self.range = 245  # miles - MEASURED, not assumed
        self.acceleration = 3.7  # 0-60 mph - DEMONSTRATED
        self.top_speed = 125  # mph - ACHIEVED

    def prove_ev_viability(self) -> ConstructiveProof:
        """The car itself IS the proof"""
        return Witness(self, performance_data=self.telemetry())
```

### 2.4 The Boring Company: Constructive vs. Classical Urban Planning

**Classical urban planning logic:**
```
Traffic solutions require either:
- More roads (expensive, NIMBY)
- Public transit (slow, limited)

¬(cheap ∧ fast ∧ direct) — by classical exhaustion

Therefore: Accept traffic as inevitable.
```

**Musk's intuitionistic attack:**
```
Reject the exhaustive disjunction.
Construct a NEW alternative: underground tunnels with autonomous vehicles.
The Boring Company IS the proof that the disjunction was incomplete.
```

**Mathematical insight:** Classical logic's Law of Excluded Middle (P ∨ ¬P) assumes we know ALL possibilities. Intuitionistic logic says: *You can only assert a disjunction if you can construct one of the disjuncts.*

---

## Part III: The Trump Paradigm — Classical Logic's Power and Limits

### 3.1 Binary Decisionism as Classical Logic

> *"You're either with us, or you're against us."*

This IS the Law of Excluded Middle applied to loyalty:
```
∀person: Loyal(person) ∨ ¬Loyal(person)
```

**Classical advantages:**
- Fast decision-making
- Clear coalition formation
- Eliminates ambiguity

**Intuitionistic critique:**
```
-- In intuitionistic logic, this is NOT provable:
loyalty_excluded_middle :: Either (Loyal person) (Not (Loyal person))
loyalty_excluded_middle = ???  -- Cannot construct without evidence

-- You can only assert what you can PROVE:
proven_loyal :: ProofOfLoyalty -> Loyal person
proven_disloyal :: ProofOfDisloyal -> Not (Loyal person)
uncertain :: Neither  -- This state EXISTS intuitionistically
```

### 3.2 The Art of the Deal: Proof by Contradiction vs. Construction

**Classical negotiation (proof by contradiction):**
```
"This is my final offer."
"Assume you reject it → you get nothing → contradiction with your interests"
"Therefore, you accept." ∎
```

**Intuitionistic negotiation:**
```
"Here is a concrete proposal: [specific terms]"
"Here is how both parties benefit: [constructed mutual gain]"
"The deal itself is the witness that agreement is possible."
```

**Key insight:** Trump's negotiation style often relies on *tertium non datur* (no third option) — a classical axiom. Intuitionistic logic permits: "Neither accept nor reject; construct a new option."

### 3.3 "Fake News" and the Double Negation Problem

**Classical logic allows:**
```
¬¬True(News) → True(News)
"It's not the case that this news is not true" → "This news is true"
```

**Intuitionistic logic demands:**
```
To prove True(News), you must CONSTRUCT:
1. Primary sources
2. Verifiable evidence
3. Reproducible methodology

¬¬True(News) only means: "Assuming this news is false leads to contradiction"
This does NOT construct actual truth.
```

### 3.4 The Border Wall as Constructive vs. Classical Security

**Classical security argument:**
```
Assume the border is secure without a wall.
Derive contradiction (illegal crossings occur).
Therefore: ¬(Secure without wall)
By classical logic: Wall → Secure (by elimination)
```

**Intuitionistic security argument:**
```haskell
-- Must construct ACTUAL security mechanism
data BorderSecurity =
    PhysicalBarrier Wall
  | TechnologicalSurveillance Sensors
  | PersonnelDeployment Agents
  | LegalFramework Immigration
  | Combination [BorderSecurity]

-- Proof requires demonstrating effectiveness:
prove_security :: BorderSecurity -> SecurityMetrics -> Proof Secure
prove_security mechanism metrics =
  if measured_effectiveness metrics > threshold
  then Witness (mechanism, metrics)
  else Insufficient
```

---

## Part IV: The Thiel Paradigm — Contrarian Epistemology

### 4.1 "What Important Truth Do Few People Agree With You On?"

This question IS intuitionistic epistemology:

**Classical consensus logic:**
```
Most experts believe X.
Assume ¬X → contradiction with expert consensus.
Therefore X. ∎
```

**Thiel's intuitionistic counter:**
```
Consensus is NOT a constructive proof.
To know X, you must construct understanding of X.
Popular belief in X does not constitute proof of X.
The absence of disproof is not proof.
```

### 4.2 Zero to One: The Constructive Creation of New Categories

> *"Every moment in business happens only once. The next Bill Gates will not build an operating system. The next Larry Page won't build a search engine."*

**Classical categorization:**
```
∀company: (Monopoly(company) ∨ Competition(company))
Success = Monopoly
Failure = Competition
```

**Thiel's intuitionistic insight:**
```haskell
-- You cannot prove "my company will be successful" by contradiction
-- You must CONSTRUCT the success

data ZeroToOne = NewCategory {
  uniqueValue :: UniqueValueProposition,  -- CONSTRUCTED, not assumed
  defenseability :: MoatProof,            -- DEMONSTRATED, not claimed
  secret :: NonObviousTruth               -- DISCOVERED, not derived
}

-- PayPal as constructive proof:
paypal_proof :: ZeroToOne
paypal_proof = NewCategory {
  uniqueValue = "First reliable online payment system",
  defenseability = NetworkEffects + Regulatory + Brand,
  secret = "People will trust digital payments before institutions do"
}
```

### 4.3 The Straussian Reading: Hidden Knowledge as Intuitionistic

Thiel's interest in Leo Strauss connects to intuitionistic logic:

**Classical (exoteric) knowledge:**
```
Published results are TRUE because peer-reviewed.
¬¬Peer_Reviewed(P) → True(P)  -- Double negation elimination
```

**Intuitionistic (esoteric) knowledge:**
```
-- True understanding requires construction:
type Understanding = Construction of (
  FirstPrinciples,
  PersonalDerivation,
  InternalizedMeaning
)

-- Reading Strauss's "Persecution and the Art of Writing":
esoteric_meaning :: Text -> Reader -> Maybe Understanding
esoteric_meaning text reader =
  case construct_interpretation reader text of
    Just derivation -> Just (verify derivation)
    Nothing -> Nothing  -- Meaning is not automatic
```

### 4.4 Thiel on Higher Education: Constructive vs. Credentialist

**Classical credentialism:**
```
Has_Degree(person) ∨ ¬Has_Degree(person)  -- LEM
Competent(person) ↔ Has_Degree(person)    -- Assumed equivalence
```

**Thiel Fellowship intuitionistic model:**
```python
class ThielFellow:
    """Constructive proof of competence"""

    def prove_competence(self) -> ConstructiveProof:
        return BuildSomething(
            artifact=self.company_or_project,
            impact=self.measurable_outcomes,
            witness=self.actual_users_or_customers
        )

    # The CONSTRUCTION is the proof, not the credential
    # A degree is ¬¬Competent at best (you haven't been proven incompetent)
    # A successful startup is ∃x.Competent(x) — an actual witness
```

---

## Part V: The Deep Mathematics — Ramanujan-Level Insights

### 5.1 The Curry-Howard-Lambek Correspondence

This is the profound unity underlying constructive logic:

```
╔═══════════════════╦═══════════════════╦═══════════════════╗
║    LOGIC          ║   PROGRAMMING     ║   CATEGORY THEORY ║
╠═══════════════════╬═══════════════════╬═══════════════════╣
║ Proposition       ║ Type              ║ Object            ║
║ Proof             ║ Program           ║ Morphism          ║
║ A → B             ║ Function A → B    ║ Hom(A, B)         ║
║ A ∧ B             ║ Product (A, B)    ║ A × B             ║
║ A ∨ B             ║ Sum Either A B    ║ A + B             ║
║ ⊥ (False)         ║ Empty/Void type   ║ Initial object 0  ║
║ ⊤ (True)          ║ Unit type ()      ║ Terminal object 1 ║
║ ¬A                ║ A → Void          ║ Hom(A, 0)         ║
║ ∀x.P(x)           ║ (x : A) → P x     ║ Right adjoint ∏   ║
║ ∃x.P(x)           ║ Σ(x : A). P x     ║ Left adjoint Σ    ║
╚═══════════════════╩═══════════════════╩═══════════════════╝
```

### 5.2 Heyting Algebras: The Algebraic Semantics

**Classical Boolean algebra:**
```
a ∨ ¬a = 1 (top element)     -- LEM as algebraic law
¬¬a = a                       -- Involutive negation
```

**Heyting algebra (intuitionistic):**
```
a ∨ ¬a ≤ 1 (may be strictly less)
¬¬a ≥ a (only one direction holds)

Relative pseudo-complement:
a → b = max{c : a ∧ c ≤ b}

Negation is derived:
¬a = a → ⊥ = max{c : a ∧ c ≤ ⊥} = max{c : a ∧ c = ⊥}
```

**Ramanujan-style insight:** Every Heyting algebra is the algebra of open sets of some topological space. Intuitionistic truth is *local* — it depends on *where you observe from*.

### 5.3 Kripke Semantics: Truth in Possible Worlds

```
A Kripke frame K = (W, ≤, V) where:
- W = set of "worlds" (states of knowledge)
- ≤ = accessibility relation (knowledge growth)
- V = valuation function (what's true where)

Forcing relation (⊩):
w ⊩ P        iff V(w, P) = true
w ⊩ A ∧ B    iff w ⊩ A and w ⊩ B
w ⊩ A ∨ B    iff w ⊩ A or w ⊩ B
w ⊩ A → B    iff ∀v ≥ w: (v ⊩ A ⟹ v ⊩ B)
w ⊩ ¬A       iff ∀v ≥ w: v ⊮ A
```

**Key property (Monotonicity):**
```
If w ⊩ A and w ≤ v, then v ⊩ A
"Once known, always known" — knowledge persists
```

**Why LEM fails:**
```
At world w₀: Neither P nor ¬P is forced
At w₁ ≥ w₀: P becomes forced
At w₂ ≥ w₀: ¬P becomes forced (different branch)

At w₀: Cannot assert P ∨ ¬P because we don't know which branch we're on!
```

### 5.4 Topos Theory: The Universe of Intuitionistic Mathematics

**A topos is a category that behaves like Set but with intuitionistic internal logic:**

```haskell
class Topos t where
  -- Terminal object (singleton, truth)
  terminal :: t ()

  -- Products (conjunction)
  product :: t a -> t b -> t (a, b)

  -- Exponentials (implication, function spaces)
  exponential :: t a -> t b -> t (a -> b)

  -- Subobject classifier (truth values, generalizes {True, False})
  omega :: t Omega  -- NOT necessarily Boolean!

  -- Characteristic morphism of subobjects
  chi :: Subobject a -> (a -> Omega)
```

**The subobject classifier Ω:**
- In **Set** (classical): Ω = {0, 1} = Bool
- In **Sh(X)** (sheaves on space X): Ω = open sets of X — a Heyting algebra!
- In a general topos: Ω is a Heyting algebra, not Boolean

### 5.5 The Effective Topos: Computability as Logic

**Realizability:** A number e *realizes* a formula φ if e encodes a computation proving φ.

```
e ⊩ A ∧ B       iff π₁(e) ⊩ A and π₂(e) ⊩ B
e ⊩ A → B       iff ∀d: d ⊩ A ⟹ {e}(d)↓ and {e}(d) ⊩ B
e ⊩ ∃x.φ(x)     iff π₁(e) is a witness n and π₂(e) ⊩ φ(n)
```

**The Effective Topos Eff:**
- Objects are "assemblies" — sets with computability structure
- Internal logic is intuitionistic
- **LEM fails** because there exist propositions P where neither P nor ¬P has a realizer

**Ramanujan insight:** The Effective Topos is where mathematics and computation become ONE. Every proof is a program. Every theorem is a type. Constructive mathematics is the logic of what can actually be computed.

---

## Part VI: Practical Implementation — Code as Proof

### 6.1 TypeScript: Intuitionistic Logic in Practice

```typescript
// Classical logic: boolean operations
function classicalOr<A, B>(a: boolean, b: boolean): boolean {
  return a || b;  // Just true/false, no witness
}

// Intuitionistic logic: constructive disjunction
type Either<A, B> = { tag: 'left'; value: A } | { tag: 'right'; value: B };

function intuitionisticOr<A, B>(
  proof: Either<A, B>
): A | B {
  // We MUST provide which side and the actual value
  switch (proof.tag) {
    case 'left': return proof.value;
    case 'right': return proof.value;
  }
}

// LEM would require:
function excludedMiddle<A>(): Either<A, (a: A) => never> {
  // IMPOSSIBLE TO IMPLEMENT without knowing A!
  // We cannot construct either side without information
  throw new Error("LEM is not constructively provable");
}

// But double negation INTRODUCTION is fine:
function doubleNegIntro<A>(a: A): (f: (a: A) => never) => never {
  return (f) => f(a);  // If you have A, and someone claims ¬A, derive contradiction
}

// Double negation ELIMINATION is NOT generally possible:
function doubleNegElim<A>(dna: (f: (a: A) => never) => never): A {
  // IMPOSSIBLE - we can't extract A from knowing ¬¬A
  // We'd need to call dna with something, but we don't have (A → never)
  throw new Error("DNE is not constructively provable");
}
```

### 6.2 Haskell: Dependent Types and Proofs

```haskell
{-# LANGUAGE GADTs, TypeFamilies, DataKinds, RankNTypes #-}

-- Propositional equality as a type
data a :~: b where
  Refl :: a :~: a

-- Constructive existence
data Exists (p :: k -> Type) where
  Ex :: p x -> Exists p

-- Natural numbers as types (Peano)
data Nat = Z | S Nat

-- Vector with length in type
data Vec (n :: Nat) a where
  VNil  :: Vec 'Z a
  VCons :: a -> Vec n a -> Vec ('S n) a

-- PROOF: concatenation preserves length (constructively!)
type family (m :: Nat) + (n :: Nat) :: Nat where
  'Z     + n = n
  ('S m) + n = 'S (m + n)

append :: Vec m a -> Vec n a -> Vec (m + n) a
append VNil         ys = ys
append (VCons x xs) ys = VCons x (append xs ys)
-- The TYPE is the theorem, the FUNCTION is the proof

-- Musk's first-principles: The program IS the proof
-- No assumptions, pure construction
```

### 6.3 Agda: Full Dependent Types

```agda
-- Intuitionistic logic in Agda

-- The empty type (⊥, False)
data ⊥ : Set where

-- Negation is A → ⊥
¬ : Set → Set
¬ A = A → ⊥

-- Sum type (disjunction, A ∨ B)
data _⊎_ (A B : Set) : Set where
  inj₁ : A → A ⊎ B
  inj₂ : B → A ⊎ B

-- LEM is NOT provable:
-- lem : (A : Set) → A ⊎ (¬ A)
-- lem A = ?  -- Cannot implement!

-- But LEM for decidable propositions IS provable:
data Dec (A : Set) : Set where
  yes : A → Dec A
  no  : ¬ A → Dec A

-- Natural numbers have decidable equality:
_≟_ : (m n : ℕ) → Dec (m ≡ n)
zero ≟ zero = yes refl
zero ≟ suc n = no (λ ())
suc m ≟ zero = no (λ ())
suc m ≟ suc n with m ≟ n
... | yes refl = yes refl
... | no ¬p = no (λ { refl → ¬p refl })

-- PROOF that ¬¬-elimination implies LEM
-- (showing they are classically equivalent)
dne→lem : ({A : Set} → ¬ (¬ A) → A) → {A : Set} → A ⊎ (¬ A)
dne→lem dne = dne (λ ¬lem → ¬lem (inj₂ (λ a → ¬lem (inj₁ a))))
```

### 6.4 The Musk-Thiel-Trump Code Comparison

```python
# How each leader might implement "prove success is possible"

# TRUMP STYLE: Classical assertion
def trump_prove_success():
    """Classical proof by assertion and elimination"""
    # "Either I succeed bigly, or they cheated"
    # P ∨ ¬P asserted, no construction needed
    assert True, "I always succeed. Believe me."
    return "SUCCESS - the best success, everyone says so"

# MUSK STYLE: Constructive witness
def musk_prove_success():
    """Intuitionistic proof by construction"""
    # Actually build the thing
    rocket = build_reusable_rocket()  # Witness
    land_successfully(rocket)          # Verify
    return ConstructiveProof(
        witness=rocket,
        evidence=landing_telemetry(),
        reproducible=True
    )

# THIEL STYLE: Contrarian construction
def thiel_prove_success():
    """Intuitionistic proof via contrarian insight"""
    # Find the secret that breaks conventional wisdom
    secret = discover_non_obvious_truth()

    # Construct monopoly from secret
    company = build_zero_to_one(
        insight=secret,
        moat=create_defensible_position(secret),
        category=create_new_category()  # Don't compete, create
    )

    return ConstructiveProof(
        witness=company,
        evidence=market_dominance_metrics(),
        secret=secret  # The non-obvious truth that made it possible
    )
```

---

## Part VII: The Synthesis — Intuitionistic Logic for Builders

### 7.1 The Builder's Creed (Intuitionistic)

```
I do not claim something exists until I can construct it.
I do not claim I know something until I can derive it.
I do not accept "by contradiction" as sufficient for creation.
I build witnesses, not assertions.
```

### 7.2 Practical Takeaways

**For Founders (Musk):**
- Don't prove your idea is possible by showing alternatives fail
- BUILD the prototype — it is the proof
- First principles = axioms; construction = proof; product = witness

**For Leaders (Trump):**
- Classical logic enables fast decisions but limits discovery
- LEM ("with us or against us") closes options that construction might reveal
- Sometimes the third option exists — but only if you construct it

**For Investors (Thiel):**
- Consensus is not proof; construct your own understanding
- Look for founders with constructive proofs, not pitch decks
- The secret (non-obvious truth) is an existential witness others lack

### 7.3 The Meta-Insight

**Classical Logic:** The universe of completed, static truth.
**Intuitionistic Logic:** The universe of *knowable*, *constructible*, *computable* truth.

The difference is not merely philosophical — it's the difference between:
- Claiming rockets could theoretically be reusable (classical)
- Landing a Falcon 9 on a drone ship (constructive)

Between:
- Believing success is possible because failure seems contradictory (classical)
- Having a working company generating revenue (constructive)

Between:
- Knowing a secret must exist because the market is inefficient (classical)
- Discovering the specific non-obvious truth that unlocks a monopoly (constructive)

---

## Appendix A: Ramanujan's Own Intuitionism

Ramanujan claimed many of his formulas came to him in dreams, from the goddess Namagiri. While this sounds mystical, it aligns with constructive mathematics:

**Ramanujan's process:**
1. Receive formula (intuition)
2. Verify specific cases (construction)
3. Find pattern (abstraction)
4. Prove rigorously when possible (formalization)

His famous continued fractions, modular equations, and infinite series are *constructive* — they provide explicit formulas, not existence proofs by contradiction.

```
Ramanujan's sum formula (constructive!):
1 + 2 + 3 + ... = -1/12 (Ramanujan summation)

This is NOT classical sum (which diverges)
It's a CONSTRUCTED regularization — the analytic continuation
The witness is the zeta function: ζ(-1) = -1/12
```

## Appendix B: The Topos of Business

```
The category Biz:
- Objects: Companies, Markets, Products
- Morphisms: Transactions, Partnerships, Acquisitions

The internal logic of Biz:
- Not Boolean! Markets have uncertainty, partial information
- Subobject classifier Ω = {fail, pivot, survive, scale, dominate, ...}
  (More than just True/False)

For proposition "This startup will succeed":
- Classical: Succeed ∨ ¬Succeed (one must be true now)
- Intuitionistic: Cannot assert until we have:
  - A witness (the successful company), OR
  - Proof of failure (bankruptcy, shutdown)
```

---

## Conclusion

Intuitionistic logic is not weaker than classical logic — it is *more honest*. It refuses to claim knowledge without construction, existence without witness, truth without proof.

**Elon Musk** embodies this: he doesn't argue electric cars are possible, he builds them.

**Peter Thiel** embodies this: he doesn't accept consensus as truth, he constructs contrarian understanding.

**Donald Trump** illustrates the contrast: classical binary logic is powerful for action but blind to constructed alternatives.

The deep lesson, worthy of Ramanujan's insight:

> *In intuitionistic logic, proof is not about establishing static truth — it is about constructing dynamic reality. The witness IS the proof. The company IS the theorem. The rocket landing on a drone ship IS the QED.*

Build your proofs. Construct your witnesses. The universe rewards those who create, not those who merely assert.

---

*"The mathematician does not study pure mathematics because it is useful; he studies it because he delights in it and he delights in it because it is beautiful."* — Henri Poincaré

*"The best way to predict the future is to invent it."* — Alan Kay

*"The best way to prove the future is to construct it."* — Intuitionistic Logic

---

## References

1. Brouwer, L.E.J. (1912). "Intuitionism and Formalism"
2. Heyting, A. (1930). "Die formalen Regeln der intuitionistischen Logik"
3. Kolmogorov, A.N. (1932). "Zur Deutung der intuitionistischen Logik"
4. Martin-Löf, P. (1984). "Intuitionistic Type Theory"
5. Lambek, J. & Scott, P.J. (1986). "Introduction to Higher Order Categorical Logic"
6. Johnstone, P.T. (2002). "Sketches of an Elephant: A Topos Theory Compendium"
7. Thiel, P. (2014). "Zero to One"
8. Vance, A. (2015). "Elon Musk: Tesla, SpaceX, and the Quest for a Fantastic Future"
9. Univalent Foundations Program. (2013). "Homotopy Type Theory"
