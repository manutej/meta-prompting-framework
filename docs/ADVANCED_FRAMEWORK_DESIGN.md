# Advanced Meta-Prompting Framework: Architectural Design

**Version:** 2.0
**Date:** November 2025
**Based on:** Gap analysis of current framework vs. state-of-the-art research

---

## 1. Design Philosophy

### 1.1 Core Principles

1. **Category Theory First**: All abstractions have rigorous categorical semantics
2. **Type Safety**: Compositional prompts with compile-time guarantees
3. **Declarative**: Specify what, not how (framework optimizes the how)
4. **Backward Compatible**: Extends current framework without breaking changes
5. **Production Ready**: Performance, caching, observability built-in

### 1.2 Theoretical Foundations

**Functorial Meta-Prompting** (Zhang et al.):
```
F: Task → Prompt  (functor preserving composition)
```

**RMP Monad** (Zhang et al.):
```
M: Prompt → Prompt  (endofunctor)
η: Id ⇒ M           (unit natural transformation)
μ: M∘M ⇒ M          (multiplication natural transformation)
```

**Enriched Categories** (de Wynter et al.):
```
Prompt_[0,1] where hom(P, Q) ∈ [0,1]  (quality metric)
```

**Polynomial Functors** (Spivak):
```
p = Σᵢ y^(Aᵢ)  (positions + directions for bidirectional tools)
```

---

## 2. Module Architecture

### 2.1 Five-Layer Stack

```
Layer 5: Applications
  ↓ uses
Layer 4: Optimizers & Composition
  ↓ uses
Layer 3: Prompt Modules & Constraints
  ↓ uses
Layer 2: Categorical Abstractions
  ↓ uses
Layer 1: Execution Runtime (current framework)
```

### 2.2 Dependency Graph

```
applications/
    ↓
optimizers/ ────┐
    ↓           ↓
prompts/    categorical/
    ↓           ↓
    └───→ core/ ←───┘
```

---

## 3. Layer 2: Categorical Abstractions

### 3.1 Functor Base Class

**File:** `meta_prompting_framework/categorical/functor.py`

```python
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Callable

A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')

class Functor(ABC, Generic[A]):
    """
    A functor F: C → D between categories.

    Must satisfy functor laws:
    1. F(id_A) = id_F(A)           (identity preservation)
    2. F(g ∘ f) = F(g) ∘ F(f)      (composition preservation)
    """

    @abstractmethod
    def fmap(self, f: Callable[[A], B]) -> 'Functor[B]':
        """Map a function over the functor.

        Laws:
        - fmap(id) = id
        - fmap(g . f) = fmap(g) . fmap(f)
        """
        pass

    def __rshift__(self, f: Callable[[A], B]) -> 'Functor[B]':
        """Infix operator: functor >> f === functor.fmap(f)"""
        return self.fmap(f)


class MetaPromptFunctor(Functor[A]):
    """
    Functor from Task category to Prompt category.

    F: Task → Prompt

    Example:
        task = Task("Solve x^2 + 5x + 6 = 0")
        prompt_functor = MetaPromptFunctor()
        prompt = prompt_functor.fmap(task)
    """

    def __init__(self, strategy: str = "chain_of_thought"):
        self.strategy = strategy
        self._value = None

    def fmap(self, task: Callable[[A], B]) -> 'MetaPromptFunctor[B]':
        """Maps a task transformation to a prompt transformation."""
        # Implementation will preserve composition
        transformed = task(self._value) if self._value else None
        result = MetaPromptFunctor(self.strategy)
        result._value = transformed
        return result


# Functor Laws Verification
def verify_functor_laws(F: Functor, value, f: Callable, g: Callable):
    """Property-based testing for functor laws."""
    # Law 1: fmap(id) = id
    identity = lambda x: x
    assert F.fmap(identity)._value == value, "Identity law violated"

    # Law 2: fmap(g . f) = fmap(g) . fmap(f)
    composed = lambda x: g(f(x))
    assert (F.fmap(composed)._value ==
            F.fmap(f).fmap(g)._value), "Composition law violated"
```

### 3.2 Monad Implementation

**File:** `meta_prompting_framework/categorical/monad.py`

```python
from typing import TypeVar, Generic, Callable
from .functor import Functor

T = TypeVar('T')
U = TypeVar('U')

class Monad(Functor[T]):
    """
    A monad (M, η, μ) consists of:
    - An endofunctor M: C → C
    - A unit natural transformation η: Id_C ⇒ M
    - A multiplication natural transformation μ: M∘M ⇒ M

    Must satisfy monad laws:
    1. μ ∘ (η ∘ M) = id_M           (left unit)
    2. μ ∘ (M ∘ η) = id_M           (right unit)
    3. μ ∘ (μ ∘ M) = μ ∘ (M ∘ μ)    (associativity)
    """

    def __init__(self, value: T):
        self._value = value

    @classmethod
    def unit(cls, value: T) -> 'Monad[T]':
        """
        η: A → M(A)

        Lifts a pure value into the monadic context.
        """
        return cls(value)

    @abstractmethod
    def flat_map(self, f: Callable[[T], 'Monad[U]']) -> 'Monad[U]':
        """
        Also called bind (>>=).

        Applies a monadic function and flattens the result.
        Equivalent to: fmap(f) followed by flatten.
        """
        pass

    def flatten(self) -> 'Monad[T]':
        """
        μ: M(M(A)) → M(A)

        Flattens nested monadic structure.
        """
        if isinstance(self._value, Monad):
            return self._value.flatten()
        return self

    def fmap(self, f: Callable[[T], U]) -> 'Monad[U]':
        """Functor's fmap, implemented via flat_map."""
        return self.flat_map(lambda x: Monad.unit(f(x)))

    def __rshift__(self, f: Callable[[T], 'Monad[U]']) -> 'Monad[U]':
        """Infix operator: m >> f === m.flat_map(f)"""
        return self.flat_map(f)


class RMPMonad(Monad[str]):
    """
    Recursive Meta-Prompting Monad.

    Implements Zhang et al.'s RMP as a monad where:
    - Values are prompts (strings)
    - unit embeds a prompt into meta-space
    - flat_map applies meta-improvement and flattens

    Example:
        prompt = "Solve this equation"
        rmp = RMPMonad.unit(prompt)
        improved = rmp.flat_map(improve_with_context)
    """

    def __init__(
        self,
        prompt: str,
        quality: float = 0.0,
        iteration: int = 0,
        context: dict = None
    ):
        super().__init__(prompt)
        self.quality = quality
        self.iteration = iteration
        self.context = context or {}

    @classmethod
    def unit(cls, prompt: str) -> 'RMPMonad':
        """η: Prompt → M(Prompt)"""
        return cls(prompt, quality=0.0, iteration=0, context={})

    def flat_map(self, f: Callable[[str], 'RMPMonad']) -> 'RMPMonad':
        """
        Applies meta-improvement function and flattens.

        The function f should take a prompt and return an improved RMPMonad.
        Quality is non-decreasing (max of current and next).
        Context accumulates.
        """
        next_monad = f(self._value)

        return RMPMonad(
            prompt=next_monad._value,
            quality=max(self.quality, next_monad.quality),
            iteration=self.iteration + 1,
            context={**self.context, **next_monad.context}
        )

    def __repr__(self):
        return f"RMPMonad(quality={self.quality:.2f}, iter={self.iteration})"


# Monad Laws Verification
def verify_monad_laws(M: type, value, f: Callable, g: Callable):
    """Property-based testing for monad laws."""
    m = M.unit(value)

    # Law 1: Left unit
    # unit(a).flat_map(f) === f(a)
    assert (M.unit(value).flat_map(f)._value ==
            f(value)._value), "Left unit law violated"

    # Law 2: Right unit
    # m.flat_map(unit) === m
    assert (m.flat_map(M.unit)._value ==
            m._value), "Right unit law violated"

    # Law 3: Associativity
    # m.flat_map(f).flat_map(g) === m.flat_map(lambda x: f(x).flat_map(g))
    left = m.flat_map(f).flat_map(g)
    right = m.flat_map(lambda x: f(x).flat_map(g))
    assert left._value == right._value, "Associativity law violated"
```

### 3.3 Natural Transformations

**File:** `meta_prompting_framework/categorical/natural_transformation.py`

```python
from typing import TypeVar, Generic, Callable
from .functor import Functor

A = TypeVar('A')

class NaturalTransformation(Generic[A]):
    """
    A natural transformation η: F ⇒ G between functors F, G: C → D.

    For each object A in C, provides a morphism η_A: F(A) → G(A)
    such that the naturality square commutes:

        F(A) --η_A--> G(A)
         |             |
       F(f)          G(f)
         |             |
         ↓             ↓
        F(B) --η_B--> G(B)

    Commutativity: η_B ∘ F(f) = G(f) ∘ η_A
    """

    def __init__(
        self,
        source_functor: type[Functor],
        target_functor: type[Functor],
        component: Callable[[Functor[A]], Functor[A]]
    ):
        self.source = source_functor
        self.target = target_functor
        self.component = component

    def apply(self, fa: Functor[A]) -> Functor[A]:
        """Apply the natural transformation at component A."""
        return self.component(fa)

    def __call__(self, fa: Functor[A]) -> Functor[A]:
        """Alias for apply."""
        return self.apply(fa)


# Example: Natural transformation between prompting strategies
class StrategyTransformation(NaturalTransformation):
    """
    Transforms one prompting strategy to another.

    Example: ChainOfThought ⇒ ReAct
    """

    def __init__(self, from_strategy: str, to_strategy: str):
        def transform(prompt_functor):
            # Convert prompt from one strategy to another
            new_functor = type(prompt_functor)(to_strategy)
            new_functor._value = self._convert_prompt(
                prompt_functor._value,
                from_strategy,
                to_strategy
            )
            return new_functor

        super().__init__(
            source_functor=MetaPromptFunctor,
            target_functor=MetaPromptFunctor,
            component=transform
        )
        self.from_strategy = from_strategy
        self.to_strategy = to_strategy

    def _convert_prompt(self, prompt: str, from_s: str, to_s: str) -> str:
        """Convert prompt between strategies."""
        # Implementation would use LLM to transform prompt style
        return prompt  # Placeholder


def verify_naturality(
    eta: NaturalTransformation,
    F: Functor[A],
    f: Callable[[A], B]
) -> bool:
    """
    Verify naturality square commutes:
    η_B ∘ F(f) = G(f) ∘ η_A
    """
    # Left path: F(f) then η_B
    left = eta.apply(F.fmap(f))

    # Right path: η_A then G(f)
    right = eta.apply(F).fmap(f)

    return left._value == right._value
```

### 3.4 Enriched Categories

**File:** `meta_prompting_framework/categorical/enriched.py`

```python
from typing import TypeVar, Generic, Callable, Protocol
from dataclasses import dataclass
import numpy as np

V = TypeVar('V')  # Enrichment base (e.g., [0,1], R+, probability distributions)
A = TypeVar('A')
B = TypeVar('B')

class MonoidalCategory(Protocol):
    """
    A monoidal category (V, ⊗, I) for enrichment.

    Examples:
    - ([0,1], max, 0): For quality metrics
    - (R+, +, 0): For costs
    - (Prob, *, uniform): For probability distributions
    """

    def tensor(self, a: V, b: V) -> V:
        """Monoidal product ⊗"""
        ...

    def unit(self) -> V:
        """Monoidal unit I"""
        ...


@dataclass
class QualityMetric:
    """[0,1] monoidal category for quality enrichment."""

    value: float  # 0.0 to 1.0

    def __post_init__(self):
        assert 0.0 <= self.value <= 1.0, "Quality must be in [0,1]"

    def tensor(self, other: 'QualityMetric') -> 'QualityMetric':
        """Monoidal product: max (optimistic composition)"""
        return QualityMetric(max(self.value, other.value))

    @staticmethod
    def unit() -> 'QualityMetric':
        """Monoidal unit: 0 (worst quality)"""
        return QualityMetric(0.0)

    def __mul__(self, other: 'QualityMetric') -> 'QualityMetric':
        """Infix operator for tensor"""
        return self.tensor(other)


class EnrichedCategory(Generic[V]):
    """
    A category enriched over monoidal category V.

    Instead of hom-sets hom(A,B), we have hom-objects hom(A,B) ∈ V.

    For prompts enriched over [0,1]:
    - Objects: Prompts
    - hom(P, Q): Quality of transforming P to Q
    """

    def __init__(self, monoidal_base: MonoidalCategory):
        self.base = monoidal_base
        self._hom_objects = {}  # (A, B) -> V

    def hom(self, a: A, b: B) -> V:
        """Get hom-object from A to B."""
        return self._hom_objects.get((a, b), self.base.unit())

    def set_hom(self, a: A, b: B, value: V):
        """Set hom-object from A to B."""
        self._hom_objects[(a, b)] = value

    def compose(self, a: A, b: B, c: C) -> V:
        """
        Enriched composition:
        hom(B,C) ⊗ hom(A,B) → hom(A,C)

        For quality: quality(A→C) = max(quality(A→B), quality(B→C))
        """
        ab = self.hom(a, b)
        bc = self.hom(b, c)
        ac = self.base.tensor(ab, bc)
        self.set_hom(a, c, ac)
        return ac


class QualityEnrichedPrompts(EnrichedCategory[QualityMetric]):
    """
    Prompts enriched over [0,1] quality metric.

    Usage:
        prompts = QualityEnrichedPrompts()
        prompts.set_hom("basic prompt", "improved prompt", QualityMetric(0.8))
        prompts.set_hom("improved prompt", "optimized prompt", QualityMetric(0.9))

        # Compose: quality(basic → optimized) = max(0.8, 0.9) = 0.9
        final_quality = prompts.compose(
            "basic prompt",
            "improved prompt",
            "optimized prompt"
        )
    """

    def __init__(self):
        super().__init__(QualityMetric)

    def add_prompt_refinement(
        self,
        original: str,
        refined: str,
        quality_improvement: float
    ):
        """Add a prompt refinement with quality metric."""
        self.set_hom(original, refined, QualityMetric(quality_improvement))
```

### 3.5 Polynomial Functors

**File:** `meta_prompting_framework/categorical/polynomial.py`

```python
from typing import TypeVar, Generic, Callable, List, Tuple
from dataclasses import dataclass

Position = TypeVar('Position')
Direction = TypeVar('Direction')

@dataclass
class PolynomialFunctor(Generic[Position, Direction]):
    """
    A polynomial functor p = Σᵢ y^(Aᵢ)

    Models bidirectional interaction:
    - Positions: Output states (the indices i)
    - Directions: Input requests at each position (the sets Aᵢ)

    Example (tool/agent):
    - Position: Current state after tool execution
    - Direction: What inputs the tool needs at that state
    """

    positions: List[Position]
    directions: Callable[[Position], List[Direction]]

    def map_position(self, f: Callable[[Position], Position]) -> 'PolynomialFunctor':
        """Map over positions (covariant)."""
        return PolynomialFunctor(
            positions=[f(p) for p in self.positions],
            directions=lambda p: self.directions(f(p))
        )

    def map_direction(self, f: Callable[[Direction], Direction]) -> 'PolynomialFunctor':
        """Map over directions (contravariant)."""
        return PolynomialFunctor(
            positions=self.positions,
            directions=lambda p: [f(d) for d in self.directions(p)]
        )

    def compose(
        self,
        other: 'PolynomialFunctor'
    ) -> 'PolynomialFunctor':
        """
        Polynomial composition p ◁ q.

        Output of q becomes input to p.
        """
        new_positions = []

        def new_directions(pos):
            # Directions from this position go through both functors
            return [
                d_outer
                for p_inner in other.positions
                for d_outer in self.directions(pos)
                for d_inner in other.directions(p_inner)
            ]

        for p_outer in self.positions:
            for p_inner in other.positions:
                new_positions.append((p_outer, p_inner))

        return PolynomialFunctor(new_positions, new_directions)


@dataclass
class Lens(Generic[Position, Direction]):
    """
    A lens is a special polynomial functor y^A.

    Represents a single position with A possible directions.
    Fundamental building block for tool interfaces.
    """

    get: Callable[[Position], Direction]  # Forward: extract value
    set: Callable[[Position, Direction], Position]  # Backward: update

    def to_polynomial(self) -> PolynomialFunctor:
        """Convert lens to polynomial functor."""
        return PolynomialFunctor(
            positions=[self.position],
            directions=lambda _: [self.get(self.position)]
        )

    def compose(self, other: 'Lens') -> 'Lens':
        """Lens composition (sequential access)."""
        return Lens(
            get=lambda s: other.get(self.get(s)),
            set=lambda s, d: self.set(s, other.set(self.get(s), d))
        )


# Example: Tool interface as polynomial functor
class ToolInterface:
    """
    Model a tool (e.g., database query) as a polynomial functor.

    Positions: Query results
    Directions: Needed parameters for each result
    """

    def __init__(self, tool_name: str):
        self.tool_name = tool_name

    def as_polynomial(self) -> PolynomialFunctor[str, dict]:
        """
        Represent tool as polynomial functor.

        Example (database):
        - Position "empty": needs {table, columns}
        - Position "results": needs {next_page, filters}
        """
        def directions(position: str) -> List[dict]:
            if position == "empty":
                return [{"table": str, "columns": List[str]}]
            elif position == "results":
                return [{"next_page": int, "filters": dict}]
            else:
                return []

        return PolynomialFunctor(
            positions=["empty", "results", "complete"],
            directions=directions
        )


# Wiring diagram for tool composition
def wire_tools(
    tool1: PolynomialFunctor,
    tool2: PolynomialFunctor
) -> PolynomialFunctor:
    """
    Compose tools using polynomial composition.

    Output of tool1 feeds into input of tool2.
    """
    return tool2.compose(tool1)
```

---

## 4. Layer 3: Prompt Modules & Constraints

### 4.1 Signatures (Type System)

**File:** `meta_prompting_framework/prompts/signature.py`

```python
from typing import Any, List, Dict, Optional, get_type_hints
from dataclasses import dataclass, field
from pydantic import BaseModel, Field as PydanticField

@dataclass
class Field:
    """
    A typed field in a signature.

    Similar to DSPy's InputField/OutputField.
    """

    name: str
    type: type
    description: str = ""
    required: bool = True
    default: Any = None

    def validate(self, value: Any) -> bool:
        """Validate value against type."""
        try:
            if self.type == int:
                return isinstance(value, int)
            elif self.type == float:
                return isinstance(value, (int, float))
            elif self.type == str:
                return isinstance(value, str)
            elif self.type == List[str]:
                return isinstance(value, list) and all(isinstance(x, str) for x in value)
            else:
                return isinstance(value, self.type)
        except:
            return False


class InputField(Field):
    """Input field to a prompt."""
    pass


class OutputField(Field):
    """Output field from a prompt."""
    pass


class Signature:
    """
    A signature defines the input/output structure of a prompt.

    Example:
        class QASignature(Signature):
            \"\"\"Answer questions with reasoning.\"\"\"
            question = InputField(str, "The question to answer")
            reasoning = OutputField(str, "Step-by-step reasoning")
            answer = OutputField(str, "Final answer")

    Usage:
        signature = QASignature()
        signature.validate_input({"question": "What is 2+2?"})
        signature.validate_output({"reasoning": "...", "answer": "4"})
    """

    def __init__(self):
        self._input_fields: Dict[str, InputField] = {}
        self._output_fields: Dict[str, OutputField] = {}
        self._instruction = self.__class__.__doc__ or ""

        # Extract fields from class attributes
        for name, value in self.__class__.__dict__.items():
            if isinstance(value, InputField):
                value.name = name
                self._input_fields[name] = value
            elif isinstance(value, OutputField):
                value.name = name
                self._output_fields[name] = value

    @property
    def instruction(self) -> str:
        return self._instruction

    @property
    def input_fields(self) -> Dict[str, InputField]:
        return self._input_fields

    @property
    def output_fields(self) -> Dict[str, OutputField]:
        return self._output_fields

    def validate_input(self, inputs: Dict[str, Any]) -> bool:
        """Validate input dictionary against input fields."""
        for name, field in self._input_fields.items():
            if field.required and name not in inputs:
                raise ValueError(f"Required input field '{name}' missing")
            if name in inputs and not field.validate(inputs[name]):
                raise TypeError(f"Input field '{name}' has wrong type")
        return True

    def validate_output(self, outputs: Dict[str, Any]) -> bool:
        """Validate output dictionary against output fields."""
        for name, field in self._output_fields.items():
            if field.required and name not in outputs:
                raise ValueError(f"Required output field '{name}' missing")
            if name in outputs and not field.validate(outputs[name]):
                raise TypeError(f"Output field '{name}' has wrong type")
        return True

    def format_prompt(self, inputs: Dict[str, Any]) -> str:
        """Generate prompt string from inputs."""
        self.validate_input(inputs)

        prompt_parts = [self.instruction, ""]

        for name, field in self._input_fields.items():
            value = inputs.get(name, field.default)
            prompt_parts.append(f"{field.description or name}: {value}")

        prompt_parts.append("")
        for name, field in self._output_fields.items():
            prompt_parts.append(f"{field.description or name}:")

        return "\n".join(prompt_parts)

    def parse_output(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into output fields."""
        # Simple parsing (can be overridden for complex cases)
        outputs = {}
        current_field = None
        current_value = []

        for line in response.split("\n"):
            line = line.strip()

            # Check if line starts a new field
            for name, field in self._output_fields.items():
                prefix = f"{field.description or name}:"
                if line.startswith(prefix):
                    # Save previous field
                    if current_field:
                        outputs[current_field] = "\n".join(current_value).strip()
                    # Start new field
                    current_field = name
                    current_value = [line[len(prefix):].strip()]
                    break
            else:
                # Continue current field
                if current_field:
                    current_value.append(line)

        # Save last field
        if current_field:
            outputs[current_field] = "\n".join(current_value).strip()

        self.validate_output(outputs)
        return outputs


# Example signatures
class ChainOfThoughtSignature(Signature):
    """Answer questions with step-by-step reasoning."""

    question = InputField(str, "Question")
    reasoning = OutputField(str, "Step-by-step reasoning")
    answer = OutputField(str, "Final answer")


class RAGSignature(Signature):
    """Answer questions using retrieved context."""

    question = InputField(str, "Question")
    context = InputField(List[str], "Retrieved passages")
    answer = OutputField(str, "Answer based on context")
    citations = OutputField(List[str], "Source citations")
```

### 4.2 Modules (Composable Units)

**File:** `meta_prompting_framework/prompts/module.py`

```python
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from .signature import Signature
from ..categorical.monad import RMPMonad
from ..core.llm_clients.base import BaseLLMClient

class Module(ABC):
    """
    Base class for composable prompt modules.

    A module takes a signature and implements a prompting strategy.
    Similar to DSPy's Module.
    """

    def __init__(
        self,
        signature: type[Signature],
        llm_client: Optional[BaseLLMClient] = None
    ):
        self.signature = signature()
        self.llm_client = llm_client

    @abstractmethod
    def forward(self, **inputs) -> Dict[str, Any]:
        """Execute the module with given inputs."""
        pass

    def __call__(self, **inputs) -> Dict[str, Any]:
        """Alias for forward."""
        return self.forward(**inputs)

    def compose(self, other: 'Module') -> 'Module':
        """Compose two modules sequentially."""
        return SequentialModule([self, other])


class Predict(Module):
    """
    Basic prediction module.

    Generates prompt from signature and gets LLM response.
    """

    def forward(self, **inputs) -> Dict[str, Any]:
        self.signature.validate_input(inputs)

        # Generate prompt
        prompt = self.signature.format_prompt(inputs)

        # Get LLM response
        response = self.llm_client.complete(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        # Parse outputs
        outputs = self.signature.parse_output(response.content)
        return outputs


class ChainOfThought(Module):
    """
    Chain-of-thought reasoning module.

    Augments signature with explicit reasoning step.
    """

    def forward(self, **inputs) -> Dict[str, Any]:
        self.signature.validate_input(inputs)

        # Add reasoning instruction to prompt
        enhanced_prompt = f"""{self.signature.format_prompt(inputs)}

Please think step-by-step:
1. Break down the problem
2. Solve each part
3. Combine for final answer
"""

        response = self.llm_client.complete(
            messages=[{"role": "user", "content": enhanced_prompt}],
            temperature=0.7
        )

        outputs = self.signature.parse_output(response.content)
        return outputs


class ReAct(Module):
    """
    ReAct (Reasoning + Acting) module.

    Interleaves reasoning and tool use.
    """

    def __init__(
        self,
        signature: type[Signature],
        llm_client: Optional[BaseLLMClient] = None,
        tools: Optional[Dict[str, callable]] = None,
        max_iterations: int = 5
    ):
        super().__init__(signature, llm_client)
        self.tools = tools or {}
        self.max_iterations = max_iterations

    def forward(self, **inputs) -> Dict[str, Any]:
        self.signature.validate_input(inputs)

        thought_action_history = []

        for i in range(self.max_iterations):
            # Generate thought and action
            prompt = self._build_react_prompt(inputs, thought_action_history)
            response = self.llm_client.complete(
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse thought/action/observation
            thought, action, action_input = self._parse_react_response(response.content)
            thought_action_history.append({
                "thought": thought,
                "action": action,
                "action_input": action_input
            })

            # Execute action
            if action == "Finish":
                return self.signature.parse_output(action_input)
            elif action in self.tools:
                observation = self.tools[action](action_input)
                thought_action_history[-1]["observation"] = observation
            else:
                thought_action_history[-1]["observation"] = f"Error: Unknown action {action}"

        # Fallback if max iterations reached
        return {"answer": "Could not determine answer in time"}

    def _build_react_prompt(self, inputs, history):
        """Build ReAct-style prompt with history."""
        prompt_parts = [
            "You can use the following tools:",
            *[f"- {name}: {func.__doc__}" for name, func in self.tools.items()],
            "",
            self.signature.format_prompt(inputs),
            "",
            "Use this format:",
            "Thought: [your reasoning]",
            "Action: [tool name or 'Finish']",
            "Action Input: [input to tool or final answer]",
            "",
        ]

        for entry in history:
            prompt_parts.append(f"Thought: {entry['thought']}")
            prompt_parts.append(f"Action: {entry['action']}")
            prompt_parts.append(f"Action Input: {entry['action_input']}")
            if 'observation' in entry:
                prompt_parts.append(f"Observation: {entry['observation']}")

        return "\n".join(prompt_parts)

    def _parse_react_response(self, response: str):
        """Parse thought/action/action_input from response."""
        thought, action, action_input = "", "", ""

        for line in response.split("\n"):
            if line.startswith("Thought:"):
                thought = line[8:].strip()
            elif line.startswith("Action:"):
                action = line[7:].strip()
            elif line.startswith("Action Input:"):
                action_input = line[13:].strip()

        return thought, action, action_input


class SequentialModule(Module):
    """
    Compose modules sequentially.

    Output of module[i] becomes input to module[i+1].
    """

    def __init__(self, modules: List[Module]):
        self.modules = modules
        # Use signature of last module
        super().__init__(modules[-1].signature.__class__)

    def forward(self, **inputs) -> Dict[str, Any]:
        current_inputs = inputs

        for module in self.modules:
            current_inputs = module.forward(**current_inputs)

        return current_inputs
```

### 4.3 Constraints (LMQL-like DSL)

**File:** `meta_prompting_framework/prompts/constraint.py`

```python
from typing import Any, Callable, List, Optional
from abc import ABC, abstractmethod
from enum import Enum
import re

class ConstraintType(Enum):
    """Types of constraints."""
    VALUE_IN = "in"           # Value must be in set
    TYPE_CHECK = "type"       # Type validation
    REGEX = "regex"           # Regex match
    LENGTH = "length"         # Length constraint
    RANGE = "range"           # Numeric range
    CUSTOM = "custom"         # Custom function

class Constraint(ABC):
    """
    Base class for output constraints.

    Constraints are checked during or after generation.
    """

    @abstractmethod
    def check(self, value: Any) -> bool:
        """Check if value satisfies constraint."""
        pass

    @abstractmethod
    def error_message(self, value: Any) -> str:
        """Generate error message for constraint violation."""
        pass


class InConstraint(Constraint):
    """Value must be in a set of allowed values."""

    def __init__(self, allowed_values: List[Any]):
        self.allowed_values = allowed_values

    def check(self, value: Any) -> bool:
        return value in self.allowed_values

    def error_message(self, value: Any) -> str:
        return f"Value '{value}' not in {self.allowed_values}"


class TypeConstraint(Constraint):
    """Value must be of specific type."""

    def __init__(self, expected_type: type):
        self.expected_type = expected_type

    def check(self, value: Any) -> bool:
        if self.expected_type == int:
            try:
                int(value)
                return True
            except:
                return False
        elif self.expected_type == float:
            try:
                float(value)
                return True
            except:
                return False
        return isinstance(value, self.expected_type)

    def error_message(self, value: Any) -> str:
        return f"Expected type {self.expected_type}, got {type(value)}"


class RegexConstraint(Constraint):
    """Value must match regex pattern."""

    def __init__(self, pattern: str):
        self.pattern = pattern
        self.regex = re.compile(pattern)

    def check(self, value: Any) -> bool:
        return bool(self.regex.match(str(value)))

    def error_message(self, value: Any) -> str:
        return f"Value '{value}' does not match pattern '{self.pattern}'"


class LengthConstraint(Constraint):
    """Length of value must satisfy bounds."""

    def __init__(self, min_len: Optional[int] = None, max_len: Optional[int] = None):
        self.min_len = min_len
        self.max_len = max_len

    def check(self, value: Any) -> bool:
        length = len(str(value))
        if self.min_len and length < self.min_len:
            return False
        if self.max_len and length > self.max_len:
            return False
        return True

    def error_message(self, value: Any) -> str:
        return f"Length {len(str(value))} violates bounds [{self.min_len}, {self.max_len}]"


class RangeConstraint(Constraint):
    """Numeric value must be in range."""

    def __init__(self, min_val: Optional[float] = None, max_val: Optional[float] = None):
        self.min_val = min_val
        self.max_val = max_val

    def check(self, value: Any) -> bool:
        try:
            num = float(value)
            if self.min_val and num < self.min_val:
                return False
            if self.max_val and num > self.max_val:
                return False
            return True
        except:
            return False

    def error_message(self, value: Any) -> str:
        return f"Value {value} not in range [{self.min_val}, {self.max_val}]"


class ConstraintChecker:
    """
    Applies constraints to outputs.

    Usage:
        checker = ConstraintChecker()
        checker.add_constraint("sentiment", InConstraint(["positive", "negative", "neutral"]))
        checker.add_constraint("confidence", RangeConstraint(0.0, 1.0))

        result = {"sentiment": "positive", "confidence": 0.9}
        checker.validate(result)  # Returns True
    """

    def __init__(self):
        self.constraints: Dict[str, List[Constraint]] = {}

    def add_constraint(self, field_name: str, constraint: Constraint):
        """Add constraint for a field."""
        if field_name not in self.constraints:
            self.constraints[field_name] = []
        self.constraints[field_name].append(constraint)

    def validate(self, outputs: Dict[str, Any]) -> bool:
        """Validate all outputs against constraints."""
        for field_name, constraints in self.constraints.items():
            if field_name not in outputs:
                continue

            value = outputs[field_name]
            for constraint in constraints:
                if not constraint.check(value):
                    raise ValueError(
                        f"Constraint violation for field '{field_name}': "
                        f"{constraint.error_message(value)}"
                    )
        return True

    def filter_violations(self, outputs: Dict[str, Any]) -> Dict[str, List[str]]:
        """Return all constraint violations without raising."""
        violations = {}

        for field_name, constraints in self.constraints.items():
            if field_name not in outputs:
                continue

            value = outputs[field_name]
            for constraint in constraints:
                if not constraint.check(value):
                    if field_name not in violations:
                        violations[field_name] = []
                    violations[field_name].append(constraint.error_message(value))

        return violations
```

---

## 5. Layer 4: Optimizers & Self-Improvement

### 5.1 Optimizer Interface

**File:** `meta_prompting_framework/optimizers/base.py`

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable
from ..prompts.module import Module
from ..prompts.signature import Signature

class Optimizer(ABC):
    """
    Base class for prompt optimizers.

    An optimizer takes a module and training data, and produces
    an optimized version of the module.
    """

    @abstractmethod
    def compile(
        self,
        module: Module,
        trainset: List[Dict[str, Any]],
        metric: Optional[Callable[[Dict, Dict], float]] = None
    ) -> Module:
        """
        Optimize the module using training data.

        Args:
            module: Module to optimize
            trainset: List of {input: ..., output: ...} examples
            metric: Function to score output quality (higher = better)

        Returns:
            Optimized module
        """
        pass


class Metric(ABC):
    """Base class for evaluation metrics."""

    @abstractmethod
    def score(self, prediction: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
        """Return score in [0, 1]."""
        pass


class ExactMatch(Metric):
    """Exact string match metric."""

    def score(self, prediction: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
        return 1.0 if prediction == ground_truth else 0.0


class F1Score(Metric):
    """Token-level F1 score."""

    def score(self, prediction: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
        # Tokenize both
        pred_tokens = set(str(prediction).lower().split())
        gt_tokens = set(str(ground_truth).lower().split())

        # Calculate F1
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            return 0.0

        intersection = pred_tokens & gt_tokens
        precision = len(intersection) / len(pred_tokens)
        recall = len(intersection) / len(gt_tokens)

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return f1
```

### 5.2 RMP Optimizer (Recursive Meta-Prompting)

**File:** `meta_prompting_framework/optimizers/rmp.py`

```python
from typing import List, Dict, Any, Optional, Callable
from .base import Optimizer, Metric
from ..prompts.module import Module
from ..categorical.monad import RMPMonad
from ..core.llm_clients.base import BaseLLMClient

class RMPOptimizer(Optimizer):
    """
    Recursive Meta-Prompting Optimizer.

    Implements Zhang et al.'s RMP monad to recursively improve
    the meta-prompt itself, not just the output.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        max_iterations: int = 5,
        quality_threshold: float = 0.9
    ):
        self.llm_client = llm_client
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold

    def compile(
        self,
        module: Module,
        trainset: List[Dict[str, Any]],
        metric: Optional[Callable[[Dict, Dict], float]] = None
    ) -> Module:
        """
        Optimize module by evolving its meta-prompt.

        1. Extract current meta-prompt from module
        2. Evaluate on trainset
        3. Use RMP monad to improve meta-prompt
        4. Repeat until convergence
        """
        metric = metric or F1Score()

        # Initial meta-prompt (extracted from module's signature)
        initial_prompt = self._extract_meta_prompt(module)

        # Wrap in RMP monad
        rmp = RMPMonad.unit(initial_prompt)

        for iteration in range(self.max_iterations):
            # Evaluate current module on trainset
            current_quality = self._evaluate(module, trainset, metric)

            # Check early stopping
            if current_quality >= self.quality_threshold:
                break

            # Improve meta-prompt using RMP monad
            rmp = rmp.flat_map(
                lambda prompt: self._improve_meta_prompt(
                    prompt,
                    module,
                    trainset,
                    metric,
                    current_quality
                )
            )

            # Update module with improved meta-prompt
            module = self._update_module(module, rmp._value)

        return module

    def _extract_meta_prompt(self, module: Module) -> str:
        """Extract meta-prompt from module's signature."""
        return module.signature.instruction

    def _evaluate(
        self,
        module: Module,
        trainset: List[Dict[str, Any]],
        metric: Metric
    ) -> float:
        """Evaluate module on trainset using metric."""
        scores = []

        for example in trainset:
            inputs = example.get("input", {})
            ground_truth = example.get("output", {})

            try:
                prediction = module.forward(**inputs)
                score = metric.score(prediction, ground_truth)
                scores.append(score)
            except Exception as e:
                # Failed predictions count as 0
                scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.0

    def _improve_meta_prompt(
        self,
        current_prompt: str,
        module: Module,
        trainset: List[Dict[str, Any]],
        metric: Metric,
        current_quality: float
    ) -> RMPMonad:
        """
        Use LLM to generate improved meta-prompt.

        This is the core RMP operation: the meta-prompt improves itself.
        """
        # Sample failures from trainset
        failures = self._get_failure_examples(module, trainset, metric, max_examples=3)

        # Ask LLM to improve meta-prompt
        improvement_prompt = f"""Current meta-prompt:
{current_prompt}

Current quality: {current_quality:.2%}

Failure examples:
{self._format_failures(failures)}

Generate an improved meta-prompt that addresses these failures.
Focus on:
1. Clearer instructions
2. Better reasoning scaffolding
3. Explicit error handling

Improved meta-prompt:"""

        response = self.llm_client.complete(
            messages=[{"role": "user", "content": improvement_prompt}],
            temperature=0.7
        )

        improved_prompt = response.content.strip()

        # Evaluate improvement
        temp_module = self._update_module(module, improved_prompt)
        new_quality = self._evaluate(temp_module, trainset, metric)

        return RMPMonad(
            prompt=improved_prompt,
            quality=new_quality,
            iteration=1,
            context={"previous_quality": current_quality}
        )

    def _get_failure_examples(
        self,
        module: Module,
        trainset: List[Dict[str, Any]],
        metric: Metric,
        max_examples: int = 3
    ) -> List[Dict[str, Any]]:
        """Get examples where module performed poorly."""
        failures = []

        for example in trainset:
            inputs = example.get("input", {})
            ground_truth = example.get("output", {})

            try:
                prediction = module.forward(**inputs)
                score = metric.score(prediction, ground_truth)

                if score < 0.5:  # Failed
                    failures.append({
                        "input": inputs,
                        "expected": ground_truth,
                        "actual": prediction,
                        "score": score
                    })
            except Exception as e:
                failures.append({
                    "input": inputs,
                    "expected": ground_truth,
                    "error": str(e)
                })

            if len(failures) >= max_examples:
                break

        return failures

    def _format_failures(self, failures: List[Dict[str, Any]]) -> str:
        """Format failure examples for LLM."""
        formatted = []

        for i, failure in enumerate(failures, 1):
            formatted.append(f"Example {i}:")
            formatted.append(f"  Input: {failure.get('input')}")
            formatted.append(f"  Expected: {failure.get('expected')}")
            if 'actual' in failure:
                formatted.append(f"  Actual: {failure.get('actual')}")
                formatted.append(f"  Score: {failure.get('score'):.2f}")
            if 'error' in failure:
                formatted.append(f"  Error: {failure.get('error')}")
            formatted.append("")

        return "\n".join(formatted)

    def _update_module(self, module: Module, new_instruction: str) -> Module:
        """Create new module with updated instruction."""
        # Create new signature with updated instruction
        new_signature_class = type(
            module.signature.__class__.__name__,
            (module.signature.__class__,),
            {"__doc__": new_instruction}
        )

        # Create new module instance
        new_module = type(module)(new_signature_class, module.llm_client)
        return new_module
```

### 5.3 Bootstrap Few-Shot Optimizer

**File:** `meta_prompting_framework/optimizers/bootstrap.py`

```python
from typing import List, Dict, Any, Optional, Callable
from .base import Optimizer, Metric
from ..prompts.module import Module
import random

class BootstrapFewShot(Optimizer):
    """
    Bootstrap few-shot examples from training data.

    Similar to DSPy's BootstrapFewShot optimizer.
    Automatically selects best demonstrations.
    """

    def __init__(
        self,
        llm_client,
        num_examples: int = 3,
        num_candidates: int = 10
    ):
        self.llm_client = llm_client
        self.num_examples = num_examples
        self.num_candidates = num_candidates

    def compile(
        self,
        module: Module,
        trainset: List[Dict[str, Any]],
        metric: Optional[Callable[[Dict, Dict], float]] = None
    ) -> Module:
        """
        Select best few-shot examples via bootstrapping.

        1. Generate many candidate outputs using module
        2. Select best candidates based on metric
        3. Prepend selected examples to prompts
        """
        metric = metric or F1Score()

        # Generate candidate examples
        candidates = []
        for example in trainset[:self.num_candidates]:
            inputs = example.get("input", {})
            ground_truth = example.get("output", {})

            try:
                prediction = module.forward(**inputs)
                score = metric.score(prediction, ground_truth)

                candidates.append({
                    "input": inputs,
                    "output": prediction,
                    "score": score
                })
            except:
                continue

        # Select top-k examples
        candidates.sort(key=lambda x: x["score"], reverse=True)
        selected = candidates[:self.num_examples]

        # Create few-shot module
        return FewShotModule(module, selected)


class FewShotModule(Module):
    """
    Module augmented with few-shot examples.

    Prepends examples to every prompt.
    """

    def __init__(self, base_module: Module, examples: List[Dict[str, Any]]):
        super().__init__(base_module.signature.__class__, base_module.llm_client)
        self.base_module = base_module
        self.examples = examples

    def forward(self, **inputs) -> Dict[str, Any]:
        # Build few-shot prompt
        few_shot_prompt = self._build_few_shot_prompt(inputs)

        # Use base LLM client directly
        response = self.llm_client.complete(
            messages=[{"role": "user", "content": few_shot_prompt}],
            temperature=0.7
        )

        # Parse using signature
        outputs = self.signature.parse_output(response.content)
        return outputs

    def _build_few_shot_prompt(self, inputs: Dict[str, Any]) -> str:
        """Build prompt with few-shot examples."""
        parts = [self.signature.instruction, ""]

        # Add examples
        for i, example in enumerate(self.examples, 1):
            parts.append(f"Example {i}:")
            parts.append(self.signature.format_prompt(example["input"]))
            parts.append(str(example["output"]))
            parts.append("")

        # Add current query
        parts.append("Now solve this:")
        parts.append(self.signature.format_prompt(inputs))

        return "\n".join(parts)
```

---

## 6. Implementation Roadmap

### Phase 1: Core Categorical Abstractions (Week 1-2)

**Deliverables:**
- [ ] `categorical/functor.py` - Functor base class + laws
- [ ] `categorical/monad.py` - Monad + RMPMonad
- [ ] `categorical/natural_transformation.py` - Strategy equivalence
- [ ] `categorical/enriched.py` - Quality-enriched categories
- [ ] `categorical/polynomial.py` - Tool composition

**Tests:**
- [ ] Property-based testing for functor laws
- [ ] Monad law verification
- [ ] Naturality square commutativity

### Phase 2: Prompt System (Week 3-4)

**Deliverables:**
- [ ] `prompts/signature.py` - Type-safe signatures
- [ ] `prompts/module.py` - Predict, ChainOfThought, ReAct
- [ ] `prompts/constraint.py` - Constraint DSL
- [ ] Integration with existing `core/`

**Tests:**
- [ ] Signature validation
- [ ] Module composition
- [ ] Constraint checking

### Phase 3: Optimizers (Week 5-6)

**Deliverables:**
- [ ] `optimizers/base.py` - Optimizer interface
- [ ] `optimizers/rmp.py` - RMP monad optimizer
- [ ] `optimizers/bootstrap.py` - Few-shot bootstrapping
- [ ] Metrics (ExactMatch, F1, etc.)

**Tests:**
- [ ] RMP convergence on toy tasks
- [ ] Bootstrap selection quality
- [ ] Metric correctness

### Phase 4: Applications & Benchmarks (Week 7-8)

**Deliverables:**
- [ ] `applications/benchmarks/gsm8k.py`
- [ ] `applications/benchmarks/math.py`
- [ ] Comprehensive examples
- [ ] Performance comparison vs. current framework

**Tests:**
- [ ] GSM8K accuracy ≥ 80%
- [ ] MATH accuracy ≥ 40%

### Phase 5: Production Features (Week 9-10)

**Deliverables:**
- [ ] Async/concurrent execution
- [ ] Caching layer (Redis + in-memory)
- [ ] Observability (metrics, tracing)
- [ ] Documentation + tutorials

---

## 7. Success Criteria

### Correctness
- ✅ All categorical laws verified (functor, monad, naturality)
- ✅ Type safety: No runtime type errors in composition
- ✅ Constraint satisfaction guaranteed

### Performance
- ✅ GSM8K ≥ 80% (matching DSPy)
- ✅ MATH ≥ 40% (matching Zhang et al.)
- ✅ 2x faster via caching + async

### Developer Experience
- ✅ 10-line programs for complex tasks
- ✅ Clear error messages
- ✅ Composable modules

### Research Contribution
- ✅ First enriched category implementation
- ✅ First polynomial functor tool library
- ✅ First proven RMP monad

---

## 8. Example Usage (Post-Implementation)

```python
from meta_prompting_framework.prompts.signature import Signature, InputField, OutputField
from meta_prompting_framework.prompts.module import ChainOfThought
from meta_prompting_framework.optimizers.rmp import RMPOptimizer
from meta_prompting_framework.categorical.monad import RMPMonad

# Define signature
class MathSignature(Signature):
    """Solve math problems with reasoning."""
    question = InputField(str, "Math question")
    reasoning = OutputField(str, "Step-by-step solution")
    answer = OutputField(str, "Final numeric answer")

# Create module
math_module = ChainOfThought(MathSignature, llm_client=claude)

# Optimize with RMP
optimizer = RMPOptimizer(llm_client=claude, max_iterations=5)
optimized_module = optimizer.compile(
    math_module,
    trainset=gsm8k_train,
    metric=exact_match
)

# Use optimized module
result = optimized_module(question="What is 25% of 80?")
print(result["answer"])  # "20"
```

---

## 9. Migration Path from Current Framework

### Backward Compatibility

The new framework **extends** the current one without breaking changes:

```python
# Old code still works
from meta_prompting_engine.core import MetaPromptingEngine

engine = MetaPromptingEngine(skill="math", llm_client=claude)
result = engine.execute_with_meta_prompting(
    task="Solve x^2 + 5x + 6 = 0",
    max_iterations=3
)

# New code uses enhanced features
from meta_prompting_framework.prompts.signature import Signature, InputField, OutputField
from meta_prompting_framework.prompts.module import ChainOfThought

class EquationSignature(Signature):
    """Solve quadratic equations."""
    equation = InputField(str)
    solution = OutputField(List[float])

module = ChainOfThought(EquationSignature, llm_client=claude)
result = module(equation="x^2 + 5x + 6 = 0")
```

### Gradual Migration

1. **Week 1-2:** Add categorical abstractions (no breaking changes)
2. **Week 3-4:** Add prompts/ modules (optional, current API still works)
3. **Week 5-6:** Add optimizers (optional)
4. **Week 7+:** Gradually migrate examples to new API

---

## 10. Conclusion

This design provides:

1. **Rigorous categorical foundations** (functors, monads, enriched categories, polynomial functors)
2. **Type-safe compositional prompts** (DSPy-like signatures)
3. **Constraint-based generation** (LMQL-like)
4. **True recursive self-improvement** (RMP monad)
5. **Production-ready features** (async, caching, observability)

**Next steps:** Begin Phase 1 implementation of categorical abstractions.
