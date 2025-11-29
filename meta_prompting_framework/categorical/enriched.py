"""
Enriched Category abstraction for meta-prompting framework.

Implements de Wynter et al.'s enriched categories for handling
quality metrics and stochasticity in prompting.
"""

from typing import TypeVar, Generic, Protocol, Dict, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

V = TypeVar('V')  # Enrichment base (e.g., [0,1], R+, probability distributions)
A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')


class MonoidalCategory(Protocol[V]):
    """
    A monoidal category (V, ⊗, I) for enrichment.

    A monoidal category has:
    - A tensor product ⊗: V × V → V
    - A unit object I
    - Associativity and unit laws

    Examples:
    - ([0,1], max, 0): For quality metrics (optimistic composition)
    - ([0,1], min, 1): For quality metrics (pessimistic composition)
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
    """
    [0,1] monoidal category for quality enrichment.

    Uses max as tensor product (optimistic composition):
    - When composing prompts, take the best quality
    - Represents "at least one path succeeded"

    This models de Wynter et al.'s enriched category for
    handling LLM stochasticity and quality.
    """

    value: float  # 0.0 to 1.0

    def __post_init__(self):
        assert 0.0 <= self.value <= 1.0, f"Quality must be in [0,1], got {self.value}"

    def tensor(self, other: 'QualityMetric') -> 'QualityMetric':
        """
        Monoidal product: max (optimistic composition).

        When composing two prompts, the quality is the maximum
        of their qualities.
        """
        return QualityMetric(max(self.value, other.value))

    @staticmethod
    def unit() -> 'QualityMetric':
        """Monoidal unit: 0 (worst quality)"""
        return QualityMetric(0.0)

    def __mul__(self, other: 'QualityMetric') -> 'QualityMetric':
        """Infix operator for tensor"""
        return self.tensor(other)

    def __le__(self, other: 'QualityMetric') -> bool:
        """Quality ordering"""
        return self.value <= other.value

    def __repr__(self):
        return f"Quality({self.value:.2f})"


@dataclass
class CostMetric:
    """
    R+ monoidal category for cost enrichment.

    Uses + as tensor product:
    - When composing prompts, add their costs
    - Represents total API cost

    This is useful for optimizing cost alongside quality.
    """

    value: float  # Non-negative cost

    def __post_init__(self):
        assert self.value >= 0, f"Cost must be non-negative, got {self.value}"

    def tensor(self, other: 'CostMetric') -> 'CostMetric':
        """Monoidal product: + (additive costs)"""
        return CostMetric(self.value + other.value)

    @staticmethod
    def unit() -> 'CostMetric':
        """Monoidal unit: 0 (no cost)"""
        return CostMetric(0.0)

    def __add__(self, other: 'CostMetric') -> 'CostMetric':
        """Infix operator for tensor"""
        return self.tensor(other)

    def __repr__(self):
        return f"Cost(${self.value:.4f})"


class EnrichedCategory(Generic[V]):
    """
    A category enriched over monoidal category V.

    Instead of hom-sets hom(A,B), we have hom-objects hom(A,B) ∈ V.

    For prompts enriched over [0,1]:
    - Objects: Prompts (identified by strings)
    - hom(P, Q) ∈ [0,1]: Quality of transforming P to Q

    This allows us to model quality as a first-class compositional property,
    not just a post-hoc metric.
    """

    def __init__(self, monoidal_base: type[MonoidalCategory]):
        """
        Initialize enriched category.

        Args:
            monoidal_base: The monoidal category to enrich over
        """
        self.base = monoidal_base
        self._hom_objects: Dict[Tuple[Any, Any], V] = {}

    def hom(self, a: A, b: B) -> V:
        """
        Get hom-object from A to B.

        Args:
            a: Source object
            b: Target object

        Returns:
            Hom-object in V
        """
        key = (a, b)
        if key in self._hom_objects:
            return self._hom_objects[key]
        else:
            return self.base.unit()

    def set_hom(self, a: A, b: B, value: V):
        """
        Set hom-object from A to B.

        Args:
            a: Source object
            b: Target object
            value: Hom-object value
        """
        self._hom_objects[(a, b)] = value

    def compose(self, a: A, b: B, c: C) -> V:
        """
        Enriched composition:
        hom(B,C) ⊗ hom(A,B) → hom(A,C)

        For quality: quality(A→C) = max(quality(A→B), quality(B→C))
        For cost: cost(A→C) = cost(A→B) + cost(B→C)

        Args:
            a: Source object
            b: Intermediate object
            c: Target object

        Returns:
            Composed hom-object
        """
        ab = self.hom(a, b)
        bc = self.hom(b, c)
        ac = ab.tensor(bc)
        self.set_hom(a, c, ac)
        return ac

    def identity(self, a: A) -> V:
        """
        Identity hom-object: hom(A, A)

        For quality: highest quality (1.0)
        For cost: no cost (0.0)
        """
        return self.base.unit()


class QualityEnrichedPrompts(EnrichedCategory[QualityMetric]):
    """
    Prompts enriched over [0,1] quality metric.

    This is the main application of enriched categories for meta-prompting:
    - Objects are prompts (strings)
    - hom(P, Q) is the quality of transforming P to Q
    - Composition uses max (best quality wins)

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
        """
        Add a prompt refinement with quality metric.

        Args:
            original: Original prompt
            refined: Refined prompt
            quality_improvement: Quality score [0, 1]
        """
        self.set_hom(original, refined, QualityMetric(quality_improvement))

    def get_quality(self, source: str, target: str) -> float:
        """
        Get quality of transformation from source to target.

        Args:
            source: Source prompt
            target: Target prompt

        Returns:
            Quality score [0, 1]
        """
        return self.hom(source, target).value

    def find_best_path(
        self,
        start: str,
        end: str,
        intermediates: list[str]
    ) -> Tuple[list[str], float]:
        """
        Find best path through intermediates from start to end.

        Uses enriched composition to calculate quality of each path,
        then selects the best.

        Args:
            start: Starting prompt
            end: Target prompt
            intermediates: Possible intermediate prompts

        Returns:
            Tuple of (best_path, quality)
        """
        best_path = [start, end]
        best_quality = self.get_quality(start, end)

        # Try all paths through intermediates
        for intermediate in intermediates:
            # Compose qualities
            start_to_mid = self.hom(start, intermediate)
            mid_to_end = self.hom(intermediate, end)
            total_quality = start_to_mid.tensor(mid_to_end)

            if total_quality.value > best_quality:
                best_quality = total_quality.value
                best_path = [start, intermediate, end]

        return best_path, best_quality


class CostEnrichedPrompts(EnrichedCategory[CostMetric]):
    """
    Prompts enriched over R+ cost metric.

    Useful for tracking API costs across prompt transformations.

    Usage:
        prompts = CostEnrichedPrompts()
        prompts.set_hom("prompt1", "prompt2", CostMetric(0.001))  # $0.001
        prompts.set_hom("prompt2", "prompt3", CostMetric(0.002))  # $0.002

        # Compose: cost(prompt1 → prompt3) = 0.001 + 0.002 = 0.003
        total_cost = prompts.compose("prompt1", "prompt2", "prompt3")
    """

    def __init__(self):
        super().__init__(CostMetric)

    def add_transformation_cost(
        self,
        source: str,
        target: str,
        cost: float
    ):
        """
        Add cost for transforming one prompt to another.

        Args:
            source: Source prompt
            target: Target prompt
            cost: API cost in dollars
        """
        self.set_hom(source, target, CostMetric(cost))

    def get_cost(self, source: str, target: str) -> float:
        """
        Get cost of transformation.

        Args:
            source: Source prompt
            target: Target prompt

        Returns:
            Cost in dollars
        """
        return self.hom(source, target).value


def test_enriched_categories():
    """
    Test enriched category functionality.

    Returns:
        Dictionary of test results
    """
    results = {}

    # Test QualityEnrichedPrompts
    quality_prompts = QualityEnrichedPrompts()
    quality_prompts.add_prompt_refinement("basic", "improved", 0.7)
    quality_prompts.add_prompt_refinement("improved", "optimized", 0.9)

    # Compose qualities (should use max)
    composed_quality = quality_prompts.compose("basic", "improved", "optimized")
    results["quality_composition_uses_max"] = composed_quality.value == 0.9

    # Test path finding
    quality_prompts.add_prompt_refinement("basic", "alternative", 0.6)
    quality_prompts.add_prompt_refinement("alternative", "optimized", 0.95)

    best_path, best_quality = quality_prompts.find_best_path(
        "basic", "optimized", ["improved", "alternative"]
    )
    results["best_path_selection"] = best_quality == 0.95

    # Test CostEnrichedPrompts
    cost_prompts = CostEnrichedPrompts()
    cost_prompts.add_transformation_cost("p1", "p2", 0.001)
    cost_prompts.add_transformation_cost("p2", "p3", 0.002)

    # Compose costs (should use +)
    composed_cost = cost_prompts.compose("p1", "p2", "p3")
    results["cost_composition_uses_sum"] = abs(composed_cost.value - 0.003) < 1e-6

    return results


if __name__ == "__main__":
    results = test_enriched_categories()

    print("Enriched Category Test Results:")
    print("=" * 50)
    for test_name, passed in results.items():
        print(f"  {test_name}: {'✓' if passed else '✗'}")

    print("\nAll tests passed!" if all(results.values()) else "\nSome tests failed!")

    # Demonstration
    print("\n" + "=" * 50)
    print("Demonstration: Quality-Enriched Prompt Evolution")
    print("=" * 50)

    prompts = QualityEnrichedPrompts()
    prompts.add_prompt_refinement("Solve the equation", "Solve x^2+5x+6=0 step-by-step", 0.75)
    prompts.add_prompt_refinement("Solve x^2+5x+6=0 step-by-step", "Solve x^2+5x+6=0 using quadratic formula with full reasoning", 0.92)

    print(f"\nInitial prompt: 'Solve the equation'")
    print(f"Quality after first refinement: {prompts.get_quality('Solve the equation', 'Solve x^2+5x+6=0 step-by-step')}")
    print(f"Quality after second refinement: {prompts.get_quality('Solve x^2+5x+6=0 step-by-step', 'Solve x^2+5x+6=0 using quadratic formula with full reasoning')}")

    final_quality = prompts.compose(
        "Solve the equation",
        "Solve x^2+5x+6=0 step-by-step",
        "Solve x^2+5x+6=0 using quadratic formula with full reasoning"
    )
    print(f"Composed quality (max): {final_quality.value}")
