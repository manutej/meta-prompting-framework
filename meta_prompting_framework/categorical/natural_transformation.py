"""
Natural Transformation abstraction for meta-prompting framework.

Implements mappings between functors that preserve categorical structure.
"""

from typing import TypeVar, Generic, Callable, Any
from .functor import Functor

A = TypeVar('A')
B = TypeVar('B')


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

    This ensures that the transformation is "uniform" across all objects
    and doesn't depend on the specific morphisms chosen.
    """

    def __init__(
        self,
        source_functor_class: type[Functor],
        target_functor_class: type[Functor],
        component: Callable[[Functor[A]], Functor[A]],
        name: str = "η"
    ):
        """
        Initialize a natural transformation.

        Args:
            source_functor_class: Source functor class F
            target_functor_class: Target functor class G
            component: The transformation function η_A for each object A
            name: Name of the transformation (for display)
        """
        self.source_class = source_functor_class
        self.target_class = target_functor_class
        self.component = component
        self.name = name

    def apply(self, fa: Functor[A]) -> Functor[A]:
        """
        Apply the natural transformation at component A.

        Args:
            fa: Functor F(A)

        Returns:
            Functor G(A)
        """
        return self.component(fa)

    def __call__(self, fa: Functor[A]) -> Functor[A]:
        """Alias for apply."""
        return self.apply(fa)

    def __repr__(self):
        return f"{self.name}: {self.source_class.__name__} ⇒ {self.target_class.__name__}"


class StrategyTransformation(NaturalTransformation):
    """
    Natural transformation between prompting strategies.

    Example: ChainOfThought ⇒ ReAct

    This models the fact that different prompting strategies
    are related by natural transformations - you can convert
    from one strategy to another in a systematic way.
    """

    def __init__(self, from_strategy: str, to_strategy: str):
        """
        Create transformation between strategies.

        Args:
            from_strategy: Source strategy name
            to_strategy: Target strategy name
        """
        from .functor import MetaPromptFunctor

        def transform(prompt_functor: MetaPromptFunctor) -> MetaPromptFunctor:
            """Transform prompt from one strategy to another."""
            # Keep the same value but change strategy
            new_functor = MetaPromptFunctor(
                prompt_functor._value,
                strategy=to_strategy
            )
            return new_functor

        super().__init__(
            source_functor_class=MetaPromptFunctor,
            target_functor_class=MetaPromptFunctor,
            component=transform,
            name=f"{from_strategy}→{to_strategy}"
        )
        self.from_strategy = from_strategy
        self.to_strategy = to_strategy


class IdentityTransformation(NaturalTransformation):
    """
    Identity natural transformation: id_F: F ⇒ F

    Maps every F(A) to itself.
    """

    def __init__(self, functor_class: type[Functor]):
        super().__init__(
            source_functor_class=functor_class,
            target_functor_class=functor_class,
            component=lambda fa: fa,
            name="id"
        )


class ComposedTransformation(NaturalTransformation):
    """
    Vertical composition of natural transformations.

    If η: F ⇒ G and θ: G ⇒ H, then (θ ∘ η): F ⇒ H
    """

    def __init__(self, eta: NaturalTransformation, theta: NaturalTransformation):
        """
        Compose two natural transformations vertically.

        Args:
            eta: First transformation F ⇒ G
            theta: Second transformation G ⇒ H
        """
        def composed_component(fa: Functor) -> Functor:
            ga = eta.apply(fa)
            ha = theta.apply(ga)
            return ha

        super().__init__(
            source_functor_class=eta.source_class,
            target_functor_class=theta.target_class,
            component=composed_component,
            name=f"{theta.name}∘{eta.name}"
        )
        self.eta = eta
        self.theta = theta


def verify_naturality(
    eta: NaturalTransformation,
    F: Functor[A],
    f: Callable[[A], B]
) -> bool:
    """
    Verify that the naturality square commutes:

        F(A) --η_A--> G(A)
         |             |
       F(f)          G(f)
         |             |
         ↓             ↓
        F(B) --η_B--> G(B)

    Commutativity: η_B ∘ F(f) = G(f) ∘ η_A

    Args:
        eta: Natural transformation to verify
        F: Source functor at object A
        f: Morphism from A to B

    Returns:
        True if naturality holds
    """
    # Left path: F(f) then η_B
    F_of_f = F.fmap(f)  # F(B)
    left_path = eta.apply(F_of_f)  # G(B)

    # Right path: η_A then G(f)
    eta_of_F = eta.apply(F)  # G(A)
    right_path = eta_of_F.fmap(f)  # G(B)

    # Check if values are equal
    return left_path._value == right_path._value


def test_natural_transformations():
    """
    Test naturality conditions for implemented transformations.

    Returns:
        Dictionary of test results
    """
    from .functor import MetaPromptFunctor

    results = {}

    # Test StrategyTransformation
    cot_to_react = StrategyTransformation("chain_of_thought", "react")
    F = MetaPromptFunctor("solve this problem", strategy="chain_of_thought")
    f = lambda x: x + " with details"

    results["StrategyTransformation_naturality"] = verify_naturality(
        cot_to_react, F, f
    )

    # Test IdentityTransformation
    id_transform = IdentityTransformation(MetaPromptFunctor)
    results["IdentityTransformation_naturality"] = verify_naturality(
        id_transform, F, f
    )

    # Test ComposedTransformation
    react_to_pot = StrategyTransformation("react", "program_of_thought")
    composed = ComposedTransformation(cot_to_react, react_to_pot)
    results["ComposedTransformation_naturality"] = verify_naturality(
        composed, F, f
    )

    return results


if __name__ == "__main__":
    results = test_natural_transformations()

    print("Natural Transformation Verification Results:")
    print("=" * 50)
    for test_name, passed in results.items():
        print(f"  {test_name}: {'✓' if passed else '✗'}")

    print("\nAll tests passed!" if all(results.values()) else "\nSome tests failed!")
