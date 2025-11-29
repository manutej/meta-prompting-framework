"""
Functor abstraction for meta-prompting framework.

Based on category theory foundations from Zhang et al. and de Wynter et al.
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Callable, Any

A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')


class Functor(ABC, Generic[A]):
    """
    A functor F: C → D between categories.

    Must satisfy functor laws:
    1. F(id_A) = id_F(A)           (identity preservation)
    2. F(g ∘ f) = F(g) ∘ F(f)      (composition preservation)

    These laws ensure that functors preserve the structure of categories,
    making composition predictable and lawful.
    """

    def __init__(self, value: A):
        self._value = value

    @abstractmethod
    def fmap(self, f: Callable[[A], B]) -> 'Functor[B]':
        """
        Map a function over the functor.

        Laws:
        - fmap(id) = id
        - fmap(g . f) = fmap(g) . fmap(f)

        Args:
            f: Function to map over the functor

        Returns:
            New functor with mapped value
        """
        pass

    def __rshift__(self, f: Callable[[A], B]) -> 'Functor[B]':
        """
        Infix operator: functor >> f === functor.fmap(f)

        Example:
            result = functor >> (lambda x: x + 1) >> (lambda x: x * 2)
        """
        return self.fmap(f)

    def __repr__(self):
        return f"{self.__class__.__name__}({self._value})"


class MetaPromptFunctor(Functor[A]):
    """
    Functor from Task category to Prompt category.

    F: Task → Prompt

    Implements Zhang et al.'s functorial meta-prompting where
    meta-prompting is modeled as a functor preserving composition.

    Example:
        task = Task("Solve x^2 + 5x + 6 = 0")
        prompt_functor = MetaPromptFunctor(task)
        enhanced = prompt_functor.fmap(lambda t: f"{t} with reasoning")
    """

    def __init__(self, value: A, strategy: str = "chain_of_thought"):
        super().__init__(value)
        self.strategy = strategy

    def fmap(self, f: Callable[[A], B]) -> 'MetaPromptFunctor[B]':
        """
        Maps a task transformation to a prompt transformation.

        Preserves composition: if you compose task transformations,
        the resulting prompt transformation is the composition of
        individual prompt transformations.
        """
        transformed = f(self._value)
        return MetaPromptFunctor(transformed, strategy=self.strategy)


class IdentityFunctor(Functor[A]):
    """
    Identity functor: Id(A) = A

    Maps every object to itself and every morphism to itself.
    Useful as a base case in functor composition.
    """

    def fmap(self, f: Callable[[A], B]) -> 'IdentityFunctor[B]':
        """Simply applies f to the value."""
        return IdentityFunctor(f(self._value))


class ComposedFunctor(Functor[A]):
    """
    Composition of two functors: (G ∘ F)(A) = G(F(A))

    If F: C → D and G: D → E, then G ∘ F: C → E
    """

    def __init__(self, value: A, inner: Functor, outer: Functor):
        super().__init__(value)
        self.inner = inner
        self.outer = outer

    def fmap(self, f: Callable[[A], B]) -> 'ComposedFunctor[B]':
        """
        (G ∘ F).fmap(f) = G.fmap(F.fmap(f))

        Composition of functors is itself a functor.
        """
        inner_result = self.inner.fmap(f)
        result = self.outer.fmap(lambda x: inner_result._value)
        return ComposedFunctor(result._value, self.inner, self.outer)


# Functor law verification utilities

def verify_functor_laws(
    functor_class: type[Functor],
    value: Any,
    f: Callable[[Any], Any],
    g: Callable[[Any], Any]
) -> dict[str, bool]:
    """
    Property-based testing for functor laws.

    Args:
        functor_class: The functor class to test
        value: A test value
        f: A test function
        g: Another test function

    Returns:
        Dictionary with law verification results
    """
    F = functor_class(value)

    # Law 1: fmap(id) = id
    identity = lambda x: x
    law1_passed = F.fmap(identity)._value == F._value

    # Law 2: fmap(g . f) = fmap(g) . fmap(f)
    composed = lambda x: g(f(x))
    left_side = F.fmap(composed)._value
    right_side = F.fmap(f).fmap(g)._value
    law2_passed = left_side == right_side

    return {
        "identity_law": law1_passed,
        "composition_law": law2_passed,
        "all_laws_satisfied": law1_passed and law2_passed
    }


def test_functor_laws():
    """
    Run functor law tests on all functor implementations.

    Returns:
        Dictionary mapping functor names to test results
    """
    results = {}

    # Test IdentityFunctor
    results["IdentityFunctor"] = verify_functor_laws(
        IdentityFunctor,
        value=5,
        f=lambda x: x + 1,
        g=lambda x: x * 2
    )

    # Test MetaPromptFunctor
    results["MetaPromptFunctor"] = verify_functor_laws(
        MetaPromptFunctor,
        value="task",
        f=lambda x: x + " enhanced",
        g=lambda x: x + " optimized"
    )

    return results


if __name__ == "__main__":
    # Run law verification
    results = test_functor_laws()

    print("Functor Law Verification Results:")
    print("=" * 50)
    for functor_name, laws in results.items():
        print(f"\n{functor_name}:")
        print(f"  Identity Law: {'✓' if laws['identity_law'] else '✗'}")
        print(f"  Composition Law: {'✓' if laws['composition_law'] else '✗'}")
        print(f"  All Laws: {'✓' if laws['all_laws_satisfied'] else '✗'}")
