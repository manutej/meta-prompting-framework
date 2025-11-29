"""
Monad abstraction for meta-prompting framework.

Implements Zhang et al.'s RMP (Recursive Meta-Prompting) as a monad.
"""

from typing import TypeVar, Generic, Callable, Any, Dict, List
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

    These laws ensure that monadic composition is coherent and
    that unit behaves as a proper identity.
    """

    def __init__(self, value: T):
        super().__init__(value)

    @classmethod
    def unit(cls, value: T) -> 'Monad[T]':
        """
        η: A → M(A)

        Lifts a pure value into the monadic context.
        Also called 'return' or 'pure' in some languages.

        Args:
            value: Pure value to lift

        Returns:
            Value wrapped in monadic context
        """
        return cls(value)

    def flat_map(self, f: Callable[[T], 'Monad[U]']) -> 'Monad[U]':
        """
        Also called bind (>>=) or flatMap.

        Applies a monadic function and flattens the result.
        Equivalent to: fmap(f) followed by flatten (μ).

        Args:
            f: Function from T to M[U]

        Returns:
            Flattened result M[U]
        """
        return f(self._value)

    def flatten(self) -> 'Monad[T]':
        """
        μ: M(M(A)) → M(A)

        Flattens nested monadic structure.

        Returns:
            Flattened monad
        """
        if isinstance(self._value, Monad):
            return self._value.flatten()
        return self

    def fmap(self, f: Callable[[T], U]) -> 'Monad[U]':
        """
        Functor's fmap, implemented via flat_map.

        fmap(f) = flat_map(λx. unit(f(x)))
        """
        return self.flat_map(lambda x: self.__class__.unit(f(x)))

    def __rshift__(self, f: Callable[[T], 'Monad[U]']) -> 'Monad[U]':
        """
        Infix operator: m >> f === m.flat_map(f)

        Example:
            result = monad >> f >> g >> h
        """
        return self.flat_map(f)


class RMPMonad(Monad[str]):
    """
    Recursive Meta-Prompting Monad.

    Implements Zhang et al.'s RMP as a monad where:
    - Values are prompts (strings)
    - unit (η) embeds a prompt into meta-space
    - flat_map (>>=) applies meta-improvement and flattens
    - Quality is non-decreasing (monotonic improvement)
    - Context accumulates across iterations

    This is the key innovation: the meta-prompt improves itself recursively.

    Example:
        prompt = "Solve this equation"
        rmp = RMPMonad.unit(prompt)
        improved = rmp.flat_map(improve_with_context)
        further_improved = improved.flat_map(improve_with_context)
    """

    def __init__(
        self,
        prompt: str,
        quality: float = 0.0,
        iteration: int = 0,
        context: Dict[str, Any] = None,
        history: List[str] = None
    ):
        super().__init__(prompt)
        self.quality = quality
        self.iteration = iteration
        self.context = context or {}
        self.history = history or []

    @classmethod
    def unit(cls, prompt: str) -> 'RMPMonad':
        """
        η: Prompt → M(Prompt)

        Embeds a basic prompt into the RMP meta-space with:
        - Initial quality 0.0
        - Iteration 0
        - Empty context
        - Empty history
        """
        return cls(
            prompt=prompt,
            quality=0.0,
            iteration=0,
            context={},
            history=[prompt]
        )

    def flat_map(self, f: Callable[[str], 'RMPMonad']) -> 'RMPMonad':
        """
        Applies meta-improvement function and flattens.

        The function f should take a prompt and return an improved RMPMonad.

        Key properties:
        - Quality is non-decreasing: max(current, next)
        - Context accumulates: merge current and next
        - History tracks all prompts
        - Iteration increments

        Args:
            f: Meta-improvement function

        Returns:
            Improved RMP monad
        """
        next_monad = f(self._value)

        # Ensure quality is non-decreasing (monotonic improvement)
        improved_quality = max(self.quality, next_monad.quality)

        # Accumulate context
        merged_context = {**self.context, **next_monad.context}

        # Track history
        combined_history = self.history + next_monad.history

        return RMPMonad(
            prompt=next_monad._value,
            quality=improved_quality,
            iteration=self.iteration + 1,
            context=merged_context,
            history=combined_history
        )

    def with_quality(self, quality: float) -> 'RMPMonad':
        """Set quality score (useful for optimization)."""
        return RMPMonad(
            prompt=self._value,
            quality=quality,
            iteration=self.iteration,
            context=self.context,
            history=self.history
        )

    def with_context(self, **kwargs) -> 'RMPMonad':
        """Update context with new key-value pairs."""
        new_context = {**self.context, **kwargs}
        return RMPMonad(
            prompt=self._value,
            quality=self.quality,
            iteration=self.iteration,
            context=new_context,
            history=self.history
        )

    def __repr__(self):
        return (
            f"RMPMonad(quality={self.quality:.2f}, "
            f"iter={self.iteration}, "
            f"context_keys={list(self.context.keys())})"
        )


class ListMonad(Monad[List[T]]):
    """
    List monad for modeling non-determinism.

    Useful for exploring multiple prompt variations in parallel.

    Example:
        variations = ListMonad(["prompt1", "prompt2", "prompt3"])
        enhanced = variations.flat_map(lambda p: ListMonad([p + " v1", p + " v2"]))
        # Result: ["prompt1 v1", "prompt1 v2", "prompt2 v1", ...]
    """

    def __init__(self, value: List[T]):
        super().__init__(value)

    @classmethod
    def unit(cls, value: T) -> 'ListMonad[T]':
        """Wrap single value in list."""
        return cls([value])

    def flat_map(self, f: Callable[[T], 'ListMonad[U]']) -> 'ListMonad[U]':
        """
        Apply f to each element and concatenate results.

        This is the bind operation for lists, modeling non-deterministic
        computation.
        """
        results = []
        for item in self._value:
            next_monad = f(item)
            results.extend(next_monad._value)
        return ListMonad(results)


# Monad law verification utilities

def verify_monad_laws(
    monad_class: type[Monad],
    value: Any,
    f: Callable[[Any], Monad],
    g: Callable[[Any], Monad]
) -> Dict[str, bool]:
    """
    Property-based testing for monad laws.

    Args:
        monad_class: The monad class to test
        value: A test value
        f: A monadic function
        g: Another monadic function

    Returns:
        Dictionary with law verification results
    """
    m = monad_class.unit(value)

    # Law 1: Left unit
    # unit(a).flat_map(f) === f(a)
    left_unit_left = monad_class.unit(value).flat_map(f)._value
    left_unit_right = f(value)._value
    law1_passed = left_unit_left == left_unit_right

    # Law 2: Right unit
    # m.flat_map(unit) === m
    right_unit_left = m.flat_map(monad_class.unit)._value
    right_unit_right = m._value
    law2_passed = right_unit_left == right_unit_right

    # Law 3: Associativity
    # m.flat_map(f).flat_map(g) === m.flat_map(lambda x: f(x).flat_map(g))
    assoc_left = m.flat_map(f).flat_map(g)._value
    assoc_right = m.flat_map(lambda x: f(x).flat_map(g))._value
    law3_passed = assoc_left == assoc_right

    return {
        "left_unit_law": law1_passed,
        "right_unit_law": law2_passed,
        "associativity_law": law3_passed,
        "all_laws_satisfied": law1_passed and law2_passed and law3_passed
    }


def test_monad_laws():
    """
    Run monad law tests on all monad implementations.

    Returns:
        Dictionary mapping monad names to test results
    """
    results = {}

    # Test RMPMonad
    def improve_prompt(p: str) -> RMPMonad:
        return RMPMonad(p + " improved", quality=0.5, iteration=1)

    def optimize_prompt(p: str) -> RMPMonad:
        return RMPMonad(p + " optimized", quality=0.8, iteration=1)

    results["RMPMonad"] = verify_monad_laws(
        RMPMonad,
        value="basic prompt",
        f=improve_prompt,
        g=optimize_prompt
    )

    # Test ListMonad
    def duplicate(x: int) -> ListMonad:
        return ListMonad([x, x])

    def triple(x: int) -> ListMonad:
        return ListMonad([x, x, x])

    results["ListMonad"] = verify_monad_laws(
        ListMonad,
        value=5,
        f=duplicate,
        g=triple
    )

    return results


def test_rmp_quality_monotonicity():
    """
    Test that RMP monad maintains non-decreasing quality.

    This is a key property for recursive meta-prompting:
    quality should never decrease across iterations.
    """
    rmp = RMPMonad.unit("basic prompt")

    def improve_high(p: str) -> RMPMonad:
        return RMPMonad(p + " improved", quality=0.8)

    def improve_low(p: str) -> RMPMonad:
        return RMPMonad(p + " slightly", quality=0.3)

    # First high quality, then low quality
    result1 = rmp.flat_map(improve_high).flat_map(improve_low)

    # First low quality, then high quality
    result2 = rmp.flat_map(improve_low).flat_map(improve_high)

    # Both should end up with quality 0.8 (max)
    monotonic1 = result1.quality == 0.8
    monotonic2 = result2.quality == 0.8

    return {
        "quality_monotonic_high_then_low": monotonic1,
        "quality_monotonic_low_then_high": monotonic2,
        "all_tests_passed": monotonic1 and monotonic2
    }


if __name__ == "__main__":
    # Run law verification
    results = test_monad_laws()
    monotonicity = test_rmp_quality_monotonicity()

    print("Monad Law Verification Results:")
    print("=" * 50)
    for monad_name, laws in results.items():
        print(f"\n{monad_name}:")
        print(f"  Left Unit Law: {'✓' if laws['left_unit_law'] else '✗'}")
        print(f"  Right Unit Law: {'✓' if laws['right_unit_law'] else '✗'}")
        print(f"  Associativity Law: {'✓' if laws['associativity_law'] else '✗'}")
        print(f"  All Laws: {'✓' if laws['all_laws_satisfied'] else '✗'}")

    print("\n" + "=" * 50)
    print("\nRMP Quality Monotonicity Tests:")
    print("=" * 50)
    for test_name, passed in monotonicity.items():
        print(f"  {test_name}: {'✓' if passed else '✗'}")
