"""
Categorical abstractions for meta-prompting framework.

This module implements the mathematical foundations from:
- Zhang et al. (2023): "Meta Prompting for AI Systems"
- de Wynter et al. (2025): "On Meta-Prompting"
- Spivak: Polynomial functors and dynamical systems

Provides:
- Functors: Structure-preserving mappings between categories
- Monads: Recursive meta-prompting with RMPMonad
- Natural Transformations: Strategy equivalence
- Enriched Categories: Quality metrics and cost tracking
- Polynomial Functors: Tool/agent composition with lenses
"""

from .functor import (
    Functor,
    MetaPromptFunctor,
    IdentityFunctor,
    ComposedFunctor,
    verify_functor_laws,
    test_functor_laws,
)

from .monad import (
    Monad,
    RMPMonad,
    ListMonad,
    verify_monad_laws,
    test_monad_laws,
    test_rmp_quality_monotonicity,
)

from .natural_transformation import (
    NaturalTransformation,
    StrategyTransformation,
    IdentityTransformation,
    ComposedTransformation,
    verify_naturality,
    test_natural_transformations,
)

from .enriched import (
    MonoidalCategory,
    QualityMetric,
    CostMetric,
    EnrichedCategory,
    QualityEnrichedPrompts,
    CostEnrichedPrompts,
    test_enriched_categories,
)

from .polynomial import (
    PolynomialFunctor,
    Lens,
    ToolInterface,
    wire_tools,
    test_polynomial_functors,
)

__all__ = [
    # Functors
    "Functor",
    "MetaPromptFunctor",
    "IdentityFunctor",
    "ComposedFunctor",
    "verify_functor_laws",
    "test_functor_laws",
    # Monads
    "Monad",
    "RMPMonad",
    "ListMonad",
    "verify_monad_laws",
    "test_monad_laws",
    "test_rmp_quality_monotonicity",
    # Natural Transformations
    "NaturalTransformation",
    "StrategyTransformation",
    "IdentityTransformation",
    "ComposedTransformation",
    "verify_naturality",
    "test_natural_transformations",
    # Enriched Categories
    "MonoidalCategory",
    "QualityMetric",
    "CostMetric",
    "EnrichedCategory",
    "QualityEnrichedPrompts",
    "CostEnrichedPrompts",
    "test_enriched_categories",
    # Polynomial Functors
    "PolynomialFunctor",
    "Lens",
    "ToolInterface",
    "wire_tools",
    "test_polynomial_functors",
]
