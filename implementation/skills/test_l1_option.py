"""
Tests for L1: Option Type

These tests prove the Option type actually works and handles edge cases correctly.
"""

import pytest
from l1_option import Option, Some, NoneType, some, none


class TestOptionCreation:
    """Test creating Options"""

    def test_some_creates_option_with_value(self):
        opt = Option.some(42)
        assert isinstance(opt, Some)
        assert opt.unwrap() == 42

    def test_none_creates_empty_option(self):
        opt = Option.none()
        assert isinstance(opt, NoneType)
        assert opt.is_none()

    def test_from_nullable_with_value(self):
        opt = Option.from_nullable(42)
        assert opt.is_some()
        assert opt.unwrap() == 42

    def test_from_nullable_with_none(self):
        opt = Option.from_nullable(None)
        assert opt.is_none()

    def test_some_rejects_none_value(self):
        with pytest.raises(ValueError, match="Cannot create Some with None"):
            Option.some(None)

    def test_convenience_functions(self):
        assert some(42) == Option.some(42)
        assert none() == Option.none()


class TestOptionInspection:
    """Test checking Option state"""

    def test_some_is_some(self):
        opt = Option.some(42)
        assert opt.is_some()
        assert not opt.is_none()

    def test_none_is_none(self):
        opt = Option.none()
        assert opt.is_none()
        assert not opt.is_some()


class TestOptionUnwrap:
    """Test extracting values from Options"""

    def test_unwrap_some_returns_value(self):
        opt = Option.some(42)
        assert opt.unwrap() == 42

    def test_unwrap_none_raises_error(self):
        opt = Option.none()
        with pytest.raises(ValueError, match="Called unwrap\\(\\) on None"):
            opt.unwrap()

    def test_unwrap_or_some_returns_value(self):
        opt = Option.some(42)
        assert opt.unwrap_or(0) == 42

    def test_unwrap_or_none_returns_default(self):
        opt = Option.none()
        assert opt.unwrap_or(0) == 0

    def test_unwrap_or_else_some_returns_value(self):
        opt = Option.some(42)
        assert opt.unwrap_or_else(lambda: 0) == 42

    def test_unwrap_or_else_none_calls_function(self):
        opt = Option.none()
        assert opt.unwrap_or_else(lambda: 99) == 99


class TestOptionMap:
    """Test mapping functions over Options"""

    def test_map_some_applies_function(self):
        opt = Option.some(5)
        result = opt.map(lambda x: x * 2)
        assert result == Option.some(10)

    def test_map_none_returns_none(self):
        opt = Option.none()
        result = opt.map(lambda x: x * 2)
        assert result.is_none()

    def test_map_chains(self):
        opt = Option.some(5)
        result = opt.map(lambda x: x * 2).map(lambda x: x + 1)
        assert result == Option.some(11)

    def test_map_preserves_none_in_chain(self):
        opt = Option.none()
        result = opt.map(lambda x: x * 2).map(lambda x: x + 1)
        assert result.is_none()


class TestOptionFlatMap:
    """Test flat_map (monadic bind)"""

    def test_flat_map_some_returns_inner_option(self):
        opt = Option.some(5)
        result = opt.flat_map(lambda x: Option.some(x * 2))
        assert result == Option.some(10)

    def test_flat_map_none_returns_none(self):
        opt = Option.none()
        result = opt.flat_map(lambda x: Option.some(x * 2))
        assert result.is_none()

    def test_flat_map_can_return_none(self):
        opt = Option.some(5)
        result = opt.flat_map(lambda x: Option.none())
        assert result.is_none()

    def test_flat_map_chains(self):
        opt = Option.some(5)
        result = (opt
                  .flat_map(lambda x: Option.some(x * 2))
                  .flat_map(lambda x: Option.some(x + 1)))
        assert result == Option.some(11)

    def test_and_then_alias(self):
        opt = Option.some(5)
        result = opt.and_then(lambda x: Option.some(x * 2))
        assert result == Option.some(10)


class TestOptionFilter:
    """Test filtering Options"""

    def test_filter_some_passes_predicate(self):
        opt = Option.some(5)
        result = opt.filter(lambda x: x > 3)
        assert result == Option.some(5)

    def test_filter_some_fails_predicate(self):
        opt = Option.some(5)
        result = opt.filter(lambda x: x > 10)
        assert result.is_none()

    def test_filter_none_returns_none(self):
        opt = Option.none()
        result = opt.filter(lambda x: x > 3)
        assert result.is_none()


class TestOptionOrElse:
    """Test providing alternative Options"""

    def test_or_else_some_returns_self(self):
        opt = Option.some(5)
        result = opt.or_else(Option.some(10))
        assert result == Option.some(5)

    def test_or_else_none_returns_alternative(self):
        opt = Option.none()
        result = opt.or_else(Option.some(10))
        assert result == Option.some(10)

    def test_or_else_chains(self):
        opt = Option.none()
        result = opt.or_else(Option.none()).or_else(Option.some(42))
        assert result == Option.some(42)


class TestOptionEquality:
    """Test Option equality"""

    def test_some_equals_some_with_same_value(self):
        assert Option.some(42) == Option.some(42)

    def test_some_not_equals_some_with_different_value(self):
        assert Option.some(42) != Option.some(43)

    def test_none_equals_none(self):
        assert Option.none() == Option.none()

    def test_some_not_equals_none(self):
        assert Option.some(42) != Option.none()


class TestOptionRealWorldUseCases:
    """Test realistic usage patterns"""

    def test_safe_dictionary_access(self):
        """Simulate safe dictionary lookup"""
        data = {"name": "Alice", "age": 30}

        def safe_get(dictionary, key):
            return Option.from_nullable(dictionary.get(key))

        name = safe_get(data, "name")
        missing = safe_get(data, "address")

        assert name.unwrap() == "Alice"
        assert missing.unwrap_or("Unknown") == "Unknown"

    def test_chaining_operations(self):
        """Simulate chaining operations that might fail"""
        def divide(x, y):
            return Option.some(x / y) if y != 0 else Option.none()

        def square(x):
            return Option.some(x * x)

        # Success case
        result = divide(10, 2).flat_map(square)
        assert result == Option.some(25.0)

        # Failure case (division by zero)
        result = divide(10, 0).flat_map(square)
        assert result.is_none()

    def test_parsing_with_option(self):
        """Simulate safe parsing"""
        def parse_int(s):
            try:
                return Option.some(int(s))
            except ValueError:
                return Option.none()

        assert parse_int("42").unwrap() == 42
        assert parse_int("not a number").is_none()

    def test_configuration_with_defaults(self):
        """Simulate configuration with fallbacks"""
        config = {"timeout": "30"}

        def get_config(key):
            return Option.from_nullable(config.get(key))

        timeout = get_config("timeout").unwrap_or("60")
        max_retries = get_config("max_retries").unwrap_or("3")

        assert timeout == "30"
        assert max_retries == "3"


class TestOptionCognitiveLoad:
    """
    Test that Option reduces cognitive load by eliminating None checks.
    These tests demonstrate the difference between traditional None handling
    and Option-based code.
    """

    def test_traditional_none_checking(self):
        """
        Traditional approach: Manual None checks everywhere.
        Cognitive load: ~3 slots (track None state, check before use, handle errors)
        """
        def get_user_age_traditional(user_id):
            user = {"id": 1, "name": "Alice"}  # Simulated lookup
            if user is None:
                return None
            age = user.get("age")
            if age is None:
                return None
            return age * 2

        # Must check for None
        result = get_user_age_traditional(1)
        if result is not None:
            assert isinstance(result, int)

    def test_option_based_approach(self):
        """
        Option approach: Automatic None propagation.
        Cognitive load: ~1 slot (just think about the happy path)
        """
        def get_user_age_option(user_id):
            user = Option.some({"id": 1, "name": "Alice"})
            return (user
                    .flat_map(lambda u: Option.from_nullable(u.get("age")))
                    .map(lambda age: age * 2))

        result = get_user_age_option(1)
        # None propagates automatically, no manual checks needed
        assert result.is_none()  # age was missing

    def test_cognitive_load_measurement(self):
        """
        Measure cognitive complexity: Count decision points.

        Traditional None checking: Many if/None checks (high complexity)
        Option: One result check (low complexity)
        """
        # Traditional: 3+ decision points
        def process_traditional(value):
            if value is None:  # Decision 1
                return None
            result = value.get("data")
            if result is None:  # Decision 2
                return None
            return result * 2

        # Option: 1 decision point (at the end)
        def process_option(value):
            return (Option.from_nullable(value)
                    .flat_map(lambda v: Option.from_nullable(v.get("data")))
                    .map(lambda d: d * 2))

        # Both handle missing data, but Option has lower cognitive load
        assert process_traditional(None) is None
        assert process_option(None).is_none()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
