"""
L1: Option Type - Null Safety

A practical implementation of the Option monad for handling nullable values safely.
This is the foundation of the skill hierarchy.
"""

from typing import TypeVar, Generic, Callable, Optional as PyOptional
from dataclasses import dataclass

T = TypeVar('T')
U = TypeVar('U')


class Option(Generic[T]):
    """
    Option type represents an optional value: every Option is either Some and contains a value,
    or None, and does not.

    This eliminates null pointer exceptions by making the absence of a value explicit.
    """

    def __init__(self):
        raise NotImplementedError("Use Option.some() or Option.none() instead")

    @staticmethod
    def some(value: T) -> 'Some[T]':
        """Create an Option containing a value."""
        if value is None:
            raise ValueError("Cannot create Some with None value. Use Option.none() instead.")
        return Some(value)

    @staticmethod
    def none() -> 'NoneType[T]':
        """Create an Option with no value."""
        return NoneType()

    @staticmethod
    def from_nullable(value: PyOptional[T]) -> 'Option[T]':
        """Convert a potentially None value to an Option."""
        if value is None:
            return Option.none()
        return Option.some(value)

    def is_some(self) -> bool:
        """Returns True if the Option is Some."""
        raise NotImplementedError

    def is_none(self) -> bool:
        """Returns True if the Option is None."""
        raise NotImplementedError

    def unwrap(self) -> T:
        """
        Returns the contained Some value.
        Raises ValueError if the value is None.
        """
        raise NotImplementedError

    def unwrap_or(self, default: T) -> T:
        """Returns the contained Some value or a provided default."""
        raise NotImplementedError

    def unwrap_or_else(self, fn: Callable[[], T]) -> T:
        """Returns the contained Some value or computes it from a closure."""
        raise NotImplementedError

    def map(self, fn: Callable[[T], U]) -> 'Option[U]':
        """
        Maps an Option[T] to Option[U] by applying a function to the contained value.
        If None, returns None.
        """
        raise NotImplementedError

    def flat_map(self, fn: Callable[[T], 'Option[U]']) -> 'Option[U]':
        """
        Applies a function to the contained value (if any), which itself returns an Option.
        Also known as 'bind' or 'and_then'.
        """
        raise NotImplementedError

    def filter(self, predicate: Callable[[T], bool]) -> 'Option[T]':
        """
        Returns None if the Option is None, otherwise calls predicate with the wrapped value
        and returns Some if the predicate returns true, None otherwise.
        """
        raise NotImplementedError

    def or_else(self, alternative: 'Option[T]') -> 'Option[T]':
        """Returns this Option if it contains a value, otherwise returns alternative."""
        raise NotImplementedError

    def and_then(self, fn: Callable[[T], 'Option[U]']) -> 'Option[U]':
        """Alias for flat_map."""
        return self.flat_map(fn)


@dataclass
class Some(Option[T]):
    """Represents an Option that contains a value."""
    value: T

    def is_some(self) -> bool:
        return True

    def is_none(self) -> bool:
        return False

    def unwrap(self) -> T:
        return self.value

    def unwrap_or(self, default: T) -> T:
        return self.value

    def unwrap_or_else(self, fn: Callable[[], T]) -> T:
        return self.value

    def map(self, fn: Callable[[T], U]) -> Option[U]:
        return Some(fn(self.value))

    def flat_map(self, fn: Callable[[T], Option[U]]) -> Option[U]:
        return fn(self.value)

    def filter(self, predicate: Callable[[T], bool]) -> Option[T]:
        if predicate(self.value):
            return self
        return Option.none()

    def or_else(self, alternative: Option[T]) -> Option[T]:
        return self

    def __repr__(self) -> str:
        return f"Some({self.value})"

    def __eq__(self, other) -> bool:
        if isinstance(other, Some):
            return self.value == other.value
        return False


class NoneType(Option[T]):
    """Represents an Option that contains no value."""

    def __init__(self):
        # Override parent __init__ to allow creating None instances
        pass

    def is_some(self) -> bool:
        return False

    def is_none(self) -> bool:
        return True

    def unwrap(self) -> T:
        raise ValueError("Called unwrap() on None value")

    def unwrap_or(self, default: T) -> T:
        return default

    def unwrap_or_else(self, fn: Callable[[], T]) -> T:
        return fn()

    def map(self, fn: Callable[[T], U]) -> Option[U]:
        return NoneType()

    def flat_map(self, fn: Callable[[T], Option[U]]) -> Option[U]:
        return NoneType()

    def filter(self, predicate: Callable[[T], bool]) -> Option[T]:
        return self

    def or_else(self, alternative: Option[T]) -> Option[T]:
        return alternative

    def __repr__(self) -> str:
        return "None"

    def __eq__(self, other) -> bool:
        return isinstance(other, NoneType)


# Convenience functions
def some(value: T) -> Option[T]:
    """Shorthand for Option.some()"""
    return Option.some(value)


def none() -> Option[T]:
    """Shorthand for Option.none()"""
    return Option.none()
