"""Base classes for all model objects."""

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, TypeVar


class ModelError(Exception):
    """Error related to model metadata."""

    pass


def to_label(name: str, capitalize: bool = True) -> str:
    """Converts `name` into label by replacing underscores by spaces.

    If `capitalize` is ``True`` (default) then the first letter of the label is
    capitalized.
    """
    label = name.replace("_", " ")
    if capitalize:
        label = label.capitalize()
    return label


@dataclass
class ModelObject:
    """Base class for all model objects.

    Provides common properties (name, label, description, info) that are
    standardized and type-safe.

    Attributes:
        name: The unique identifier for this model object.
        label: Human-readable label (optional, auto-generated if not provided).
        description: Longer description of the object (optional).
        info: Dictionary of additional metadata (defaults to empty dict).
    """

    name: str
    label: Optional[str] = None
    description: Optional[str] = None
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, create_label: bool = False) -> dict[str, Any]:
        """Convert to a dictionary.

        Args:
            create_label: If True and no label is set, auto-generate from name.
                         If False, use the existing label (which may be None).

        Returns:
            Dictionary representation with keys: name, label, description, info.
        """
        return {
            "name": self.name,
            "label": to_label(self.name)
            if create_label and self.label is None
            else self.label,
            "description": self.description,
            "info": dict(self.info),
        }


# Type variable for objects that have name/ref attributes
T = TypeVar("T")


class _NamedObject(Protocol):
    """Protocol for objects with name attribute."""

    name: str


class _RefObject(Protocol):
    """Protocol for objects with ref attribute."""

    ref: str


def object_dict(
    objects: list[T],
    by_ref: bool = False,
    error_message: Optional[str] = None,
    error_dict: Optional[dict[str, Any]] = None,
) -> dict[str, T]:
    """Make a dictionary from model objects where keys are object names.

    Creates an ordered dictionary from a list of model objects. Keys are
    object names by default, or object references if by_ref is True.
    Keys must be unique in the list, otherwise a ModelError is raised.

    Args:
        objects: List of model objects to convert to a dictionary.
        by_ref: If True, use object's `ref` attribute as key instead of `name`.
        error_message: Custom error message template for duplicate keys.
            Can use {key} placeholder. Defaults to "Duplicate key {key}".
        error_dict: Additional context for error message formatting.

    Returns:
        Dictionary mapping object names/refs to the objects themselves.
        Uses regular dict (Python 3.7+ preserves insertion order).

    Raises:
        ModelError: If duplicate keys are found in the objects list.

    Example:
        >>> objs = [ModelObject(name="a"), ModelObject(name="b")]
        >>> object_dict(objs)
        {'a': ModelObject(...), 'b': ModelObject(...)}
    """
    result: dict[str, T] = {}

    for obj in objects:
        key = obj.ref if by_ref else obj.name  # type: ignore[attr-defined]

        if key in result:
            msg = error_message or "Duplicate key {key}"
            ctx = error_dict or {}
            raise ModelError(msg.format(key=key, **ctx))

        result[key] = obj

    return result
