"""Base classes for model objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


def to_label(name: str, capitalize: bool = True) -> str:
    """Convert name into label by replacing underscores with spaces.

    Args:
        name: The name to convert.
        capitalize: Whether to capitalize the first letter (default: True).

    Returns:
        The formatted label string.
    """
    label = name.replace("_", " ")
    if capitalize:
        label = label.capitalize()
    return label


@dataclass
class ModelObject:
    """Base class for all model objects.

    Provides common properties (name, label, description, info) that are
    standardized and type-safe across all model objects.

    Attributes:
        name: The object's name/identifier.
        label: Optional human-readable label.
        description: Optional description text.
        info: Dictionary for arbitrary metadata (default empty dict).
    """

    name: str
    label: Optional[str] = None
    description: Optional[str] = None
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, create_label: bool = False, **options: Any) -> dict[str, Any]:
        """Convert to a dictionary representation.

        Args:
            create_label: If True, generate a label from name if label is None.
            **options: Additional options (ignored for compatibility).

        Returns:
            Dictionary with name, label, description, and info fields.
        """
        result: dict[str, Any] = {
            "name": self.name,
            "info": self.info,
        }

        if create_label:
            result["label"] = self.label or to_label(self.name)
        else:
            result["label"] = self.label

        result["description"] = self.description

        # Remove None values to match IgnoringDictionary behavior
        return {k: v for k, v in result.items() if v is not None}
