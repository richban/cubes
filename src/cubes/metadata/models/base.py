"""Base classes for all model objects."""

from dataclasses import dataclass, field
from typing import Any, Optional


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
