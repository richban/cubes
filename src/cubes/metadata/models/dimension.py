"""Dimension model classes for levels, hierarchies, and dimensions.

Provides domain dataclasses for dimension metadata with business logic
for level ownership, hierarchy management, and attribute indexing.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

from cubes.metadata.models.attributes import Attribute
from cubes.metadata.models.base import ModelError, to_label


@dataclass
class Level:
    """Hierarchy level representing a granularity level within a dimension.

    Levels define the granularity of data within a dimension hierarchy.
    Each level has a set of attributes, with a key attribute used for
    grouping, an optional label attribute for display, and an order
    attribute for sorting.

    Attributes:
        name: Level name, used as identifier.
        attributes: List of Attribute objects for this level.
        key: The key attribute used for grouping (defaults to first attribute).
        label_attribute: Attribute used for display labels (defaults to second
            attribute if available, otherwise first).
        order_attribute: Attribute used for sorting (defaults to first attribute).
        order: Default sort order ('asc' or 'desc', optional).
        cardinality: Approximate member count (tiny, low, medium, high, optional).
        role: Special role within dimensions (optional).
        nonadditive: Non-additivity type (None, 'all', or 'time', optional).
        info: Custom information dictionary (defaults to empty dict).
        description: Longer description of the level (optional).
        label: Human-readable label (optional, auto-generated if not provided).

    Example:
        >>> level = Level(
        ...     name="month",
        ...     attributes=[
        ...         Attribute(name="month"),
        ...         Attribute(name="month_name")
        ...     ]
        ... )
        >>> level.key.name
        'month'
        >>> level.label_attribute.name
        'month_name'
        >>> level.has_details
        True
    """

    name: str
    attributes: list[Attribute]
    label: Optional[str] = None
    description: Optional[str] = None
    info: dict[str, Any] = field(default_factory=dict)
    key: Union[Attribute, str, None] = None
    label_attribute: Union[Attribute, str, None] = None
    order_attribute: Union[Attribute, str, None] = None
    order: Optional[Literal["asc", "desc"]] = None
    cardinality: Optional[Literal["tiny", "low", "medium", "high"]] = None
    role: Optional[str] = None
    nonadditive: Optional[Literal["all", "time"]] = None

    def __post_init__(self) -> None:
        """Initialize computed fields and resolve attribute references.

        Validates that attributes list is not empty and resolves key,
        label_attribute, and order_attribute references to actual
        Attribute objects.

        Raises:
            ModelError: If attributes list is empty.
        """
        # Validate attributes list is not empty
        if not self.attributes:
            raise ModelError(f"Level '{self.name}' must have at least one attribute")

        # Resolve key attribute reference
        if self.key is None:
            # Default to first attribute
            object.__setattr__(self, "key", self.attributes[0])
        elif isinstance(self.key, str):
            # Resolve string reference to Attribute object
            object.__setattr__(self, "key", self.attribute(self.key))

        # Resolve label_attribute reference
        if self.label_attribute is None:
            # Default to second attribute if available, otherwise first
            if len(self.attributes) > 1:
                object.__setattr__(self, "label_attribute", self.attributes[1])
            else:
                object.__setattr__(self, "label_attribute", self.key)
        elif isinstance(self.label_attribute, str):
            # Resolve string reference to Attribute object
            object.__setattr__(
                self, "label_attribute", self.attribute(self.label_attribute)
            )

        # Resolve order_attribute reference
        if self.order_attribute is None:
            # Default to first attribute
            object.__setattr__(self, "order_attribute", self.attributes[0])
        elif isinstance(self.order_attribute, str):
            # Resolve string reference to Attribute object
            object.__setattr__(
                self, "order_attribute", self.attribute(self.order_attribute)
            )

        # Normalize nonadditive value
        if self.nonadditive is not None:
            normalized = self.nonadditive.lower()
            if normalized == "none":
                object.__setattr__(self, "nonadditive", None)
            elif normalized in ("all", "any"):
                object.__setattr__(self, "nonadditive", "all")
            elif normalized == "time":
                object.__setattr__(self, "nonadditive", "time")
            else:
                raise ModelError(
                    f"Level '{self.name}' has unknown non-additive type "
                    f"'{self.nonadditive}'. Must be 'none', 'all', 'any', or 'time'."
                )

        # Normalize order to lowercase
        if self.order is not None:
            normalized = self.order.lower()
            if normalized not in ("asc", "desc"):
                raise ModelError(
                    f"Level '{self.name}' has unknown ordering '{self.order}'. "
                    "Must be 'asc' or 'desc'."
                )
            object.__setattr__(self, "order", normalized)

    def __eq__(self, other: object) -> bool:
        """Check equality with another Level.

        Two levels are equal if they have the same name, label, key,
        cardinality, role, label_attribute, order_attribute, nonadditive
        state, and attributes.

        Args:
            other: Another object to compare with.

        Returns:
            True if equal, False otherwise.
        """
        if not isinstance(other, Level):
            return False

        return (
            self.name == other.name
            and self.label == other.label
            and self.key == other.key
            and self.cardinality == other.cardinality
            and self.role == other.role
            and self.label_attribute == other.label_attribute
            and self.order_attribute == other.order_attribute
            and self.nonadditive == other.nonadditive
            and self.attributes == other.attributes
        )

    def __hash__(self) -> int:
        """Return hash based on name for use in sets and dicts."""
        return hash(self.name)

    def __str__(self) -> str:
        """Return string representation (name)."""
        return self.name

    def __repr__(self) -> str:
        """Return detailed representation."""
        return f"Level({self.to_dict()!r})"

    def __deepcopy__(self, memo: dict[int, Any]) -> "Level":
        """Create a deep copy of this level.

        Args:
            memo: Memoization dict for deep copy.

        Returns:
            A new Level instance with copied values.
        """
        # Deep copy all attributes
        copied_attrs = [copy.deepcopy(attr, memo) for attr in self.attributes]

        # Create new level with copied attributes, but store names for key attrs
        # They will be resolved in __post_init__
        key_name: Optional[str] = None
        if self.key:
            key_name = self.key.name if isinstance(self.key, Attribute) else self.key

        label_attr_name: Optional[str] = None
        if self.label_attribute:
            label_attr_name = (
                self.label_attribute.name
                if isinstance(self.label_attribute, Attribute)
                else self.label_attribute
            )

        order_attr_name: Optional[str] = None
        if self.order_attribute:
            order_attr_name = (
                self.order_attribute.name
                if isinstance(self.order_attribute, Attribute)
                else self.order_attribute
            )

        result = Level(
            name=self.name,
            attributes=copied_attrs,
            label=self.label,
            description=self.description,
            info=copy.deepcopy(self.info, memo),
            key=key_name,
            label_attribute=label_attr_name,
            order_attribute=order_attr_name,
            order=self.order,
            cardinality=self.cardinality,
            role=self.role,
            nonadditive=self.nonadditive,
        )
        memo[id(self)] = result
        return result

    @property
    def has_details(self) -> bool:
        """Check if level has more than one attribute.

        Levels with only one attribute are considered "flat" and don't
        have additional details. Levels with multiple attributes have
        details beyond just the key.

        Returns:
            True if level has more than one attribute, False otherwise.
        """
        return len(self.attributes) > 1

    def attribute(self, name: str) -> Attribute:
        """Get attribute by name.

        Looks up an attribute within this level's attributes by its name.

        Args:
            name: The attribute name to look up.

        Returns:
            The Attribute object with the matching name.

        Raises:
            ModelError: If no attribute with the given name exists.
        """
        for attr in self.attributes:
            if attr.name == name:
                return attr
        raise ModelError(
            f"Level '{self.name}' has no attribute '{name}'. "
            f"Available attributes: {[a.name for a in self.attributes]}"
        )

    def to_dict(
        self, create_label: bool = False, full_attribute_names: bool = False
    ) -> dict[str, Any]:
        """Convert to a dictionary representation.

        Args:
            create_label: If True and no label is set, auto-generate from name.
            full_attribute_names: If True, use full refs for key attributes
                instead of just names.

        Returns:
            Dictionary with all level properties.
        """
        result: dict[str, Any] = {
            "name": self.name,
            "label": self.label
            if self.label
            else (to_label(self.name) if create_label else None),
            "description": self.description,
            "info": dict(self.info),
        }

        # Add Level-specific fields
        # Handle Union[Attribute, str, None] types for key attributes
        def _get_key_value(
            attr: Union[Attribute, str, None], use_ref: bool
        ) -> Optional[str]:
            if attr is None:
                return None
            if isinstance(attr, Attribute):
                return attr.ref if use_ref else attr.name
            return attr  # It's already a string

        result["key"] = _get_key_value(self.key, full_attribute_names)
        result["label_attribute"] = _get_key_value(
            self.label_attribute, full_attribute_names
        )
        result["order_attribute"] = _get_key_value(
            self.order_attribute, full_attribute_names
        )

        result["order"] = self.order
        result["cardinality"] = self.cardinality
        result["nonadditive"] = self.nonadditive
        result["role"] = self.role
        result["attributes"] = [
            attr.to_dict(create_label=create_label) for attr in self.attributes
        ]

        return result

    @classmethod
    def from_schema(cls, schema: Any) -> "Level":
        """Create a Level instance from a validated LevelSchema.

        This factory method creates a Level from a Pydantic schema,
        converting attribute dictionaries to Attribute objects.

        Args:
            schema: A validated LevelSchema instance.

        Returns:
            A new Level instance with resolved attributes.
        """
        from cubes.metadata.schemas.attributes import AttributeSchema
        from cubes.metadata.schemas.dimension import LevelSchema

        # Convert attributes from schema to Attribute objects
        attributes: list[Attribute] = []
        for attr_input in schema.attributes:
            if isinstance(attr_input, AttributeSchema):
                # Create Attribute from AttributeSchema
                attr = Attribute(
                    name=attr_input.name,
                    label=attr_input.label,
                    description=attr_input.description,
                    order=attr_input.order,
                    info=dict(attr_input.info),
                    format=attr_input.format,
                    missing_value=attr_input.missing_value,
                    expression=attr_input.expression,
                )
            elif isinstance(attr_input, dict):
                # Create Attribute from dict
                attr = Attribute(
                    name=attr_input.get("name", ""),
                    label=attr_input.get("label"),
                    description=attr_input.get("description"),
                    order=attr_input.get("order"),
                    info=dict(attr_input.get("info", {})),
                    format=attr_input.get("format"),
                    missing_value=attr_input.get("missing_value"),
                    expression=attr_input.get("expression"),
                )
            else:
                # String - create simple attribute
                attr = Attribute(name=str(attr_input))
            attributes.append(attr)

        # Create level with attribute names for key fields (will be resolved in __post_init__)
        return cls(
            name=schema.name,
            attributes=attributes,
            label=schema.label,
            description=schema.description,
            info=dict(schema.info),
            key=schema.key,
            label_attribute=schema.label_attribute,
            order_attribute=schema.order_attribute,
            order=schema.order,
            cardinality=schema.cardinality,
            role=schema.role,
            nonadditive=schema.nonadditive,
        )


# Dimension stub for forward references from attributes module
@dataclass
class Dimension:
    """Placeholder Dimension class for forward references.

    This stub is replaced by the full implementation in US-016/US-017.
    """

    name: str
