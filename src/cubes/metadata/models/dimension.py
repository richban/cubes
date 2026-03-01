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


class HierarchyError(ModelError):
    """Error raised for hierarchy-related issues.

    This exception is raised when operations on hierarchies fail,
    such as invalid level access, out-of-bounds depth requests,
    or invalid path operations.

    Example:
        >>> raise HierarchyError("Depth 5 exceeds hierarchy length")
    """

    pass


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


@dataclass
class Hierarchy:
    """Dimension hierarchy - specifies order of dimension levels.

    Hierarchies define the drill-down and roll-up paths within a dimension.
    Each hierarchy contains an ordered list of levels, from most general
    to most specific.

    Attributes:
        name: Hierarchy name, used as identifier.
        levels: Ordered list of Level objects (from most to least granular).
        label: Human-readable label (optional, auto-generated if not provided).
        description: Longer description of the hierarchy (optional).
        info: Custom information dictionary (defaults to empty dict).

    Example:
        >>> hierarchy = Hierarchy(
        ...     name="default",
        ...     levels=[
        ...         Level(name="year", attributes=[Attribute(name="year")]),
        ...         Level(name="month", attributes=[Attribute(name="month")]),
        ...         Level(name="day", attributes=[Attribute(name="day")])
        ...     ]
        ... )
        >>> len(hierarchy)
        3
        >>> hierarchy.level_names
        ['year', 'month', 'day']
    """

    name: str
    levels: list[Level]
    label: Optional[str] = None
    description: Optional[str] = None
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate hierarchy initialization.

        Ensures that the levels list is not empty.

        Raises:
            ModelError: If levels list is empty.
        """
        if not self.levels:
            raise ModelError(f"Hierarchy '{self.name}' must have at least one level")

    def __eq__(self, other: object) -> bool:
        """Check equality with another Hierarchy.

        Two hierarchies are equal if they have the same name, label,
        and levels.

        Args:
            other: Another object to compare with.

        Returns:
            True if equal, False otherwise.
        """
        if not isinstance(other, Hierarchy):
            return False

        return (
            self.name == other.name
            and self.label == other.label
            and self.levels == other.levels
        )

    def __hash__(self) -> int:
        """Return hash based on name for use in sets and dicts."""
        return hash(self.name)

    def __str__(self) -> str:
        """Return string representation (name)."""
        return self.name

    def __len__(self) -> int:
        """Return the number of levels in the hierarchy."""
        return len(self.levels)

    def __getitem__(self, index: int) -> Level:
        """Get level by index.

        Args:
            index: The level index (0-based).

        Returns:
            The Level at the specified index.

        Raises:
            HierarchyError: If index is out of range.
        """
        try:
            return self.levels[index]
        except IndexError:
            raise HierarchyError(
                f"Hierarchy '{self.name}' has only {len(self.levels)} levels, "
                f"asking for deeper level"
            )

    def __contains__(self, item: Union[Level, str]) -> bool:
        """Check if level or level name is in hierarchy.

        Args:
            item: Either a Level object or a level name string.

        Returns:
            True if the level is in the hierarchy, False otherwise.
        """
        if isinstance(item, Level):
            return item in self.levels
        return item in [level.name for level in self.levels]

    def __deepcopy__(self, memo: dict[int, Any]) -> "Hierarchy":
        """Create a deep copy of this hierarchy.

        Creates an independent copy with all levels and attributes copied.
        Level attributes have their dimension disowned (set to None).

        Args:
            memo: Memoization dict for deep copy.

        Returns:
            A new Hierarchy instance with copied levels.
        """
        copied_levels = [copy.deepcopy(level, memo) for level in self.levels]
        result = Hierarchy(
            name=self.name,
            levels=copied_levels,
            label=self.label,
            description=self.description,
            info=copy.deepcopy(self.info, memo),
        )
        memo[id(self)] = result
        return result

    @property
    def level_names(self) -> list[str]:
        """Return list of level names in order.

        Returns:
            List of level names.
        """
        return [level.name for level in self.levels]

    def keys(self, depth: Optional[int] = None) -> list[Attribute]:
        """Return key attributes for levels up to specified depth.

        Args:
            depth: Number of levels to include (None for all levels).

        Returns:
            List of key attributes for the requested levels.
        """
        if depth is not None:
            levels = self.levels[:depth]
        else:
            levels = self.levels
        # After Level.__post_init__, key is always an Attribute
        return [level.key for level in levels]  # type: ignore[misc]

    def levels_for_path(self, path: list[Any], drilldown: bool = False) -> list[Level]:
        """Return levels for given path.

        Args:
            path: List of path elements representing current drill-down.
            drilldown: If True, include one additional level for drill-down.

        Returns:
            List of levels up to path depth (plus one if drilldown).

        Raises:
            HierarchyError: If path is longer than hierarchy levels.
        """
        depth = 0 if not path else len(path)
        return self.levels_for_depth(depth, drilldown)

    def levels_for_depth(
        self, depth: Optional[int], drilldown: bool = False
    ) -> list[Level]:
        """Return levels up to specified depth.

        Args:
            depth: The depth (number of levels) to return.
            drilldown: If True, include one additional level.

        Returns:
            List of levels up to the specified depth.

        Raises:
            HierarchyError: If depth exceeds hierarchy length.
        """
        depth = depth or 0
        extend = 1 if drilldown else 0

        if depth + extend > len(self.levels):
            raise HierarchyError(
                f"Depth {depth} is longer than hierarchy "
                f"levels {self.level_names} (drilldown: {drilldown})"
            )

        return self.levels[: depth + extend]

    def next_level(self, level: Union[Level, str, None]) -> Optional[Level]:
        """Get the next level after the specified level.

        Args:
            level: The current level (Level object, name string, or None).

        Returns:
            The next level, or None if current level is the last.
            If level is None, returns the first level.
        """
        if level is None:
            return self.levels[0]

        index = self.level_index(level)
        if index + 1 >= len(self.levels):
            return None
        return self.levels[index + 1]

    def previous_level(self, level: Union[Level, str, None]) -> Optional[Level]:
        """Get the previous level before the specified level.

        Args:
            level: The current level (Level object, name string, or None).

        Returns:
            The previous level, or None if current level is the first
            or if level is None.
        """
        if level is None:
            return None

        index = self.level_index(level)
        if index == 0:
            return None
        return self.levels[index - 1]

    def level_index(self, level: Union[Level, str]) -> int:
        """Get the index of a level in the hierarchy.

        Args:
            level: The level (Level object or name string) to find.

        Returns:
            The 0-based index of the level.

        Raises:
            HierarchyError: If level is not found in hierarchy.
        """
        level_name = str(level)
        try:
            return self.level_names.index(level_name)
        except ValueError:
            raise HierarchyError(
                f"Level {level_name} is not part of hierarchy {self.name}"
            )

    def is_last(self, level: Union[Level, str]) -> bool:
        """Check if level is the last (most granular) in the hierarchy.

        Args:
            level: The level to check (Level object or name string).

        Returns:
            True if level is the last level, False otherwise.
        """
        return level == self.levels[-1] or str(level) == self.levels[-1].name

    def rollup(
        self, path: list[Any], level: Optional[Union[Level, str]] = None
    ) -> list[Any]:
        """Roll up a path to a specified level.

        Removes elements from the end of the path to reach the target level.
        If no target level is specified, rolls up by one level.

        Args:
            path: The current path (list of values).
            level: Target level to roll up to (optional). If None,
                rolls up one level.

        Returns:
            The rolled-up path (truncated list).

        Raises:
            HierarchyError: If target level is deeper than the path
                or if level is not found.
        """
        if level is not None:
            last = self.level_index(level) + 1
            if last > len(path):
                raise HierarchyError(
                    f"Can not roll-up: level '{level}' - it is "
                    f"deeper than deepest element of path {path}"
                )
        else:
            if len(path) > 0:
                last = len(path) - 1
            else:
                return []

        return path[:last]

    def path_is_base(self, path: Optional[list[Any]]) -> bool:
        """Check if path reaches the base (bottom) of the hierarchy.

        A base path has one element for each level in the hierarchy,
        meaning no further drill-down is possible.

        Args:
            path: The path to check.

        Returns:
            True if path reaches hierarchy bottom, False otherwise.
        """
        return path is not None and len(path) == len(self.levels)

    def key_attributes(self) -> list[Attribute]:
        """Return key attributes for all levels.

        Returns:
            List of key attributes from all levels.
        """
        # After Level.__post_init__, key is always an Attribute
        return [level.key for level in self.levels]  # type: ignore[misc]

    @property
    def all_attributes(self) -> list[Attribute]:
        """Return all attributes from all levels as a flattened list.

        Returns:
            Flattened list of all attributes from all levels.
        """
        attributes: list[Attribute] = []
        for level in self.levels:
            attributes.extend(level.attributes)
        return attributes

    def to_dict(self, create_label: bool = False) -> dict[str, Any]:
        """Convert to a dictionary representation.

        Args:
            create_label: If True and no label is set, auto-generate from name.

        Returns:
            Dictionary with all hierarchy properties.
        """
        return {
            "name": self.name,
            "label": self.label
            if self.label
            else (to_label(self.name) if create_label else None),
            "description": self.description,
            "info": dict(self.info),
            "levels": [
                level.to_dict(create_label=create_label) for level in self.levels
            ],
        }

    @classmethod
    def from_schema(cls, schema: Any) -> "Hierarchy":
        """Create a Hierarchy instance from a validated HierarchySchema.

        This factory method creates a Hierarchy from a Pydantic schema,
        converting level definitions to Level objects.

        Args:
            schema: A validated HierarchySchema instance.

        Returns:
            A new Hierarchy instance with resolved levels.
        """
        from cubes.metadata.schemas.dimension import HierarchySchema, LevelSchema

        levels: list[Level] = []
        for level_input in schema.levels:
            if isinstance(level_input, LevelSchema):
                levels.append(Level.from_schema(level_input))
            elif isinstance(level_input, dict):
                # Convert dict to LevelSchema then to Level
                level_schema = LevelSchema.model_validate(level_input)
                levels.append(Level.from_schema(level_schema))
            else:
                # String - this shouldn't happen in normal flow as
                # schema validation converts strings to LevelSchema
                raise ModelError(
                    f"Cannot create hierarchy from string level: {level_input}"
                )

        return cls(
            name=schema.name,
            levels=levels,
            label=schema.label,
            description=schema.description,
            info=dict(schema.info),
        )


# Dimension stub for forward references from attributes module
@dataclass
class Dimension:
    """Placeholder Dimension class for forward references.

    This stub is replaced by the full implementation in US-016/US-017.
    """

    name: str
