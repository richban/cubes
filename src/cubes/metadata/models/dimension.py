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

    def to_dict(
        self, create_label: bool = False, depth: Optional[int] = None
    ) -> dict[str, Any]:
        """Convert to a dictionary representation.

        Args:
            create_label: If True and no label is set, auto-generate from name.
            depth: If specified, truncate levels to this depth.

        Returns:
            Dictionary with all hierarchy properties.
        """
        levels = [str(level) for level in self.levels]
        if depth:
            levels = levels[:depth]

        return {
            "name": self.name,
            "label": self.label
            if self.label
            else (to_label(self.name) if create_label else None),
            "description": self.description,
            "info": dict(self.info),
            "levels": levels,
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
            elif isinstance(level_input, str):
                # String level name - create simple level with that name
                # This will be resolved to actual level object by Dimension.from_schema
                levels.append(
                    Level(name=level_input, attributes=[Attribute(name=level_input)])
                )
            else:
                raise ModelError(f"Cannot create hierarchy from level: {level_input}")

        return cls(
            name=schema.name,
            levels=levels,
            label=schema.label,
            description=schema.description,
            info=dict(schema.info),
        )


# Default level roles for time dimensions
_DEFAULT_LEVEL_ROLES: dict[str, tuple[str, ...]] = {
    "time": (
        "year",
        "quarter",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "week",
        "weeknum",
        "dow",
        "isoyear",
        "isoweek",
        "isoweekday",
    )
}


@dataclass
class Dimension:
    """Cube dimension representing a business entity like date, product, etc.

    Dimensions contain levels and hierarchies that define how data can be
    analyzed and drilled into. Each dimension owns its levels and their
    attributes, allowing for proper scoping and referencing.

    Attributes:
        name: Dimension name, used as identifier.
        levels: List of Level objects defining granularity levels.
        hierarchies: List of Hierarchy objects defining drill-down paths.
        default_hierarchy_name: Name of the default hierarchy to use.
        label: Human-readable label (optional, auto-generated if not provided).
        description: Longer description of the dimension (optional).
        info: Custom information dictionary (defaults to empty dict).
        role: Special dimension role, e.g., "time" for time dimensions.
        cardinality: Approximate member count (tiny, low, medium, high).
        category: Logical dimension group for UI organization.
        nonadditive: Non-additivity type (None, 'all', or 'time').
        master: Original dimension this was cloned from (optional).

    Example:
        >>> dim = Dimension(
        ...     name="date",
        ...     levels=[
        ...         Level(name="year", attributes=[Attribute(name="year")]),
        ...         Level(name="month", attributes=[Attribute(name="month")]),
        ...     ]
        ... )
        >>> dim.is_flat
        False
        >>> len(dim.levels)
        2
    """

    name: str
    levels: list[Level]
    hierarchies: Optional[list[Hierarchy]] = None
    default_hierarchy_name: Optional[str] = None
    label: Optional[str] = None
    description: Optional[str] = None
    info: dict[str, Any] = field(default_factory=dict)
    role: Optional[str] = None
    cardinality: Optional[Literal["tiny", "low", "medium", "high"]] = None
    category: Optional[str] = None
    nonadditive: Optional[Literal["all", "time"]] = None
    master: Optional["Dimension"] = None

    # Internal dictionaries for fast lookups (initialized in __post_init__)
    _level_dict: dict[str, Level] = field(default_factory=dict, init=False, repr=False)
    _hierarchy_dict: dict[str, Hierarchy] = field(
        default_factory=dict, init=False, repr=False
    )
    _attributes: dict[str, Attribute] = field(
        default_factory=dict, init=False, repr=False
    )
    _attributes_by_ref: dict[str, Attribute] = field(
        default_factory=dict, init=False, repr=False
    )
    _default_hierarchy: Optional[Hierarchy] = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Initialize computed fields and build lookup dictionaries.

        This method:
        1. Validates that levels list is not empty
        2. Builds level and hierarchy dictionaries
        3. Sets default level roles for time dimensions
        4. Collects all attributes and builds lookup maps
        5. Assigns dimension ownership to all level attributes
        6. Validates and sets the default hierarchy
        """
        if not self.levels:
            raise ModelError(f"Dimension '{self.name}' must have at least one level")

        # Build levels dictionary
        object.__setattr__(self, "_levels", {})
        for level in self.levels:
            if level.name in self._level_dict:
                raise ModelError(
                    f"Duplicate level '{level.name}' in dimension '{self.name}'"
                )
            self._level_dict[level.name] = level

        # Set default level roles for time dimensions
        if self.role is not None:
            default_roles = _DEFAULT_LEVEL_ROLES.get(self.role)
            if default_roles:
                for level in self._level_dict.values():
                    if level.name in default_roles:
                        object.__setattr__(level, "role", level.name)

        # Collect all attributes and build lookup maps
        object.__setattr__(self, "_attributes", {})
        object.__setattr__(self, "_attributes_by_ref", {})

        for level in self.levels:
            for attr in level.attributes:
                # Check if attribute is owned by another dimension
                if attr.dimension is not None and attr.dimension is not self:
                    raise ModelError(
                        f"Dimension '{self.name}' cannot claim attribute "
                        f"'{attr.name}' because it is owned by another "
                        f"dimension '{attr.dimension.name}'."
                    )
                # Assign dimension ownership
                attr.dimension = self
                self._attributes[attr.name] = attr
                self._attributes_by_ref[attr.ref] = attr

        # Build hierarchies
        object.__setattr__(self, "_hierarchy_dict", {})

        if self.hierarchies:
            for hier in self.hierarchies:
                if hier.name in self._hierarchy_dict:
                    raise ModelError(
                        f"Duplicate hierarchy '{hier.name}' in dimension '{self.name}'"
                    )
                self._hierarchy_dict[hier.name] = hier
        else:
            # Create default hierarchy from all levels
            default_hier = Hierarchy(name="default", levels=self.levels)
            self._hierarchy_dict["default"] = default_hier
            object.__setattr__(self, "hierarchies", [default_hier])

        # Filter levels to only those used in hierarchies
        used_level_names: set[str] = set()
        for hier in self._hierarchy_dict.values():
            for level in hier.levels:
                used_level_names.add(level.name)

        # Rebuild _level_dict with only used levels
        filtered_levels = {
            name: level
            for name, level in self._level_dict.items()
            if name in used_level_names
        }
        object.__setattr__(self, "_level_dict", filtered_levels)

        # Also update the levels field to match
        object.__setattr__(self, "levels", list(filtered_levels.values()))

        # Normalize nonadditive
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
                    f"Dimension '{self.name}' has unknown non-additive type "
                    f"'{self.nonadditive}'. Must be 'none', 'all', 'any', or 'time'."
                )

        # Validate and set default hierarchy
        default_name = self.default_hierarchy_name or "default"
        if default_name in self._hierarchy_dict:
            object.__setattr__(
                self, "_default_hierarchy", self._hierarchy_dict[default_name]
            )
            object.__setattr__(self, "default_hierarchy_name", default_name)
        elif self._hierarchy_dict:
            # Use first hierarchy if default not found
            first_hier = list(self._hierarchy_dict.values())[0]
            object.__setattr__(self, "_default_hierarchy", first_hier)
            object.__setattr__(self, "default_hierarchy_name", first_hier.name)

    def __eq__(self, other: object) -> bool:
        """Check equality with another Dimension.

        Two dimensions are equal if they have the same name, role, label,
        description, cardinality, category, default_hierarchy_name, levels,
        and hierarchies.

        Args:
            other: Another object to compare with.

        Returns:
            True if equal, False otherwise.
        """
        if other is None or not isinstance(other, Dimension):
            return False

        return (
            self.name == other.name
            and self.role == other.role
            and self.label == other.label
            and self.description == other.description
            and self.cardinality == other.cardinality
            and self.category == other.category
            and self.default_hierarchy_name == other.default_hierarchy_name
            and self._level_dict == other._level_dict
            and self._hierarchy_dict == other._hierarchy_dict
        )

    def __hash__(self) -> int:
        """Return hash based on name for use in sets and dicts."""
        return hash(self.name)

    def __str__(self) -> str:
        """Return string representation (name)."""
        return self.name

    def __repr__(self) -> str:
        """Return detailed representation."""
        return f"<dimension: {{name: '{self.name}', levels: {list(self._level_dict.keys())}}}>"

    @property
    def level_names(self) -> list[str]:
        """Get list of level names.

        Order is not guaranteed; use a hierarchy for known ordering.

        Returns:
            List of level name strings.
        """
        return list(self._level_dict.keys())

    @property
    def is_flat(self) -> bool:
        """Check if dimension has only one level.

        Returns:
            True if dimension has only one level, False otherwise.
        """
        if self.master:
            return self.master.is_flat
        return len(self._level_dict) == 1

    @property
    def has_details(self) -> bool:
        """Check if any level has more than one attribute.

        Returns:
            True if any level has details (multiple attributes).
        """
        if self.master:
            return self.master.has_details
        return any(level.has_details for level in self._level_dict.values())

    @property
    def key_attributes(self) -> list[Attribute]:
        """Return all dimension key attributes.

        Order is not guaranteed; use a hierarchy for known ordering.

        Returns:
            List of key Attribute objects from all levels.
        """
        # After Level.__post_init__, key is always an Attribute
        return [level.key for level in self._level_dict.values()]  # type: ignore[misc]

    @property
    def attributes(self) -> list[Attribute]:
        """Return all dimension attributes.

        Order is not guaranteed; use a hierarchy for known ordering.
        Order of attributes within level is preserved.

        Returns:
            List of all Attribute objects.
        """
        return list(self._attributes.values())

    def level(self, obj: Union[str, Level]) -> Level:
        """Get level by name or as Level object.

        This method coalesces values, accepting either a level name string
        or a Level object.

        Args:
            obj: Level name (string) or Level object.

        Returns:
            The Level object.

        Raises:
            KeyError: If level name is not found.
            ValueError: If obj is neither string nor Level.
        """
        if isinstance(obj, str):
            if obj not in self._level_dict:
                raise KeyError(f"No level {obj} in dimension {self.name}")
            return self._level_dict[obj]
        elif isinstance(obj, Level):
            return obj
        else:
            raise ValueError(
                f"Unknown level object {obj} (should be a string or Level)"
            )

    def hierarchy(self, obj: Optional[Union[str, Hierarchy]] = None) -> Hierarchy:
        """Get hierarchy by name or as Hierarchy object.

        If obj is None, returns the default hierarchy.

        Args:
            obj: Hierarchy name (string), Hierarchy object, or None for default.

        Returns:
            The Hierarchy object.

        Raises:
            ModelError: If hierarchy name is not found.
            ValueError: If obj is neither string, Hierarchy, nor None.
        """
        if obj is None:
            if self._default_hierarchy is None:
                raise ModelError(
                    f"No default hierarchy set for dimension '{self.name}'"
                )
            return self._default_hierarchy
        if isinstance(obj, str):
            if obj not in self._hierarchy_dict:
                raise ModelError(f"No hierarchy {obj} in dimension {self.name}")
            return self._hierarchy_dict[obj]
        elif isinstance(obj, Hierarchy):
            return obj
        else:
            raise ValueError(
                f"Unknown hierarchy object {obj} (should be a string or Hierarchy)"
            )

    def attribute(self, name: str, by_ref: bool = False) -> Attribute:
        """Get dimension attribute.

        Args:
            name: Attribute name (default) or attribute reference if by_ref=True.
            by_ref: If True, look up by attribute reference (e.g., "dim.attr").

        Returns:
            The Attribute object.

        Raises:
            KeyError: If attribute is not found.
        """
        if by_ref:
            if name not in self._attributes_by_ref:
                raise KeyError(f"Unknown attribute '{name}' in dimension '{self.name}'")
            return self._attributes_by_ref[name]
        else:
            if name not in self._attributes:
                raise KeyError(f"Unknown attribute '{name}' in dimension '{self.name}'")
            return self._attributes[name]

    def clone(
        self,
        hierarchies: Optional[list[str]] = None,
        exclude_hierarchies: Optional[list[str]] = None,
        nonadditive: Optional[str] = None,
        default_hierarchy_name: Optional[str] = None,
        cardinality: Optional[str] = None,
        alias: Optional[str] = None,
    ) -> "Dimension":
        """Create a clone of this dimension with modifications.

        The clone has this dimension set as its master. Level attributes
        are unowned (dimension set to None) in the clone.

        Args:
            hierarchies: Limit to only these hierarchy names. If empty list,
                raises error.
            exclude_hierarchies: Exclude these hierarchy names.
            nonadditive: Override nonadditive value.
            default_hierarchy_name: Override default hierarchy name.
            cardinality: Override cardinality.
            alias: New name for the cloned dimension.

        Returns:
            A new Dimension instance.

        Raises:
            ModelError: If trying to remove all hierarchies.
        """
        if hierarchies == []:
            raise ModelError(
                f"Cannot remove all hierarchies from dimension '{self.name}'"
            )

        # Determine which hierarchies to include
        if hierarchies:
            linked = [self.hierarchy(name) for name in hierarchies]
        elif exclude_hierarchies:
            linked = [
                hier
                for hier in self._hierarchy_dict.values()
                if hier.name not in exclude_hierarchies
            ]
        else:
            linked = list(self._hierarchy_dict.values())

        if not linked:
            raise ModelError(f"No hierarchies to clone in dimension '{self.name}'")

        # Deep copy hierarchies
        copied_hierarchies = [copy.deepcopy(hier) for hier in linked]

        # Get only levels used in the hierarchies
        levels: list[Level] = []
        seen: set[str] = set()
        for hier in copied_hierarchies:
            for level in hier.levels:
                if level.name not in seen:
                    levels.append(level)
                    seen.add(level.name)

        # Unown level attributes (deepcopy already disowns them)
        # No additional work needed as deepcopy handles this

        # Determine default hierarchy name
        if not default_hierarchy_name:
            if any(
                hier.name == self.default_hierarchy_name for hier in copied_hierarchies
            ):
                default_hierarchy_name = self.default_hierarchy_name
            else:
                default_hierarchy_name = copied_hierarchies[0].name

        # Determine new name
        new_name = alias or self.name

        return Dimension(
            name=new_name,
            levels=levels,
            hierarchies=copied_hierarchies,
            default_hierarchy_name=default_hierarchy_name,
            label=self.label,
            description=self.description,
            info=copy.deepcopy(self.info),
            role=self.role,
            cardinality=cardinality or self.cardinality,  # type: ignore[arg-type]
            category=self.category,
            master=self,
            nonadditive=nonadditive or self.nonadditive,  # type: ignore[arg-type]
        )

    def to_dict(
        self,
        create_label: bool = False,
        hierarchy_limits: Optional[dict[str, Optional[str]]] = None,
    ) -> dict[str, Any]:
        """Convert to a dictionary representation.

        Args:
            create_label: If True and no label is set, auto-generate from name.
            hierarchy_limits: Dict mapping hierarchy names to level names.
                If level is None, the hierarchy is excluded. Otherwise,
                the hierarchy is truncated at that level.

        Returns:
            Dictionary with all dimension properties.
        """
        result: dict[str, Any] = {
            "name": self.name,
            "label": self.label
            if self.label
            else (to_label(self.name) if create_label else None),
            "description": self.description,
            "info": dict(self.info),
            "default_hierarchy_name": self.hierarchy().name,
            "role": self.role,
            "cardinality": self.cardinality,
            "category": self.category,
        }

        # Add levels
        result["levels"] = [
            level.to_dict(create_label=create_label) for level in self.levels
        ]

        # Add hierarchies with optional depth restrictions
        hierarchy_limits = hierarchy_limits or {}
        hierarchies_out: list[dict[str, Any]] = []

        for name, hier in self._hierarchy_dict.items():
            if name in hierarchy_limits:
                level_limit = hierarchy_limits[name]
                if level_limit is None:
                    # Skip this hierarchy entirely
                    continue
                # Truncate hierarchy at specified level
                depth = hier.level_index(level_limit) + 1
                hier_dict = hier.to_dict(create_label=create_label)
                hier_dict["levels"] = hier_dict["levels"][:depth]
                hierarchies_out.append(hier_dict)
            else:
                hierarchies_out.append(hier.to_dict(create_label=create_label))

        result["hierarchies"] = hierarchies_out

        # Add derived properties for convenience (read-only)
        result["is_flat"] = self.is_flat
        result["has_details"] = self.has_details

        return result

    @classmethod
    def from_schema(
        cls, schema: Any, templates: Optional[dict[str, "Dimension"]] = None
    ) -> "Dimension":
        """Create a Dimension instance from a validated DimensionSchema.

        This factory method handles template inheritance and creates a
        fully resolved Dimension from a Pydantic schema.

        Args:
            schema: A validated DimensionSchema instance.
            templates: Optional dict of template dimensions by name.

        Returns:
            A new Dimension instance.

        Raises:
            ModelError: If template is specified but not found.
        """
        from cubes.metadata.schemas.dimension import (
            DimensionSchema,
            HierarchySchema,
            LevelSchema,
        )

        templates = templates or {}
        template: Optional["Dimension"] = None

        # Handle template inheritance
        if hasattr(schema, "template") and schema.template:
            template_name = schema.template
            if template_name not in templates:
                raise ModelError(f"Template dimension '{template_name}' not found")
            template = templates[template_name]

            # Deep copy template's levels
            levels = [copy.deepcopy(level) for level in template.levels]

            # Create copy of template's hierarchies with references to new levels
            hierarchies: list[Hierarchy] = []
            level_dict = {level.name: level for level in levels}

            for hier in template._hierarchy_dict.values():
                hier_levels = [level_dict[level.name] for level in hier.levels]
                hier_copy = Hierarchy(
                    name=hier.name,
                    levels=hier_levels,
                    label=hier.label,
                    info=copy.deepcopy(hier.info),
                    description=hier.description,
                )
                hierarchies.append(hier_copy)

            # Inherit other properties from template
            default_hierarchy_name = template.default_hierarchy_name
            label = template.label
            description = template.description
            info = dict(template.info)
            cardinality = template.cardinality
            role = template.role
            category = template.category
            nonadditive = template.nonadditive
        else:
            levels = []
            hierarchies = []
            default_hierarchy_name = None
            label = None
            description = None
            info = {}
            cardinality = None
            role = None
            category = None
            nonadditive = None

        # Override with schema values
        if hasattr(schema, "name"):
            name = schema.name
        else:
            name = schema.get("name", "")

        label = getattr(schema, "label", None) or label
        description = getattr(schema, "description", None) or description
        info = {**info, **getattr(schema, "info", {})}
        role = getattr(schema, "role", None) or role
        category = getattr(schema, "category", None) or category
        cardinality = getattr(schema, "cardinality", None) or cardinality
        nonadditive = getattr(schema, "nonadditive", None) or nonadditive

        # Process levels from schema
        if hasattr(schema, "levels") and schema.levels:
            # If explicit levels are provided, they replace template levels
            schema_levels: list[Level] = []
            for level_input in schema.levels:
                if isinstance(level_input, str):
                    # Level name reference - inherit from template
                    if (
                        not templates
                        or not hasattr(schema, "template")
                        or not schema.template
                        or template is None
                    ):
                        raise ModelError(
                            f"Cannot specify level name '{level_input}' without template"
                        )
                    level = template.level(level_input)
                    schema_levels.append(copy.deepcopy(level))
                elif isinstance(level_input, LevelSchema):
                    # Create from LevelSchema
                    level = Level.from_schema(level_input)
                    # Merge with template level info if available
                    if hasattr(schema, "template") and schema.template and template:
                        try:
                            template_level = template.level(level.name)
                            merged_info = copy.deepcopy(template_level.info)
                            merged_info.update(level.info)
                            object.__setattr__(level, "info", merged_info)
                        except KeyError:
                            pass
                    schema_levels.append(level)
                elif isinstance(level_input, dict):
                    # Create from dict
                    level_schema = LevelSchema.model_validate(level_input)
                    level = Level.from_schema(level_schema)
                    schema_levels.append(level)
            # Replace template levels with schema levels
            levels = schema_levels

        # Process hierarchies from schema
        if hasattr(schema, "hierarchies") and schema.hierarchies:
            new_hierarchies: list[Hierarchy] = []
            for hier_input in schema.hierarchies:
                if isinstance(hier_input, str):
                    # Hierarchy name reference - inherit from template
                    if (
                        not hasattr(schema, "template")
                        or not schema.template
                        or template is None
                    ):
                        raise ModelError(
                            f"Cannot specify hierarchy name '{hier_input}' without template"
                        )
                    hier = template.hierarchy(hier_input)
                    new_hierarchies.append(copy.deepcopy(hier))
                elif isinstance(hier_input, HierarchySchema):
                    # Create from HierarchySchema
                    hier = Hierarchy.from_schema(hier_input)
                    # Resolve level references to actual level objects
                    resolved_levels: list[Level] = []
                    level_dict = {level.name: level for level in levels}
                    for level in hier.levels:
                        if level.name in level_dict:
                            resolved_levels.append(level_dict[level.name])
                        else:
                            raise ModelError(
                                f"Hierarchy '{hier.name}' references unknown level "
                                f"'{level.name}'"
                            )
                    object.__setattr__(hier, "levels", resolved_levels)
                    new_hierarchies.append(hier)
                elif isinstance(hier_input, dict):
                    # Create from dict
                    hier_schema = HierarchySchema.model_validate(hier_input)
                    hier = Hierarchy.from_schema(hier_schema)
                    # Resolve level references
                    resolved_levels = []
                    level_dict = {level.name: level for level in levels}
                    for level in hier.levels:
                        if level.name in level_dict:
                            resolved_levels.append(level_dict[level.name])
                        else:
                            raise ModelError(
                                f"Hierarchy '{hier.name}' references unknown level "
                                f"'{level.name}'"
                            )
                    object.__setattr__(hier, "levels", resolved_levels)
                    new_hierarchies.append(hier)
            hierarchies = new_hierarchies
        else:
            # Filter hierarchies to only those using local levels
            level_names = {level.name for level in levels}
            hierarchies = [
                hier
                for hier in hierarchies
                if all(level.name in level_names for level in hier.levels)
            ]

        # Override default hierarchy name from schema
        if hasattr(schema, "default_hierarchy_name") and schema.default_hierarchy_name:
            default_hierarchy_name = schema.default_hierarchy_name

        return cls(
            name=name,
            levels=levels,
            hierarchies=hierarchies if hierarchies else None,
            default_hierarchy_name=default_hierarchy_name,
            label=label,
            description=description,
            info=info,
            role=role,
            cardinality=cardinality,
            category=category,
            nonadditive=nonadditive,
        )
