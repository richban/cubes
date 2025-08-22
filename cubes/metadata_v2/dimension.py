"""
Pydantic-based dimension classes for Cubes metadata V2.

This module provides a modern, Pydantic-based implementation of a Dimension,
focusing on a balance of declarative validation and clear, self-contained
business logic using Pydantic validators.
"""

import copy

from pydantic import Field, computed_field, field_validator, model_validator

from ..errors import ArgumentError, HierarchyError, ModelError, NoSuchAttributeError
from .attributes import Attribute
from .base import MetadataObject


class Level(MetadataObject):
    """Represents a hierarchy level, containing attributes that define it."""

    attributes: list[Attribute] = Field(default_factory=list, min_length=1)
    key: str | None = None
    label_attribute: str | None = None
    order_attribute: str | None = None
    order: str | None = None
    cardinality: str | None = None
    role: str | None = None
    nonadditive: str | None = None

    @model_validator(mode="after")
    def validate_attributes_exist(self):
        """Validate that key and label attributes exist in the attributes list."""
        attr_names = [attr.name for attr in self.attributes]

        if self.key and self.key not in attr_names:
            raise ModelError(
                f"Key attribute '{self.key}' not found in level '{self.name}' attributes"
            )

        if self.label_attribute and self.label_attribute not in attr_names:
            raise ModelError(
                f"Label attribute '{self.label_attribute}' not found in level '{self.name}' attributes"
            )

        if self.order_attribute and self.order_attribute not in attr_names:
            raise ModelError(
                f"Order attribute '{self.order_attribute}' not found in level '{self.name}' attributes"
            )

        # Validate nonadditive values
        if self.nonadditive and self.nonadditive not in [None, "time", "all"]:
            raise ModelError(
                f"Invalid nonadditive value '{self.nonadditive}' for level '{self.name}'"
            )

        return self

    @field_validator("attributes", mode="before")
    @classmethod
    def convert_string_attributes(cls, v, info):
        """Convert string attributes to Attribute objects."""
        if not v:
            name = info.data.get("name")
            return [Attribute(name=name)] if name else []

        return [
            Attribute.model_validate(attr)
            if isinstance(attr, dict)
            else Attribute(name=attr)
            if isinstance(attr, str)
            else attr
            for attr in v
        ]

    @model_validator(mode="after")
    def ensure_attributes_exist(self):
        """Ensure attributes exist, creating default from name if needed."""
        if not self.attributes and self.name:
            self.attributes = [Attribute(name=self.name)]
        elif not self.attributes:
            raise ValueError("Level must have at least one attribute or a name")
        return self

    @model_validator(mode="after")
    def set_attribute_defaults(self):
        """Set default key and label_attribute after validation."""
        if not self.key:
            self.key = self.attributes[0].name

        if not self.label_attribute:
            if len(self.attributes) > 1:
                self.label_attribute = self.attributes[1].name
            else:
                self.label_attribute = self.key

        return self

    def attribute(self, name: str) -> Attribute:
        """Get attribute by name."""
        for attr in self.attributes:
            if attr.name == name:
                return attr
        raise NoSuchAttributeError(
            f"Attribute '{name}' not found in level '{self.name}'"
        )

    @computed_field
    @property
    def has_details(self) -> bool:
        """Returns True when level has more than one attribute."""
        return len(self.attributes) > 1

    @computed_field
    @property
    def key_attribute(self) -> Attribute:
        """Get the key attribute object."""
        return self.attribute(self.key)

    def __deepcopy__(self, memo):
        """Deep copy implementation."""
        return Level(
            name=self.name,
            attributes=copy.deepcopy(self.attributes, memo),
            key=self.key,
            label_attribute=self.label_attribute,
            order_attribute=self.order_attribute,
            order=self.order,
            cardinality=self.cardinality,
            role=self.role,
            nonadditive=self.nonadditive,
            label=self.label,
            description=self.description,
            info=copy.deepcopy(self.info, memo),
        )

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"<Level(name='{self.name}', attributes={len(self.attributes)})>"


class Hierarchy(MetadataObject):
    """Defines an ordered arrangement of levels within a dimension."""

    levels: list[Level] = Field(min_length=1)

    # Use Pydantic private attributes for the cache
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data):
        super().__init__(**data)
        self._level_index_cache: dict[str, int] = {}

    @model_validator(mode="after")
    def validate_hierarchy(self):
        """Validate hierarchy structure."""
        if not self.levels:
            raise ModelError(f"Hierarchy '{self.name}' must have at least one level")

        # Check for duplicate level names and build cache
        level_names = []
        self._level_index_cache = {}

        for i, level in enumerate(self.levels):
            if level.name in level_names:
                raise ModelError(f"Hierarchy '{self.name}' has duplicate level names")
            level_names.append(level.name)
            self._level_index_cache[level.name] = i

        return self

    @field_validator("levels", mode="before")
    @classmethod
    def convert_string_levels(cls, v):
        """Convert string levels to Level objects."""
        if not v:
            return []

        result = []
        for level in v:
            if isinstance(level, str):
                result.append(Level(name=level))
            elif isinstance(level, Level):
                result.append(level)
            elif isinstance(level, dict):
                # Let Pydantic handle dict validation to Level
                result.append(Level.model_validate(level))
            else:
                raise ValueError(
                    f"Levels must be strings, dicts, or Level objects, got {type(level)}"
                )
        return result

    def __str__(self) -> str:
        return self.name

    def __len__(self) -> int:
        return len(self.levels)

    def __getitem__(self, item: int | str) -> Level:
        if isinstance(item, int):
            try:
                return self.levels[item]
            except IndexError as err:
                raise HierarchyError(
                    f"Hierarchy '{self.name}' has only {len(self.levels)} levels, "
                    f"asking for level at index {item}"
                ) from err
        elif isinstance(item, str):
            for level in self.levels:
                if level.name == item:
                    return level
            raise HierarchyError(f"Level '{item}' not found in hierarchy '{self.name}'")
        else:
            raise TypeError("Hierarchy item must be an integer index or a string name.")

    def __contains__(self, item: str | Level) -> bool:
        if isinstance(item, Level):
            return item in self.levels
        elif isinstance(item, str):
            return any(level.name == item for level in self.levels)
        return False

    @computed_field
    @property
    def level_names(self) -> list[str]:
        """Get list of level names in hierarchy order."""
        return [level.name for level in self.levels]

    def level(self, obj: str | Level) -> Level:
        """Get level by name or as Level object."""
        if isinstance(obj, str):
            for level in self.levels:
                if level.name == obj:
                    return level
            raise HierarchyError(f"No level '{obj}' in hierarchy '{self.name}'")
        elif isinstance(obj, Level):
            if obj not in self.levels:
                raise HierarchyError(
                    f"Level object '{obj.name}' not found in hierarchy '{self.name}'"
                )
            return obj
        else:
            raise TypeError("Level object must be a string or Level instance.")

    def level_index(self, level: str | Level) -> int:
        """Get order index of level. Can be used for ordering and comparing
        levels within hierarchy."""
        if isinstance(level, str):
            for i, lv in enumerate(self.levels):
                if lv.name == level:
                    return i
            raise HierarchyError(f"No level '{level}' in hierarchy '{self.name}'")
        elif isinstance(level, Level):
            if level not in self.levels:
                raise HierarchyError(
                    f"Level object '{level.name}' not found in hierarchy '{self.name}'"
                )
            return self.levels.index(level)
        else:
            raise TypeError("Level object must be a string or Level instance.")

    def next_level(self, level: str | Level | None) -> Level | None:
        """Returns next level in hierarchy after `level`. If `level` is last
        level, returns ``None``. If `level` is ``None``, then the first level
        is returned."""
        if level is None:
            return self.levels[0] if self.levels else None

        try:
            index = self.level_index(level)
            if index + 1 < len(self.levels):
                return self.levels[index + 1]
            else:
                return None
        except HierarchyError:
            return None  # Level not in hierarchy

    def previous_level(self, level: str | Level | None) -> Level | None:
        """Returns previous level in hierarchy after `level`. If `level` is
        first level or ``None``, returns ``None``"""
        if level is None:
            return None

        try:
            index = self.level_index(level)
            if index > 0:
                return self.levels[index - 1]
            else:
                return None
        except HierarchyError:
            return None  # Level not in hierarchy

    def is_last(self, level: str | Level) -> bool:
        """Returns `True` if `level` is last level of the hierarchy."""
        level_obj = self.level(level)
        return level_obj == self.levels[-1]

    def levels_for_depth(self, depth: int, drilldown: bool = False) -> list[Level]:
        """Returns levels for given `depth`. If `depth` is longer than
        hierarchy levels, `cubes.HierarchyError` exception is raised"""
        if depth < 0:
            raise ArgumentError("Depth cannot be negative.")

        extend = 1 if drilldown else 0

        if depth + extend > len(self.levels):
            raise HierarchyError(
                f"Depth {depth} is longer than hierarchy levels {self.level_names} (drilldown: {drilldown})"
            )

        return self.levels[0 : depth + extend]

    def levels_for_path(self, path: list[str], drilldown: bool = False) -> list[Level]:
        """Returns levels for given path. If path is longer than hierarchy
        levels, `cubes.ArgumentError` exception is raised"""
        depth = len(path)
        return self.levels_for_depth(depth, drilldown)

    def rollup(self, path: list[str], level: str | Level | None = None) -> list[str]:
        """Rolls-up the path to the `level`. If `level` is ``None`` then path
        is rolled-up only one level.

        If `level` is deeper than last level of `path` the
        `cubes.HierarchyError` exception is raised. If `level` is the same as
        `path` level, nothing happens."""
        if level:
            target_index = self.level_index(level)
            if target_index + 1 > len(path):
                raise HierarchyError(
                    f"Can not roll-up: level '{level}' is deeper than deepest element of path {path}"
                )
            return path[0 : target_index + 1]
        else:
            if not path:
                return []
            return path[0 : len(path) - 1]

    def path_is_base(self, path: list[str]) -> bool:
        """Returns True if path is base path for the hierarchy. Base path is a
        path where there are no more levels to be added - no drill down
        possible."""
        return path is not None and len(path) == len(self.levels)

    @computed_field
    @property
    def key_attributes(self) -> list[Attribute]:
        """Return key attributes for all levels in hierarchy order."""
        return [level.key_attribute for level in self.levels]

    @computed_field
    @property
    def all_attributes(self) -> list[Attribute]:
        """Return all dimension attributes as a single list."""
        attributes = []
        for level in self.levels:
            attributes.extend(level.attributes)
        return attributes

    def keys(self, depth: int | None = None) -> list[str]:
        """Return names of keys for all levels in the hierarchy to `depth`.
        If `depth` is `None` then all levels are returned."""
        if depth is not None:
            levels = self.levels[0:depth]
        else:
            levels = self.levels
        return [level.key for level in levels]

    def __hash__(self):
        return hash(self.name)

    def __deepcopy__(self, memo):
        """Deep copy implementation."""
        return Hierarchy(
            name=self.name,
            levels=copy.deepcopy(self.levels, memo),
            label=self.label,
            description=self.description,
            info=copy.deepcopy(self.info, memo),
        )


class Dimension(MetadataObject):
    """Represents a cube dimension, which defines its own intrinsic hierarchy."""

    levels: list[Level] = Field(default_factory=list, min_length=1)
    hierarchies: list[Hierarchy] = Field(default_factory=list)
    default_hierarchy_name: str | None = None
    role: str | None = None
    cardinality: str | None = None
    category: str | None = None
    nonadditive: str | None = None

    @field_validator("levels", mode="before")
    @classmethod
    def convert_levels_input(cls, v):
        """Convert level inputs to Level objects."""
        if not v:
            return []

        if isinstance(v, list):
            result = []
            for level in v:
                if isinstance(level, str):
                    result.append(Level(name=level))
                elif isinstance(level, Level):
                    result.append(level)
                elif isinstance(level, dict):
                    result.append(Level.model_validate(level))
                else:
                    raise ValueError(
                        f"Levels must be strings, dicts, or Level objects, got {type(level)}"
                    )
            return result

        raise ValueError(f"Levels must be a list, got {type(v)}")

    @field_validator("hierarchies", mode="before")
    @classmethod
    def convert_hierarchies_input(cls, v):
        """Convert hierarchy inputs to Hierarchy objects."""
        if not v:
            return []

        # Handle modern format: [{"name": "...", "levels": [...]}]
        if isinstance(v, list):
            result = []
            for hierarchy in v:
                if isinstance(hierarchy, str):
                    result.append(Hierarchy(name=hierarchy))
                elif isinstance(hierarchy, Hierarchy):
                    result.append(hierarchy)
                elif isinstance(hierarchy, dict):
                    result.append(Hierarchy.model_validate(hierarchy))
                else:
                    raise ValueError(
                        f"Hierarchies must be strings, dicts, or Hierarchy objects, got {type(hierarchy)}"
                    )
            return result

        raise ValueError(f"Hierarchies must be a list, got {type(v)}")

    @property
    def is_flat(self) -> bool:
        """Returns True if the dimension has only one level."""
        return len(self.levels) == 1

    @model_validator(mode="after")
    def validate_dimension(self):
        """Validate dimension structure and relationships."""
        # Name is required for dimensions
        if not self.name:
            raise ValueError("Dimension name is required")

        # Create default level if none exist
        if not self.levels:
            default_level = Level(name=self.name)
            self.levels = [default_level]

        # Create default hierarchy if none exist
        if not self.hierarchies and self.levels:
            default_hierarchy = Hierarchy(name="default", levels=self.levels)
            self.hierarchies = [default_hierarchy]

        # Set default hierarchy name if not provided
        if not self.default_hierarchy_name and self.hierarchies:
            self.default_hierarchy_name = self.hierarchies[0].name
        elif not self.default_hierarchy_name:
            self.default_hierarchy_name = "default"

        # Set ownership of attributes
        for level in self.levels:
            for attr in level.attributes:
                attr.dimension = self

        # Validate that all hierarchy level references exist in dimension levels
        dimension_level_names = set(level.name for level in self.levels)
        for hierarchy in self.hierarchies:
            for hierarchy_level in hierarchy.levels:
                if hierarchy_level.name not in dimension_level_names:
                    raise ModelError(
                        f"Hierarchy '{hierarchy.name}' references level '{hierarchy_level.name}' "
                        f"which is not defined in dimension '{self.name}'"
                    )

        # Ensure default hierarchy exists if specified
        if self.default_hierarchy_name:
            hierarchy_names = [h.name for h in self.hierarchies]
            if self.default_hierarchy_name not in hierarchy_names:
                raise ModelError(
                    f"Default hierarchy '{self.default_hierarchy_name}' not found in dimension '{self.name}'"
                )
        elif self.hierarchies:
            # Set default hierarchy to first one if not specified
            self.default_hierarchy_name = self.hierarchies[0].name

        # Validate nonadditive values
        if self.nonadditive and self.nonadditive not in [None, "time", "all"]:
            raise ModelError(
                f"Invalid nonadditive value '{self.nonadditive}' for dimension '{self.name}'"
            )

        # Check for duplicate level names
        level_names = [level.name for level in self.levels]
        if len(level_names) != len(set(level_names)):
            raise ModelError(f"Dimension '{self.name}' has duplicate level names")

        # Check for duplicate hierarchy names
        hierarchy_names = [h.name for h in self.hierarchies]
        if len(hierarchy_names) != len(set(hierarchy_names)):
            raise ModelError(f"Dimension '{self.name}' has duplicate hierarchy names")

        return self

    def level(self, obj: str | Level) -> Level:
        """Get level by name or as Level object."""
        if isinstance(obj, str):
            for level in self.levels:
                if level.name == obj:
                    return level
            raise ModelError(f"No level '{obj}' in dimension '{self.name}'")
        elif isinstance(obj, Level):
            if obj not in self.levels:
                raise ModelError(
                    f"Level object '{obj.name}' not found in dimension '{self.name}'"
                )
            return obj
        else:
            raise ValueError("Level object must be a string or Level instance")

    def hierarchy(self, name: str | None = None) -> Hierarchy:
        """Gets a hierarchy by name or the default hierarchy."""
        name = name or self.default_hierarchy_name
        for h in self.hierarchies:
            if h.name == name:
                return h
        raise HierarchyError(f"Hierarchy '{name}' not found in dimension '{self.name}'")

    def attribute(self, name: str, by_ref: bool = False) -> Attribute:
        """Get dimension attribute by name or reference."""
        for level in self.levels:
            for attr in level.attributes:
                if by_ref and hasattr(attr, "ref") and attr.ref == name:
                    return attr
                elif not by_ref and attr.name == name:
                    return attr
        raise NoSuchAttributeError(
            f"Unknown attribute '{name}' in dimension '{self.name}'", name
        )

    @computed_field
    @property
    def level_names(self) -> list[str]:
        """Get list of level names. Order is not guaranteed, use a hierarchy for known order."""
        return [level.name for level in self.levels]

    @computed_field
    @property
    def has_details(self) -> bool:
        """Returns True when each level has only one attribute, usually key."""
        return any(level.has_details for level in self.levels)

    @computed_field
    @property
    def key_attributes(self) -> list[Attribute]:
        """Return all dimension key attributes, regardless of hierarchy."""
        return [level.key_attribute for level in self.levels]

    @computed_field
    @property
    def attributes(self) -> list[Attribute]:
        """Return all dimension attributes regardless of hierarchy."""
        attrs = []
        for level in self.levels:
            attrs.extend(level.attributes)
        return attrs

    def clone(
        self,
        hierarchies: list[str] | None = None,
        exclude_hierarchies: list[str] | None = None,
        nonadditive: str | None = None,
        default_hierarchy_name: str | None = None,
        cardinality: str | None = None,
        alias: str | None = None,
        **extra,
    ) -> "Dimension":
        """Returns a clone of the dimension with some modifications."""

        if hierarchies == []:
            raise ModelError(
                f"Cannot remove all hierarchies from dimension '{self.name}'"
            )

        # Filter hierarchies
        if hierarchies:
            filtered_hierarchies = []
            for name in hierarchies:
                h = self.hierarchy(name)
                filtered_hierarchies.append(h)
        elif exclude_hierarchies:
            filtered_hierarchies = []
            for h in self.hierarchies:
                if h.name not in exclude_hierarchies:
                    filtered_hierarchies.append(h)
        else:
            filtered_hierarchies = self.hierarchies.copy()

        if not filtered_hierarchies:
            raise ModelError("No hierarchies to clone")

        # Deep copy hierarchies and collect levels
        cloned_hierarchies = copy.deepcopy(filtered_hierarchies)
        used_levels = set()
        levels = []

        for hier in cloned_hierarchies:
            for level in hier.levels:
                if level.name not in used_levels:
                    levels.append(level)
                    used_levels.add(level.name)

        # Determine default hierarchy name
        if not default_hierarchy_name:
            if self.default_hierarchy_name in [h.name for h in cloned_hierarchies]:
                default_hierarchy_name = self.default_hierarchy_name
            else:
                default_hierarchy_name = cloned_hierarchies[0].name

        name = alias or self.name

        return Dimension(
            name=name,
            levels=levels,
            hierarchies=cloned_hierarchies,
            default_hierarchy_name=default_hierarchy_name,
            role=self.role,
            cardinality=cardinality or self.cardinality,
            category=self.category,
            nonadditive=nonadditive or self.nonadditive,
            label=self.label,
            description=self.description,
            info=copy.deepcopy(self.info),
            **extra,
        )

    def validate(self) -> list[tuple]:
        """Validate dimension structure and return list of validation results.
        Each result is a tuple of (severity, message)."""
        results = []

        if not self.levels:
            results.append(("error", f"No levels in dimension '{self.name}'"))
            return results

        if not self.hierarchies:
            results.append(("error", f"No hierarchies in dimension '{self.name}'"))

        # Validate default hierarchy
        if self.default_hierarchy_name:
            hierarchy_names = [h.name for h in self.hierarchies]
            if self.default_hierarchy_name not in hierarchy_names:
                results.append(
                    (
                        "error",
                        f"Default hierarchy '{self.default_hierarchy_name}' does not exist in dimension '{self.name}'",
                    )
                )

        # Validate levels and attributes
        for level in self.levels:
            if not level.attributes:
                results.append(
                    (
                        "error",
                        f"Level '{level.name}' in dimension '{self.name}' has no attributes",
                    )
                )
                continue

            # Check key attribute exists
            try:
                _ = level.key_attribute
            except NoSuchAttributeError:
                results.append(
                    (
                        "error",
                        f"Key attribute '{level.key}' not found in level '{level.name}' attributes",
                    )
                )

        return results

    def __eq__(self, other: "Dimension"):
        if not isinstance(other, Dimension):
            return False
        return (
            self.name == other.name
            and self.role == other.role
            and self.label == other.label
            and self.description == other.description
            and self.cardinality == other.cardinality
            and self.category == other.category
            and self.default_hierarchy_name == other.default_hierarchy_name
            and self.levels == other.levels
            and self.hierarchies == other.hierarchies
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return f"<Dimension(name='{self.name}', levels={len(self.levels)}, hierarchies={len(self.hierarchies)})>"
