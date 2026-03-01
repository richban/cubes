"""Pydantic schemas for dimension metadata.

Provides input validation and parsing for dimension levels, hierarchies,
and dimensions. Handles default value resolution and attribute normalization.
"""

from typing import Any, Optional

from typing import Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cubes.metadata.schemas.attributes import AttributeInput, AttributeSchema


def _coerce_string_to_level_dict(value: Any) -> Any:
    """Coerce string level names to {name: string, attributes: [string]} dictionaries.

    This allows levels to be specified as simple strings like "my_level"
    instead of full dictionaries. The string becomes both the level name
    and its only attribute.

    Args:
        value: Input value, either a string (level name) or dict.

    Returns:
        Dictionary with "name" key and "attributes" list if input was string,
        otherwise original value.
    """
    if isinstance(value, str):
        return {"name": value, "attributes": [value]}
    return value


class LevelSchema(BaseModel):
    """Pydantic schema for validating level metadata.

    This schema handles raw JSON/dict input validation for hierarchy levels
    with automatic default resolution for key, label_attribute, and
    order_attribute fields.

    Attributes:
        name: Level name, used as identifier (required).
        attributes: List of attribute definitions for this level (required).
        key: Name of the key attribute, defaults to first attribute.
        label_attribute: Name of the label attribute, defaults to second
            attribute if available, otherwise first.
        order_attribute: Name of the order attribute, defaults to first attribute.
        order: Default sort order for the level, "asc" or "desc".
        cardinality: Approximate number of members (tiny, low, medium, high).
        role: Role of the level within special dimensions.
        nonadditive: Non-additivity type: "none", "all", or "time".
        info: Custom information dictionary for app-specific metadata.
        description: Longer description of the level.
        label: Human-readable label displayed to users.
    """

    model_config = ConfigDict(
        # Allow extra fields for forward compatibility
        extra="allow",
        # Don't validate assignments to avoid recursion issues with validators
        validate_assignment=False,
        # Don't allow mutation after creation
        frozen=False,
    )

    name: str = Field(..., description="Level name, used as identifier")
    attributes: list[AttributeInput] = Field(
        ..., description="List of attribute definitions for this level"
    )
    key: Optional[str] = Field(
        None, description="Name of the key attribute (defaults to first attribute)"
    )
    label_attribute: Optional[str] = Field(
        None,
        description="Name of the label attribute (defaults to second attribute or first if only one)",
    )
    order_attribute: Optional[str] = Field(
        None, description="Name of the order attribute (defaults to first attribute)"
    )
    order: Optional[str] = Field(None, description="Default sort order: asc or desc")
    cardinality: Optional[str] = Field(
        None, description="Approximate cardinality: tiny, low, medium, high"
    )
    role: Optional[str] = Field(
        None, description="Role of the level within special dimensions"
    )
    nonadditive: Optional[str] = Field(
        None, description="Non-additivity type: none, all, or time"
    )
    info: dict[str, Any] = Field(
        default_factory=dict, description="Custom information dictionary"
    )
    description: Optional[str] = Field(None, description="Longer description")
    label: Optional[str] = Field(None, description="Human-readable label")

    @classmethod
    def model_validate(
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        from_attributes: bool | None = None,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "LevelSchema":
        """Validate and create a LevelSchema instance.

        Supports string inputs by coercing them to {"name": string,
        "attributes": [string]} dicts.

        Args:
            obj: Input object, can be dict, string, or LevelSchema instance.
            strict: Whether to enforce strict validation.
            from_attributes: Whether to extract data from object attributes.
            context: Additional context for validation.
            **kwargs: Additional arguments passed to parent class.

        Returns:
            Validated LevelSchema instance.
        """
        # Coerce string to dict before validation
        if isinstance(obj, str):
            obj = {"name": obj, "attributes": [obj]}
        return super().model_validate(
            obj,
            strict=strict,
            from_attributes=from_attributes,
            context=context,
            **kwargs,
        )

    @model_validator(mode="after")
    def _set_default_attributes(self) -> "LevelSchema":
        """Set default values for key, label_attribute, and order_attribute.

        After validation, ensures that:
        - key defaults to the first attribute's name
        - label_attribute defaults to the second attribute's name (or first if only one)
        - order_attribute defaults to the first attribute's name

        Raises:
            ValueError: If attributes list is empty.
        """
        if not self.attributes:
            raise ValueError(f"Level '{self.name}' must have at least one attribute")

        # Get attribute names for reference
        attr_names = []
        for attr in self.attributes:
            if isinstance(attr, AttributeSchema):
                attr_names.append(attr.name)
            elif isinstance(attr, dict):
                attr_names.append(attr.get("name", ""))
            else:
                attr_names.append(str(attr))

        # Set default key to first attribute
        if self.key is None:
            self.key = attr_names[0]

        # Set default label_attribute:
        # - If more than one attribute, use second attribute
        # - Otherwise use first attribute
        if self.label_attribute is None:
            if len(attr_names) > 1:
                self.label_attribute = attr_names[1]
            else:
                self.label_attribute = attr_names[0]

        # Set default order_attribute to first attribute
        if self.order_attribute is None:
            self.order_attribute = attr_names[0]

        return self

    @model_validator(mode="after")
    def _validate_attribute_references(self) -> "LevelSchema":
        """Validate that key, label_attribute, and order_attribute reference existing attributes.

        Raises:
            ValueError: If any reference points to a non-existent attribute.
        """
        if not self.attributes:
            return self

        # Get valid attribute names
        valid_names = set()
        for attr in self.attributes:
            if isinstance(attr, AttributeSchema):
                valid_names.add(attr.name)
            elif isinstance(attr, dict):
                valid_names.add(attr.get("name", ""))

        # Validate key reference
        if self.key and self.key not in valid_names:
            raise ValueError(
                f"Level '{self.name}' key '{self.key}' does not reference "
                f"an existing attribute. Valid attributes: {sorted(valid_names)}"
            )

        # Validate label_attribute reference
        if self.label_attribute and self.label_attribute not in valid_names:
            raise ValueError(
                f"Level '{self.name}' label_attribute '{self.label_attribute}' "
                f"does not reference an existing attribute. "
                f"Valid attributes: {sorted(valid_names)}"
            )

        # Validate order_attribute reference
        if self.order_attribute and self.order_attribute not in valid_names:
            raise ValueError(
                f"Level '{self.name}' order_attribute '{self.order_attribute}' "
                f"does not reference an existing attribute. "
                f"Valid attributes: {sorted(valid_names)}"
            )

        return self

    @model_validator(mode="after")
    def _normalize_nonadditive(self) -> "LevelSchema":
        """Normalize nonadditive value to standard form.

        Handles normalization:
        - "none" -> None (fully additive)
        - "all", "any" -> "all"
        - "time" -> "time"
        """
        if self.nonadditive is None:
            return self

        lower_value = self.nonadditive.lower()

        if lower_value == "none":
            self.nonadditive = None
        elif lower_value in ["all", "any"]:
            self.nonadditive = "all"
        elif lower_value == "time":
            self.nonadditive = "time"
        else:
            raise ValueError(
                f"Level '{self.name}' has unknown non-additive type "
                f"'{self.nonadditive}'. Must be 'none', 'all', 'any', or 'time'."
            )

        return self

    @model_validator(mode="after")
    def _normalize_order(self) -> "LevelSchema":
        """Normalize order value to lowercase asc/desc.

        Handles case-insensitive input like "ASC", "DESC".
        """
        if self.order is None:
            return self

        lower_value = self.order.lower()

        if lower_value.startswith("asc"):
            self.order = "asc"
        elif lower_value.startswith("desc"):
            self.order = "desc"
        else:
            raise ValueError(
                f"Level '{self.name}' has unknown ordering '{self.order}'. "
                f"Must be 'asc' or 'desc'."
            )

        return self


# Type alias for hierarchy level references
# Can be a level name (string), a full LevelSchema, or a dict representing a level
HierarchyLevelInput = Union[str, "LevelSchema", dict[str, Any]]


class HierarchySchema(BaseModel):
    """Pydantic schema for validating hierarchy metadata.

    This schema handles raw JSON/dict input validation for dimension hierarchies.
    Hierarchies define the ordering of levels for drill-down and roll-up operations.

    Attributes:
        name: Hierarchy name, used as identifier (required).
        levels: Ordered list of level references. Can be level names (strings)
            or full level definitions. Must contain at least one level.
        label: Human-readable label displayed to users.
        info: Custom information dictionary for app-specific metadata.
        description: Longer description of the hierarchy.
    """

    model_config = ConfigDict(
        # Allow extra fields for forward compatibility
        extra="allow",
        # Don't validate assignments to avoid recursion issues with validators
        validate_assignment=False,
        # Don't allow mutation after creation
        frozen=False,
    )

    name: str = Field(..., description="Hierarchy name, used as identifier")
    levels: list[HierarchyLevelInput] = Field(
        ..., description="Ordered list of level references (names or level definitions)"
    )
    label: Optional[str] = Field(None, description="Human-readable label")
    info: dict[str, Any] = Field(
        default_factory=dict, description="Custom information dictionary"
    )
    description: Optional[str] = Field(None, description="Longer description")

    @model_validator(mode="after")
    def _validate_levels_not_empty(self) -> "HierarchySchema":
        """Validate that levels list is not empty.

        Raises:
            ValueError: If levels list is empty.
        """
        if not self.levels:
            raise ValueError(f"Hierarchy '{self.name}' must have at least one level")
        return self

    @model_validator(mode="after")
    def _coerce_level_dicts(self) -> "HierarchySchema":
        """Convert level dicts to LevelSchema instances.

        Processes the levels list and converts any dictionaries to LevelSchema
        instances while preserving string references.
        """
        coerced_levels: list[HierarchyLevelInput] = []
        for level in self.levels:
            if isinstance(level, dict):
                # Convert dict to LevelSchema
                coerced_levels.append(LevelSchema.model_validate(level))
            else:
                # Keep as-is (string or LevelSchema)
                coerced_levels.append(level)
        self.levels = coerced_levels
        return self


# Type alias for dimension level input
# Can be a level name (string) or full level definition
DimensionLevelInput = Union[str, "LevelSchema", dict[str, Any]]

# Type alias for dimension hierarchy input
# Can be a hierarchy name (string) or full hierarchy definition
DimensionHierarchyInput = Union[str, "HierarchySchema", dict[str, Any]]


class DimensionSchema(BaseModel):
    """Pydantic schema for validating dimension metadata.

    This schema handles raw JSON/dict input validation for dimensions including
    template inheritance, level/hierarchy definitions, and default resolution.
    Ensures that dimensions have valid structure with mutually exclusive
    levels vs attributes alternatives.

    Attributes:
        name: Dimension name, used as identifier (required).
        template: Name of a template dimension to inherit from.
        levels: List of level definitions. Mutually exclusive with attributes.
        hierarchies: List of hierarchy definitions. If not provided, a default
            hierarchy is created from all levels.
        default_hierarchy_name: Name of the default hierarchy to use.
        attributes: Alternative to levels for flat dimensions.
        label: Human-readable label displayed to users.
        description: Longer description of the dimension.
        info: Custom information dictionary for app-specific metadata.
        role: Role of the dimension (e.g., "time" for time dimensions).
        cardinality: Approximate number of dimension members.
        category: Logical dimension group for UI organization.
        nonadditive: Non-additivity type: "none", "all", or "time".
    """

    model_config = ConfigDict(
        # Allow extra fields for forward compatibility
        extra="allow",
        # Don't validate assignments to avoid recursion issues with validators
        validate_assignment=False,
        # Don't allow mutation after creation
        frozen=False,
    )

    name: str = Field(..., description="Dimension name, used as identifier")
    template: Optional[str] = Field(
        None, description="Name of template dimension to inherit from"
    )
    levels: Optional[list[DimensionLevelInput]] = Field(
        None,
        description="List of level definitions (mutually exclusive with attributes)",
    )
    hierarchies: Optional[list[DimensionHierarchyInput]] = Field(
        None, description="List of hierarchy definitions"
    )
    default_hierarchy_name: Optional[str] = Field(
        None, description="Name of the default hierarchy"
    )
    attributes: Optional[list[AttributeInput]] = Field(
        None, description="Alternative to levels for flat dimensions"
    )
    label: Optional[str] = Field(None, description="Human-readable label")
    description: Optional[str] = Field(None, description="Longer description")
    info: dict[str, Any] = Field(
        default_factory=dict, description="Custom information dictionary"
    )
    role: Optional[str] = Field(None, description="Role of dimension (e.g., 'time')")
    cardinality: Optional[str] = Field(
        None, description="Approximate cardinality: tiny, low, medium, high"
    )
    category: Optional[str] = Field(None, description="Logical dimension group")
    nonadditive: Optional[str] = Field(
        None, description="Non-additivity type: none, all, or time"
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_dimension_string(cls, data: Any) -> Any:
        """Coerce string dimension names to {"name": string} dicts.

        This allows dimensions to be specified as simple strings like "my_dim"
        which will create a dimension with that name and a single level with
        the same name.

        Args:
            data: Input data, can be dict or string.

        Returns:
            Dictionary with name and default level.
        """
        if isinstance(data, str):
            return {"name": data, "levels": [data]}
        return data

    @model_validator(mode="after")
    def _validate_mutual_exclusion(self) -> "DimensionSchema":
        """Validate that levels and attributes are mutually exclusive.

        Raises:
            ValueError: If both levels and attributes are provided.
        """
        if self.levels is not None and self.attributes is not None:
            raise ValueError(
                f"Dimension '{self.name}' cannot have both 'levels' and "
                f"'attributes' fields. Use one or the other."
            )
        return self

    @model_validator(mode="after")
    def _coerce_levels(self) -> "DimensionSchema":
        """Convert level strings and dicts to LevelSchema instances.

        Processes the levels list and converts any strings or dictionaries
        to LevelSchema instances.
        """
        if self.levels is None:
            return self

        coerced_levels: list[DimensionLevelInput] = []
        for level in self.levels:
            if isinstance(level, str):
                # Convert string to LevelSchema
                coerced_levels.append(LevelSchema.model_validate(level))
            elif isinstance(level, dict):
                # Convert dict to LevelSchema
                coerced_levels.append(LevelSchema.model_validate(level))
            else:
                # Keep as-is (already LevelSchema)
                coerced_levels.append(level)
        self.levels = coerced_levels
        return self

    @model_validator(mode="after")
    def _coerce_hierarchies(self) -> "DimensionSchema":
        """Convert hierarchy strings and dicts to HierarchySchema instances.

        Processes the hierarchies list and converts any dictionaries
        to HierarchySchema instances while preserving string references.
        """
        if self.hierarchies is None:
            return self

        coerced_hierarchies: list[DimensionHierarchyInput] = []
        for hier in self.hierarchies:
            if isinstance(hier, dict):
                # Convert dict to HierarchySchema
                coerced_hierarchies.append(HierarchySchema.model_validate(hier))
            else:
                # Keep as-is (string or HierarchySchema)
                coerced_hierarchies.append(hier)
        self.hierarchies = coerced_hierarchies
        return self

    @model_validator(mode="after")
    def _normalize_nonadditive(self) -> "DimensionSchema":
        """Normalize nonadditive value to standard form.

        Handles normalization:
        - "none" -> None (fully additive)
        - "all", "any" -> "all"
        - "time" -> "time"
        """
        if self.nonadditive is None:
            return self

        lower_value = self.nonadditive.lower()

        if lower_value == "none":
            self.nonadditive = None
        elif lower_value in ["all", "any"]:
            self.nonadditive = "all"
        elif lower_value == "time":
            self.nonadditive = "time"
        else:
            raise ValueError(
                f"Dimension '{self.name}' has unknown non-additive type "
                f"'{self.nonadditive}'. Must be 'none', 'all', 'any', or 'time'."
            )

        return self

    @model_validator(mode="after")
    def _create_default_hierarchy(self) -> "DimensionSchema":
        """Create default hierarchy if no hierarchies provided.

        If hierarchies list is empty or None, create a single default
        hierarchy containing all levels.
        """
        if self.hierarchies is not None and self.hierarchies:
            return self

        # Only create default if we have levels
        if self.levels:
            # Extract level names
            level_names: list[Any] = []
            for level in self.levels:
                if isinstance(level, LevelSchema):
                    level_names.append(level.name)
                elif isinstance(level, dict):
                    level_names.append(level.get("name", ""))
                else:
                    level_names.append(str(level))

            if level_names:
                default_hier = HierarchySchema(
                    name="default",
                    levels=level_names,
                    label=None,
                    info={},
                    description=None,
                )
                # Use object.__setattr__ to bypass frozen check
                object.__setattr__(self, "hierarchies", [default_hier])

        return self

    @model_validator(mode="after")
    def _set_default_hierarchy_name(self) -> "DimensionSchema":
        """Set default hierarchy name if not provided.

        If no default_hierarchy_name is set, use "default" if a default
        hierarchy exists, otherwise use the name of the first hierarchy.
        """
        if self.default_hierarchy_name is not None:
            return self

        if self.hierarchies:
            # Check for hierarchy named "default"
            for hier in self.hierarchies:
                if isinstance(hier, HierarchySchema) and hier.name == "default":
                    self.default_hierarchy_name = "default"
                    return self
                elif isinstance(hier, str) and hier == "default":
                    self.default_hierarchy_name = "default"
                    return self

            # Use first hierarchy name
            first_hier = self.hierarchies[0]
            if isinstance(first_hier, HierarchySchema):
                self.default_hierarchy_name = first_hier.name
            else:
                self.default_hierarchy_name = str(first_hier)

        return self
