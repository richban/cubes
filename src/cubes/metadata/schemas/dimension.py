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
