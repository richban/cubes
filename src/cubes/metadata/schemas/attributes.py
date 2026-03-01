"""Pydantic schemas for attribute definitions.

Provides input validation and parsing for dimension attributes, measures,
and measure aggregates. Handles normalization of order values and coercion
of string inputs to structured dictionaries.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator
from typing_extensions import Annotated


def _coerce_string_to_attribute_dict(value: Any) -> Any:
    """Coerce string attribute names to {name: string} dictionaries.

    This allows attributes to be specified as simple strings like "my_attr"
    instead of full dictionaries {"name": "my_attr"}.

    Args:
        value: Input value, either a string (attribute name) or dict.

    Returns:
        Dictionary with "name" key if input was string, otherwise original value.
    """
    if isinstance(value, str):
        return {"name": value}
    return value


class AttributeSchema(BaseModel):
    """Pydantic schema for validating attribute metadata.

    This schema handles raw JSON/dict input validation for attributes
    with normalization of order values and support for string coercion.

    Attributes:
        name: Attribute name, used as identifier (required).
        label: Human-readable label displayed to users (optional).
        description: Longer description of the attribute (optional).
        order: Default sort order, either "asc" or "desc" (optional).
        info: Custom information dictionary for app-specific metadata (default {}).
        format: Application-specific display format information (optional).
        missing_value: Value to use when data is NULL (optional).
        expression: Arithmetic expression for computed attributes (optional).
    """

    model_config = ConfigDict(
        # Allow extra fields for forward compatibility
        extra="allow",
        # Validate assignments to catch errors early
        validate_assignment=True,
        # Don't allow mutation after creation
        frozen=False,
    )

    name: str = Field(..., description="Attribute name, used as identifier")
    label: Optional[str] = Field(None, description="Human-readable label")
    description: Optional[str] = Field(None, description="Longer description")
    order: Optional[Literal["asc", "desc"]] = Field(
        None, description="Default sort order: asc or desc"
    )
    info: dict[str, Any] = Field(
        default_factory=dict, description="Custom information dictionary"
    )
    format: Optional[str] = Field(
        None, description="Application-specific display format"
    )
    missing_value: Optional[Any] = Field(
        None, description="Value to use when data is NULL"
    )
    expression: Optional[str] = Field(
        None, description="Arithmetic expression for computed attributes"
    )

    @classmethod
    def model_validate(
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        from_attributes: bool | None = None,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "AttributeSchema":
        """Validate and create an AttributeSchema instance.

        Supports string inputs by coercing them to {"name": string} dicts.

        Args:
            obj: Input object, can be dict, string, or AttributeSchema instance.
            strict: Whether to enforce strict validation.
            from_attributes: Whether to extract data from object attributes.
            context: Additional context for validation.
            **kwargs: Additional arguments passed to parent class.

        Returns:
            Validated AttributeSchema instance.
        """
        # Coerce string to dict before validation
        if isinstance(obj, str):
            obj = {"name": obj}
        return super().model_validate(
            obj,
            strict=strict,
            from_attributes=from_attributes,
            context=context,
            **kwargs,
        )

    @field_validator("order", mode="before")
    @classmethod
    def _normalize_order(cls, value: Any) -> Any:
        """Normalize order value to lowercase asc/desc.

        Handles case-insensitive input like "ASC", "DESC", "ascending", etc.

        Args:
            value: Raw order value from input.

        Returns:
            Normalized order string or None.

        Raises:
            ValueError: If order value is not recognized.
        """
        if value is None:
            return None

        if not isinstance(value, str):
            raise ValueError(f"Order must be a string, got {type(value).__name__}")

        lower_value = value.lower()

        if lower_value.startswith("asc"):
            return "asc"
        elif lower_value.startswith("desc"):
            return "desc"
        else:
            raise ValueError(f"Unknown ordering '{value}'. Must be 'asc' or 'desc'.")


# Type alias for attributes that can be strings or full schemas
AttributeInput = Annotated[
    AttributeSchema,
    BeforeValidator(_coerce_string_to_attribute_dict),
]
