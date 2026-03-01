"""Pydantic schemas for attribute definitions.

Provides input validation and parsing for dimension attributes, measures,
and measure aggregates. Handles normalization of order values and coercion
of string inputs to structured dictionaries.
"""

from typing import Any, Literal, Optional

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from typing_extensions import Annotated

# Supported aggregation functions for measure validation
SUPPORTED_AGGREGATE_FUNCTIONS = {
    "sum",
    "count",
    "count_nonempty",
    "min",
    "max",
    "avg",
    "identity",
    "mean",
    "std",
    "var",
    "median",
}


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


class MeasureSchema(AttributeSchema):
    """Pydantic schema for validating measure metadata.

    This schema extends AttributeSchema with measure-specific fields
    for formula, aggregates, nonadditive behavior, and window size.

    Attributes:
        formula: Name of a formula for the measure (optional).
        aggregates: List of default aggregate functions (optional, default ["sum"]).
        nonadditive: Non-additivity type: "none", "all", or "time" (optional).
        window_size: Size of rolling window for time-based aggregations (optional).
    """

    model_config = ConfigDict(
        # Allow extra fields for forward compatibility
        extra="allow",
        # Validate assignments to catch errors early
        validate_assignment=True,
        # Don't allow mutation after creation
        frozen=False,
    )

    formula: Optional[str] = Field(
        None, description="Name of a formula for the measure"
    )
    aggregates: list[str] = Field(
        default_factory=lambda: ["sum"],
        description="List of default aggregate functions",
    )
    nonadditive: Optional[Literal["none", "all", "time"]] = Field(
        None, description="Non-additivity type: none, all, or time"
    )
    window_size: Optional[int] = Field(
        None, description="Size of rolling window for time-based aggregations"
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
    ) -> "MeasureSchema":
        """Validate and create a MeasureSchema instance.

        Supports string inputs by coercing them to {"name": string} dicts.

        Args:
            obj: Input object, can be dict, string, or MeasureSchema instance.
            strict: Whether to enforce strict validation.
            from_attributes: Whether to extract data from object attributes.
            context: Additional context for validation.
            **kwargs: Additional arguments passed to parent class.

        Returns:
            Validated MeasureSchema instance.
        """
        # Coerce string to dict before validation
        if isinstance(obj, str):
            obj = {"name": obj}
        result = super().model_validate(
            obj,
            strict=strict,
            from_attributes=from_attributes,
            context=context,
            **kwargs,
        )
        # Cast to MeasureSchema since mypy doesn't understand super() returns subclass
        return result  # type: ignore[return-value]

    @field_validator("nonadditive", mode="before")
    @classmethod
    def _normalize_nonadditive(cls, value: Any) -> Any:
        """Normalize nonadditive value to standard form.

        Handles normalization:
        - "none" -> None (fully additive)
        - "all", "any" -> "all"
        - "time" -> "time"

        Args:
            value: Raw nonadditive value from input.

        Returns:
            Normalized nonadditive string or None.

        Raises:
            ValueError: If nonadditive value is not recognized.
        """
        if value is None:
            return None

        if not isinstance(value, str):
            raise ValueError(
                f"Nonadditive must be a string, got {type(value).__name__}"
            )

        lower_value = value.lower()

        if lower_value == "none":
            return None
        elif lower_value in ["all", "any"]:
            return "all"
        elif lower_value == "time":
            return "time"
        else:
            raise ValueError(
                f"Unknown non-additive type '{value}'. "
                "Must be 'none', 'all', 'any', or 'time'."
            )

    @model_validator(mode="after")
    def _validate_aggregates(self) -> "MeasureSchema":
        """Validate that aggregate functions are supported.

        Raises:
            ValueError: If any aggregate function is not supported.
        """
        if self.aggregates:
            invalid = []
            for agg in self.aggregates:
                if agg not in SUPPORTED_AGGREGATE_FUNCTIONS:
                    invalid.append(agg)
            if invalid:
                raise ValueError(
                    f"Unsupported aggregate functions: {', '.join(invalid)}. "
                    f"Supported: {', '.join(sorted(SUPPORTED_AGGREGATE_FUNCTIONS))}"
                )
        return self


# Type alias for measures that can be strings or full schemas
MeasureInput = Annotated[
    MeasureSchema,
    BeforeValidator(_coerce_string_to_attribute_dict),
]


class MeasureAggregateSchema(AttributeSchema):
    """Pydantic schema for validating measure aggregate metadata.

    This schema extends AttributeSchema with aggregate-specific fields
    for function, formula, measure reference, and nonadditive behavior.
    Enforces mutual exclusivity between measure and expression.

    Attributes:
        function: Aggregation function name (e.g., "sum", "count", optional).
        formula: Name of a formula for the aggregate (optional).
        measure: Name of the measure being aggregated (optional).
        nonadditive: Non-additivity type: "none", "all", or "time" (optional).
        window_size: Size of rolling window for time-based aggregations (optional).
    """

    model_config = ConfigDict(
        # Allow extra fields for forward compatibility
        extra="allow",
        # Validate assignments to catch errors early
        validate_assignment=True,
        # Don't allow mutation after creation
        frozen=False,
    )

    function: Optional[str] = Field(
        None, description="Aggregation function name (e.g., sum, count)"
    )
    formula: Optional[str] = Field(
        None, description="Name of a formula for the aggregate"
    )
    measure: Optional[str] = Field(
        None, description="Name of the measure being aggregated"
    )
    nonadditive: Optional[Literal["none", "all", "time"]] = Field(
        None, description="Non-additivity type: none, all, or time"
    )
    window_size: Optional[int] = Field(
        None, description="Size of rolling window for time-based aggregations"
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
    ) -> "MeasureAggregateSchema":
        """Validate and create a MeasureAggregateSchema instance.

        Supports string inputs by coercing them to {"name": string} dicts.

        Args:
            obj: Input object, can be dict, string, or MeasureAggregateSchema instance.
            strict: Whether to enforce strict validation.
            from_attributes: Whether to extract data from object attributes.
            context: Additional context for validation.
            **kwargs: Additional arguments passed to parent class.

        Returns:
            Validated MeasureAggregateSchema instance.
        """
        # Coerce string to dict before validation
        if isinstance(obj, str):
            obj = {"name": obj}
        result = super().model_validate(
            obj,
            strict=strict,
            from_attributes=from_attributes,
            context=context,
            **kwargs,
        )
        # Cast to MeasureAggregateSchema since mypy doesn't understand super() returns subclass
        return result  # type: ignore[return-value]

    @field_validator("nonadditive", mode="before")
    @classmethod
    def _normalize_nonadditive(cls, value: Any) -> Any:
        """Normalize nonadditive value to standard form.

        Handles normalization:
        - "none" -> None (fully additive)
        - "all", "any" -> "all"
        - "time" -> "time"

        Args:
            value: Raw nonadditive value from input.

        Returns:
            Normalized nonadditive string or None.

        Raises:
            ValueError: If nonadditive value is not recognized.
        """
        if value is None:
            return None

        if not isinstance(value, str):
            raise ValueError(
                f"Nonadditive must be a string, got {type(value).__name__}"
            )

        lower_value = value.lower()

        if lower_value == "none":
            return None
        elif lower_value in ["all", "any"]:
            return "all"
        elif lower_value == "time":
            return "time"
        else:
            raise ValueError(
                f"Unknown non-additive type '{value}'. "
                "Must be 'none', 'all', 'any', or 'time'."
            )

    @model_validator(mode="after")
    def _validate_measure_expression_mutual_exclusivity(
        self,
    ) -> "MeasureAggregateSchema":
        """Validate that measure and expression are mutually exclusive.

        An aggregate can have either a measure (with optional function) OR an
        expression, but not both. This is a fundamental constraint of the
        aggregate model.

        Raises:
            ValueError: If both measure and expression are set.
        """
        if self.measure and self.expression:
            raise ValueError(
                f"Aggregate '{self.name}' cannot have both 'measure' and 'expression' set. "
                "An aggregate can either aggregate a measure OR compute an expression, "
                "but not both."
            )
        return self

    @model_validator(mode="after")
    def _validate_function(self) -> "MeasureAggregateSchema":
        """Validate that function name is supported if set.

        Raises:
            ValueError: If function is not in supported aggregation functions.
        """
        if self.function and self.function not in SUPPORTED_AGGREGATE_FUNCTIONS:
            raise ValueError(
                f"Unsupported aggregate function '{self.function}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_AGGREGATE_FUNCTIONS))}"
            )
        return self


# Type alias for measure aggregates that can be strings or full schemas
MeasureAggregateInput = Annotated[
    MeasureAggregateSchema,
    BeforeValidator(_coerce_string_to_attribute_dict),
]
