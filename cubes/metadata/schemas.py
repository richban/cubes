"""Pydantic schemas for metadata validation and serialization."""

from typing import Any, Literal

from expressions import inspect_variables
from pydantic import BaseModel, ConfigDict, Field, field_validator


class BaseMetadataModel(BaseModel):
    """Base model for all metadata objects with common fields."""

    model_config = ConfigDict(
        extra="forbid",  # Don't allow extra fields
        validate_assignment=True,  # Validate on assignment
        str_strip_whitespace=True,  # Strip whitespace from strings
    )

    name: str = Field(..., min_length=1, description="Object name (identifier)")
    label: str | None = Field(None, description="Human-readable label")
    description: str | None = Field(None, description="Detailed description")
    info: dict[str, Any] | None = Field(
        default_factory=dict, description="Custom metadata"
    )


class AttributeSchema(BaseMetadataModel):
    """Schema for dimension attributes."""

    order: Literal["asc", "desc"] | None = Field(None, description="Default sort order")
    format: str | None = Field(None, description="Display format")
    missing_value: str | int | float | bool | None = Field(
        None, description="Value for NULL data"
    )
    expression: str | None = Field(None, description="Calculated field expression")
    locales: list[str] | None = Field(
        default_factory=list, description="Available locales"
    )

    @field_validator("expression")
    @classmethod
    def validate_expression(cls, v: str | None) -> str | None:
        """Validate that expression syntax is parseable."""
        if v is not None:
            try:
                # Test that the expression can be parsed
                inspect_variables(v)
            except Exception as e:
                raise ValueError(f"Invalid expression syntax: {e}")
        return v


class MeasureSchema(BaseMetadataModel):
    """Schema for cube measures."""

    aggregates: list[str] | None = Field(
        default_factory=lambda: ["sum"], description="Available aggregation functions"
    )
    formula: str | None = Field(None, description="Formula name for calculation")
    expression: str | None = Field(None, description="Arithmetic expression")
    nonadditive: Literal["none", "time", "all"] | None = Field(
        "none", description="Non-additivity type"
    )
    window_size: int | None = Field(
        None, gt=0, description="Window size for calculations"
    )
    format: str | None = Field(None, description="Display format")
    missing_value: str | int | float | bool | None = Field(
        None, description="Value for NULL data"
    )

    @field_validator("expression")
    @classmethod
    def validate_expression(cls, v: str | None) -> str | None:
        """Validate that expression syntax is parseable."""
        if v is not None:
            try:
                inspect_variables(v)
            except Exception as e:
                raise ValueError(f"Invalid expression syntax: {e}")
        return v

    @field_validator("aggregates")
    @classmethod
    def validate_aggregates(cls, v: list[str]) -> list[str]:
        """Validate aggregation function names."""
        valid_functions = {
            "sum",
            "avg",
            "min",
            "max",
            "count",
            "count_nonempty",
            "stddev",
            "variance",
            "identity",
        }
        for agg in v:
            if agg not in valid_functions:
                raise ValueError(f"Unknown aggregation function: {agg}")
        return v


class MeasureAggregateSchema(BaseMetadataModel):
    """Schema for measure aggregates."""

    function: str | None = Field(None, description="Aggregation function")
    formula: str | None = Field(None, description="Formula name")
    measure: str | None = Field(None, description="Source measure name")
    expression: str | None = Field(None, description="Arithmetic expression")
    nonadditive: Literal["none", "time", "all"] | None = Field(
        None, description="Non-additivity"
    )
    window_size: int | None = Field(None, gt=0, description="Window size")
    format: str | None = Field(None, description="Display format")

    @field_validator("expression")
    @classmethod
    def validate_expression(cls, v: str | None) -> str | None:
        if v is not None:
            try:
                inspect_variables(v)
            except Exception as e:
                raise ValueError(f"Invalid expression syntax: {e}")
        return v


class LevelSchema(BaseMetadataModel):
    """Schema for dimension levels."""

    attributes: list[str | AttributeSchema] = Field(
        ..., min_items=1, description="Level attributes"
    )
    key: str | None = Field(None, description="Key attribute name")
    order_attribute: str | None = Field(None, description="Ordering attribute")
    order: Literal["asc", "desc"] | None = Field(None, description="Default order")
    cardinality: Literal["tiny", "low", "medium", "high"] | None = Field(
        None, description="Cardinality hint"
    )
    role: str | None = Field(None, description="Semantic role (e.g., 'time')")


class HierarchySchema(BaseMetadataModel):
    """Schema for dimension hierarchies."""

    levels: list[str | LevelSchema] = Field(
        ..., min_items=1, description="Hierarchy levels"
    )


class DimensionSchema(BaseMetadataModel):
    """Schema for dimensions."""

    levels: list[str | LevelSchema] | None = Field(None, description="Dimension levels")
    hierarchies: list[str | HierarchySchema] | None = Field(
        None, description="Dimension hierarchies"
    )
    default_hierarchy_name: str | None = Field(None, description="Default hierarchy")
    role: str | None = Field(None, description="Dimension role")
    cardinality: Literal["tiny", "low", "medium", "high"] | None = Field(
        None, description="Cardinality hint"
    )
    category: str | None = Field(None, description="Dimension category")
    nonadditive: Literal["none", "time", "all"] | None = Field(
        None, description="Non-additivity"
    )


class DimensionLinkSchema(BaseModel):
    """Schema for dimension links in cubes."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Dimension name")
    hierarchies: list[str] | None = Field(None, description="Included hierarchies")
    nonadditive: Literal["none", "time", "all"] | None = Field(
        None, description="Override non-additivity"
    )
    cardinality: Literal["tiny", "low", "medium", "high"] | None = Field(
        None, description="Override cardinality"
    )
    default_hierarchy_name: str | None = Field(
        None, description="Override default hierarchy"
    )


class CubeSchema(BaseMetadataModel):
    """Schema for cubes."""

    # Logical structure
    measures: list[str | MeasureSchema] | None = Field(
        default_factory=list, description="Cube measures"
    )
    aggregates: list[str | MeasureAggregateSchema] | None = Field(
        default_factory=list, description="Measure aggregates"
    )
    dimensions: list[str | DimensionLinkSchema] | None = Field(
        default_factory=list, description="Linked dimensions"
    )
    details: list[str | AttributeSchema] | None = Field(
        default_factory=list, description="Detail attributes"
    )

    # Physical structure
    fact: str | None = Field(None, description="Fact table name")
    key: str | None = Field(None, description="Fact key field")
    mappings: dict[str, Any] | None = Field(
        None, description="Logical to physical mappings"
    )
    joins: list[dict[str, Any]] | None = Field(None, description="Join specifications")

    # Metadata
    store: str | None = Field(None, description="Data store name")
    category: str | None = Field(None, description="Cube category")
    locale: str | None = Field(None, description="Default locale")
    browser_options: dict[str, Any] | None = Field(
        None, description="Backend-specific options"
    )


class ModelSchema(BaseModel):
    """Schema for complete OLAP models."""

    model_config = ConfigDict(extra="forbid")

    cubes: list[CubeSchema] | None = Field(
        default_factory=list, description="Model cubes"
    )
    dimensions: list[DimensionSchema] | None = Field(
        default_factory=list, description="Model dimensions"
    )
    locale: str | None = Field(None, description="Default model locale")
    info: dict[str, Any] | None = Field(
        default_factory=dict, description="Model metadata"
    )


# Validation functions for backward compatibility
def validate_cube_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Validate cube metadata using pydantic schema."""
    cube = CubeSchema.model_validate(metadata)
    return cube.model_dump(exclude_none=True)


def validate_dimension_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Validate dimension metadata using pydantic schema."""
    dimension = DimensionSchema.model_validate(metadata)
    return dimension.model_dump(exclude_none=True)


def validate_measure_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Validate measure metadata using pydantic schema."""
    measure = MeasureSchema.model_validate(metadata)
    return measure.model_dump(exclude_none=True)


def validate_model_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Validate complete model metadata using pydantic schema."""
    model = ModelSchema.model_validate(metadata)
    return model.model_dump(exclude_none=True)
