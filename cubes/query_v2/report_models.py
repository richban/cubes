"""
Pydantic models for type-safe OLAP report query validation.

This module provides strongly-typed query specifications for the report() method,
replacing dictionary-based validation with compile-time type safety and automatic validation.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class OrderSpec(BaseModel):
    """Type-safe ordering specification."""

    attribute: str = Field(..., description="Attribute name to order by")
    direction: Literal["asc", "desc"] | None = Field(None, description="Sort direction")

    class Config:
        extra = "forbid"


class BaseQuery(BaseModel):
    """Base class for all query types with common fields."""

    rollup: str | None = Field(None, description="Dimension level to roll up to")

    class Config:
        extra = "forbid"


class AggregateQuery(BaseQuery):
    """Type-safe aggregate query specification."""

    query: Literal["aggregate"] = "aggregate"
    aggregates: list[str] | None = Field(None, description="List of aggregate measures")
    drilldown: list[str] | None = Field(None, description="Dimensions to drill down")
    split: dict[str, Any] | None = Field(None, description="Split cell specification")
    order: list[str | tuple[str, str]] | None = Field(
        None, description="Ordering specification"
    )
    page: int | None = Field(None, ge=1, description="Page number (1-based)")
    page_size: int | None = Field(None, ge=1, le=10000, description="Page size limit")

    @model_validator(mode="before")
    @classmethod
    def validate_aggregates_and_pagination(cls, values):
        if isinstance(values, dict):
            aggregates = values.get("aggregates")
            if aggregates is not None and len(aggregates) == 0:
                raise ValueError("Aggregates list cannot be empty")

            page = values.get("page")
            if page is not None and page <= 0:
                raise ValueError("Page must be positive")

            page_size = values.get("page_size")
            if page_size is not None and page_size <= 0:
                raise ValueError("Page size must be positive")

        return values

    @model_validator(mode="after")
    def validate_pagination_consistency(self):
        if self.page is not None and self.page_size is None:
            raise ValueError("page_size is required when page is specified")

        return self


class FactsQuery(BaseQuery):
    """Type-safe facts query specification."""

    query: Literal["facts"] = "facts"
    fields: list[str] | None = Field(None, description="Fields to include")
    order: list[str | tuple[str, str]] | None = Field(
        None, description="Ordering specification"
    )
    page: int | None = Field(None, ge=1, description="Page number (1-based)")
    page_size: int | None = Field(None, ge=1, le=50000, description="Page size limit")

    @model_validator(mode="after")
    def validate_pagination_consistency(self):
        if self.page is not None and self.page_size is None:
            raise ValueError("page_size is required when page is specified")

        return self


class FactQuery(BaseQuery):
    """Type-safe single fact query specification."""

    query: Literal["fact"] = "fact"
    key: str | int = Field(..., description="Fact key/ID to retrieve")

    # Backward compatibility aliases
    id: str | int | None = Field(None, description="Deprecated: use 'key' instead")

    @model_validator(mode="before")
    @classmethod
    def validate_key_or_id(cls, values):
        if isinstance(values, dict):
            key = values.get("key")
            id_val = values.get("id")

            if key is None and id_val is None:
                raise ValueError("Either 'key' or 'id' must be provided")

            # Use id as fallback for key
            if key is None:
                values["key"] = id_val

        return values


class MembersQuery(BaseQuery):
    """Type-safe members/values query specification."""

    query: Literal["members", "values"] = "members"
    dimension: str = Field(..., description="Dimension name to query")
    depth: int | None = Field(None, ge=1, description="Hierarchy depth")
    level: str | None = Field(None, description="Specific level name")
    hierarchy: str | None = Field(None, description="Hierarchy name")
    attributes: list[str] | None = Field(None, description="Attributes to include")
    order: list[str | tuple[str, str]] | None = Field(
        None, description="Ordering specification"
    )
    page: int | None = Field(None, ge=1, description="Page number (1-based)")
    page_size: int | None = Field(None, ge=1, le=10000, description="Page size limit")

    @model_validator(mode="after")
    def validate_depth_and_level(self):
        if self.depth is not None and self.level is not None:
            raise ValueError("Cannot specify both 'depth' and 'level'")

        return self

    @model_validator(mode="after")
    def validate_pagination_consistency(self):
        if self.page is not None and self.page_size is None:
            raise ValueError("page_size is required when page is specified")

        return self


class DetailsQuery(BaseQuery):
    """Type-safe cell details query specification."""

    query: Literal["details"] = "details"
    dimension: str | None = Field(None, description="Specific dimension for details")


class CellQuery(BaseQuery):
    """Type-safe cell query specification."""

    query: Literal["cell"] = "cell"
    dimension: str | None = Field(None, description="Specific dimension for details")


# Simplified for now - using Any to avoid forward reference issues
QuerySpec = Any


class ReportRequest(BaseModel):
    """Type-safe report request with multiple queries."""

    queries: dict[str, Any] = Field(
        ..., description="Named queries to execute", min_items=1
    )

    @model_validator(mode="before")
    @classmethod
    def validate_query_names(cls, values):
        if isinstance(values, dict):
            queries = values.get("queries", {})
            # Ensure query names are valid identifiers
            for name in queries.keys():
                if not name.replace("_", "").replace("-", "").isalnum():
                    raise ValueError(f"Query name '{name}' contains invalid characters")
        return values

    class Config:
        extra = "forbid"
        # Enable discriminator for union types
        use_enum_values = True


class ReportResult(BaseModel):
    """Type-safe report result container."""

    results: dict[str, Any] = Field(..., description="Query results by name")
    errors: dict[str, str] = Field(
        default_factory=dict, description="Errors by query name"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Execution metadata"
    )

    @property
    def success_count(self) -> int:
        """Number of successful queries."""
        return len(self.results) - len(self.errors)

    @property
    def error_count(self) -> int:
        """Number of failed queries."""
        return len(self.errors)

    @property
    def has_errors(self) -> bool:
        """Whether any queries failed."""
        return len(self.errors) > 0

    class Config:
        extra = "forbid"


