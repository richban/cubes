"""
Pydantic-based attribute classes for Cubes metadata V2.

This module provides Pydantic implementations of AttributeBase, Attribute,
Measure, and MeasureAggregate classes with improved validation and type safety.
"""

from enum import Enum
from typing import Any, ClassVar

from expressions import inspect_variables
from pydantic import (
    Field,
    PrivateAttr,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic.types import PositiveInt

from ..errors import ModelError
from .base import MetadataObject


class OrderType(str, Enum):
    """Valid ordering types for attributes"""

    ASC = "asc"
    DESC = "desc"


class NonAdditiveType(str, Enum):
    """Valid non-additive types for measures"""

    TIME = "time"
    ANY = "any"


class AttributeBase(MetadataObject):
    # Class constants for ordering
    ASC: ClassVar[str] = "asc"
    DESC: ClassVar[str] = "desc"

    # Additional fields beyond base MetadataObject
    format: str | None = Field(
        None, description="Application-specific display format information"
    )
    order: OrderType | None = Field(
        None, description="Default ordering: 'asc' or 'desc'"
    )
    missing_value: str | int | float | bool | None = Field(
        None, description="Value to use when data source has NULL"
    )
    expression: str | None = Field(
        None, description="Arithmetic expression for computing this attribute"
    )

    @field_validator("order", mode="before")
    @classmethod
    def validate_order(cls, v):
        """Validate and normalize ordering values"""
        if v is None:
            return v

        # Handle case variations
        v_lower = str(v).lower()
        if v_lower.startswith("asc"):
            return OrderType.ASC
        elif v_lower.startswith("desc"):
            return OrderType.DESC
        else:
            raise ValueError(f"Unknown ordering '{v}' - must be 'asc' or 'desc'")

    @computed_field
    @property
    def ref(self) -> str:
        """The unique, fully-qualified reference for the attribute."""
        return self.name

    @computed_field
    @property
    def is_base(self) -> bool:
        """Returns True if attribute is base (not computed from expression)"""
        return not bool(self.expression)

    @computed_field
    @property
    def dependencies(self) -> set[str]:
        """Set of attributes this attribute depends on through expressions"""
        if not self.expression:
            return set()
        return inspect_variables(self.expression)

    def __str__(self) -> str:
        """String representation using ref"""
        return str(self.ref)

    def __hash__(self) -> int:
        """Hash based on ref"""
        return hash(self.ref)


class Attribute(AttributeBase):
    """Dimension attribute object. Also used as fact detail."""

    _dimension: Any | None = PrivateAttr(default=None)

    @computed_field
    @property
    def ref(self) -> str:
        """The unique, fully-qualified reference for the attribute."""
        if self.dimension:
            return f"{self.dimension.name}.{self.name}"
        return self.name

    @property
    def dimension(self):
        """Get the parent dimension"""
        return self._dimension

    @dimension.setter
    def dimension(self, dimension):
        """Set the parent dimension and update ref accordingly"""
        self._dimension = dimension


class Measure(AttributeBase):
    """
    Cube measure attribute – a numerical attribute that can be aggregated.
    """

    formula: str | None = Field(None, description="Name of a formula for the measure")
    aggregates: list[str] | None = Field(
        None, description="List of default aggregate functions for this measure"
    )
    nonadditive: NonAdditiveType | None = Field(
        None, description="Non-additivity type: 'time', 'any', or None"
    )
    window_size: PositiveInt | None = Field(
        None, description="Window size for windowed aggregations"
    )

    @field_validator("aggregates", mode="before")
    @classmethod
    def validate_aggregates(cls, v):
        """Ensure aggregates is a list when provided"""
        if v is None:
            return v
        if isinstance(v, str):
            return [v]
        return list(v)

    @field_validator("nonadditive", mode="before")
    @classmethod
    def validate_nonadditive(cls, v):
        """Validate and normalize non-additive values"""
        if v is None or v == "none":
            return None
        elif v in ["all", "any"]:
            return NonAdditiveType.ANY
        elif v == "time":
            return NonAdditiveType.TIME
        else:
            raise ValueError(f"Unknown non-additive measure type '{v}'")

    def default_aggregates(self) -> list["MeasureAggregate"]:
        """
        Creates default measure aggregates from this measure's aggregate list.
        If no aggregates are specified, defaults to 'sum'.
        """
        aggregates = []
        agg_list = self.aggregates or ["sum"]

        for agg in agg_list:
            if agg == "identity":
                name = str(self.name)
                measure = None
                function = None
            else:
                name = f"{self.name}_{agg}"
                measure = str(self.name)
                function = agg

            aggregate = MeasureAggregate(
                name=name,
                label=self.label,
                description=self.description,
                order=self.order,
                info=self.info,
                format=self.format,
                measure=measure,
                function=function,
                window_size=self.window_size,
                nonadditive=self.nonadditive,
            )

            aggregates.append(aggregate)

        return aggregates


class MeasureAggregate(AttributeBase):
    """
    Measure aggregate - represents an aggregated measure value.
    """

    function: str | None = Field(
        None, description="Aggregation function name (sum, avg, min, max, count, etc.)"
    )
    formula: str | None = Field(
        None, description="Name of a formula containing arithmetic expression"
    )
    measure: str | None = Field(
        None, description="Source measure name for this aggregate"
    )
    nonadditive: NonAdditiveType | None = Field(
        None, description="Non-additivity behavior inherited from measure"
    )
    window_size: PositiveInt | None = Field(
        None, description="Window size for windowed aggregations"
    )

    @field_validator("nonadditive", mode="before")
    @classmethod
    def validate_nonadditive(cls, v):
        """Validate non-additive values"""
        if v is None or v == "none":
            return None
        elif v in ["all", "any"]:
            return NonAdditiveType.ANY
        elif v == "time":
            return NonAdditiveType.TIME
        else:
            raise ValueError(f"Unknown non-additive type '{v}'")

    @model_validator(mode="after")
    def validate_measure_expression_exclusivity(self):
        """Ensure measure and expression are not both specified"""
        if self.measure and self.expression:
            raise ValueError(
                f"Aggregate '{self.name}' cannot have both measure and expression set"
            )
        return self

    @computed_field
    @property
    def is_base(self) -> bool:
        """Returns True if aggregate is base (no expression or function)"""
        return not bool(self.expression) and not bool(self.function)

    @computed_field
    @property
    def dependencies(self) -> set[str]:
        """Set of attributes this aggregate depends on"""
        if self.measure:
            if self.expression:
                raise ModelError(
                    f"Aggregate '{self.ref}' has both measure and expression set"
                )
            return {self.measure}

        if not self.expression:
            return set()

        return inspect_variables(self.expression)
