"""
Pydantic-based attribute classes for Cubes metadata V2.

This module provides Pydantic implementations of AttributeBase, Attribute,
Measure, and MeasureAggregate classes with improved validation and type safety.
"""

from typing import Any, Dict, List, Optional, Set, Union, ClassVar
from enum import Enum

from pydantic import Field, field_validator, model_validator, computed_field, PrivateAttr, ConfigDict
from pydantic.types import PositiveInt

from expressions import inspect_variables

from ..errors import ArgumentError, ModelError
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

    # Localization support
    localizable_attributes: ClassVar[List[str]] = ["label", "description", "format"]

    # Additional fields beyond base MetadataObject
    format: Optional[str] = Field(
        None, description="Application-specific display format information"
    )
    order: Optional[OrderType] = Field(
        None, description="Default ordering: 'asc' or 'desc'"
    )
    missing_value: Optional[Union[str, int, float, bool]] = Field(
        None, description="Value to use when data source has NULL"
    )
    expression: Optional[str] = Field(
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
    def dependencies(self) -> Set[str]:
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

    def is_localizable(self) -> bool:
        """Check if attribute supports localization - base implementation"""
        return False

    def localized_ref(self, locale: Optional[str]) -> str:
        """Returns localized attribute reference for locale"""
        if locale:
            if not hasattr(self, "locales") or not self.locales:
                raise ArgumentError(
                    f"Attribute '{self.name}' is not localizable "
                    f"(localization {locale} requested)"
                )
            elif locale not in self.locales:
                raise ArgumentError(
                    f"Attribute '{self.name}' has no localization {locale} "
                    f"(has: {self.locales})"
                )
            else:
                locale_suffix = "." + locale
        else:
            locale_suffix = ""

        return str(self.ref) + locale_suffix

    @classmethod
    def from_metadata(
        cls,
        metadata: Union[str, Dict[str, Any], "AttributeBase"],
        templates: Optional[Dict] = None,
    ):
        """Create instance from metadata with enhanced validation"""
        if isinstance(metadata, str):
            return cls(name=metadata)
        elif isinstance(metadata, cls):
            return metadata.model_copy()
        elif isinstance(metadata, dict):
            if "name" not in metadata:
                raise ModelError(
                    "Model objects metadata require at least name to be present."
                )
            return cls(**metadata)
        else:
            raise ArgumentError(
                f"Invalid metadata type for {cls.__name__}: {type(metadata)}"
            )


class Attribute(AttributeBase):
    """Dimension attribute object. Also used as fact detail."""

    _dimension: Optional[Any] = PrivateAttr(default=None)

    locales: List[str] = Field(
        default_factory=list, description="List of supported locales for this attribute"
    )

    @field_validator("locales", mode="before")
    @classmethod
    def validate_locales(cls, v):
        """Ensure locales is a list"""
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return list(v) if v else []

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

    def is_localizable(self) -> bool:
        """Check if attribute supports localization"""
        return bool(self.locales)


class Measure(AttributeBase):
    """
    Cube measure attribute – a numerical attribute that can be aggregated.
    """

    formula: Optional[str] = Field(
        None, description="Name of a formula for the measure"
    )
    aggregates: Optional[List[str]] = Field(
        None, description="List of default aggregate functions for this measure"
    )
    nonadditive: Optional[NonAdditiveType] = Field(
        None, description="Non-additivity type: 'time', 'any', or None"
    )
    window_size: Optional[PositiveInt] = Field(
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

    def default_aggregates(self) -> List["MeasureAggregate"]:
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

    function: Optional[str] = Field(
        None, description="Aggregation function name (sum, avg, min, max, count, etc.)"
    )
    formula: Optional[str] = Field(
        None, description="Name of a formula containing arithmetic expression"
    )
    measure: Optional[str] = Field(
        None, description="Source measure name for this aggregate"
    )
    nonadditive: Optional[NonAdditiveType] = Field(
        None, description="Non-additivity behavior inherited from measure"
    )
    window_size: Optional[PositiveInt] = Field(
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
    def dependencies(self) -> Set[str]:
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


# Export metadata expansion utility for compatibility
def expand_attribute_metadata(metadata: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Fixes metadata of an attribute. If metadata is a string it will be
    converted into a dictionary with key "name" set to the string value.
    """
    if isinstance(metadata, str):
        return {"name": metadata}
    return metadata
