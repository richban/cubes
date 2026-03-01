"""Attribute models for dimension attributes, measures, and aggregates."""

from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional, Set

from cubes.metadata.models.base import ModelError, ModelObject, to_label

if TYPE_CHECKING:
    from cubes.metadata.models.dimension import Dimension


def inspect_variables(expression: str) -> Set[str]:
    """Extract variable names from an arithmetic expression.

    Parses an expression string and returns a set of identifier names
    that appear in the expression, excluding Python keywords and constants.

    Args:
        expression: An arithmetic expression string (e.g., "amount * price")

    Returns:
        Set of variable names used in the expression.

    Example:
        >>> inspect_variables("a + b * 2")
        {'a', 'b'}
        >>> inspect_variables("amount * price / quantity")
        {'amount', 'price', 'quantity'}
    """
    if not expression:
        return set()

    # Python keywords and constants to exclude
    keywords = {
        "and",
        "as",
        "assert",
        "break",
        "class",
        "continue",
        "def",
        "del",
        "elif",
        "else",
        "except",
        "False",
        "finally",
        "for",
        "from",
        "global",
        "if",
        "import",
        "in",
        "is",
        "lambda",
        "None",
        "nonlocal",
        "not",
        "or",
        "pass",
        "raise",
        "return",
        "True",
        "try",
        "while",
        "with",
        "yield",
    }

    # Pattern to match identifiers (variable names)
    # Matches word characters and dots for attribute access like "foo.bar"
    pattern = r"[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*"

    # Find all potential identifiers
    matches = re.findall(pattern, expression)

    # Filter out keywords, numeric literals, and empty strings
    variables = {
        match
        for match in matches
        if match not in keywords and not match.replace(".", "").isdigit()
    }

    return variables


@dataclass
class AttributeBase(ModelObject):
    """Base class for dimension attributes, measures and measure aggregates.

    Provides common properties for attributes including ordering, formatting,
    missing value handling, and expression-based computation.

    Attributes:
        name: Attribute name, used as identifier.
        label: Human-readable label (optional, auto-generated if not provided).
        description: Longer description of the attribute (optional).
        order: Default order of this attribute ('asc' or 'desc', optional).
        info: Custom information dictionary (defaults to empty dict).
        format: Application-specific display format (optional).
        missing_value: Value to use when data is NULL (optional).
        expression: Arithmetic expression for computed attributes (optional).
        ref: Computed reference string (defaults to name).

    Example:
        >>> attr = AttributeBase(name="amount", expression="price * quantity")
        >>> attr.ref
        'amount'
        >>> attr.is_base
        False
        >>> attr.dependencies
        {'price', 'quantity'}
    """

    name: str
    label: Optional[str] = None
    description: Optional[str] = None
    order: Optional[Literal["asc", "desc"]] = None
    info: dict[str, Any] = field(default_factory=dict)
    format: Optional[str] = None
    missing_value: Any = None
    expression: Optional[str] = None
    ref: str = field(init=False)

    def __post_init__(self) -> None:
        """Initialize computed fields after dataclass initialization."""
        # Set ref to name initially
        object.__setattr__(self, "ref", self.name)

        # Normalize order to lowercase
        if self.order is not None:
            normalized = self.order.lower()
            if normalized not in ("asc", "desc"):
                raise ValueError(
                    f"Unknown ordering '{self.order}' for attribute '{self.name}'. "
                    "Use 'asc' or 'desc'."
                )
            object.__setattr__(self, "order", normalized)

    def __eq__(self, other: object) -> bool:
        """Check equality based on all relevant attributes.

        Two AttributeBase instances are equal if they have the same name,
        label, description, info, format, expression, and missing_value.
        """
        if not isinstance(other, AttributeBase):
            return False

        return (
            self.name == other.name
            and self.label == other.label
            and self.description == other.description
            and self.info == other.info
            and self.format == other.format
            and self.expression == other.expression
            and self.missing_value == other.missing_value
        )

    def __hash__(self) -> int:
        """Return hash based on ref for use in sets and dicts."""
        return hash(self.ref)

    def __str__(self) -> str:
        """Return string representation (ref)."""
        return self.ref

    def __repr__(self) -> str:
        """Return detailed representation."""
        return f"{self.__class__.__name__}({self.to_dict()!r})"

    @property
    def is_base(self) -> bool:
        """Check if this is a base attribute (has no expression).

        Base attributes are directly mapped to data source columns.
        Derived attributes have expressions that reference other attributes.

        Returns:
            True if the attribute has no expression, False otherwise.
        """
        return self.expression is None or self.expression == ""

    @property
    def dependencies(self) -> Set[str]:
        """Get set of attributes this attribute depends on.

        For attributes with expressions, returns the set of variable names
        referenced in the expression. For base attributes, returns empty set.

        Returns:
            Set of attribute references that this attribute depends on.
        """
        if self.is_base:
            return set()

        return inspect_variables(self.expression or "")

    def to_dict(self, create_label: bool = False) -> dict[str, Any]:
        """Convert to a dictionary representation.

        Args:
            create_label: If True and no label is set, auto-generate from name.

        Returns:
            Dictionary with all attribute properties.
        """
        # Get base dict from parent
        base_dict = super().to_dict(create_label=create_label)

        # Add AttributeBase-specific fields
        base_dict.update(
            {
                "format": self.format,
                "order": self.order,
                "missing_value": self.missing_value,
                "expression": self.expression,
                "ref": self.ref,
            }
        )

        return base_dict


@dataclass
class Attribute(AttributeBase):
    """Dimension attribute object. Also used as fact detail.

    Represents an attribute within a dimension or fact. Attributes have
    an optional dimension reference that prefixes their ref when owned.

    Attributes:
        name: Attribute name, used as identifier.
        label: Human-readable label (optional, auto-generated if not provided).
        description: Longer description of the attribute (optional).
        order: Default order of this attribute ('asc' or 'desc', optional).
        info: Custom information dictionary (defaults to empty dict).
        format: Application-specific display format (optional).
        missing_value: Value to use when data is NULL (optional).
        expression: Arithmetic expression for computed attributes (optional).
        ref: Computed reference string (defaults to name or dimension.name).
        dimension: The dimension that owns this attribute (optional).

    Example:
        >>> attr = Attribute(name="name")
        >>> attr.ref
        'name'
        >>> # When assigned to a dimension, ref includes dimension prefix
        >>> # attr.dimension = Dimension(name="customer")
        >>> # attr.ref
        # 'customer.name'
    """

    _dimension: Optional["Dimension"] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize computed fields after dataclass initialization."""
        # Call parent's post_init first
        super().__post_init__()
        # Initialize _dimension to None
        object.__setattr__(self, "_dimension", None)

    @property
    def dimension(self) -> Optional["Dimension"]:
        """Get the dimension that owns this attribute."""
        return self._dimension

    @dimension.setter
    def dimension(self, dimension: Optional["Dimension"]) -> None:
        """Set the dimension owner and update ref accordingly.

        When a dimension is assigned, the ref is updated to include the
        dimension name prefix (dimension.name + '.' + name).
        When dimension is None, ref reverts to just the attribute name.

        Args:
            dimension: The dimension that owns this attribute, or None.
        """
        if dimension is not None:
            # Update ref to include dimension prefix
            object.__setattr__(self, "ref", f"{dimension.name}.{self.name}")
        else:
            # Revert ref to just the name
            object.__setattr__(self, "ref", self.name)
        object.__setattr__(self, "_dimension", dimension)

    def __deepcopy__(self, memo: dict[int, Any]) -> "Attribute":
        """Create a deep copy of this attribute.

        Note: Copied attributes are disowned from their dimension.
        The new attribute must have its dimension assigned separately.

        Args:
            memo: Memoization dict for deep copy.

        Returns:
            A new Attribute instance with copied values but no dimension.
        """
        # Create new instance without dimension
        result = Attribute(
            name=self.name,
            label=self.label,
            description=self.description,
            order=self.order,
            info=copy.deepcopy(self.info, memo),
            format=self.format,
            missing_value=self.missing_value,
            expression=self.expression,
        )
        # Ensure ref is set correctly (without dimension prefix)
        object.__setattr__(result, "ref", self.name)
        memo[id(self)] = result
        return result

    def __eq__(self, other: object) -> bool:
        """Check equality with another Attribute.

        Two attributes are equal if their base properties are equal.
        Note: dimension (owner) is not compared.
        """
        if not isinstance(other, Attribute):
            return False
        # Compare using parent class equality
        if not super().__eq__(other):
            return False
        return True

    def __hash__(self) -> int:
        """Return hash based on ref for use in sets and dicts."""
        return hash(self.ref)

    def to_dict(self, create_label: bool = False) -> dict[str, Any]:
        """Convert to a dictionary representation.

        Args:
            create_label: If True and no label is set, auto-generate from name.

        Returns:
            Dictionary with all attribute properties.
        """
        # Get base dict from parent
        base_dict = super().to_dict(create_label=create_label)

        return base_dict


@dataclass
class Measure(AttributeBase):
    """Cube measure attribute – a numerical attribute that can be aggregated.

    Measures represent numerical values in a cube that can be aggregated using
    various functions (sum, count, avg, etc.). They can have formulas for
    computed measures and specify their default aggregation behaviors.

    Attributes:
        name: Measure name, used as identifier.
        label: Human-readable label (optional).
        description: Longer description of the measure (optional).
        order: Default sort order ('asc' or 'desc', optional).
        info: Custom information dictionary (defaults to empty dict).
        format: Application-specific display format (optional).
        missing_value: Value to use when data is NULL (optional).
        expression: Arithmetic expression for computed measures (optional).
        ref: Computed reference string (defaults to name).
        formula: Name of a formula for the measure (optional).
        aggregates: List of default aggregate functions (defaults to ["sum"]).
        nonadditive: Non-additivity type: None, "all", or "time" (optional).
        window_size: Size of rolling window for time-based aggregations (optional).

    Example:
        >>> measure = Measure(name="amount")
        >>> measure.aggregates
        ['sum']
        >>> measure.nonadditive is None
        True
        >>> aggs = measure.default_aggregates()
        >>> len(aggs)
        1
        >>> aggs[0].name
        'amount_sum'
    """

    formula: Optional[str] = None
    aggregates: list[str] = field(default_factory=lambda: ["sum"])
    nonadditive: Optional[Literal["all", "time"]] = None
    window_size: Optional[int] = None

    def __post_init__(self) -> None:
        """Initialize computed fields after dataclass initialization."""
        # Call parent's post_init first
        super().__post_init__()

        # Normalize nonadditive value
        # "none" -> None, "all"/"any" -> "all", "time" -> "time"
        if self.nonadditive is not None:
            normalized = self.nonadditive.lower()
            if normalized == "none":
                object.__setattr__(self, "nonadditive", None)
            elif normalized in ("all", "any"):
                object.__setattr__(self, "nonadditive", "all")
            elif normalized == "time":
                object.__setattr__(self, "nonadditive", "time")
            else:
                raise ValueError(
                    f"Unknown non-additive type '{self.nonadditive}'. "
                    "Must be 'none', 'all', 'any', or 'time'."
                )

        # Ensure aggregates is a list
        if not self.aggregates:
            object.__setattr__(self, "aggregates", ["sum"])

    def __deepcopy__(self, memo: dict[int, Any]) -> "Measure":
        """Create a deep copy of this measure.

        Args:
            memo: Memoization dict for deep copy.

        Returns:
            A new Measure instance with copied values.
        """
        result = Measure(
            name=self.name,
            label=self.label,
            description=self.description,
            order=self.order,
            info=copy.deepcopy(self.info, memo),
            format=self.format,
            missing_value=self.missing_value,
            expression=self.expression,
            formula=self.formula,
            aggregates=copy.deepcopy(self.aggregates, memo),
            nonadditive=self.nonadditive,
            window_size=self.window_size,
        )
        # Ensure ref is set correctly
        object.__setattr__(result, "ref", self.ref)
        memo[id(self)] = result
        return result

    def __eq__(self, other: object) -> bool:
        """Check equality with another Measure.

        Two measures are equal if their base properties and measure-specific
        properties are equal.
        """
        if not isinstance(other, Measure):
            return False
        # Compare using parent class equality
        if not super().__eq__(other):
            return False
        return (
            self.aggregates == other.aggregates
            and self.formula == other.formula
            and self.nonadditive == other.nonadditive
            and self.window_size == other.window_size
        )

    def __hash__(self) -> int:
        """Return hash based on ref for use in sets and dicts."""
        return hash(self.ref)

    def default_aggregates(self) -> list["MeasureAggregate"]:
        """Create default measure aggregates from this measure.

        This is a convenience method that generates MeasureAggregate objects
        based on the measure's aggregates list. If no aggregates are specified,
        then only 'sum' is assumed.

        For the 'identity' aggregate, the aggregate name is just the measure name,
        measure is None, and function is None.

        For other aggregates, the name is '{measure_name}_{function}', measure is
        the measure name, and function is the aggregate function name.

        Returns:
            List of MeasureAggregate instances.

        Example:
            >>> measure = Measure(name="amount", aggregates=["sum", "min"])
            >>> aggs = measure.default_aggregates()
            >>> aggs[0].name
            'amount_sum'
            >>> aggs[1].name
            'amount_min'
        """
        aggregates: list["MeasureAggregate"] = []

        for agg in self.aggregates or ["sum"]:
            if agg == "identity":
                name = self.name
                measure_name: Optional[str] = None
                function: Optional[str] = None
            else:
                name = f"{self.name}_{agg}"
                measure_name = self.name
                function = agg

            aggregate = MeasureAggregate(
                name=name,
                label=None,
                description=self.description,
                order=self.order,
                info=copy.deepcopy(self.info),
                format=self.format,
                measure=measure_name,
                function=function,
                window_size=self.window_size,
            )

            aggregates.append(aggregate)

        return aggregates

    def to_dict(self, create_label: bool = False) -> dict[str, Any]:
        """Convert to a dictionary representation.

        Args:
            create_label: If True and no label is set, auto-generate from name.

        Returns:
            Dictionary with all measure properties.
        """
        # Get base dict from parent
        base_dict = super().to_dict(create_label=create_label)

        # Add Measure-specific fields
        base_dict.update(
            {
                "formula": self.formula,
                "aggregates": list(self.aggregates) if self.aggregates else None,
                "nonadditive": self.nonadditive,
                "window_size": self.window_size,
            }
        )

        return base_dict


@dataclass
class MeasureAggregate(AttributeBase):
    """Measure aggregate object representing an aggregated measure value.

    Measure aggregates represent the result of applying an aggregation function
    to a measure. They can be simple (e.g., sum of amount) or computed
    (e.g., sum(amount) / count(1)).

    Attributes:
        name: Aggregate name, used as identifier.
        label: Human-readable label (optional).
        description: Longer description of the aggregate (optional).
        order: Default sort order ('asc' or 'desc', optional).
        info: Custom information dictionary (defaults to empty dict).
        format: Application-specific display format (optional).
        missing_value: Value to use when data is NULL (optional).
        expression: Arithmetic expression for computed aggregates (optional).
        ref: Computed reference string (defaults to name).
        function: Aggregation function name (e.g., "sum", "count", optional).
        formula: Name of a formula for the aggregate (optional).
        measure: Name of the measure being aggregated (optional).
        nonadditive: Additive behavior inherited from the measure (optional).
        window_size: Size of rolling window for time-based aggregations (optional).

    Note:
        An aggregate can have either a measure (with function) OR an expression,
        but not both. This is enforced by validation.

    Example:
        >>> agg = MeasureAggregate(name="amount_sum", measure="amount", function="sum")
        >>> agg.measure
        'amount'
        >>> agg.function
        'sum'
    """

    function: Optional[str] = None
    formula: Optional[str] = None
    measure: Optional[str] = None
    nonadditive: Optional[Literal["all", "time"]] = None
    window_size: Optional[int] = None

    def __post_init__(self) -> None:
        """Initialize computed fields after dataclass initialization."""
        # Call parent's post_init first
        super().__post_init__()

    def __deepcopy__(self, memo: dict[int, Any]) -> "MeasureAggregate":
        """Create a deep copy of this aggregate.

        Args:
            memo: Memoization dict for deep copy.

        Returns:
            A new MeasureAggregate instance with copied values.
        """
        result = MeasureAggregate(
            name=self.name,
            label=self.label,
            description=self.description,
            order=self.order,
            info=copy.deepcopy(self.info, memo),
            format=self.format,
            missing_value=self.missing_value,
            expression=self.expression,
            function=self.function,
            formula=self.formula,
            measure=self.measure,
            nonadditive=self.nonadditive,
            window_size=self.window_size,
        )
        # Ensure ref is set correctly
        object.__setattr__(result, "ref", self.ref)
        memo[id(self)] = result
        return result

    def __eq__(self, other: object) -> bool:
        """Check equality with another MeasureAggregate.

        Two aggregates are equal if their base properties and aggregate-specific
        properties are equal.
        """
        if not isinstance(other, MeasureAggregate):
            return False
        # Compare using parent class equality
        if not super().__eq__(other):
            return False
        return (
            self.function == other.function
            and self.measure == other.measure
            and self.formula == other.formula
            and self.nonadditive == other.nonadditive
            and self.window_size == other.window_size
        )

    def __hash__(self) -> int:
        """Return hash based on ref for use in sets and dicts."""
        return hash(self.ref)

    @property
    def is_base(self) -> bool:
        """Check if this is a base aggregate.

        A base aggregate has no expression and no aggregation function.
        These are typically backend-provided aggregates.

        Returns:
            True if no expression AND no function, False otherwise.
        """
        return not self.expression and not self.function

    @property
    def dependencies(self) -> Set[str]:
        """Get set of attributes this aggregate depends on.

        If the aggregate has a measure, returns a set containing just the
        measure name. If the aggregate has an expression, returns the
        variable names from the expression. If neither, returns empty set.

        Raises:
            ModelError: If both measure and expression are set.

        Returns:
            Set of attribute/measure references that this aggregate depends on.
        """
        if self.measure:
            if self.expression:
                raise ModelError(
                    f"Aggregate '{self.ref}' has both measure and expression set"
                )
            return {self.measure}

        if not self.expression:
            return set()

        return inspect_variables(self.expression)

    def to_dict(self, create_label: bool = False) -> dict[str, Any]:
        """Convert to a dictionary representation.

        Args:
            create_label: If True and no label is set, auto-generate from name.

        Returns:
            Dictionary with all aggregate properties.
        """
        # Get base dict from parent
        base_dict = super().to_dict(create_label=create_label)

        # Add MeasureAggregate-specific fields
        base_dict.update(
            {
                "function": self.function,
                "formula": self.formula,
                "measure": self.measure,
                "nonadditive": self.nonadditive,
                "window_size": self.window_size,
            }
        )

        return base_dict
