"""Attribute models for dimension attributes, measures, and aggregates."""

from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional, Set

from cubes.metadata.models.base import ModelObject, to_label

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
