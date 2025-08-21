"""
Modern Cube implementation for Cubes metadata V2.

Complete rewrite using Pydantic for validation, type safety, and modern Python patterns.
This implementation preserves all functionality from the legacy cube.py while adding
significant improvements in performance, type safety, and developer experience.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import (
    ConfigDict,
    Field,
    PrivateAttr,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic.types import PositiveInt

from ..errors import (
    ArgumentError,
    ModelError,
    NoSuchAttributeError,
    NoSuchDimensionError,
)
from .attributes import Attribute, Measure, MeasureAggregate
from .base import MetadataObject
from .dimension import Dimension

__all__ = ["Cube", "CubeType", "DEFAULT_FACT_COUNT_AGGREGATE"]

# Default aggregate when no measures or aggregates are defined
DEFAULT_FACT_COUNT_AGGREGATE = {
    "name": "fact_count",
    "label": "Count",
    "function": "count",
}

# Template labels for automatically generated aggregates
IMPLICIT_AGGREGATE_LABELS = {
    "sum": "Sum of {measure}",
    "count": "Record Count",
    "count_nonempty": "Non-empty count of {measure}",
    "min": "{measure} Minimum",
    "max": "{measure} Maximum",
    "avg": "Average of {measure}",
}


class CubeType(str, Enum):
    """Semantic types of cubes for better understanding and processing"""

    FACT = "fact"  # Fact table-based cube with detailed data
    AGGREGATE = "aggregate"  # Pre-aggregated cube for performance
    VIRTUAL = "virtual"  # Virtual cube combining other cubes


class Cube(MetadataObject):
    """
    Modern cube definition representing a logical data structure for multidimensional analysis.

    A cube defines:
    - Which dimensions can be used for slicing/dicing
    - What measures can be aggregated
    - How data should be accessed and processed
    - Physical mapping to backend storage

    This implementation uses Pydantic for automatic validation and type safety
    while preserving all functionality from the legacy cube implementation.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="allow",
        use_enum_values=True,
        arbitrary_types_allowed=True,
    )

    # ========================================
    # Core Logical Properties (Dictionary-based for O(1) lookups)
    # ========================================

    # Private dictionary storage for efficient lookups
    _dimensions: dict[str, Dimension] = PrivateAttr(default_factory=dict)
    _measures: dict[str, Measure] = PrivateAttr(default_factory=dict)
    _aggregates: dict[str, MeasureAggregate] = PrivateAttr(default_factory=dict)

    # Public fields for initialization (will be processed by validators)
    dimensions: list[Dimension] = Field(
        default_factory=list, description="Dimensions available for slicing and dicing"
    )

    measures: list[Measure] = Field(
        default_factory=list, description="Numerical measures that can be aggregated"
    )

    aggregates: list[MeasureAggregate] = Field(
        default_factory=list, description="Pre-computed or derived aggregates"
    )

    details: list[Attribute] = Field(
        default_factory=list,
        description="Detail attributes for drill-through operations",
    )

    # ========================================
    # Cube Metadata and Classification
    # ========================================

    cube_type: CubeType = Field(
        default=CubeType.FACT, description="Type of cube: fact, aggregate, or virtual"
    )

    category: str | None = Field(
        default=None, description="Category for organizing cubes"
    )

    tags: set[str] = Field(
        default_factory=set, description="Tags for categorization and search"
    )

    # ========================================
    # Physical Backend Properties
    # ========================================

    fact_table: str | None = Field(
        default=None,
        description="Name of the fact table/collection/dataset",
        alias="fact",  # Support legacy 'fact' name
    )

    key_field: str | None = Field(
        default=None,
        description="Primary key field name",
        alias="key",  # Support legacy 'key' name
    )

    mappings: dict[str, Any] | None = Field(
        default=None, description="Backend-specific logical to physical mapping"
    )

    joins: list[dict[str, Any]] | None = Field(
        default=None, description="Backend-specific join specifications"
    )

    browser_options: dict[str, Any] = Field(
        default_factory=dict, description="Backend browser configuration options"
    )

    store_name: str | None = Field(
        default=None, description="Name of the data store containing this cube"
    )

    # ========================================
    # Advanced Features
    # ========================================

    nonadditive_dimensions: set[str] | None = Field(
        default=None, description="Dimensions where measures are non-additive"
    )

    estimated_size: PositiveInt | None = Field(
        default=None, description="Estimated number of fact records for optimization"
    )

    auto_generate_aggregates: bool = Field(
        default=True,
        description="Automatically generate default aggregates for measures",
    )

    # Legacy support for dimension links
    dimension_links: dict[str, Any] = Field(
        default_factory=dict,
        description="Dimension linking specifications (legacy support)",
        exclude=True,  # Don't serialize this field
    )

    # ========================================
    # Validation and Setup
    # ========================================

    @field_validator("dimensions", mode="before")
    @classmethod
    def validate_dimensions(cls, v: Any) -> list[Dimension]:
        """Convert various dimension inputs to Dimension objects"""
        if not v:
            return []

        result = []
        for dim in v:
            if isinstance(dim, Dimension):
                result.append(dim)
            elif isinstance(dim, str):
                result.append(Dimension(name=dim))
            elif isinstance(dim, dict):
                result.append(Dimension.model_validate(dim))
            else:
                raise ValueError(f"Invalid dimension type: {type(dim)}")

        return result

    @field_validator("measures", mode="before")
    @classmethod
    def validate_measures(cls, v: Any) -> list[Measure]:
        """Convert various measure inputs to Measure objects"""
        if not v:
            return []

        result = []
        for measure in v:
            if isinstance(measure, Measure):
                result.append(measure)
            elif isinstance(measure, str):
                result.append(Measure(name=measure))
            elif isinstance(measure, dict):
                result.append(Measure.model_validate(measure))
            else:
                raise ValueError(f"Invalid measure type: {type(measure)}")

        return result

    @field_validator("aggregates", mode="before")
    @classmethod
    def validate_aggregates(cls, v: Any) -> list[MeasureAggregate]:
        """Convert various aggregate inputs to MeasureAggregate objects"""
        if not v:
            return []

        result = []
        for agg in v:
            if isinstance(agg, MeasureAggregate):
                result.append(agg)
            elif isinstance(agg, str):
                result.append(MeasureAggregate(name=agg))
            elif isinstance(agg, dict):
                result.append(MeasureAggregate.model_validate(agg))
            else:
                raise ValueError(f"Invalid aggregate type: {type(agg)}")

        return result

    @field_validator("details", mode="before")
    @classmethod
    def validate_details(cls, v: Any) -> list[Attribute]:
        """Convert various detail inputs to Attribute objects"""
        if not v:
            return []

        result = []
        for detail in v:
            if isinstance(detail, Attribute):
                result.append(detail)
            elif isinstance(detail, str):
                result.append(Attribute(name=detail))
            elif isinstance(detail, dict):
                result.append(Attribute.model_validate(detail))
            else:
                raise ValueError(f"Invalid detail type: {type(detail)}")

        return result

    @field_validator("nonadditive_dimensions", mode="before")
    @classmethod
    def validate_nonadditive_dimensions(cls, v: Any) -> set[str] | None:
        """Convert nonadditive dimensions to set of strings"""
        if v is None:
            return None
        if isinstance(v, str):
            return {v}
        if hasattr(v, "__iter__"):
            return set(str(item) for item in v)
        return {str(v)}

    @field_validator("tags", mode="before")
    @classmethod
    def validate_tags(cls, v: Any) -> set[str]:
        """Convert tags to set of strings"""
        if not v:
            return set()
        if isinstance(v, str):
            return {v}
        if hasattr(v, "__iter__"):
            return set(str(tag) for tag in v)
        return {str(v)}

    @model_validator(mode="after")
    def setup_cube_and_validate(self) -> Cube:
        """Post-validation setup, populate dictionaries, and comprehensive validation"""

        # Step 1: Populate internal dictionaries from lists for O(1) lookups
        self._dimensions.clear()
        self._measures.clear()
        self._aggregates.clear()

        for dim in self.dimensions:
            if dim.name in self._dimensions:
                raise ValueError(
                    f"Duplicate dimension '{dim.name}' in cube '{self.name}'"
                )
            self._dimensions[dim.name] = dim

        for measure in self.measures:
            if measure.name in self._measures:
                raise ValueError(
                    f"Duplicate measure '{measure.name}' in cube '{self.name}'"
                )
            self._measures[measure.name] = measure

        for agg in self.aggregates:
            if agg.name in self._aggregates:
                raise ValueError(
                    f"Duplicate aggregate '{agg.name}' in cube '{self.name}'"
                )
            self._aggregates[agg.name] = agg

        # Step 2: Apply cube-level nonadditive to measures if not already set
        if self.nonadditive_dimensions:
            for measure in self.measures:
                if not hasattr(measure, "nonadditive") or measure.nonadditive is None:
                    measure.nonadditive = "any"

        # Step 3: Auto-generate aggregates if requested and no aggregates exist
        if self.auto_generate_aggregates and not self.aggregates and self.measures:
            generated_aggregates = self._generate_default_aggregates()
            self.aggregates.extend(generated_aggregates)
            # Add to dictionary too
            for agg in generated_aggregates:
                self._aggregates[agg.name] = agg

        # Step 4: If no measures or aggregates, add default fact count
        if not self.measures and not self.aggregates:
            default_agg = MeasureAggregate.model_validate(DEFAULT_FACT_COUNT_AGGREGATE)
            self.aggregates = [default_agg]
            self._aggregates[default_agg.name] = default_agg

        # Step 5: Generate automatic labels for aggregates if missing
        self._apply_automatic_aggregate_labels()

        # Step 6: Comprehensive validation
        self._validate_cube_integrity()

        return self

    def _validate_cube_integrity(self) -> None:
        """Comprehensive cube integrity validation"""

        # Validate aggregate references to measures
        for agg in self.aggregates:
            if (
                hasattr(agg, "measure")
                and agg.measure
                and agg.measure not in self._measures
            ):
                available_measures = list(self._measures.keys())
                suggestion = ""
                if available_measures:
                    import difflib

                    matches = difflib.get_close_matches(
                        agg.measure, available_measures, n=1
                    )
                    if matches:
                        suggestion = f" Did you mean '{matches[0]}'?"

                raise ValueError(
                    f"Aggregate '{agg.name}' references unknown measure '{agg.measure}'.{suggestion} "
                    f"Available measures: {', '.join(available_measures) if available_measures else 'none'}"
                )

        # Validate nonadditive dimension references
        if self.nonadditive_dimensions:
            available_dims = set(self._dimensions.keys())
            invalid_dims = self.nonadditive_dimensions - available_dims
            if invalid_dims:
                suggestion = ""
                if available_dims:
                    import difflib

                    # Get suggestions for each invalid dimension
                    suggestions = []
                    for invalid_dim in invalid_dims:
                        matches = difflib.get_close_matches(
                            invalid_dim, list(available_dims), n=1
                        )
                        if matches:
                            suggestions.append(f"'{invalid_dim}' -> '{matches[0]}'")
                    if suggestions:
                        suggestion = f" Suggestions: {', '.join(suggestions)}."

                raise ValueError(
                    f"Nonadditive dimensions {sorted(invalid_dims)} not found in cube dimensions.{suggestion} "
                    f"Available dimensions: {', '.join(sorted(available_dims)) if available_dims else 'none'}"
                )

        # Validate that measures and details don't have conflicting names
        detail_names = {d.name for d in self.details}
        measure_names = set(self._measures.keys())
        conflicts = detail_names & measure_names
        if conflicts:
            raise ValueError(
                f"Detail attributes have conflicting names with measures: {sorted(conflicts)}. "
                "Details and measures must have unique names."
            )

        # Validate cube has some form of data
        if not self._measures and not self._aggregates:
            raise ValueError(
                f"Cube '{self.name}' has no measures or aggregates. "
                "A cube must have at least one measure or aggregate to be useful."
            )

    # ========================================
    # Computed Properties
    # ========================================

    @computed_field
    @property
    def dimension_names(self) -> list[str]:
        """Names of all dimensions in order"""
        return list(self._dimensions.keys())

    @computed_field
    @property
    def measure_names(self) -> list[str]:
        """Names of all measures in order"""
        return list(self._measures.keys())

    @computed_field
    @property
    def aggregate_names(self) -> list[str]:
        """Names of all aggregates in order"""
        return list(self._aggregates.keys())

    @computed_field
    @property
    def all_attributes(self) -> list[Attribute]:
        """All attributes: dimensions, measures, aggregates, and details"""
        result = []

        # Dimension attributes
        for dimension in self.dimensions:
            if hasattr(dimension, "all_attributes"):
                result.extend(dimension.all_attributes)
            elif hasattr(dimension, "attributes"):
                result.extend(dimension.attributes)

        # Detail, measure, and aggregate attributes
        result.extend(self.details)
        result.extend(self.measures)
        result.extend(self.aggregates)

        return result

    @computed_field
    @property
    def all_fact_attributes(self) -> list[Attribute]:
        """Attributes for fact queries: dimensions, measures, and details"""
        result = []

        # Dimension attributes
        for dimension in self.dimensions:
            if hasattr(dimension, "all_attributes"):
                result.extend(dimension.all_attributes)
            elif hasattr(dimension, "attributes"):
                result.extend(dimension.attributes)

        # Fact-level attributes only
        result.extend(self.details)
        result.extend(self.measures)

        return result

    @computed_field
    @property
    def all_aggregate_attributes(self) -> list[Attribute]:
        """Attributes for aggregate queries: dimensions and aggregates"""
        result = []

        # Dimension attributes
        for dimension in self.dimensions:
            if hasattr(dimension, "all_attributes"):
                result.extend(dimension.all_attributes)
            elif hasattr(dimension, "attributes"):
                result.extend(dimension.attributes)

        # Aggregate attributes only
        result.extend(self.aggregates)

        return result

    @computed_field
    @property
    def all_dimension_keys(self) -> list[Attribute]:
        """Key attributes from all dimensions"""
        result = []
        for dimension in self.dimensions:
            if hasattr(dimension, "key_attributes"):
                result.extend(dimension.key_attributes)
        return result

    @computed_field
    @property
    def base_attributes(self) -> list[Attribute]:
        """
        Returns attributes that are not derived from other attributes.

        Base attributes don't depend on other cube attributes, variables or
        parameters. Any attribute with an expression is considered derived.

        Returns:
            List of base (non-derived) attributes
        """
        return [
            attr
            for attr in self.all_attributes
            if hasattr(attr, "is_base") and attr.is_base
        ]

    @computed_field
    @property
    def attribute_dependencies(self) -> dict[str, set[str]]:
        """
        Dictionary of dependencies between attributes.

        Maps attribute references to the attributes they depend on through expressions.
        For example, if attribute 'a' has expression 'b + c', returns: {"a": {"b", "c"}}

        Returns:
            Dictionary mapping attribute refs to their dependencies
        """
        dependencies = {}

        # Include all attributes and aggregates
        all_attrs = self.all_attributes
        if hasattr(self, "all_aggregate_attributes"):
            # Avoid duplication if aggregates are already in all_attributes
            aggregate_attrs = self.all_aggregate_attributes
            all_refs = {getattr(attr, "ref", attr.name) for attr in all_attrs}
            for attr in aggregate_attrs:
                attr_ref = getattr(attr, "ref", attr.name)
                if attr_ref not in all_refs:
                    all_attrs.append(attr)

        for attr in all_attrs:
            attr_ref = getattr(attr, "ref", attr.name)
            if hasattr(attr, "dependencies"):
                dependencies[attr_ref] = attr.dependencies
            else:
                dependencies[attr_ref] = set()

        return dependencies

    # ========================================
    # Private Helper Methods
    # ========================================

    @classmethod
    def _validate_dimension_input(
        cls, dimension: str | dict[str, Any] | Dimension, **options: Any
    ) -> Dimension:
        """Convert various dimension inputs to Dimension objects (reuses field validator logic)"""
        if isinstance(dimension, str):
            return Dimension(name=dimension, **options)
        elif isinstance(dimension, dict):
            dim_data = dict(dimension)
            dim_data.update(options)
            return Dimension.model_validate(dim_data)
        elif isinstance(dimension, Dimension):
            return dimension
        else:
            raise ArgumentError(f"Invalid dimension type: {type(dimension)}")

    @classmethod
    def _validate_measure_input(
        cls, measure: str | dict[str, Any] | Measure
    ) -> Measure:
        """Convert various measure inputs to Measure objects (reuses field validator logic)"""
        if isinstance(measure, str):
            return Measure(name=measure)
        elif isinstance(measure, dict):
            return Measure.model_validate(measure)
        elif isinstance(measure, Measure):
            return measure
        else:
            raise ArgumentError(f"Invalid measure type: {type(measure)}")

    @classmethod
    def _validate_aggregate_input(
        cls, aggregate: str | dict[str, Any] | MeasureAggregate
    ) -> MeasureAggregate:
        """Convert various aggregate inputs to MeasureAggregate objects (reuses field validator logic)"""
        if isinstance(aggregate, str):
            return MeasureAggregate(name=aggregate)
        elif isinstance(aggregate, dict):
            return MeasureAggregate.model_validate(aggregate)
        elif isinstance(aggregate, MeasureAggregate):
            return aggregate
        else:
            raise ArgumentError(f"Invalid aggregate type: {type(aggregate)}")

    def _generate_default_aggregates(self) -> list[MeasureAggregate]:
        """Generate default aggregates for all measures"""
        aggregates = []
        for measure in self.measures:
            if hasattr(measure, "default_aggregates"):
                aggregates.extend(measure.default_aggregates())
            else:
                # Fallback: create sum aggregate for measure
                aggregates.append(
                    MeasureAggregate(
                        name=f"{measure.name}_sum",
                        label=f"Sum of {measure.get_label()}",
                        function="sum",
                        measure=measure.name,
                    )
                )
        return aggregates

    def _apply_automatic_aggregate_labels(self) -> None:
        """Apply automatic labels to aggregates that don't have labels"""
        measure_dict = {m.name: m for m in self.measures if m.name}

        for aggregate in self.aggregates:
            if aggregate.label is None and hasattr(aggregate, "function"):
                # Try to find the measure this aggregate refers to
                measure = None
                if hasattr(aggregate, "measure") and aggregate.measure:
                    measure = measure_dict.get(aggregate.measure)

                # Generate label based on function and measure
                function = aggregate.function
                if function in IMPLICIT_AGGREGATE_LABELS:
                    template = IMPLICIT_AGGREGATE_LABELS[function]
                    if measure:
                        measure_label = measure.get_label()
                        aggregate.label = template.format(measure=measure_label)
                    elif "{measure}" not in template:
                        aggregate.label = template

    # ========================================
    # Public API Methods
    # ========================================

    def has_dimension(self, name: str) -> bool:
        """Check if cube has dimension with given name (O(1) lookup)"""
        return name in self._dimensions

    def get_dimension(self, name: str) -> Dimension:
        """Get dimension by name (alias for dimension method)"""
        return self.dimension(name)

    def has_measure(self, name: str) -> bool:
        """Check if cube has measure with given name (O(1) lookup)"""
        return name in self._measures

    def get_measure(self, name: str) -> Measure:
        """Get measure by name (alias for measure method)"""
        return self.measure(name)

    def has_aggregate(self, name: str) -> bool:
        """Check if cube has aggregate with given name (O(1) lookup)"""
        return name in self._aggregates

    def get_aggregate(self, name: str) -> MeasureAggregate:
        """Get aggregate by name (alias for aggregate method)"""
        return self.aggregate(name)

    def dimension(self, name: str) -> Dimension:
        """
        Get dimension by name (O(1) lookup).

        Args:
            name: Dimension name

        Returns:
            Dimension object

        Raises:
            NoSuchDimensionError: If dimension not found
        """
        if not name:
            raise NoSuchDimensionError(
                f"Requested dimension should not be none (cube '{self.name}')"
            )

        try:
            return self._dimensions[name]
        except KeyError:
            # Provide helpful error message with suggestions
            available = list(self._dimensions.keys())
            suggestion = ""
            if available:
                import difflib

                matches = difflib.get_close_matches(name, available, n=1)
                if matches:
                    suggestion = f" Did you mean '{matches[0]}'?"

            raise NoSuchDimensionError(
                f"Cube '{self.name}' has no dimension '{name}'.{suggestion} "
                f"Available dimensions: {', '.join(available) if available else 'none'}"
            )

    def measure(self, name: str) -> Measure:
        """
        Get measure by name (O(1) lookup).

        Args:
            name: Measure name

        Returns:
            Measure object

        Raises:
            NoSuchAttributeError: If measure not found
        """
        name = str(name)
        try:
            return self._measures[name]
        except KeyError:
            # Provide helpful error message with suggestions
            available = list(self._measures.keys())
            suggestion = ""
            if available:
                import difflib

                matches = difflib.get_close_matches(name, available, n=1)
                if matches:
                    suggestion = f" Did you mean '{matches[0]}'?"

            raise NoSuchAttributeError(
                f"Cube '{self.name}' has no measure '{name}'.{suggestion} "
                f"Available measures: {', '.join(available) if available else 'none'}"
            )

    def aggregate(self, name: str) -> MeasureAggregate:
        """
        Get aggregate by name (O(1) lookup).

        Args:
            name: Aggregate name

        Returns:
            MeasureAggregate object

        Raises:
            NoSuchAttributeError: If aggregate not found
        """
        name = str(name)
        try:
            return self._aggregates[name]
        except KeyError:
            # Provide helpful error message with suggestions
            available = list(self._aggregates.keys())
            suggestion = ""
            if available:
                import difflib

                matches = difflib.get_close_matches(name, available, n=1)
                if matches:
                    suggestion = f" Did you mean '{matches[0]}'?"

            raise NoSuchAttributeError(
                f"Cube '{self.name}' has no aggregate '{name}'.{suggestion} "
                f"Available aggregates: {', '.join(available) if available else 'none'}"
            )

    def get_measures(self, names: list[str] | None = None) -> list[Measure]:
        """
        Get list of measures by name.

        Args:
            names: List of measure names, or None for all measures

        Returns:
            List of Measure objects
        """
        if names is None:
            return list(self.measures)

        return [self.measure(name) for name in names]

    def get_aggregates(self, names: list[str] | None = None) -> list[MeasureAggregate]:
        """
        Get list of aggregates by name.

        Args:
            names: List of aggregate names, or None for all aggregates

        Returns:
            List of MeasureAggregate objects
        """
        if names is None:
            return list(self.aggregates)

        return [self.aggregate(name) for name in names]

    def aggregates_for_measure(self, measure_name: str) -> list[MeasureAggregate]:
        """
        Get all aggregates that use the specified measure.

        Args:
            measure_name: Name of the measure

        Returns:
            List of aggregates that reference the measure
        """
        result = []
        for agg in self.aggregates:
            if hasattr(agg, "measure") and agg.measure == measure_name:
                result.append(agg)
        return result

    def attribute(self, ref: str) -> Attribute:
        """
        Get any attribute by reference.

        Args:
            ref: Attribute reference (simple name or dimension.attribute format)

        Returns:
            The requested attribute

        Raises:
            NoSuchAttributeError: If attribute not found
        """
        ref = str(ref)

        # Try direct match first (measures, aggregates, details)
        for attr in self.measures + self.aggregates + self.details:
            if attr.name == ref:
                return attr
            # Also check ref property if available
            if hasattr(attr, "ref") and attr.ref == ref:
                return attr

        # Try dimension attributes with dot notation
        if "." in ref:
            dim_name, attr_name = ref.split(".", 1)
            try:
                dimension = self.dimension(dim_name)
                if hasattr(dimension, "attribute"):
                    return dimension.attribute(attr_name)
                elif hasattr(dimension, "get_attribute"):
                    return dimension.get_attribute(attr_name)
            except (NoSuchDimensionError, AttributeError, NoSuchAttributeError):
                pass

        # Try looking in all dimension attributes without dot notation
        for dimension in self.dimensions:
            if hasattr(dimension, "all_attributes"):
                attrs = dimension.all_attributes
            elif hasattr(dimension, "attributes"):
                attrs = dimension.attributes
            else:
                continue

            for attr in attrs:
                if attr.name == ref:
                    return attr
                if hasattr(attr, "ref") and attr.ref == ref:
                    return attr

        raise NoSuchAttributeError(f"Cube '{self.name}' has no attribute '{ref}'")

    def get_attribute(self, ref: str) -> Attribute:
        """Get any attribute by reference (alias for attribute method)"""
        return self.attribute(ref)

    def get_attributes(
        self, attributes: list[str] | None = None, aggregated: bool = False
    ) -> list[Attribute]:
        """
        Get list of cube attributes.

        Args:
            attributes: List of attribute references, or None for all
            aggregated: If True, return aggregate attributes; otherwise fact attributes

        Returns:
            List of Attribute objects
        """
        if not attributes:
            if aggregated:
                return self.all_aggregate_attributes
            else:
                return self.all_fact_attributes

        return [self.attribute(ref) for ref in attributes]

    def collect_dependencies(self, attributes: list[str]) -> list[Attribute]:
        """
        Collect all original and dependent cube attributes sorted by dependency.

        Args:
            attributes: List of attribute references

        Returns:
            List of attributes sorted by dependency (independents first)
        """
        # This is a simplified version - full dependency analysis would require
        # parsing expressions and building dependency graphs
        attr_objects = self.get_attributes(attributes)

        # For now, just return the attributes (dependency analysis would go here)
        return attr_objects

    # ========================================
    # Dimension Management
    # ========================================

    def add_dimension(
        self, dimension: str | dict[str, Any] | Dimension, **link_options: Any
    ) -> None:
        """
        Add dimension to cube.

        Args:
            dimension: Dimension to add (name, dict, or Dimension object)
            **link_options: Additional dimension linking options
        """
        new_dim = self._validate_dimension_input(dimension, **link_options)

        if not new_dim.name:
            raise ArgumentError("Dimension must have a name")

        # Check for duplicates
        if self.has_dimension(new_dim.name):
            raise ModelError(
                f"Dimension '{new_dim.name}' already exists in cube '{self.name}'"
            )

        self.dimensions.append(new_dim)
        self._dimensions[new_dim.name] = new_dim

    def link_dimension(self, dimension: Dimension) -> None:
        """
        Link dimension to cube with any specified link options.

        Args:
            dimension: Dimension to link
        """
        link_options = self.dimension_links.get(dimension.name, {})

        if link_options:
            # Clone dimension with link options
            if hasattr(dimension, "clone"):
                dimension = dimension.clone(**link_options)
            else:
                # Fallback: create new dimension with updated properties
                dim_data = dimension.model_dump()
                dim_data.update(link_options)
                dimension = Dimension.model_validate(dim_data)

        self.add_dimension(dimension)

    def remove_dimension(self, name: str) -> None:
        """Remove dimension by name (O(1) lookup + O(n) list removal)"""
        if name not in self._dimensions:
            available = list(self._dimensions.keys())
            suggestion = ""
            if available:
                import difflib

                matches = difflib.get_close_matches(name, available, n=1)
                if matches:
                    suggestion = f" Did you mean '{matches[0]}'?"

            raise NoSuchDimensionError(
                f"Dimension '{name}' not found in cube '{self.name}'.{suggestion} "
                f"Available dimensions: {', '.join(available) if available else 'none'}"
            )

        # Remove from dictionary (O(1))
        dimension_to_remove = self._dimensions.pop(name)

        # Remove from list (O(n))
        self.dimensions.remove(dimension_to_remove)

    # ========================================
    # Measure and Aggregate Management
    # ========================================

    def add_measure(self, measure: str | dict[str, Any] | Measure) -> None:
        """Add measure to cube"""
        new_measure = self._validate_measure_input(measure)

        if not new_measure.name:
            raise ArgumentError("Measure must have a name")

        # Check for duplicates
        if self.has_measure(new_measure.name):
            raise ModelError(
                f"Measure '{new_measure.name}' already exists in cube '{self.name}'"
            )

        self.measures.append(new_measure)
        self._measures[new_measure.name] = new_measure

        # Auto-generate aggregates if enabled
        if self.auto_generate_aggregates:
            new_aggregates = []
            if hasattr(new_measure, "default_aggregates"):
                new_aggregates = new_measure.default_aggregates()
            else:
                # Fallback aggregate
                new_aggregates = [
                    MeasureAggregate(
                        name=f"{new_measure.name}_sum",
                        label=f"Sum of {new_measure.get_label()}",
                        function="sum",
                        measure=new_measure.name,
                    )
                ]

            for agg in new_aggregates:
                if not self.has_aggregate(agg.name):
                    self.aggregates.append(agg)
                    self._aggregates[agg.name] = agg

    def remove_measure(self, name: str) -> None:
        """Remove measure by name (O(1) lookup + O(n) list removal)"""
        if name not in self._measures:
            available = list(self._measures.keys())
            suggestion = ""
            if available:
                import difflib

                matches = difflib.get_close_matches(name, available, n=1)
                if matches:
                    suggestion = f" Did you mean '{matches[0]}'?"

            raise NoSuchAttributeError(
                f"Measure '{name}' not found in cube '{self.name}'.{suggestion} "
                f"Available measures: {', '.join(available) if available else 'none'}"
            )

        # Remove from dictionary (O(1))
        measure_to_remove = self._measures.pop(name)

        # Remove from list (O(n))
        self.measures.remove(measure_to_remove)

    def add_aggregate(self, aggregate: str | dict[str, Any] | MeasureAggregate) -> None:
        """Add aggregate to cube"""
        new_agg = self._validate_aggregate_input(aggregate)

        if not new_agg.name:
            raise ArgumentError("Aggregate must have a name")

        # Check for duplicates
        if self.has_aggregate(new_agg.name):
            raise ModelError(
                f"Aggregate '{new_agg.name}' already exists in cube '{self.name}'"
            )

        self.aggregates.append(new_agg)
        self._aggregates[new_agg.name] = new_agg

    def remove_aggregate(self, name: str) -> None:
        """Remove aggregate by name (O(1) lookup + O(n) list removal)"""
        if name not in self._aggregates:
            available = list(self._aggregates.keys())
            suggestion = ""
            if available:
                import difflib

                matches = difflib.get_close_matches(name, available, n=1)
                if matches:
                    suggestion = f" Did you mean '{matches[0]}'?"

            raise NoSuchAttributeError(
                f"Aggregate '{name}' not found in cube '{self.name}'.{suggestion} "
                f"Available aggregates: {', '.join(available) if available else 'none'}"
            )

        # Remove from dictionary (O(1))
        aggregate_to_remove = self._aggregates.pop(name)

        # Remove from list (O(n))
        self.aggregates.remove(aggregate_to_remove)

    # ========================================
    # Validation and Integrity
    # ========================================

    def validate_references(self) -> list[str]:
        """
        Validate all references in the cube.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check aggregate measure references
        measure_names = set(self.measure_names)
        for aggregate in self.aggregates:
            if (
                hasattr(aggregate, "measure")
                and aggregate.measure
                and aggregate.measure not in measure_names
            ):
                errors.append(
                    f"Aggregate '{aggregate.name}' references unknown "
                    f"measure '{aggregate.measure}'"
                )

        # Check nonadditive dimension references
        if self.nonadditive_dimensions:
            dim_names = set(self.dimension_names)
            for dim_name in self.nonadditive_dimensions:
                if dim_name not in dim_names:
                    errors.append(
                        f"Nonadditive dimension '{dim_name}' not found "
                        f"in cube dimensions"
                    )

        return errors

    def validate(self) -> list[tuple[str, str]]:
        """
        Validate cube consistency (legacy format).

        Returns:
            List of (level, message) tuples
        """
        results = []

        # Validate reference integrity
        ref_errors = self.validate_references()
        for error in ref_errors:
            results.append(("error", error))

        # Check for duplicate names within collections
        measure_names = set()
        for measure in self.measures:
            if measure.name in measure_names:
                results.append(
                    (
                        "error",
                        f"Duplicate measure '{measure.name}' in cube '{self.name}'",
                    )
                )
            else:
                measure_names.add(measure.name)

        detail_names = set()
        for detail in self.details:
            if detail.name in detail_names:
                results.append(
                    ("error", f"Duplicate detail '{detail.name}' in cube '{self.name}'")
                )
            elif detail.name in measure_names:
                results.append(
                    (
                        "error",
                        f"Detail '{detail.name}' conflicts with measure "
                        f"name in cube '{self.name}'",
                    )
                )
            else:
                detail_names.add(detail.name)

        return results

    # ========================================
    # Localization
    # ========================================

    def localizable_dictionary(self) -> dict[str, Any]:
        """
        Return dictionary of localizable strings.

        Returns:
            Dictionary with localizable properties for the cube and its components
        """
        result = {}

        # Cube-level localizable properties
        if self.label:
            result["label"] = self.label
        if self.description:
            result["description"] = self.description
        if self.category:
            result["category"] = self.category

        # Measures
        if self.measures:
            measures_dict = {}
            for measure in self.measures:
                if hasattr(measure, "localizable_dictionary"):
                    measure_locals = measure.localizable_dictionary()
                    if measure_locals:
                        measures_dict[measure.name] = measure_locals
                elif measure.label:
                    measures_dict[measure.name] = {"label": measure.label}
            if measures_dict:
                result["measures"] = measures_dict

        # Aggregates
        if self.aggregates:
            aggregates_dict = {}
            for aggregate in self.aggregates:
                if hasattr(aggregate, "localizable_dictionary"):
                    agg_locals = aggregate.localizable_dictionary()
                    if agg_locals:
                        aggregates_dict[aggregate.name] = agg_locals
                elif aggregate.label:
                    aggregates_dict[aggregate.name] = {"label": aggregate.label}
            if aggregates_dict:
                result["aggregates"] = aggregates_dict

        # Details
        if self.details:
            details_dict = {}
            for detail in self.details:
                if hasattr(detail, "localizable_dictionary"):
                    detail_locals = detail.localizable_dictionary()
                    if detail_locals:
                        details_dict[detail.name] = detail_locals
                elif detail.label:
                    details_dict[detail.name] = {"label": detail.label}
            if details_dict:
                result["details"] = details_dict

        return result

    def localize(self, translations: dict[str, Any]) -> None:
        """
        Apply localization translations.

        Args:
            translations: Dictionary with translated strings
        """
        # Apply cube-level translations
        if "label" in translations:
            self.label = translations["label"]
        if "description" in translations:
            self.description = translations["description"]
        if "category" in translations:
            self.category = translations["category"]

        # Apply measure translations
        if "measures" in translations:
            measure_translations = translations["measures"]
            for measure in self.measures:
                if measure.name in measure_translations:
                    measure_trans = measure_translations[measure.name]
                    if hasattr(measure, "localize"):
                        measure.localize(measure_trans)
                    elif "label" in measure_trans:
                        measure.label = measure_trans["label"]

        # Apply aggregate translations
        if "aggregates" in translations:
            agg_translations = translations["aggregates"]
            for aggregate in self.aggregates:
                if aggregate.name in agg_translations:
                    agg_trans = agg_translations[aggregate.name]
                    if hasattr(aggregate, "localize"):
                        aggregate.localize(agg_trans)
                    elif "label" in agg_trans:
                        aggregate.label = agg_trans["label"]

        # Apply detail translations
        if "details" in translations:
            detail_translations = translations["details"]
            for detail in self.details:
                if detail.name in detail_translations:
                    detail_trans = detail_translations[detail.name]
                    if hasattr(detail, "localize"):
                        detail.localize(detail_trans)
                    elif "label" in detail_trans:
                        detail.label = detail_trans["label"]

    # ========================================
    # Serialization
    # ========================================

    def to_dict(self, **options: Any) -> dict[str, Any]:
        """
        Convert cube to dictionary representation.

        Args:
            **options: Serialization options
                - with_mappings: Include physical properties (default True)
                - expand_dimensions: Include full dimension definitions
                - create_label: Generate labels from names
        """
        # Filter options to only pass valid Pydantic options
        pydantic_options = {
            k: v
            for k, v in options.items()
            if k
            in {
                "exclude",
                "include",
                "by_alias",
                "exclude_unset",
                "exclude_defaults",
                "exclude_none",
                "round_trip",
                "warnings",
            }
        }

        # Use Pydantic's model_dump as base
        result = self.model_dump(
            exclude_none=True,
            by_alias=True,  # Use field aliases (fact, key)
            exclude={"dimension_links"},  # Exclude internal fields
            **pydantic_options,
        )

        # Legacy field name compatibility
        if "fact_table" in result:
            result["fact"] = result.pop("fact_table")
        if "key_field" in result:
            result["key"] = result.pop("key_field")

        # Handle dimension serialization
        if options.get("expand_dimensions", False):
            # Include full dimension definitions
            result["dimensions"] = [dim.to_dict(**options) for dim in self.dimensions]
        else:
            # Just dimension names (legacy default)
            result["dimensions"] = self.dimension_names

        # Include/exclude physical mappings
        if not options.get("with_mappings", True):
            for key in ["mappings", "fact", "joins", "browser_options"]:
                result.pop(key, None)

        # Convert sets to lists for JSON serialization
        if "tags" in result:
            result["tags"] = list(result["tags"])
        if "nonadditive_dimensions" in result:
            result["nonadditive_dimensions"] = list(result["nonadditive_dimensions"])

        return result

    @classmethod
    def from_metadata(cls, metadata: str | dict[str, Any]) -> Cube:
        """
        Create cube from metadata definition.

        Args:
            metadata: Either a cube name string or a dictionary with cube definition

        Returns:
            Cube instance
        """
        if isinstance(metadata, str):
            # Simple name-only cube
            return cls(name=metadata)
        elif isinstance(metadata, dict):
            # Full dictionary definition
            return cls.model_validate(metadata)
        else:
            raise ArgumentError(f"Invalid metadata type: {type(metadata)}")


# Utility Functions
# ========================================


# ========================================
# Factory Functions
# ========================================


def merge_cubes(*cubes: Cube, name: str) -> Cube:
    """
    Merge multiple cubes into a single virtual cube.

    Args:
        *cubes: Cubes to merge
        name: Name for merged cube

    Returns:
        New virtual cube containing all components
    """
    if not cubes:
        raise ArgumentError("At least one cube required for merging")

    # Collect unique components by name
    all_dimensions = {}
    all_measures = {}
    all_aggregates = {}
    all_details = {}

    for cube in cubes:
        for dim in cube.dimensions:
            if dim.name:
                all_dimensions[dim.name] = dim
        for measure in cube.measures:
            if measure.name:
                all_measures[measure.name] = measure
        for agg in cube.aggregates:
            if agg.name:
                all_aggregates[agg.name] = agg
        for detail in cube.details:
            if detail.name:
                all_details[detail.name] = detail

    return Cube(
        name=name,
        dimensions=list(all_dimensions.values()),
        measures=list(all_measures.values()),
        aggregates=list(all_aggregates.values()),
        details=list(all_details.values()),
        cube_type=CubeType.VIRTUAL,
        auto_generate_aggregates=False,  # Already have aggregates
    )


def create_star_schema_cube(
    name: str,
    fact_table: str,
    dimensions: list[str | dict | Dimension],
    measures: list[str | dict | Measure],
    **kwargs: Any,
) -> Cube:
    """
    Create cube optimized for star schema.

    Args:
        name: Cube name
        fact_table: Fact table name
        dimensions: Dimension definitions
        measures: Measure definitions
        **kwargs: Additional cube options

    Returns:
        Star schema cube configuration
    """
    return Cube(
        name=name,
        fact_table=fact_table,
        dimensions=dimensions,
        measures=measures,
        cube_type=CubeType.FACT,
        **kwargs,
    )
