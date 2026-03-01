"""Cube model for OLAP metadata.

Provides the Cube domain dataclass for representing analytical cubes
with dimensions, measures, aggregates, and their relationships.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

from cubes.metadata.models.attributes import (
    Attribute,
    AttributeBase,
    Measure,
    MeasureAggregate,
    collect_dependencies,
)
from cubes.metadata.models.base import (
    ArgumentError,
    ModelError,
    object_dict,
    to_label,
)
from cubes.metadata.models.dimension import Dimension


@dataclass
class Cube:
    """Logical representation of a cube for OLAP analysis.

    A cube represents a multidimensional data structure with dimensions,
    measures, and aggregates. It provides methods for attribute lookup,
    dimension linking, and dependency collection.

    Attributes:
        name: Cube name, used as identifier.
        measures: List of measure objects (numerical attributes).
        aggregates: List of measure aggregate objects.
        details: List of detail attribute objects.
        dimensions: List of dimension objects linked to the cube.
        mappings: Backend-specific logical to physical mapping dictionary.
        joins: Backend-specific join specifications.
        fact: Fact table/collection/dataset name.
        key: Fact key field name.
        browser_options: Dictionary of backend-specific browser options.
        store: Data store name where the cube belongs (can be string or object).
        dimension_links: List of dimension link specifications.
        label: Human-readable label (optional, auto-generated if not provided).
        description: Longer description of the cube (optional).
        info: Custom information dictionary (defaults to empty dict).
        category: Logical cube group for UI organization.
        basename: Base name of the cube (same as name by default).
        store_name: Name of the store if store was provided as string.

    Example:
        >>> cube = Cube(
        ...     name="sales",
        ...     measures=[Measure(name="amount")],
        ...     dimensions=[Dimension(name="date", levels=[...])]
        ... )
        >>> len(cube.measures)
        1
        >>> cube.measure("amount").name
        'amount'
    """

    name: str
    measures: list[Measure] = field(default_factory=list)
    aggregates: list[MeasureAggregate] = field(default_factory=list)
    details: list[Attribute] = field(default_factory=list)
    dimensions: list[Dimension] = field(default_factory=list)
    mappings: dict[str, Any] = field(default_factory=dict)
    joins: list[dict[str, Any]] = field(default_factory=list)
    fact: Optional[str] = None
    key: Optional[str] = None
    browser_options: dict[str, Any] = field(default_factory=dict)
    store: Optional[Union[str, object]] = None
    dimension_links: list[dict[str, Any]] = field(default_factory=list)
    label: Optional[str] = None
    description: Optional[str] = None
    info: dict[str, Any] = field(default_factory=dict)
    category: Optional[str] = None
    basename: str = field(init=False)
    store_name: Optional[str] = field(init=False)

    # Internal dictionaries for fast lookups (initialized in __post_init__)
    _measures: dict[str, Measure] = field(default_factory=dict, init=False, repr=False)
    _aggregates: dict[str, MeasureAggregate] = field(
        default_factory=dict, init=False, repr=False
    )
    _dimensions: dict[str, Dimension] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Initialize computed fields and build lookup dictionaries.

        This method:
        1. Validates measures and aggregates are correct types
        2. Builds measure and aggregate lookup dictionaries
        3. Builds dimension lookup dictionary
        4. Handles store string vs object
        5. Sets basename and store_name
        """
        # Set basename to name
        object.__setattr__(self, "basename", self.name)

        # Handle store string vs object
        if isinstance(self.store, str):
            object.__setattr__(self, "store_name", self.store)
            object.__setattr__(self, "store", None)
        else:
            object.__setattr__(self, "store_name", None)

        # Validate and build measures dictionary
        object.__setattr__(self, "_measures", {})
        for measure in self.measures:
            if not isinstance(measure, Measure):
                raise ModelError(
                    f"Measure '{measure}' in cube '{self.name}' is not a "
                    f"Measure instance. It is '{type(measure).__name__}'"
                )
            if measure.name in self._measures:
                raise ModelError(
                    f"Duplicate measure '{measure.name}' in cube '{self.name}'"
                )
            self._measures[measure.name] = measure

        # Validate and build aggregates dictionary
        object.__setattr__(self, "_aggregates", {})
        for aggregate in self.aggregates:
            if not isinstance(aggregate, MeasureAggregate):
                raise ModelError(
                    f"Aggregate '{aggregate}' in cube '{self.name}' is not a "
                    f"MeasureAggregate instance. It is '{type(aggregate).__name__}'"
                )
            if aggregate.name in self._aggregates:
                raise ModelError(
                    f"Duplicate aggregate '{aggregate.name}' in cube '{self.name}'"
                )
            self._aggregates[aggregate.name] = aggregate

        # Validate and build dimensions dictionary
        object.__setattr__(self, "_dimensions", {})
        for dim in self.dimensions:
            if not isinstance(dim, Dimension):
                raise ModelError(
                    f"Dimension '{dim}' in cube '{self.name}' is not a "
                    f"Dimension instance. It is '{type(dim).__name__}'"
                )
            if dim.name in self._dimensions:
                raise ModelError(
                    f"Duplicate dimension '{dim.name}' in cube '{self.name}'"
                )
            self._dimensions[dim.name] = dim

    @property
    def measures_list(self) -> list[Measure]:
        """Return list of all measures."""
        return list(self._measures.values())

    @property
    def aggregates_list(self) -> list[MeasureAggregate]:
        """Return list of all aggregates."""
        return list(self._aggregates.values())

    @property
    def dimensions_list(self) -> list[Dimension]:
        """Return list of all dimensions."""
        return list(self._dimensions.values())

    @property
    def all_dimension_keys(self) -> list[Attribute]:
        """Return all key attributes from all dimensions.

        Returns:
            List of key attributes from all dimension levels.
        """
        keys: list[Attribute] = []
        for dim in self._dimensions.values():
            keys.extend(dim.key_attributes)
        return keys

    def measure(self, name: str) -> Measure:
        """Get measure by name.

        Args:
            name: Measure name.

        Returns:
            The Measure object.

        Raises:
            KeyError: If measure not found.
        """
        if name not in self._measures:
            raise KeyError(f"Cube '{self.name}' has no measure '{name}'")
        return self._measures[name]

    def get_measures(
        self, measures: Optional[list[Union[str, Measure]]] = None
    ) -> list[Measure]:
        """Get list of measures.

        Args:
            measures: List of measure names or Measure objects. If None,
                returns all measures.

        Returns:
            List of Measure objects.
        """
        if measures is None:
            return self.measures_list

        result: list[Measure] = []
        for m in measures:
            if isinstance(m, Measure):
                result.append(m)
            else:
                result.append(self.measure(str(m)))
        return result

    def aggregate(self, name: str) -> MeasureAggregate:
        """Get aggregate by name.

        Args:
            name: Aggregate name.

        Returns:
            The MeasureAggregate object.

        Raises:
            KeyError: If aggregate not found.
        """
        if name not in self._aggregates:
            raise KeyError(f"Cube '{self.name}' has no aggregate '{name}'")
        return self._aggregates[name]

    def get_aggregates(
        self, names: Optional[list[Union[str, MeasureAggregate]]] = None
    ) -> list[MeasureAggregate]:
        """Get list of aggregates.

        Args:
            names: List of aggregate names or MeasureAggregate objects. If None,
                returns all aggregates.

        Returns:
            List of MeasureAggregate objects.
        """
        if names is None:
            return self.aggregates_list

        result: list[MeasureAggregate] = []
        for name in names:
            if isinstance(name, MeasureAggregate):
                result.append(name)
            else:
                result.append(self.aggregate(str(name)))
        return result

    def aggregates_for_measure(self, name: str) -> list[MeasureAggregate]:
        """Get aggregates that reference a specific measure.

        Args:
            name: Measure name.

        Returns:
            List of aggregates that have this measure.
        """
        return [agg for agg in self._aggregates.values() if agg.measure == name]

    def dimension(self, obj: Union[str, Dimension]) -> Dimension:
        """Get dimension by name or as Dimension object.

        Args:
            obj: Dimension name (string) or Dimension object.

        Returns:
            The Dimension object.

        Raises:
            KeyError: If dimension name not found.
            ValueError: If obj is neither string nor Dimension.
        """
        if isinstance(obj, str):
            if obj not in self._dimensions:
                raise KeyError(f"Cube '{self.name}' has no dimension '{obj}'")
            return self._dimensions[obj]
        elif isinstance(obj, Dimension):
            return obj
        else:
            raise ValueError(
                f"Unknown dimension object {obj} (should be a string or Dimension)"
            )

    @property
    def all_attributes(self) -> list[AttributeBase]:
        """Return all cube attributes including dimensions, details, measures, and aggregates.

        Returns:
            List of all attributes from dimensions, details, measures, and aggregates.
        """
        attributes: list[AttributeBase] = []

        # Add dimension attributes
        for dim in self._dimensions.values():
            attributes.extend(dim.attributes)

        # Add details, measures, and aggregates
        attributes.extend(self.details)
        attributes.extend(self._measures.values())
        attributes.extend(self._aggregates.values())

        return attributes

    @property
    def base_attributes(self) -> list[AttributeBase]:
        """Return attributes that are not derived (no expression).

        Returns:
            List of base attributes.
        """
        return [attr for attr in self.all_attributes if attr.is_base]

    @property
    def all_fact_attributes(self) -> list[AttributeBase]:
        """Return all fact attributes (dimensions, details, measures).

        Returns:
            List of fact attributes (excludes aggregates).
        """
        attributes: list[AttributeBase] = []

        # Add dimension attributes
        for dim in self._dimensions.values():
            attributes.extend(dim.attributes)

        # Add details and measures
        attributes.extend(self.details)
        attributes.extend(self._measures.values())

        return attributes

    @property
    def attribute_dependencies(self) -> dict[str, set[str]]:
        """Return dictionary of attribute dependencies.

        Returns:
            Dictionary mapping attribute refs to their dependency sets.
        """
        return {attr.ref: attr.dependencies for attr in self.all_attributes}

    @property
    def all_aggregate_attributes(self) -> list[AttributeBase]:
        """Return all attributes for aggregation (dimensions + aggregates).

        Returns:
            List of dimension attributes and aggregates.
        """
        attributes: list[AttributeBase] = []

        # Add dimension attributes
        for dim in self._dimensions.values():
            attributes.extend(dim.attributes)

        # Add aggregates
        attributes.extend(self._aggregates.values())

        return attributes

    def attribute(self, name: str) -> AttributeBase:
        """Get attribute by name across dimensions, details, and measures.

        Args:
            name: Attribute name or reference.

        Returns:
            The AttributeBase object.

        Raises:
            KeyError: If attribute not found.
        """
        # Try dimensions first (by ref)
        for dim in self._dimensions.values():
            try:
                return dim.attribute(name, by_ref=True)
            except KeyError:
                continue

        # Try details
        for detail in self.details:
            if detail.name == name or detail.ref == name:
                return detail

        # Try measures
        for measure in self._measures.values():
            if measure.name == name or measure.ref == name:
                return measure

        raise KeyError(f"Cube '{self.name}' has no attribute '{name}'")

    def get_attributes(
        self,
        attributes: Optional[list[Union[str, AttributeBase]]] = None,
        aggregated: bool = False,
    ) -> list[AttributeBase]:
        """Get list of cube attributes.

        Args:
            attributes: List of attribute names or objects. If None, returns all.
            aggregated: If True, return aggregated attributes (dimensions + aggregates).
                If False, return fact attributes (dimensions + details + measures).

        Returns:
            List of AttributeBase objects.
        """
        if not attributes:
            if aggregated:
                return self.all_aggregate_attributes
            else:
                return self.all_fact_attributes

        # Build lookup dict of all attributes
        everything: dict[str, AttributeBase] = {}
        for attr in self.all_attributes:
            everything[attr.ref] = attr
            everything[attr.name] = attr

        result: list[AttributeBase] = []
        for attr_input in attributes:
            attr_name = str(attr_input)
            if attr_name not in everything:
                raise KeyError(f"Unknown attribute '{attr_name}' in cube '{self.name}'")
            result.append(everything[attr_name])

        return result

    def collect_dependencies(
        self, attributes: list[Union[str, AttributeBase]]
    ) -> list[AttributeBase]:
        """Collect all attributes with their dependencies in topological order.

        Args:
            attributes: List of attribute names or objects.

        Returns:
            List of attributes sorted by dependency (dependencies first).
        """
        # Get the attribute objects
        attrs = self.get_attributes(attributes)

        # Get sorted refs
        sorted_refs = collect_dependencies(attrs, self.all_attributes)

        # Convert refs back to attribute objects
        # sorted_refs is list[str], need to cast for type checker
        refs: list[Union[str, AttributeBase]] = list(sorted_refs)
        return self.get_attributes(refs)

    def link_dimension(self, dimension: Dimension) -> None:
        """Link a dimension to the cube with optional cloning.

        If the cube has dimension_links for this dimension, the dimension
        will be cloned according to the link specification.

        Args:
            dimension: The Dimension object to link.
        """
        # Find link specification for this dimension
        link: Optional[dict[str, Any]] = None
        for dl in self.dimension_links:
            if dl.get("name") == dimension.name:
                link = dl
                break

        if link:
            # Clone dimension with link parameters
            clone_kwargs: dict[str, Any] = {}
            if "hierarchies" in link:
                clone_kwargs["hierarchies"] = link["hierarchies"]
            if "exclude_hierarchies" in link:
                clone_kwargs["exclude_hierarchies"] = link["exclude_hierarchies"]
            if "nonadditive" in link:
                clone_kwargs["nonadditive"] = link["nonadditive"]
            if "cardinality" in link:
                clone_kwargs["cardinality"] = link["cardinality"]
            if "default_hierarchy_name" in link:
                clone_kwargs["default_hierarchy_name"] = link["default_hierarchy_name"]
            if "alias" in link:
                clone_kwargs["alias"] = link["alias"]

            dimension = dimension.clone(**clone_kwargs)

        self._add_dimension(dimension)

    def _add_dimension(self, dimension: Dimension) -> None:
        """Add a dimension to the cube.

        Args:
            dimension: The Dimension object to add.

        Raises:
            ArgumentError: If dimension is None or not a Dimension instance.
        """
        if dimension is None:
            raise ArgumentError(f"Trying to add None dimension to cube '{self.name}'")

        if not isinstance(dimension, Dimension):
            raise ArgumentError(
                f"Dimension added to cube '{self.name}' is not a Dimension instance. "
                f"It is '{type(dimension).__name__}'"
            )

        self._dimensions[dimension.name] = dimension

    @property
    def distilled_hierarchies(self) -> dict[tuple[str, Optional[str]], list[str]]:
        """Return dictionary of hierarchies with their level key refs.

        Returns:
            Dictionary mapping (dimension_name, hierarchy_name) tuples to
            lists of level key attribute refs. If hierarchy_name is None,
            it represents the default hierarchy.
        """
        hierarchies: dict[tuple[str, Optional[str]], list[str]] = {}

        for dim in self._dimensions.values():
            # Dimension.__post_init__ always ensures hierarchies is set
            dim_hierarchies = dim.hierarchies if dim.hierarchies is not None else []
            for hier in dim_hierarchies:
                key = (dim.name, hier.name)
                # hier.keys() returns key attributes
                level_keys = [k.ref for k in hier.keys()]
                hierarchies[key] = level_keys

                # Also add entry for default hierarchy with None as name
                if dim.default_hierarchy_name == hier.name:
                    hierarchies[(dim.name, None)] = level_keys

        return hierarchies

    def to_dict(
        self,
        create_label: bool = False,
        expand_dimensions: bool = False,
        hierarchy_limits: Optional[dict[str, Optional[str]]] = None,
        with_mappings: bool = True,
    ) -> dict[str, Any]:
        """Convert to a dictionary representation.

        Args:
            create_label: If True and no label is set, auto-generate from name.
            expand_dimensions: If True, expand dimensions to full dicts.
                If False, just list dimension names.
            hierarchy_limits: Dict mapping dimension names to hierarchy limits.
                Each limit can be a level name to truncate at, or None to exclude.
            with_mappings: If True, include mappings, fact, joins, browser_options.

        Returns:
            Dictionary with all cube properties.
        """
        result: dict[str, Any] = {
            "name": self.name,
            "label": self.label
            if self.label
            else (to_label(self.name) if create_label else None),
            "description": self.description,
            "info": dict(self.info),
            "category": self.category,
        }

        # Add aggregates
        result["aggregates"] = [
            agg.to_dict(create_label=create_label) for agg in self._aggregates.values()
        ]

        # Add measures
        result["measures"] = [
            m.to_dict(create_label=create_label) for m in self._measures.values()
        ]

        # Add details
        result["details"] = [d.to_dict(create_label=create_label) for d in self.details]

        # Add dimensions
        if expand_dimensions:
            hierarchy_limits = hierarchy_limits or {}
            dims: list[dict[str, Any]] = []
            for dim in self._dimensions.values():
                limit = hierarchy_limits.get(dim.name)
                if limit is None:
                    # No limit, include full dimension
                    dims.append(dim.to_dict(create_label=create_label))
                else:
                    # Truncate or exclude
                    if limit == "":
                        # Empty string means exclude
                        continue
                    dims.append(
                        dim.to_dict(
                            create_label=create_label, hierarchy_limits={limit: limit}
                        )
                    )
            result["dimensions"] = dims
        else:
            result["dimensions"] = [dim.name for dim in self._dimensions.values()]

        # Add physical properties if requested
        if with_mappings:
            result["mappings"] = dict(self.mappings)
            result["fact"] = self.fact
            result["joins"] = list(self.joins)
            result["browser_options"] = dict(self.browser_options)

        result["key"] = self.key

        return result

    @classmethod
    def from_schema(
        cls,
        schema: Any,
        dimensions: Optional[dict[str, Dimension]] = None,
        templates: Optional[dict[str, Dimension]] = None,
    ) -> "Cube":
        """Create a Cube instance from a validated CubeSchema.

        This factory method creates a Cube from a Pydantic schema,
        linking dimensions as specified.

        Args:
            schema: A validated CubeSchema instance.
            dimensions: Optional dict of available dimensions by name.
            templates: Optional dict of template dimensions for inheritance.

        Returns:
            A new Cube instance with linked dimensions.

        Raises:
            ModelError: If a dimension specified in links is not found.
        """
        from cubes.metadata.schemas.attributes import (
            MeasureAggregateSchema,
            MeasureSchema,
        )
        from cubes.metadata.schemas.cube import CubeSchema

        # Handle both schema object and dict
        if isinstance(schema, dict):
            schema = CubeSchema.model_validate(schema)

        dimensions = dimensions or {}
        templates = templates or {}

        # Create measures from schema
        measures: list[Measure] = []

        def _normalize_nonadditive(
            val: Optional[str],
        ) -> Optional[Literal["all", "time"]]:
            """Normalize nonadditive value - 'none' becomes None."""
            if val is None or val.lower() == "none":
                return None
            return val  # type: ignore[return-value]

        for measure_input in schema.measures:
            if isinstance(measure_input, MeasureSchema):
                measure = Measure(
                    name=measure_input.name,
                    label=measure_input.label,
                    description=measure_input.description,
                    order=measure_input.order,
                    info=dict(measure_input.info),
                    format=measure_input.format,
                    missing_value=measure_input.missing_value,
                    expression=measure_input.expression,
                    formula=measure_input.formula,
                    aggregates=list(measure_input.aggregates)
                    if measure_input.aggregates
                    else ["sum"],
                    nonadditive=_normalize_nonadditive(measure_input.nonadditive),
                    window_size=measure_input.window_size,
                )
            elif isinstance(measure_input, dict):
                measure = Measure(
                    name=measure_input.get("name", ""),
                    label=measure_input.get("label"),
                    description=measure_input.get("description"),
                    order=measure_input.get("order"),
                    info=dict(measure_input.get("info", {})),
                    format=measure_input.get("format"),
                    missing_value=measure_input.get("missing_value"),
                    expression=measure_input.get("expression"),
                    formula=measure_input.get("formula"),
                    aggregates=measure_input.get("aggregates", ["sum"]),
                    nonadditive=_normalize_nonadditive(
                        measure_input.get("nonadditive")
                    ),
                    window_size=measure_input.get("window_size"),
                )
            else:
                # String - simple measure name
                measure = Measure(name=str(measure_input))
            measures.append(measure)

        # Create aggregates from schema
        aggregates: list[MeasureAggregate] = []
        for agg_input in schema.aggregates:
            if isinstance(agg_input, MeasureAggregateSchema):
                aggregate = MeasureAggregate(
                    name=agg_input.name,
                    label=agg_input.label,
                    description=agg_input.description,
                    order=agg_input.order,
                    info=dict(agg_input.info),
                    format=agg_input.format,
                    missing_value=agg_input.missing_value,
                    expression=agg_input.expression,
                    function=agg_input.function,
                    formula=agg_input.formula,
                    measure=agg_input.measure,
                    nonadditive=_normalize_nonadditive(agg_input.nonadditive),
                    window_size=agg_input.window_size,
                )
            elif isinstance(agg_input, dict):
                aggregate = MeasureAggregate(
                    name=agg_input.get("name", ""),
                    label=agg_input.get("label"),
                    description=agg_input.get("description"),
                    order=agg_input.get("order"),
                    info=dict(agg_input.get("info", {})),
                    format=agg_input.get("format"),
                    missing_value=agg_input.get("missing_value"),
                    expression=agg_input.get("expression"),
                    function=agg_input.get("function"),
                    formula=agg_input.get("formula"),
                    measure=agg_input.get("measure"),
                    nonadditive=_normalize_nonadditive(agg_input.get("nonadditive")),
                    window_size=agg_input.get("window_size"),
                )
            else:
                # String - simple aggregate name
                aggregate = MeasureAggregate(name=str(agg_input))
            aggregates.append(aggregate)

        # Create details from schema (treat as simple attributes)
        details: list[Attribute] = []
        for detail_input in schema.details:
            if isinstance(detail_input, dict):
                detail = Attribute(
                    name=detail_input.get("name", ""),
                    label=detail_input.get("label"),
                    description=detail_input.get("description"),
                    order=detail_input.get("order"),
                    info=dict(detail_input.get("info", {})),
                    format=detail_input.get("format"),
                    missing_value=detail_input.get("missing_value"),
                    expression=detail_input.get("expression"),
                )
            else:
                # String - simple attribute name
                detail = Attribute(name=str(detail_input))
            details.append(detail)

        # Process dimension links
        dimension_links: list[dict[str, Any]] = []
        for link in schema.dimensions:
            if isinstance(link, dict):
                dimension_links.append(dict(link))
            else:
                # String dimension name
                dimension_links.append({"name": str(link)})

        # Create cube
        cube = cls(
            name=schema.name,
            measures=measures,
            aggregates=aggregates,
            details=details,
            mappings=dict(schema.mappings) if schema.mappings else {},
            joins=list(schema.joins) if schema.joins else [],
            fact=schema.fact,
            key=schema.key,
            browser_options=dict(schema.browser_options)
            if schema.browser_options
            else {},
            store=schema.store,
            dimension_links=dimension_links,
            label=schema.label,
            description=schema.description,
            info=dict(schema.info) if schema.info else {},
            category=schema.category,
        )

        # Link dimensions if provided
        for link in dimension_links:
            dim_name = link.get("name")
            if dim_name and dim_name in dimensions:
                cube.link_dimension(dimensions[dim_name])

        return cube

    def __eq__(self, other: object) -> bool:
        """Check equality with another Cube.

        Two cubes are equal if they have the same name, dimensions,
        measures, aggregates, and details.

        Args:
            other: Another object to compare with.

        Returns:
            True if equal, False otherwise.
        """
        if other is None or not isinstance(other, Cube):
            return False

        return (
            self.name == other.name
            and self.label == other.label
            and self.description == other.description
            and self._dimensions == other._dimensions
            and self._measures == other._measures
            and self._aggregates == other._aggregates
            and self.details == other.details
            and self.mappings == other.mappings
            and self.joins == other.joins
            and self.browser_options == other.browser_options
            and self.info == other.info
        )

    def __hash__(self) -> int:
        """Return hash based on name for use in sets and dicts."""
        return hash(self.name)

    def __str__(self) -> str:
        """Return string representation (name)."""
        return self.name

    def __repr__(self) -> str:
        """Return detailed representation."""
        return f"<cube: {self.name}>"
