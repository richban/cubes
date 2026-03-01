"""Pydantic schemas for cube metadata.

Provides input validation and parsing for cube definitions including
dimension links, measures, aggregates, and implicit aggregate generation.
"""

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cubes.metadata.schemas.attributes import (
    MeasureAggregateInput,
    MeasureAggregateSchema,
    MeasureInput,
    MeasureSchema,
)

# Default fact count aggregate when no measures/aggregates specified
DEFAULT_FACT_COUNT_AGGREGATE = {
    "name": "fact_count",
    "label": "Count",
    "function": "count",
}

# Template strings for implicit aggregate labels
IMPLICIT_AGGREGATE_LABELS = {
    "sum": "Sum of {measure}",
    "count": "Record Count",
    "count_nonempty": "Non-empty count of {measure}",
    "min": "{measure} Minimum",
    "max": "{measure} Maximum",
    "avg": "Average of {measure}",
}


def _expand_dimension_link(link: Any) -> dict[str, Any]:
    """Expand a dimension link from string or dict to normalized dict.

    Args:
        link: Dimension link, either a string (dimension name) or dict.

    Returns:
        Normalized dimension link dictionary with at least a 'name' key.
    """
    if isinstance(link, str):
        return {"name": link}
    elif isinstance(link, dict):
        if "name" not in link:
            raise ValueError("Dimension link must have a 'name' field")
        return dict(link)
    else:
        raise ValueError(
            f"Dimension link must be string or dict, got {type(link).__name__}"
        )


def _get_measure_aggregate_label(function: str, measure_name: str) -> str:
    """Generate a label for an implicit aggregate.

    Args:
        function: Aggregate function name.
        measure_name: Name of the measure being aggregated.

    Returns:
        Human-readable label for the aggregate.
    """
    template = IMPLICIT_AGGREGATE_LABELS.get(function, "{measure}")
    return template.format(measure=measure_name)


class CubeSchema(BaseModel):
    """Pydantic schema for validating cube metadata.

    This schema handles raw JSON/dict input validation for cubes including
    dimension links, measures, aggregates, and implicit aggregate generation.

    Attributes:
        name: Cube name, used as identifier (required).
        dimensions: List of dimension links. Can be dimension names (strings)
            or dictionaries with link configuration.
        measures: List of measure definitions for the cube.
        aggregates: List of aggregate definitions for the cube.
        details: List of detail attribute definitions.
        label: Human-readable label displayed to users.
        description: Longer description of the cube.
        info: Custom information dictionary for app-specific metadata.
        category: Logical cube group for UI organization.
        mappings: Backend-specific logical to physical mapping dictionary.
        joins: Backend-specific join specifications.
        fact: Fact table/collection/dataset name.
        key: Fact key field name.
        browser_options: Dictionary of backend-specific browser options.
        store: Data store name where the cube belongs.
        implicit_aggregates: If True, generate implicit aggregates from measures.
    """

    model_config = ConfigDict(
        # Allow extra fields for forward compatibility
        extra="allow",
        # Don't validate assignments to avoid recursion issues with validators
        validate_assignment=False,
        # Don't allow mutation after creation
        frozen=False,
    )

    name: str = Field(..., description="Cube name, used as identifier")
    dimensions: list[Any] = Field(
        default_factory=list,
        description="List of dimension links (strings or dicts)",
    )
    measures: list[MeasureInput] = Field(
        default_factory=list,
        description="List of measure definitions",
    )
    aggregates: list[MeasureAggregateInput] = Field(
        default_factory=list,
        description="List of aggregate definitions",
    )
    details: list[Any] = Field(
        default_factory=list,
        description="List of detail attribute definitions",
    )
    label: Optional[str] = Field(None, description="Human-readable label")
    description: Optional[str] = Field(None, description="Longer description")
    info: dict[str, Any] = Field(
        default_factory=dict, description="Custom information dictionary"
    )
    category: Optional[str] = Field(None, description="Logical cube group")
    mappings: dict[str, Any] = Field(
        default_factory=dict,
        description="Backend-specific logical to physical mapping",
    )
    joins: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Backend-specific join specifications",
    )
    fact: Optional[str] = Field(None, description="Fact table/collection name")
    key: Optional[str] = Field(None, description="Fact key field name")
    browser_options: dict[str, Any] = Field(
        default_factory=dict,
        description="Backend-specific browser options",
    )
    store: Optional[str] = Field(None, description="Data store name")
    implicit_aggregates: bool = Field(
        False,
        description="Generate implicit aggregates from measures",
    )

    @model_validator(mode="after")
    def _expand_dimension_links(self) -> "CubeSchema":
        """Expand dimension links from strings to normalized dictionaries.

        Converts string dimension names to {"name": name} dicts and
        validates that all links have required fields.
        """
        if not self.dimensions:
            return self

        expanded_links: list[dict[str, Any]] = []
        for link in self.dimensions:
            expanded = _expand_dimension_link(link)
            expanded_links.append(expanded)

        self.dimensions = expanded_links
        return self

    @model_validator(mode="after")
    def _propagate_nonadditive_to_measures(self) -> "CubeSchema":
        """Propagate nonadditive property from cube to measures.

        If the cube has a nonadditive property and measures don't have
        their own nonadditive setting, inherit it from the cube.
        """
        # Check if cube has nonadditive in extra fields (not explicitly defined)
        cube_nonadditive = getattr(self, "nonadditive", None)
        if cube_nonadditive is None:
            return self

        # Propagate to measures that don't have their own nonadditive setting
        updated_measures: list[MeasureSchema] = []
        for measure in self.measures:
            if isinstance(measure, dict):
                # Convert dict to MeasureSchema if needed
                if "nonadditive" not in measure:
                    measure["nonadditive"] = cube_nonadditive
                updated_measures.append(MeasureSchema.model_validate(measure))
            elif isinstance(measure, MeasureSchema):
                if measure.nonadditive is None:
                    # Create a copy with nonadditive set
                    measure_dict = measure.model_dump()
                    measure_dict["nonadditive"] = cube_nonadditive
                    updated_measures.append(MeasureSchema.model_validate(measure_dict))
                else:
                    updated_measures.append(measure)

        self.measures = updated_measures
        return self

    @model_validator(mode="before")
    @classmethod
    def _generate_default_fact_count(cls, data: Any) -> Any:
        """Generate default fact_count aggregate if no measures or aggregates.

        If the cube has neither measures nor aggregates specified, create a
        default fact_count aggregate to ensure the cube is usable.

        This uses mode="before" to check if the 'aggregates' key exists in
        the raw data, distinguishing between "not provided" and "empty list".
        """
        if isinstance(data, dict):
            has_measures = "measures" in data and data["measures"]
            has_aggregates = "aggregates" in data  # Key exists (even if empty)

            if not has_measures and not has_aggregates:
                data = dict(data)
                data["aggregates"] = [DEFAULT_FACT_COUNT_AGGREGATE]

        return data

    @model_validator(mode="after")
    def _generate_implicit_aggregates(self) -> "CubeSchema":
        """Generate implicit aggregates from measures if enabled.

        If implicit_aggregates is True, create aggregates for each measure
        based on its default_aggregates. Existing aggregates are not duplicated.
        """
        if not self.implicit_aggregates or not self.measures:
            return self

        # Build set of existing aggregate names for deduplication
        existing_names = set()
        existing_signatures = set()  # (function, measure) tuples

        for agg in self.aggregates:
            if isinstance(agg, MeasureAggregateSchema):
                existing_names.add(agg.name)
                if agg.function and agg.measure:
                    existing_signatures.add((agg.function, agg.measure))
            elif isinstance(agg, dict):
                existing_names.add(agg.get("name", ""))
                func = agg.get("function")
                measure = agg.get("measure")
                if func and measure:
                    existing_signatures.add((func, measure))

        # Generate implicit aggregates from measures
        new_aggregates: list[MeasureAggregateSchema] = []

        for measure in self.measures:
            if isinstance(measure, dict):
                measure_name = measure.get("name", "")
                measure_aggregates = measure.get("aggregates", ["sum"])
            else:
                measure_name = measure.name
                measure_aggregates = (
                    measure.aggregates if measure.aggregates else ["sum"]
                )

            for function in measure_aggregates:
                # Check if aggregate already exists by name
                if function == "identity":
                    agg_name = measure_name
                else:
                    agg_name = f"{measure_name}_{function}"

                if agg_name in existing_names:
                    continue

                # Check if aggregate exists by signature
                if (function, measure_name) in existing_signatures:
                    continue

                # Create the implicit aggregate
                if function == "identity":
                    # Identity aggregate has no function or measure
                    agg_data = {
                        "name": agg_name,
                        "measure": None,
                        "function": None,
                    }
                else:
                    agg_data = {
                        "name": agg_name,
                        "measure": measure_name,
                        "function": function,
                    }

                # Generate label
                label = _get_measure_aggregate_label(function, measure_name)
                agg_data["label"] = label

                # Inherit nonadditive from measure if available
                if isinstance(measure, dict):
                    measure_nonadditive = measure.get("nonadditive")
                else:
                    measure_nonadditive = measure.nonadditive

                if measure_nonadditive:
                    agg_data["nonadditive"] = measure_nonadditive

                new_aggregates.append(MeasureAggregateSchema.model_validate(agg_data))

        # Add new aggregates to the list
        self.aggregates = list(self.aggregates) + new_aggregates

        return self
