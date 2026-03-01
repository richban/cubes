"""Tests for the Cube domain dataclass.

Comprehensive tests covering initialization, properties, lookup methods,
attribute collection, dimension linking, and serialization.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from cubes.metadata.models.attributes import (
    Attribute,
    Measure,
    MeasureAggregate,
)
from cubes.metadata.models.base import ArgumentError, ModelError
from cubes.metadata.models.cube import Cube
from cubes.metadata.models.dimension import Dimension, Hierarchy, Level


# Test Fixtures


@pytest.fixture
def simple_level() -> Level:
    """Create a simple level for testing."""
    return Level(
        name="month",
        attributes=[Attribute(name="month"), Attribute(name="month_name")],
    )


@pytest.fixture
def simple_dimension(simple_level: Level) -> Dimension:
    """Create a simple dimension for testing."""
    return Dimension(
        name="date",
        levels=[simple_level],
        hierarchies=[Hierarchy(name="default", levels=[simple_level])],
    )


@pytest.fixture
def product_dimension() -> Dimension:
    """Create a product dimension for testing."""
    level = Level(
        name="product",
        attributes=[
            Attribute(name="id"),
            Attribute(name="name"),
            Attribute(name="description"),
        ],
    )
    return Dimension(
        name="product",
        levels=[level],
        hierarchies=[Hierarchy(name="default", levels=[level])],
    )


@pytest.fixture
def simple_measure() -> Measure:
    """Create a simple measure for testing."""
    return Measure(name="amount")


@pytest.fixture
def simple_cube(simple_dimension: Dimension, simple_measure: Measure) -> Cube:
    """Create a simple cube for testing."""
    return Cube(
        name="sales",
        measures=[simple_measure],
        dimensions=[simple_dimension],
    )


# Initialization Tests


class TestCubeInitialization:
    """Test cube initialization and __post_init__."""

    def test_basic_initialization(self) -> None:
        """Test basic cube creation with name only."""
        cube = Cube(name="test")
        assert cube.name == "test"
        assert cube.basename == "test"
        assert cube.measures_list == []
        assert cube.aggregates_list == []
        assert cube.dimensions_list == []

    def test_initialization_with_measures(self) -> None:
        """Test cube creation with measures."""
        measures = [Measure(name="amount"), Measure(name="quantity")]
        cube = Cube(name="sales", measures=measures)
        assert len(cube.measures_list) == 2
        assert cube.measure("amount").name == "amount"
        assert cube.measure("quantity").name == "quantity"

    def test_initialization_with_aggregates(self) -> None:
        """Test cube creation with aggregates."""
        aggregates = [
            MeasureAggregate(name="amount_sum", measure="amount", function="sum")
        ]
        cube = Cube(name="sales", aggregates=aggregates)
        assert len(cube.aggregates_list) == 1
        assert cube.aggregate("amount_sum").name == "amount_sum"

    def test_initialization_with_dimensions(self, simple_dimension: Dimension) -> None:
        """Test cube creation with dimensions."""
        cube = Cube(name="sales", dimensions=[simple_dimension])
        assert len(cube.dimensions_list) == 1
        assert cube.dimension("date").name == "date"

    def test_initialization_with_details(self) -> None:
        """Test cube creation with detail attributes."""
        details = [Attribute(name="detail1"), Attribute(name="detail2")]
        cube = Cube(name="sales", details=details)
        assert len(cube.details) == 2

    def test_store_as_string(self) -> None:
        """Test store initialization with string value."""
        cube = Cube(name="sales", store="my_store")
        assert cube.store is None
        assert cube.store_name == "my_store"

    def test_store_as_object(self) -> None:
        """Test store initialization with object value."""
        store_obj = object()
        cube = Cube(name="sales", store=store_obj)
        assert cube.store is store_obj
        assert cube.store_name is None

    def test_duplicate_measure_error(self) -> None:
        """Test that duplicate measures raise ModelError."""
        with pytest.raises(ModelError, match="Duplicate measure"):
            Cube(
                name="sales", measures=[Measure(name="amount"), Measure(name="amount")]
            )

    def test_duplicate_aggregate_error(self) -> None:
        """Test that duplicate aggregates raise ModelError."""
        with pytest.raises(ModelError, match="Duplicate aggregate"):
            Cube(
                name="sales",
                aggregates=[
                    MeasureAggregate(name="sum", measure="amount", function="sum"),
                    MeasureAggregate(name="sum", measure="quantity", function="sum"),
                ],
            )

    def test_duplicate_dimension_error(self) -> None:
        """Test that duplicate dimensions raise ModelError."""
        dim = Dimension(
            name="date",
            levels=[Level(name="year", attributes=[Attribute(name="year")])],
        )
        with pytest.raises(ModelError, match="Duplicate dimension"):
            Cube(name="sales", dimensions=[dim, dim])

    def test_invalid_measure_type_error(self) -> None:
        """Test that invalid measure type raises ModelError."""
        with pytest.raises(ModelError, match="is not a Measure instance"):
            Cube(name="sales", measures=["invalid"])  # type: ignore[list-item]

    def test_invalid_dimension_type_error(self) -> None:
        """Test that invalid dimension type raises ModelError."""
        with pytest.raises(ModelError, match="is not a Dimension instance"):
            Cube(name="sales", dimensions=["invalid"])  # type: ignore[list-item]


# Property Tests


class TestCubeProperties:
    """Test cube property getters."""

    def test_measures_property(
        self, simple_cube: Cube, simple_measure: Measure
    ) -> None:
        """Test measures_list property returns correct measures."""
        measures = simple_cube.measures_list
        assert len(measures) == 1
        assert measures[0].name == "amount"

    def test_aggregates_property(self) -> None:
        """Test aggregates_list property returns correct aggregates."""
        agg = MeasureAggregate(name="sum", measure="amount", function="sum")
        cube = Cube(name="sales", aggregates=[agg])
        aggregates = cube.aggregates_list
        assert len(aggregates) == 1
        assert aggregates[0].name == "sum"

    def test_dimensions_property(
        self, simple_cube: Cube, simple_dimension: Dimension
    ) -> None:
        """Test dimensions_list property returns correct dimensions."""
        dims = simple_cube.dimensions_list
        assert len(dims) == 1
        assert dims[0].name == "date"

    def test_all_dimension_keys(self, simple_cube: Cube) -> None:
        """Test all_dimension_keys property returns dimension keys."""
        keys = simple_cube.all_dimension_keys
        assert len(keys) == 1
        assert keys[0].name == "month"  # Key of the single level

    def test_all_attributes(self, simple_cube: Cube) -> None:
        """Test all_attributes property includes all attributes."""
        attrs = simple_cube.all_attributes
        # Should have: date.month, date.month_name (from dimension), amount (measure)
        assert len(attrs) == 3
        refs = [a.ref for a in attrs]
        assert "date.month" in refs
        assert "date.month_name" in refs
        assert "amount" in refs

    def test_base_attributes(self) -> None:
        """Test base_attributes property excludes derived attributes."""
        derived_measure = Measure(name="total", expression="amount * 2")
        base_measure = Measure(name="amount")
        cube = Cube(name="sales", measures=[base_measure, derived_measure])
        base_attrs = cube.base_attributes
        assert len(base_attrs) == 1
        assert base_attrs[0].name == "amount"

    def test_all_fact_attributes(self, simple_cube: Cube) -> None:
        """Test all_fact_attributes excludes aggregates."""
        agg = MeasureAggregate(name="sum", measure="amount", function="sum")
        cube = Cube(
            name="sales",
            measures=[Measure(name="amount")],
            aggregates=[agg],
            dimensions=[simple_cube.dimensions_list[0]],
        )
        fact_attrs = cube.all_fact_attributes
        # Should include dimension attributes and measures, but not aggregates
        refs = [a.ref for a in fact_attrs]
        assert "amount" in refs
        assert "sum" not in refs

    def test_all_aggregate_attributes(self, simple_cube: Cube) -> None:
        """Test all_aggregate_attributes includes dimensions and aggregates."""
        agg = MeasureAggregate(name="sum", measure="amount", function="sum")
        cube = Cube(
            name="sales",
            measures=[Measure(name="amount")],
            aggregates=[agg],
            dimensions=simple_cube.dimensions_list,
        )
        agg_attrs = cube.all_aggregate_attributes
        # Should include dimension attributes and aggregates, but not measures
        refs = [a.ref for a in agg_attrs]
        assert "date.month" in refs
        assert "sum" in refs
        assert "amount" not in refs

    def test_attribute_dependencies(self) -> None:
        """Test attribute_dependencies property."""
        derived = Measure(name="total", expression="amount + tax")
        base = Measure(name="amount")
        cube = Cube(name="sales", measures=[base, derived])
        deps = cube.attribute_dependencies
        assert "total" in deps
        assert deps["total"] == {"amount", "tax"}
        assert deps["amount"] == set()


# Lookup Method Tests


class TestCubeLookups:
    """Test cube lookup methods."""

    def test_measure_lookup(self, simple_cube: Cube) -> None:
        """Test measure() method finds measure by name."""
        measure = simple_cube.measure("amount")
        assert measure.name == "amount"

    def test_measure_lookup_not_found(self, simple_cube: Cube) -> None:
        """Test measure() raises KeyError for unknown measure."""
        with pytest.raises(KeyError, match="has no measure 'unknown'"):
            simple_cube.measure("unknown")

    def test_get_measures_all(self, simple_cube: Cube) -> None:
        """Test get_measures() returns all when no argument."""
        measures = simple_cube.get_measures()
        assert len(measures) == 1

    def test_get_measures_by_name(self, simple_cube: Cube) -> None:
        """Test get_measures() with list of names."""
        cube = Cube(
            name="sales",
            measures=[Measure(name="amount"), Measure(name="quantity")],
        )
        measures = cube.get_measures(["amount"])
        assert len(measures) == 1
        assert measures[0].name == "amount"

    def test_get_measures_by_object(
        self, simple_cube: Cube, simple_measure: Measure
    ) -> None:
        """Test get_measures() with list of Measure objects."""
        measures = simple_cube.get_measures([simple_measure])
        assert len(measures) == 1
        assert measures[0].name == "amount"

    def test_aggregate_lookup(self) -> None:
        """Test aggregate() method finds aggregate by name."""
        agg = MeasureAggregate(name="sum", measure="amount", function="sum")
        cube = Cube(name="sales", aggregates=[agg])
        found = cube.aggregate("sum")
        assert found.name == "sum"

    def test_aggregate_lookup_not_found(self, simple_cube: Cube) -> None:
        """Test aggregate() raises KeyError for unknown aggregate."""
        with pytest.raises(KeyError, match="has no aggregate 'unknown'"):
            simple_cube.aggregate("unknown")

    def test_get_aggregates_all(self) -> None:
        """Test get_aggregates() returns all when no argument."""
        agg = MeasureAggregate(name="sum", measure="amount", function="sum")
        cube = Cube(name="sales", aggregates=[agg])
        aggregates = cube.get_aggregates()
        assert len(aggregates) == 1

    def test_get_aggregates_by_name(self) -> None:
        """Test get_aggregates() with list of names."""
        agg1 = MeasureAggregate(name="sum", measure="amount", function="sum")
        agg2 = MeasureAggregate(name="count", measure="amount", function="count")
        cube = Cube(name="sales", aggregates=[agg1, agg2])
        aggregates = cube.get_aggregates(["sum"])
        assert len(aggregates) == 1
        assert aggregates[0].name == "sum"

    def test_aggregates_for_measure(self) -> None:
        """Test aggregates_for_measure() finds aggregates for a measure."""
        agg1 = MeasureAggregate(name="amount_sum", measure="amount", function="sum")
        agg2 = MeasureAggregate(name="amount_count", measure="amount", function="count")
        agg3 = MeasureAggregate(name="quantity_sum", measure="quantity", function="sum")
        cube = Cube(name="sales", aggregates=[agg1, agg2, agg3])
        amount_aggs = cube.aggregates_for_measure("amount")
        assert len(amount_aggs) == 2
        assert all(a.measure == "amount" for a in amount_aggs)

    def test_dimension_lookup_by_name(self, simple_cube: Cube) -> None:
        """Test dimension() method finds dimension by name."""
        dim = simple_cube.dimension("date")
        assert dim.name == "date"

    def test_dimension_lookup_by_object(
        self, simple_cube: Cube, simple_dimension: Dimension
    ) -> None:
        """Test dimension() method accepts Dimension object."""
        dim = simple_cube.dimension(simple_dimension)
        assert dim.name == "date"

    def test_dimension_lookup_not_found(self, simple_cube: Cube) -> None:
        """Test dimension() raises KeyError for unknown dimension."""
        with pytest.raises(KeyError, match="has no dimension 'unknown'"):
            simple_cube.dimension("unknown")

    def test_dimension_lookup_invalid_type(self, simple_cube: Cube) -> None:
        """Test dimension() raises ValueError for invalid type."""
        with pytest.raises(ValueError, match="Unknown dimension object"):
            simple_cube.dimension(123)  # type: ignore[arg-type]


# Attribute Lookup Tests


class TestCubeAttributeLookups:
    """Test cube attribute lookup methods."""

    def test_attribute_lookup_in_dimension(self, simple_cube: Cube) -> None:
        """Test attribute() finds attributes in dimensions."""
        attr = simple_cube.attribute("date.month")
        assert attr.name == "month"

    def test_attribute_lookup_in_details(self) -> None:
        """Test attribute() finds attributes in details."""
        detail = Attribute(name="detail_attr")
        cube = Cube(name="sales", details=[detail])
        attr = cube.attribute("detail_attr")
        assert attr.name == "detail_attr"

    def test_attribute_lookup_in_measures(self, simple_cube: Cube) -> None:
        """Test attribute() finds attributes in measures."""
        attr = simple_cube.attribute("amount")
        assert attr.name == "amount"

    def test_attribute_lookup_not_found(self, simple_cube: Cube) -> None:
        """Test attribute() raises KeyError for unknown attribute."""
        with pytest.raises(KeyError, match="has no attribute 'unknown'"):
            simple_cube.attribute("unknown")

    def test_get_attributes_all_fact(self, simple_cube: Cube) -> None:
        """Test get_attributes() returns all fact attributes when no args."""
        attrs = simple_cube.get_attributes()
        refs = [a.ref for a in attrs]
        assert "amount" in refs
        assert "date.month" in refs

    def test_get_attributes_all_aggregated(self, simple_cube: Cube) -> None:
        """Test get_attributes() with aggregated=True."""
        agg = MeasureAggregate(name="sum", measure="amount", function="sum")
        cube = Cube(
            name="sales",
            measures=[Measure(name="amount")],
            aggregates=[agg],
            dimensions=simple_cube.dimensions_list,
        )
        attrs = cube.get_attributes(aggregated=True)
        refs = [a.ref for a in attrs]
        assert "sum" in refs
        assert "amount" not in refs  # Measures not in aggregated

    def test_get_attributes_by_name(self, simple_cube: Cube) -> None:
        """Test get_attributes() with specific attribute names."""
        attrs = simple_cube.get_attributes(["amount"])
        assert len(attrs) == 1
        assert attrs[0].name == "amount"

    def test_get_attributes_by_object(self, simple_cube: Cube) -> None:
        """Test get_attributes() with attribute objects."""
        measure = simple_cube.measure("amount")
        attrs = simple_cube.get_attributes([measure])
        assert len(attrs) == 1
        assert attrs[0].name == "amount"

    def test_get_attributes_not_found(self, simple_cube: Cube) -> None:
        """Test get_attributes() raises KeyError for unknown attribute."""
        with pytest.raises(KeyError, match="Unknown attribute"):
            simple_cube.get_attributes(["unknown"])


# Dependency Tests


class TestCubeDependencies:
    """Test cube dependency collection."""

    def test_collect_dependencies(self) -> None:
        """Test collect_dependencies() returns attributes in dependency order."""
        # Create measures with dependencies: total depends on amount and tax
        amount = Measure(name="amount")
        tax = Measure(name="tax")
        total = Measure(name="total", expression="amount + tax")
        cube = Cube(name="sales", measures=[total, amount, tax])

        # Get total with its dependencies
        attrs = cube.collect_dependencies(["total"])
        refs = [a.ref for a in attrs]

        # Should include dependencies first
        assert "total" in refs
        assert "amount" in refs
        assert "tax" in refs
        # total should come after its dependencies
        assert refs.index("total") > refs.index("amount")
        assert refs.index("total") > refs.index("tax")

    def test_collect_dependencies_with_objects(self) -> None:
        """Test collect_dependencies() with attribute objects."""
        amount = Measure(name="amount")
        total = Measure(name="total", expression="amount * 2")
        cube = Cube(name="sales", measures=[amount, total])

        attrs = cube.collect_dependencies([total])
        refs = [a.ref for a in attrs]

        assert "amount" in refs
        assert "total" in refs


# Dimension Linking Tests


class TestCubeDimensionLinking:
    """Test cube dimension linking functionality."""

    def test_add_dimension(
        self, simple_cube: Cube, product_dimension: Dimension
    ) -> None:
        """Test _add_dimension() adds a dimension."""
        simple_cube._add_dimension(product_dimension)
        assert "product" in [d.name for d in simple_cube.dimensions_list]

    def test_add_dimension_none(self, simple_cube: Cube) -> None:
        """Test _add_dimension() raises error for None."""
        with pytest.raises(ArgumentError, match="Trying to add None dimension"):
            simple_cube._add_dimension(None)  # type: ignore[arg-type]

    def test_add_dimension_invalid_type(self, simple_cube: Cube) -> None:
        """Test _add_dimension() raises error for invalid type."""
        with pytest.raises(ArgumentError, match="is not a Dimension instance"):
            simple_cube._add_dimension("invalid")  # type: ignore[arg-type]

    def test_link_dimension_simple(
        self, simple_cube: Cube, product_dimension: Dimension
    ) -> None:
        """Test link_dimension() adds dimension without link config."""
        simple_cube.link_dimension(product_dimension)
        assert "product" in [d.name for d in simple_cube.dimensions_list]

    def test_link_dimension_with_clone(self) -> None:
        """Test link_dimension() clones dimension when link specifies."""
        level = Level(name="year", attributes=[Attribute(name="year")])
        dim = Dimension(
            name="date",
            levels=[level],
            hierarchies=[
                Hierarchy(name="default", levels=[level]),
                Hierarchy(name="alternative", levels=[level]),
            ],
        )

        cube = Cube(
            name="sales",
            dimension_links=[{"name": "date", "hierarchies": ["default"]}],
        )

        cube.link_dimension(dim)

        linked_dim = cube.dimension("date")
        assert len(linked_dim.hierarchies) == 1
        assert linked_dim.hierarchies[0].name == "default"

    def test_link_dimension_with_alias(self) -> None:
        """Test link_dimension() with alias creates dimension clone with new name."""
        level = Level(name="year", attributes=[Attribute(name="year")])
        dim = Dimension(
            name="date",
            levels=[level],
            hierarchies=[Hierarchy(name="default", levels=[level])],
        )

        cube = Cube(
            name="sales",
            dimension_links=[{"name": "date", "alias": "order_date"}],
        )

        cube.link_dimension(dim)

        # Should be accessible by alias
        assert "order_date" in [d.name for d in cube.dimensions_list]


# Distilled Hierarchies Tests


class TestCubeDistilledHierarchies:
    """Test cube distilled_hierarchies property."""

    def test_distilled_hierarchies(self, simple_cube: Cube) -> None:
        """Test distilled_hierarchies returns correct structure."""
        hierarchies = simple_cube.distilled_hierarchies

        # Should have (dim_name, hier_name) -> [level_keys]
        assert ("date", "default") in hierarchies
        # Should also have (dim_name, None) for default
        assert ("date", None) in hierarchies

        # Keys should be attribute refs
        assert len(hierarchies[("date", "default")]) == 1
        assert hierarchies[("date", "default")][0] == "date.month"

    def test_distilled_hierarchies_multiple_dimensions(
        self, simple_cube: Cube, product_dimension: Dimension
    ) -> None:
        """Test distilled_hierarchies with multiple dimensions."""
        simple_cube._add_dimension(product_dimension)

        hierarchies = simple_cube.distilled_hierarchies

        assert ("date", "default") in hierarchies
        assert ("product", "default") in hierarchies


# Serialization Tests


class TestCubeSerialization:
    """Test cube to_dict() method."""

    def test_to_dict_basic(self, simple_cube: Cube) -> None:
        """Test basic to_dict() output."""
        result = simple_cube.to_dict()

        assert result["name"] == "sales"
        assert "measures" in result
        assert "aggregates" in result
        assert "details" in result
        assert "dimensions" in result

    def test_to_dict_create_label(self) -> None:
        """Test to_dict() with create_label=True."""
        cube = Cube(name="my_cube")
        result = cube.to_dict(create_label=True)
        assert result["label"] == "My cube"

    def test_to_dict_expand_dimensions(self, simple_cube: Cube) -> None:
        """Test to_dict() with expand_dimensions=True."""
        result = simple_cube.to_dict(expand_dimensions=True)

        # Dimensions should be dicts, not just names
        assert isinstance(result["dimensions"][0], dict)
        assert result["dimensions"][0]["name"] == "date"

    def test_to_dict_no_expand_dimensions(self, simple_cube: Cube) -> None:
        """Test to_dict() with expand_dimensions=False (default)."""
        result = simple_cube.to_dict(expand_dimensions=False)

        # Dimensions should be just names
        assert result["dimensions"] == ["date"]

    def test_to_dict_with_mappings(self, simple_cube: Cube) -> None:
        """Test to_dict() with with_mappings=True."""
        cube = Cube(
            name="sales",
            mappings={"amount": "fact.amount"},
            fact="fact_table",
        )
        result = cube.to_dict(with_mappings=True)

        assert result["mappings"] == {"amount": "fact.amount"}
        assert result["fact"] == "fact_table"

    def test_to_dict_without_mappings(self, simple_cube: Cube) -> None:
        """Test to_dict() with with_mappings=False."""
        cube = Cube(
            name="sales",
            mappings={"amount": "fact.amount"},
        )
        result = cube.to_dict(with_mappings=False)

        assert "mappings" not in result


# from_schema Factory Tests


class TestCubeFromSchema:
    """Test Cube.from_schema() factory method."""

    def test_from_schema_basic(self) -> None:
        """Test from_schema() with basic cube schema."""
        schema = {
            "name": "sales",
            "measures": [{"name": "amount"}],
            "dimensions": [],
        }

        cube = Cube.from_schema(schema)

        assert cube.name == "sales"
        assert len(cube.measures_list) == 1
        assert cube.measure("amount").name == "amount"

    def test_from_schema_with_dimensions(self, simple_dimension: Dimension) -> None:
        """Test from_schema() with dimension linking."""
        schema = {
            "name": "sales",
            "measures": [{"name": "amount"}],
            "dimensions": ["date"],
        }

        dimensions = {"date": simple_dimension}
        cube = Cube.from_schema(schema, dimensions=dimensions)

        assert len(cube.dimensions_list) == 1
        assert cube.dimension("date").name == "date"

    def test_from_schema_with_aggregates(self) -> None:
        """Test from_schema() with aggregates."""
        schema = {
            "name": "sales",
            "measures": [{"name": "amount"}],
            "aggregates": [
                {"name": "amount_sum", "measure": "amount", "function": "sum"}
            ],
            "dimensions": [],
        }

        cube = Cube.from_schema(schema)

        assert len(cube.aggregates_list) == 1
        assert cube.aggregate("amount_sum").name == "amount_sum"

    def test_from_schema_with_details(self) -> None:
        """Test from_schema() with details."""
        schema = {
            "name": "sales",
            "measures": [],
            "details": [{"name": "detail_attr"}],
            "dimensions": [],
        }

        cube = Cube.from_schema(schema)

        assert len(cube.details) == 1
        assert cube.details[0].name == "detail_attr"

    def test_from_schema_with_physical_props(self) -> None:
        """Test from_schema() with physical properties."""
        schema = {
            "name": "sales",
            "fact": "sales_fact",
            "key": "id",
            "mappings": {"amount": "fact.amount"},
            "joins": [{"master": "fact", "detail": "dim"}],
            "browser_options": {"option1": "value1"},
            "store": "main_store",
        }

        cube = Cube.from_schema(schema)

        assert cube.fact == "sales_fact"
        assert cube.key == "id"
        assert cube.mappings == {"amount": "fact.amount"}
        assert cube.joins == [{"master": "fact", "detail": "dim"}]
        assert cube.browser_options == {"option1": "value1"}
        assert cube.store_name == "main_store"


# Equality and String Representation Tests


class TestCubeEquality:
    """Test cube equality and string representation."""

    def test_eq_same(self, simple_cube: Cube) -> None:
        """Test equality with identical cube."""
        cube2 = copy.deepcopy(simple_cube)
        assert simple_cube == cube2

    def test_eq_different_name(self, simple_cube: Cube) -> None:
        """Test inequality with different name."""
        cube2 = copy.deepcopy(simple_cube)
        object.__setattr__(cube2, "name", "different")
        assert simple_cube != cube2

    def test_eq_different_measures(self, simple_cube: Cube) -> None:
        """Test inequality with different measures."""
        cube2 = Cube(
            name=simple_cube.name,
            measures=[Measure(name="different")],
            dimensions=simple_cube.dimensions_list,
        )
        assert simple_cube != cube2

    def test_eq_none(self, simple_cube: Cube) -> None:
        """Test inequality with None."""
        assert simple_cube != None  # noqa: E711

    def test_eq_other_type(self, simple_cube: Cube) -> None:
        """Test inequality with other type."""
        assert simple_cube != "not a cube"

    def test_hash(self, simple_cube: Cube) -> None:
        """Test hash based on name."""
        assert hash(simple_cube) == hash("sales")

    def test_str(self, simple_cube: Cube) -> None:
        """Test string representation."""
        assert str(simple_cube) == "sales"

    def test_repr(self, simple_cube: Cube) -> None:
        """Test repr representation."""
        assert repr(simple_cube) == "<cube: sales>"


# Edge Case Tests


class TestCubeEdgeCases:
    """Test cube edge cases and boundary conditions."""

    def test_empty_cube(self) -> None:
        """Test cube with no attributes."""
        cube = Cube(name="empty")
        assert cube.name == "empty"
        assert cube.measures_list == []
        assert cube.aggregates_list == []
        assert cube.dimensions_list == []
        assert cube.all_attributes == []

    def test_cube_with_only_details(self) -> None:
        """Test cube with only detail attributes."""
        cube = Cube(
            name="details_only",
            details=[Attribute(name="detail1"), Attribute(name="detail2")],
        )
        assert len(cube.details) == 2
        assert len(cube.all_fact_attributes) == 2

    def test_cube_with_only_dimensions(self, simple_dimension: Dimension) -> None:
        """Test cube with only dimensions."""
        cube = Cube(name="dims_only", dimensions=[simple_dimension])
        assert len(cube.dimensions_list) == 1
        assert len(cube.all_dimension_keys) == 1

    def test_multiple_dimensions(self) -> None:
        """Test cube with multiple dimensions."""
        dim1 = Dimension(
            name="date",
            levels=[Level(name="year", attributes=[Attribute(name="year")])],
        )
        dim2 = Dimension(
            name="product",
            levels=[Level(name="product", attributes=[Attribute(name="product")])],
        )

        cube = Cube(name="multi", dimensions=[dim1, dim2])

        assert len(cube.dimensions_list) == 2
        assert cube.dimension("date").name == "date"
        assert cube.dimension("product").name == "product"

    def test_complex_attribute_lookup(self) -> None:
        """Test attribute lookup across multiple dimensions and measures."""
        dim1 = Dimension(
            name="date",
            levels=[Level(name="year", attributes=[Attribute(name="year")])],
        )
        dim2 = Dimension(
            name="product",
            levels=[Level(name="product", attributes=[Attribute(name="product")])],
        )

        cube = Cube(
            name="complex",
            dimensions=[dim1, dim2],
            measures=[Measure(name="amount"), Measure(name="quantity")],
            details=[Attribute(name="detail")],
        )

        # Should find attributes from dimensions
        assert cube.attribute("date.year").name == "year"
        assert cube.attribute("product.product").name == "product"

        # Should find measures
        assert cube.attribute("amount").name == "amount"

        # Should find details
        assert cube.attribute("detail").name == "detail"
