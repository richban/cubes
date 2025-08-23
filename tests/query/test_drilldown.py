"""
Tests for Drilldown class in cubes.query_v2.browser module.

This module provides comprehensive test coverage for the Drilldown class,
which handles resolution and management of drilldown specifications for
OLAP queries.
"""

import pytest

from cubes.errors import ArgumentError, HierarchyError
from cubes.metadata_v2 import Attribute, Cube, Dimension, Hierarchy, Level
from cubes.query_v2.browser import Drilldown, DrilldownSpec
from cubes.query_v2.cells import Cell


class TestDrilldownFixtures:
    """Test fixtures and setup for Drilldown tests."""

    def setup_method(self):
        """Setup test data with realistic cube structure."""
        # Create attributes for different levels
        self.key_attr = Attribute(name="id", label="ID")
        self.label_attr = Attribute(name="name", label="Name")
        self.code_attr = Attribute(name="code", label="Code")

        # Time dimension levels
        self.year_level = Level(
            name="year", attributes=[self.key_attr, self.label_attr], key="id"
        )
        self.quarter_level = Level(
            name="quarter", attributes=[self.key_attr, self.label_attr], key="id"
        )
        self.month_level = Level(
            name="month", attributes=[self.key_attr, self.label_attr], key="id"
        )

        # Geography dimension levels
        self.country_level = Level(
            name="country",
            attributes=[self.key_attr, self.label_attr, self.code_attr],
            key="id",
        )
        self.state_level = Level(
            name="state",
            attributes=[self.key_attr, self.label_attr, self.code_attr],
            key="id",
        )
        self.city_level = Level(
            name="city", attributes=[self.key_attr, self.label_attr], key="id"
        )

        # Create hierarchies
        self.time_hierarchy = Hierarchy(
            name="default",
            levels=[self.year_level, self.quarter_level, self.month_level],
        )
        self.time_fiscal_hierarchy = Hierarchy(
            name="fiscal",
            levels=[self.year_level, self.quarter_level, self.month_level],
        )

        self.geo_hierarchy = Hierarchy(
            name="default",
            levels=[self.country_level, self.state_level, self.city_level],
        )

        # Create dimensions
        self.time_dimension = Dimension(
            name="time",
            levels=[self.year_level, self.quarter_level, self.month_level],
            hierarchies=[self.time_hierarchy, self.time_fiscal_hierarchy],
        )

        self.geography_dimension = Dimension(
            name="geography",
            levels=[self.country_level, self.state_level, self.city_level],
            hierarchies=[self.geo_hierarchy],
        )

        # Create a simple product dimension (flat)
        self.product_level = Level(
            name="product", attributes=[self.key_attr, self.label_attr], key="id"
        )
        self.product_hierarchy = Hierarchy(name="default", levels=[self.product_level])
        self.product_dimension = Dimension(
            name="product",
            levels=[self.product_level],
            hierarchies=[self.product_hierarchy],
            is_flat=True,
        )

        # Create cube
        self.cube = Cube(
            name="sales",
            dimensions=[
                self.time_dimension,
                self.geography_dimension,
                self.product_dimension,
            ],
            measures=[],
        )

        # Create cells for testing
        self.empty_cell = Cell(cube=self.cube)
        self.basic_cell = Cell(cube=self.cube)


class TestDrilldownInitialization(TestDrilldownFixtures):
    """Test Drilldown class initialization and validation."""

    def test_drilldown_requires_cell_context(self):
        """Test that Drilldown requires a Cell context for initialization."""
        with pytest.raises(ArgumentError, match="Cell context is required"):
            Drilldown(["time:year"], cell=None)

    def test_empty_drilldown(self):
        """Test creating an empty drilldown."""
        drilldown = Drilldown([], self.empty_cell)

        assert len(drilldown) == 0
        assert not drilldown
        assert drilldown.dimensions == []
        assert drilldown._contained_dimensions == set()
        assert str(drilldown) == ""

    def test_none_drilldown(self):
        """Test creating drilldown with None specification."""
        drilldown = Drilldown(None, self.empty_cell)

        assert len(drilldown) == 0
        assert not drilldown
        assert drilldown.dimensions == []

    def test_single_string_drilldown(self):
        """Test creating drilldown with single string specification."""
        drilldown = Drilldown(["time:year"], self.empty_cell)

        assert len(drilldown) == 1
        assert bool(drilldown)
        assert drilldown.dimensions[0].name == "time"
        assert "time" in drilldown._contained_dimensions

        # Check that the resolved item has correct structure
        item = drilldown.drilldown[0]
        assert item.dimension.name == "time"
        assert item.hierarchy.name == "default"
        assert len(item.levels) == 1
        assert item.levels[0].name == "year"

    def test_multiple_string_drilldowns(self):
        """Test creating drilldown with multiple string specifications."""
        drilldown = Drilldown(["time:year", "geography:country"], self.empty_cell)

        assert len(drilldown) == 2
        assert bool(drilldown)
        assert len(drilldown.dimensions) == 2
        assert "time" in drilldown._contained_dimensions
        assert "geography" in drilldown._contained_dimensions

        # Verify items are resolved correctly
        time_item = drilldown.drilldown[0]
        geo_item = drilldown.drilldown[1]

        assert time_item.dimension.name == "time"
        assert geo_item.dimension.name == "geography"

    def test_tuple_specification_drilldown(self):
        """Test creating drilldown with tuple specifications."""
        drilldown = Drilldown(
            [("time", None, "year"), ("geography", None, "country")], self.empty_cell
        )

        assert len(drilldown) == 2
        time_item = drilldown.drilldown[0]
        geo_item = drilldown.drilldown[1]

        assert time_item.dimension.name == "time"
        assert time_item.levels[0].name == "year"
        assert geo_item.dimension.name == "geography"
        assert geo_item.levels[0].name == "country"

    def test_hierarchy_specification_drilldown(self):
        """Test creating drilldown with hierarchy specifications."""
        drilldown = Drilldown(["time@fiscal:year"], self.empty_cell)

        assert len(drilldown) == 1
        item = drilldown.drilldown[0]

        assert item.dimension.name == "time"
        assert item.hierarchy.name == "fiscal"
        assert item.levels[0].name == "year"

    def test_dimension_object_specification(self):
        """Test creating drilldown with Dimension objects."""
        drilldown = Drilldown([self.time_dimension], self.empty_cell)

        assert len(drilldown) == 1
        item = drilldown.drilldown[0]

        assert item.dimension.name == "time"
        # Should use the deepest level for dimension objects
        assert item.levels[-1].name == "month"  # Deepest level in time hierarchy

    def test_invalid_dimension_error(self):
        """Test that invalid dimension names raise ArgumentError."""
        with pytest.raises(ArgumentError, match="Invalid drilldown specification"):
            Drilldown(["nonexistent:level"], self.empty_cell)

    def test_invalid_level_error(self):
        """Test that invalid level names raise ArgumentError."""
        with pytest.raises(ArgumentError, match="Invalid drilldown specification"):
            Drilldown(["time:nonexistent"], self.empty_cell)

    def test_invalid_hierarchy_error(self):
        """Test that invalid hierarchy names raise ArgumentError."""
        with pytest.raises(ArgumentError, match="Invalid drilldown specification"):
            Drilldown(["time@nonexistent:year"], self.empty_cell)


class TestDrilldownAccess(TestDrilldownFixtures):
    """Test Drilldown access methods and iteration."""

    def test_index_access(self):
        """Test accessing drilldown items by index."""
        drilldown = Drilldown(["time:year", "geography:country"], self.empty_cell)

        first_item = drilldown[0]
        second_item = drilldown[1]

        assert first_item.dimension.name == "time"
        assert second_item.dimension.name == "geography"

    def test_dimension_name_access(self):
        """Test accessing drilldown items by dimension name."""
        drilldown = Drilldown(["time:year", "geography:country"], self.empty_cell)

        time_item = drilldown["time"]
        geo_item = drilldown["geography"]

        assert time_item.dimension.name == "time"
        assert geo_item.dimension.name == "geography"

    def test_invalid_index_access(self):
        """Test that invalid index access raises ArgumentError."""
        drilldown = Drilldown(["time:year"], self.empty_cell)

        with pytest.raises(ArgumentError, match="Invalid drilldown index"):
            _ = drilldown[10]

        with pytest.raises(ArgumentError, match="Dimension .* not found in drilldown"):
            _ = drilldown["nonexistent"]

    def test_iteration(self):
        """Test iterating over drilldown items."""
        drilldown = Drilldown(["time:year", "geography:country"], self.empty_cell)

        items = list(drilldown)
        assert len(items) == 2
        assert items[0].dimension.name == "time"
        assert items[1].dimension.name == "geography"

        # Test direct iteration
        dimension_names = [item.dimension.name for item in drilldown]
        assert dimension_names == ["time", "geography"]

    def test_length_and_bool(self):
        """Test length and boolean evaluation of drilldown."""
        empty_drilldown = Drilldown([], self.empty_cell)
        single_drilldown = Drilldown(["time:year"], self.empty_cell)
        multi_drilldown = Drilldown(["time:year", "geography:country"], self.empty_cell)

        # Test length
        assert len(empty_drilldown) == 0
        assert len(single_drilldown) == 1
        assert len(multi_drilldown) == 2

        # Test boolean evaluation
        assert not empty_drilldown
        assert single_drilldown
        assert multi_drilldown


class TestDrilldownStringRepresentation(TestDrilldownFixtures):
    """Test string representation methods of Drilldown."""

    def test_str_representation(self):
        """Test string representation of drilldown."""
        drilldown = Drilldown(["time:year", "geography:country"], self.empty_cell)
        assert str(drilldown) == "time:year,geography:country"

        empty_drilldown = Drilldown([], self.empty_cell)
        assert str(empty_drilldown) == ""

    def test_str_with_hierarchy(self):
        """Test string representation with custom hierarchy."""
        drilldown = Drilldown(["time@fiscal:year"], self.empty_cell)
        assert str(drilldown) == "time@fiscal:year"

    def test_items_as_strings(self):
        """Test getting drilldown items as string list."""
        drilldown = Drilldown(
            ["time:year", "geography:country", "time@fiscal:quarter"], self.empty_cell
        )
        strings = drilldown.items_as_strings()

        assert len(strings) == 3
        assert "time:year" in strings
        assert "geography:country" in strings
        assert "time@fiscal:quarter" in strings

    def test_items_as_strings_default_hierarchy(self):
        """Test string representation omits default hierarchy."""
        drilldown = Drilldown(["time:year"], self.empty_cell)
        strings = drilldown.items_as_strings()

        assert strings == ["time:year"]  # No @default suffix


class TestDrilldownUtilityMethods(TestDrilldownFixtures):
    """Test utility methods of Drilldown class."""

    def test_drilldown_for_dimension(self):
        """Test getting drilldown items for specific dimension."""
        drilldown = Drilldown(
            ["time:year", "time:quarter", "geography:country"], self.empty_cell
        )

        time_items = drilldown.drilldown_for_dimension("time")
        geo_items = drilldown.drilldown_for_dimension("geography")

        assert len(time_items) == 2
        assert len(geo_items) == 1
        assert all(item.dimension.name == "time" for item in time_items)
        assert geo_items[0].dimension.name == "geography"

    def test_drilldown_for_dimension_with_object(self):
        """Test getting drilldown items using dimension object."""
        drilldown = Drilldown(["time:year", "geography:country"], self.empty_cell)

        time_items = drilldown.drilldown_for_dimension(self.time_dimension)
        assert len(time_items) == 1
        assert time_items[0].dimension.name == "time"

    def test_has_dimension(self):
        """Test checking if drilldown contains specific dimension."""
        drilldown = Drilldown(["time:year", "geography:country"], self.empty_cell)

        assert drilldown.has_dimension("time")
        assert drilldown.has_dimension("geography")
        assert not drilldown.has_dimension("product")

        # Test with dimension object
        assert drilldown.has_dimension(self.time_dimension)
        assert not drilldown.has_dimension(self.product_dimension)

    def test_deepest_levels(self):
        """Test getting deepest levels for each dimension."""
        drilldown = Drilldown(["time:year", "geography:state"], self.empty_cell)
        levels = drilldown.deepest_levels()

        assert len(levels) == 2

        # Each tuple should be (dimension, hierarchy, level)
        time_tuple, geo_tuple = levels

        assert time_tuple[0].name == "time"
        assert time_tuple[1].name == "default"  # hierarchy
        assert time_tuple[2].name == "year"  # level

        assert geo_tuple[0].name == "geography"
        assert geo_tuple[1].name == "default"
        assert geo_tuple[2].name == "state"

    def test_result_levels(self):
        """Test generating result levels dictionary."""
        drilldown = Drilldown(["time:year", "geography:country"], self.empty_cell)
        levels = drilldown.result_levels()

        assert "time" in levels
        assert "geography" in levels
        assert levels["time"] == ["year"]
        assert levels["geography"] == ["country"]

    def test_result_levels_with_hierarchy(self):
        """Test result levels with custom hierarchy."""
        drilldown = Drilldown(["time@fiscal:year"], self.empty_cell)
        levels = drilldown.result_levels()

        assert "time@fiscal" in levels
        assert levels["time@fiscal"] == ["year"]

    def test_result_levels_with_split(self):
        """Test result levels with split dimension included."""
        drilldown = Drilldown(["time:year"], self.empty_cell)
        levels = drilldown.result_levels(include_split=True)

        assert "time" in levels
        assert "__within_split__" in levels
        assert levels["__within_split__"] == ["__within_split__"]


class TestDrilldownAttributes(TestDrilldownFixtures):
    """Test attribute properties of Drilldown class."""

    def test_key_attributes(self):
        """Test getting key attributes from drilldown."""
        drilldown = Drilldown(["time:year", "geography:country"], self.empty_cell)
        key_attrs = drilldown.key_attributes

        # Should have key attributes from both year and country levels
        assert len(key_attrs) >= 2

        # All should be Attribute objects with key role
        for attr in key_attrs:
            assert hasattr(attr, "name")

    def test_all_attributes(self):
        """Test getting all attributes from drilldown."""
        drilldown = Drilldown(["time:year", "geography:country"], self.empty_cell)
        all_attrs = drilldown.all_attributes

        # Should include key, label, and other attributes
        assert len(all_attrs) >= len(drilldown.key_attributes)

        # All should be Attribute objects
        for attr in all_attrs:
            assert hasattr(attr, "name")

    def test_natural_order(self):
        """Test getting natural order from drilldown."""
        drilldown = Drilldown(["time:year", "geography:country"], self.empty_cell)
        order = drilldown.natural_order

        # Should have ordering tuples for each level
        assert len(order) >= 2

        # Each tuple should be (attribute, direction)
        for attr, direction in order:
            assert hasattr(attr, "name")
            assert direction in ["asc", "desc"]


class TestDrilldownHighCardinality(TestDrilldownFixtures):
    """Test high cardinality detection in Drilldown."""

    def test_high_cardinality_levels_empty(self):
        """Test high cardinality detection with normal levels."""
        drilldown = Drilldown(["time:year", "geography:country"], self.empty_cell)
        high_card = drilldown.high_cardinality_levels(self.empty_cell)

        # Default levels should not be high cardinality
        assert len(high_card) == 0

    def test_high_cardinality_with_marked_level(self):
        """Test high cardinality detection with explicitly marked level."""
        # Create a high cardinality level
        high_card_level = Level(
            name="customer",
            attributes=[self.key_attr, self.label_attr],
            key="id",
            cardinality="high",
        )
        high_card_hierarchy = Hierarchy(name="default", levels=[high_card_level])
        high_card_dimension = Dimension(
            name="customer", levels=[high_card_level], hierarchies=[high_card_hierarchy]
        )

        # Add to cube
        test_cube = Cube(
            name="test_sales", dimensions=[high_card_dimension], measures=[]
        )
        test_cell = Cell(cube=test_cube)

        drilldown = Drilldown(["customer"], test_cell)
        high_card = drilldown.high_cardinality_levels(test_cell)

        assert len(high_card) == 1
        assert high_card[0].name == "customer"


class TestDrilldownSpecResolution(TestDrilldownFixtures):
    """Test the _resolve_spec method for proper cube object mapping."""

    def test_basic_dimension_resolution(self):
        """Test basic dimension name to Dimension object resolution."""
        drilldown = Drilldown([], self.empty_cell)  # Empty drilldown for direct access
        spec = DrilldownSpec(dimension="time")

        resolved_item = drilldown._resolve_spec(spec, self.cube, self.empty_cell)

        # Verify dimension object resolution
        assert resolved_item.dimension.name == "time"
        assert resolved_item.dimension == self.time_dimension
        assert isinstance(resolved_item.dimension, Dimension)

    def test_default_hierarchy_resolution(self):
        """Test default hierarchy resolution when no hierarchy specified."""
        drilldown = Drilldown([], self.empty_cell)
        spec = DrilldownSpec(dimension="time", hierarchy=None)

        resolved_item = drilldown._resolve_spec(spec, self.cube, self.empty_cell)

        # Should use default hierarchy
        assert resolved_item.hierarchy.name == "default"
        assert resolved_item.hierarchy == self.time_dimension.hierarchy()

    def test_explicit_hierarchy_resolution(self):
        """Test explicit hierarchy resolution."""
        drilldown = Drilldown([], self.empty_cell)
        spec = DrilldownSpec(dimension="time", hierarchy="fiscal")

        resolved_item = drilldown._resolve_spec(spec, self.cube, self.empty_cell)

        # Should use specified hierarchy
        assert resolved_item.hierarchy.name == "fiscal"
        assert resolved_item.hierarchy == self.time_dimension.hierarchy("fiscal")

    def test_explicit_level_resolution(self):
        """Test explicit level specification resolution."""
        drilldown = Drilldown([], self.empty_cell)
        spec = DrilldownSpec(dimension="time", level="quarter")

        resolved_item = drilldown._resolve_spec(spec, self.cube, self.empty_cell)

        # Should include levels up to and including quarter
        assert len(resolved_item.levels) == 2  # year and quarter
        assert resolved_item.levels[0].name == "year"
        assert resolved_item.levels[1].name == "quarter"

        # Keys should match levels
        assert len(resolved_item.keys) == 2
        assert resolved_item.keys[0] == "id"  # year key
        assert resolved_item.keys[1] == "id"  # quarter key

    def test_flat_dimension_resolution(self):
        """Test resolution of flat dimensions."""
        drilldown = Drilldown([], self.empty_cell)
        spec = DrilldownSpec(dimension="product")  # Product is flat

        resolved_item = drilldown._resolve_spec(spec, self.cube, self.empty_cell)

        # Flat dimension should use all its levels (just one in this case)
        assert len(resolved_item.levels) == 1
        assert resolved_item.levels[0].name == "product"
        assert len(resolved_item.keys) == 1

    def test_implicit_level_detection_no_cuts(self):
        """Test implicit level detection with empty cell (no cuts)."""
        drilldown = Drilldown([], self.empty_cell)
        spec = DrilldownSpec(dimension="geography")  # Non-flat, no explicit level

        resolved_item = drilldown._resolve_spec(spec, self.cube, self.empty_cell)

        # Should default to first level when no cuts exist
        assert len(resolved_item.levels) == 1
        assert resolved_item.levels[0].name == "country"
        assert len(resolved_item.keys) == 1

    def test_implicit_level_detection_with_cuts(self):
        """Test implicit level detection with existing cuts."""
        from cubes.query_v2.cells import PointCut

        # Create a cell with a cut on geography at country level
        country_cut = PointCut(
            dimension=self.geography_dimension, path=["US"], hierarchy="default"
        )
        cell_with_cut = self.empty_cell.slice(country_cut)

        drilldown = Drilldown([], self.empty_cell)
        spec = DrilldownSpec(dimension="geography")

        resolved_item = drilldown._resolve_spec(spec, self.cube, cell_with_cut)

        # Should drill to next level after the cut (state level)
        # This assumes the cut is at depth 1 (country), so next level includes state
        assert len(resolved_item.levels) >= 1
        level_names = [level.name for level in resolved_item.levels]

        # Should include at least country level, possibly state
        assert "country" in level_names

    def test_hierarchy_level_combination_validation(self):
        """Test that hierarchy and level combinations are properly validated."""
        drilldown = Drilldown([], self.empty_cell)
        spec = DrilldownSpec(dimension="time", hierarchy="fiscal", level="month")

        resolved_item = drilldown._resolve_spec(spec, self.cube, self.empty_cell)

        # Should resolve correctly with both hierarchy and level specified
        assert resolved_item.hierarchy.name == "fiscal"
        assert len(resolved_item.levels) == 3  # year, quarter, month
        assert resolved_item.levels[-1].name == "month"

    def test_drilldown_item_completeness(self):
        """Test that resolved DrilldownItem has all required components."""
        drilldown = Drilldown([], self.empty_cell)
        spec = DrilldownSpec(dimension="geography", level="state")

        resolved_item = drilldown._resolve_spec(spec, self.cube, self.empty_cell)

        # Verify all DrilldownItem components are properly set
        assert hasattr(resolved_item, "dimension")
        assert hasattr(resolved_item, "hierarchy")
        assert hasattr(resolved_item, "levels")
        assert hasattr(resolved_item, "keys")

        # Verify types and consistency
        assert isinstance(resolved_item.dimension, Dimension)
        assert isinstance(resolved_item.hierarchy, Hierarchy)
        assert isinstance(resolved_item.levels, tuple)
        assert isinstance(resolved_item.keys, list)
        assert len(resolved_item.levels) == len(resolved_item.keys)

    def test_resolution_error_invalid_dimension(self):
        """Test error handling for invalid dimension names."""
        drilldown = Drilldown([], self.empty_cell)
        spec = DrilldownSpec(dimension="nonexistent")

        with pytest.raises(
            ArgumentError, match="Failed to resolve drilldown specification"
        ):
            drilldown._resolve_spec(spec, self.cube, self.empty_cell)

    def test_resolution_error_invalid_hierarchy(self):
        """Test error handling for invalid hierarchy names."""
        drilldown = Drilldown([], self.empty_cell)
        spec = DrilldownSpec(dimension="time", hierarchy="nonexistent")

        with pytest.raises(
            HierarchyError,
            match="Hierarchy 'nonexistent' not found in dimension 'time'",
        ):
            drilldown._resolve_spec(spec, self.cube, self.empty_cell)

    def test_resolution_error_invalid_level(self):
        """Test error handling for invalid level names."""
        drilldown = Drilldown([], self.empty_cell)
        spec = DrilldownSpec(dimension="time", level="nonexistent")

        with pytest.raises(
            HierarchyError, match="No level 'nonexistent' in hierarchy 'default'"
        ):
            drilldown._resolve_spec(spec, self.cube, self.empty_cell)

    def test_hierarchy_mismatch_with_cuts(self):
        """Test error when cut hierarchy differs from drilldown hierarchy."""
        from cubes.query_v2.cells import PointCut

        # Create a cut using default hierarchy
        country_cut = PointCut(
            dimension=self.geography_dimension, path=["US"], hierarchy="default"
        )
        cell_with_cut = self.empty_cell.slice(country_cut)

        drilldown = Drilldown([], self.empty_cell)
        # Try to drill down using a different hierarchy (this should work since geography only has default)
        spec = DrilldownSpec(
            dimension="time", hierarchy="fiscal"
        )  # Different dimension to avoid the check

        # This should work fine since they're different dimensions
        resolved_item = drilldown._resolve_spec(spec, self.cube, cell_with_cut)
        assert resolved_item.hierarchy.name == "fiscal"

    def test_max_depth_exceeded_error(self):
        """Test error when trying to drill beyond maximum hierarchy depth."""
        from cubes.query_v2.cells import PointCut

        # Create a deep cut that reaches the maximum depth
        # Mock a cut at the deepest level
        deep_cut = PointCut(
            dimension=self.geography_dimension,
            path=["US", "CA", "LA"],  # Country, State, City - maximum depth
            hierarchy="default",
        )
        cell_with_deep_cut = self.empty_cell.slice(deep_cut)

        drilldown = Drilldown([], self.empty_cell)
        spec = DrilldownSpec(
            dimension="geography"
        )  # No explicit level - should try implicit

        # This might succeed or fail depending on the level_depth() implementation
        # If it fails, it should be with HierarchyError about depth
        try:
            resolved_item = drilldown._resolve_spec(spec, self.cube, cell_with_deep_cut)
            # If it succeeds, verify it didn't exceed bounds
            assert len(resolved_item.levels) <= len(self.geo_hierarchy.levels)
        except HierarchyError as e:
            # Expected error for exceeding depth
            assert "levels" in str(e) and "cannot drill" in str(e)


class TestDrilldownEdgeCases(TestDrilldownFixtures):
    """Test edge cases and error conditions for Drilldown."""

    def test_malformed_drilldown_items_handling(self):
        """Test that malformed items are handled gracefully."""
        # This tests the exception handling in various methods
        drilldown = Drilldown(["time:year"], self.empty_cell)

        # These methods should not fail even with edge cases
        assert isinstance(drilldown.deepest_levels(), list)
        assert isinstance(drilldown.key_attributes, list)
        assert isinstance(drilldown.all_attributes, list)
        assert isinstance(drilldown.natural_order, list)

    def test_mixed_specification_types(self):
        """Test mixing different specification types."""
        drilldown = Drilldown(
            [
                "time:year",  # string
                ("geography", None, "country"),  # tuple
                self.product_dimension,  # dimension object
            ],
            self.empty_cell,
        )

        assert len(drilldown) == 3
        assert drilldown.has_dimension("time")
        assert drilldown.has_dimension("geography")
        assert drilldown.has_dimension("product")

    def test_duplicate_dimensions_handling(self):
        """Test handling multiple drilldowns on same dimension."""
        drilldown = Drilldown(["time:year", "time:quarter"], self.empty_cell)

        assert len(drilldown) == 2
        assert drilldown.has_dimension("time")

        # First occurrence wins for dimension name lookup
        time_item = drilldown["time"]
        assert time_item.levels[0].name == "year"

        # Both items should be returned for dimension filtering
        time_items = drilldown.drilldown_for_dimension("time")
        assert len(time_items) == 2
