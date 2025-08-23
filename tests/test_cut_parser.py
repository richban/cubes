"""
Tests for CutParser class from cubes.query_v2.parsers.

This module tests the parsing of legacy cut string formats into modern
Pydantic-validated Cut objects, ensuring 100% compatibility with existing
cubes query formats while providing comprehensive error handling.
"""

import pytest

from cubes.errors import ArgumentError
from cubes.metadata_v2 import Attribute, Cube, Dimension, Hierarchy, Level
from cubes.query_v2.cells import PointCut, RangeCut, SetCut
from cubes.query_v2.parsers import CutParser


class TestCutParser:
    """Test the CutParser class."""

    def setup_method(self):
        """Setup test data including dimensions and cube."""
        # Create attributes for testing
        key_attr = Attribute(name="id", label="ID")
        label_attr = Attribute(name="name", label="Name")

        # Create multi-level dimensions for testing
        year_level = Level(name="year", attributes=[key_attr, label_attr], key="id")
        quarter_level = Level(
            name="quarter", attributes=[key_attr, label_attr], key="id"
        )
        month_level = Level(name="month", attributes=[key_attr, label_attr], key="id")

        category_level = Level(
            name="category", attributes=[key_attr, label_attr], key="id"
        )
        subcategory_level = Level(
            name="subcategory", attributes=[key_attr, label_attr], key="id"
        )

        country_level = Level(
            name="country", attributes=[key_attr, label_attr], key="id"
        )
        state_level = Level(name="state", attributes=[key_attr, label_attr], key="id")
        city_level = Level(name="city", attributes=[key_attr, label_attr], key="id")

        # Create hierarchies
        default_hierarchy = Hierarchy(
            name="default", levels=[year_level, quarter_level, month_level]
        )
        fiscal_hierarchy = Hierarchy(name="fiscal", levels=[year_level, quarter_level])

        product_hierarchy = Hierarchy(
            name="default", levels=[category_level, subcategory_level]
        )

        geo_hierarchy = Hierarchy(
            name="default", levels=[country_level, state_level, city_level]
        )
        admin_hierarchy = Hierarchy(
            name="administrative", levels=[country_level, state_level]
        )

        # Create dimensions
        self.date_dimension = Dimension(
            name="date",
            levels=[year_level, quarter_level, month_level],
            hierarchies=[default_hierarchy, fiscal_hierarchy],
        )

        self.product_dimension = Dimension(
            name="product",
            levels=[category_level, subcategory_level],
            hierarchies=[product_hierarchy],
        )

        self.geography_dimension = Dimension(
            name="geography",
            levels=[country_level, state_level, city_level],
            hierarchies=[geo_hierarchy, admin_hierarchy],
        )

        # Create a cube
        self.cube = Cube(
            name="sales",
            dimensions=[
                self.date_dimension,
                self.product_dimension,
                self.geography_dimension,
            ],
            measures=[],
        )

        # Create parser instance
        self.parser = CutParser(self.cube)

    def test_empty_cuts_string(self):
        """Test parsing empty cuts string."""
        cuts = self.parser.parse_cuts("")
        assert cuts == []

        cuts = self.parser.parse_cuts(None)
        assert cuts == []

    def test_parse_single_point_cut(self):
        """Test parsing a single point cut."""
        cuts = self.parser.parse_cuts("date:2020")

        assert len(cuts) == 1
        assert isinstance(cuts[0], PointCut)
        assert cuts[0].dimension.name == "date"
        assert cuts[0].path == ["2020"]
        assert cuts[0].hierarchy is None
        assert not cuts[0].invert

    def test_parse_multi_level_point_cut(self):
        """Test parsing point cut with multiple path levels."""
        cuts = self.parser.parse_cuts("date:2020,Q1")

        assert len(cuts) == 1
        cut = cuts[0]
        assert isinstance(cut, PointCut)
        assert cut.dimension.name == "date"
        assert cut.path == ["2020", "Q1"]
        assert cut.level_depth() == 2

    def test_parse_point_cut_with_hierarchy(self):
        """Test parsing point cut with explicit hierarchy specification."""
        cuts = self.parser.parse_cuts("date@fiscal:2020")

        assert len(cuts) == 1
        cut = cuts[0]
        assert isinstance(cut, PointCut)
        assert cut.dimension.name == "date"
        assert cut.hierarchy == "fiscal"
        assert cut.path == ["2020"]

    def test_parse_inverted_point_cut(self):
        """Test parsing inverted point cut."""
        cuts = self.parser.parse_cuts("!date:2020")

        assert len(cuts) == 1
        cut = cuts[0]
        assert isinstance(cut, PointCut)
        assert cut.dimension.name == "date"
        assert cut.path == ["2020"]
        assert cut.invert

    def test_parse_inverted_cut_with_hierarchy(self):
        """Test parsing inverted cut with hierarchy."""
        cuts = self.parser.parse_cuts("!date@fiscal:2020,Q1")

        assert len(cuts) == 1
        cut = cuts[0]
        assert isinstance(cut, PointCut)
        assert cut.dimension.name == "date"
        assert cut.hierarchy == "fiscal"
        assert cut.path == ["2020", "Q1"]
        assert cut.invert

    def test_parse_empty_path_cut(self):
        """Test parsing cut with empty path."""
        cuts = self.parser.parse_cuts("date:")

        assert len(cuts) == 1
        cut = cuts[0]
        assert isinstance(cut, PointCut)
        assert cut.dimension.name == "date"
        assert cut.path == [""]

    def test_parse_range_cut_full(self):
        """Test parsing a full range cut with both boundaries."""
        cuts = self.parser.parse_cuts("date:2020-2023")

        assert len(cuts) == 1
        cut = cuts[0]
        assert isinstance(cut, RangeCut)
        assert cut.dimension.name == "date"
        assert cut.from_path == ["2020"]
        assert cut.to_path == ["2023"]
        assert not cut.invert

    def test_parse_range_cut_multi_level(self):
        """Test parsing range cut with multi-level paths."""
        cuts = self.parser.parse_cuts("date:2020,Q1-2023,Q4")

        assert len(cuts) == 1
        cut = cuts[0]
        assert isinstance(cut, RangeCut)
        assert cut.dimension.name == "date"
        assert cut.from_path == ["2020", "Q1"]
        assert cut.to_path == ["2023", "Q4"]

    def test_parse_range_cut_from_only(self):
        """Test parsing range cut with only 'from' boundary."""
        cuts = self.parser.parse_cuts("date:2020-")

        assert len(cuts) == 1
        cut = cuts[0]
        assert isinstance(cut, RangeCut)
        assert cut.dimension.name == "date"
        assert cut.from_path == ["2020"]
        assert cut.to_path is None

    def test_parse_range_cut_to_only(self):
        """Test parsing range cut with only 'to' boundary."""
        cuts = self.parser.parse_cuts("date:-2023")

        assert len(cuts) == 1
        cut = cuts[0]
        assert isinstance(cut, RangeCut)
        assert cut.dimension.name == "date"
        assert cut.from_path is None
        assert cut.to_path == ["2023"]

    def test_parse_inverted_range_cut(self):
        """Test parsing inverted range cut."""
        cuts = self.parser.parse_cuts("!date:2020-2023")

        assert len(cuts) == 1
        cut = cuts[0]
        assert isinstance(cut, RangeCut)
        assert cut.dimension.name == "date"
        assert cut.from_path == ["2020"]
        assert cut.to_path == ["2023"]
        assert cut.invert

    def test_parse_set_cut_simple(self):
        """Test parsing a simple set cut."""
        cuts = self.parser.parse_cuts("date:2020;2021;2022")

        assert len(cuts) == 1
        cut = cuts[0]
        assert isinstance(cut, SetCut)
        assert cut.dimension.name == "date"
        assert cut.paths == [["2020"], ["2021"], ["2022"]]
        assert not cut.invert

    def test_parse_set_cut_multi_level(self):
        """Test parsing set cut with multi-level paths."""
        cuts = self.parser.parse_cuts("date:2020,Q1;2020,Q2;2021,Q1")

        assert len(cuts) == 1
        cut = cuts[0]
        assert isinstance(cut, SetCut)
        assert cut.dimension.name == "date"
        assert cut.paths == [["2020", "Q1"], ["2020", "Q2"], ["2021", "Q1"]]

    def test_parse_inverted_set_cut(self):
        """Test parsing inverted set cut."""
        cuts = self.parser.parse_cuts("!date:2020;2021;2022")

        assert len(cuts) == 1
        cut = cuts[0]
        assert isinstance(cut, SetCut)
        assert cut.dimension.name == "date"
        assert cut.paths == [["2020"], ["2021"], ["2022"]]
        assert cut.invert

    def test_parse_multiple_cuts(self):
        """Test parsing multiple cuts separated by |."""
        cuts = self.parser.parse_cuts("date:2020|product:electronics|geography:USA")

        assert len(cuts) == 3

        # Check date cut
        date_cut = cuts[0]
        assert isinstance(date_cut, PointCut)
        assert date_cut.dimension.name == "date"
        assert date_cut.path == ["2020"]

        # Check product cut
        product_cut = cuts[1]
        assert isinstance(product_cut, PointCut)
        assert product_cut.dimension.name == "product"
        assert product_cut.path == ["electronics"]

        # Check geography cut
        geo_cut = cuts[2]
        assert isinstance(geo_cut, PointCut)
        assert geo_cut.dimension.name == "geography"
        assert geo_cut.path == ["USA"]

    def test_parse_mixed_cut_types(self):
        """Test parsing different types of cuts together."""
        cuts = self.parser.parse_cuts(
            "date:2020-2023|product:electronics;books|geography:USA,CA"
        )

        assert len(cuts) == 3

        # Range cut
        assert isinstance(cuts[0], RangeCut)
        assert cuts[0].dimension.name == "date"

        # Set cut
        assert isinstance(cuts[1], SetCut)
        assert cuts[1].dimension.name == "product"
        assert cuts[1].paths == [["electronics"], ["books"]]

        # Point cut
        assert isinstance(cuts[2], PointCut)
        assert cuts[2].dimension.name == "geography"
        assert cuts[2].path == ["USA", "CA"]

    def test_parse_cuts_with_converters(self):
        """Test parsing cuts with member converters."""

        def date_converter(dimension, hierarchy, path):
            # Convert years to integers, then back to strings (simulating conversion)
            return [
                str(int(p) + 1000) if p.isdigit() and len(p) == 4 else p for p in path
            ]

        member_converters = {"date": date_converter}
        cuts = self.parser.parse_cuts("date:2020", member_converters=member_converters)

        assert len(cuts) == 1
        cut = cuts[0]
        assert isinstance(cut, PointCut)
        assert cut.dimension.name == "date"
        assert cut.path == ["3020"]  # 2020 + 1000

    def test_parse_cuts_with_role_converters(self):
        """Test parsing cuts with role-based converters."""
        # Add role to dimension for testing
        self.date_dimension.role = "time"

        def time_converter(dimension, hierarchy, path):
            return [f"TIME_{p}" for p in path]

        role_member_converters = {"time": time_converter}
        cuts = self.parser.parse_cuts(
            "date:2020", role_member_converters=role_member_converters
        )

        assert len(cuts) == 1
        cut = cuts[0]
        assert isinstance(cut, PointCut)
        assert cut.dimension.name == "date"
        assert cut.path == ["TIME_2020"]

    def test_error_invalid_dimension(self):
        """Test error handling for unknown dimension."""
        with pytest.raises(ArgumentError, match="Unknown dimension 'unknown'"):
            self.parser.parse_cuts("unknown:value")

    def test_error_invalid_cut_format(self):
        """Test error handling for invalid cut format."""
        with pytest.raises(ArgumentError, match="Invalid cut format"):
            self.parser.parse_cuts("invalid_format_no_colon")

    def test_error_invalid_dimension_specification(self):
        """Test error handling for invalid dimension specification."""
        with pytest.raises(ArgumentError, match="Invalid dimension specification"):
            self.parser.parse_cuts("@invalid@spec:value")

    def test_error_empty_range_cut(self):
        """Test error handling for empty range cut."""
        with pytest.raises(ArgumentError, match="at least one range boundary required"):
            self.parser.parse_cuts("date:-")

    def test_error_empty_set_cut(self):
        """Test error handling for empty set cut."""
        with pytest.raises(ArgumentError, match="Set cut cannot be empty"):
            self.parser.parse_cuts("date:;;;")  # Only empty elements

    def test_error_invalid_hierarchy(self):
        """Test error handling for invalid hierarchy name."""
        with pytest.raises(ArgumentError, match="Unknown dimension"):
            # This should fail during dimension lookup, not hierarchy validation
            self.parser.parse_cuts("nonexistent@hierarchy:value")

    def test_error_converter_failure(self):
        """Test error handling when converter fails."""

        def failing_converter(dimension, hierarchy, path):
            raise ValueError("Converter failed")

        member_converters = {"date": failing_converter}
        with pytest.raises(ArgumentError, match="Path converter failed"):
            self.parser.parse_cuts("date:2020", member_converters=member_converters)

    def test_error_context_in_multiple_cuts(self):
        """Test error provides context about which cut failed."""
        with pytest.raises(ArgumentError, match="Failed to parse cut 2 of 3"):
            self.parser.parse_cuts("date:2020|unknown:value|product:electronics")

    def test_parse_cuts_whitespace_handling(self):
        """Test that parser handles whitespace correctly."""
        cuts = self.parser.parse_cuts("  date:2020  |  product:electronics  ")

        assert len(cuts) == 2
        assert cuts[0].dimension.name == "date"
        assert cuts[1].dimension.name == "product"

    def test_parse_cuts_empty_segments(self):
        """Test handling of empty cut segments."""
        cuts = self.parser.parse_cuts("date:2020||product:electronics")

        # Should skip empty segments
        assert len(cuts) == 2
        assert cuts[0].dimension.name == "date"
        assert cuts[1].dimension.name == "product"

    def test_parse_single_cut_direct(self):
        """Test parsing single cut directly via parse_single_cut method."""
        cut = self.parser.parse_single_cut("date:2020,Q1")

        assert isinstance(cut, PointCut)
        assert cut.dimension.name == "date"
        assert cut.path == ["2020", "Q1"]

    def test_parse_single_cut_with_converters(self):
        """Test parse_single_cut with converters."""

        def upper_converter(dimension, hierarchy, path):
            return [p.upper() for p in path]

        member_converters = {"product": upper_converter}
        cut = self.parser.parse_single_cut(
            "product:electronics", member_converters=member_converters
        )

        assert isinstance(cut, PointCut)
        assert cut.dimension.name == "product"
        assert cut.path == ["ELECTRONICS"]

    def test_regex_patterns_match_correctly(self):
        """Test that regex patterns correctly identify cut types."""
        # Test set pattern detection
        assert self.parser.RE_SET.match("value1;value2")
        assert not self.parser.RE_SET.match("value1")
        assert not self.parser.RE_SET.match("value1-value2")

        # Test range pattern detection
        assert self.parser.RE_RANGE.match("value1-value2")
        assert not self.parser.RE_RANGE.match("value1")
        assert not self.parser.RE_RANGE.match("value1;value2")

    def test_dimension_hierarchy_invert_parsing(self):
        """Test detailed parsing of dimension specification patterns."""
        # Test all combinations
        test_cases = [
            ("dimension", ("dimension", None, False)),
            ("!dimension", ("dimension", None, True)),
            ("dimension@hierarchy", ("dimension", "hierarchy", False)),
            ("!dimension@hierarchy", ("dimension", "hierarchy", True)),
        ]

        for spec, expected in test_cases:
            result = self.parser._parse_dim_hierarchy_invert(spec)
            assert result == expected, f"Failed for spec: {spec}"

    def test_path_parsing_with_commas(self):
        """Test path parsing handles comma-separated values correctly."""
        path = self.parser._parse_path("2020,Q1,January")
        assert path == ["2020", "Q1", "January"]

        # Test empty path
        path = self.parser._parse_path("")
        assert path == []

        # Test path with empty components (legacy behavior)
        path = self.parser._parse_path("2020,,Q1")
        assert path == ["2020", "", "Q1"]

    def test_converter_selection_priority(self):
        """Test converter selection gives priority to dimension name over role."""
        # Set up dimension with role
        self.date_dimension.role = "time"

        def dim_converter(dimension, hierarchy, path):
            return [f"DIM_{p}" for p in path]

        def role_converter(dimension, hierarchy, path):
            return [f"ROLE_{p}" for p in path]

        member_converters = {"date": dim_converter}
        role_member_converters = {"time": role_converter}

        cuts = self.parser.parse_cuts(
            "date:2020",
            member_converters=member_converters,
            role_member_converters=role_member_converters,
        )

        # Should use dimension-specific converter, not role converter
        assert cuts[0].path == ["DIM_2020"]

    def test_converter_fallback_to_role(self):
        """Test converter falls back to role when no dimension-specific converter."""
        # Set up dimension with role
        self.date_dimension.role = "time"

        def role_converter(dimension, hierarchy, path):
            return [f"ROLE_{p}" for p in path]

        role_member_converters = {"time": role_converter}

        cuts = self.parser.parse_cuts(
            "date:2020", role_member_converters=role_member_converters
        )

        # Should use role converter
        assert cuts[0].path == ["ROLE_2020"]
