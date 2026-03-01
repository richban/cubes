"""Tests for string_to_dimension_level parser function."""

import pytest

from cubes.metadata.models.base import ArgumentError, string_to_dimension_level


class TestStringToDimensionLevel:
    """Tests for string_to_dimension_level parser function."""

    def test_dimension_only(self):
        """Parse string with dimension only."""
        result = string_to_dimension_level("date")
        assert result == ("date", None, None)

    def test_dimension_with_hierarchy(self):
        """Parse string with dimension and hierarchy."""
        result = string_to_dimension_level("date@yearly")
        assert result == ("date", "yearly", None)

    def test_dimension_with_hierarchy_and_level(self):
        """Parse string with dimension, hierarchy, and level."""
        result = string_to_dimension_level("date@yearly:year")
        assert result == ("date", "yearly", "year")

    def test_dimension_with_level_no_hierarchy(self):
        """Parse string with dimension and level but no hierarchy."""
        result = string_to_dimension_level("date:month")
        assert result == ("date", None, "month")

    def test_empty_string_raises_error(self):
        """Empty string should raise ArgumentError."""
        with pytest.raises(ArgumentError) as exc_info:
            string_to_dimension_level("")
        assert "should not be empty" in str(exc_info.value)

    def test_invalid_format_raises_error(self):
        """Invalid format should raise ArgumentError."""
        with pytest.raises(ArgumentError) as exc_info:
            string_to_dimension_level("invalid@")
        assert "does not match drilldown level pattern" in str(exc_info.value)

    def test_invalid_characters_raises_error(self):
        """String with invalid characters should raise ArgumentError."""
        with pytest.raises(ArgumentError) as exc_info:
            string_to_dimension_level("date@hier archy")
        assert "does not match drilldown level pattern" in str(exc_info.value)

    def test_dimension_with_underscores(self):
        """Parse dimension name with underscores."""
        result = string_to_dimension_level("customer_type")
        assert result == ("customer_type", None, None)

    def test_dimension_with_numbers(self):
        """Parse dimension name with numbers."""
        result = string_to_dimension_level("date2")
        assert result == ("date2", None, None)

    def test_dimension_starting_with_number(self):
        """Parse dimension name starting with number."""
        result = string_to_dimension_level("2d")
        assert result == ("2d", None, None)

    def test_hierarchy_with_underscores(self):
        """Parse hierarchy name with underscores."""
        result = string_to_dimension_level("date@fiscal_year")
        assert result == ("date", "fiscal_year", None)

    def test_level_with_underscores(self):
        """Parse level name with underscores."""
        result = string_to_dimension_level("date:week_number")
        assert result == ("date", None, "week_number")

    def test_all_parts_with_numbers(self):
        """Parse all parts with numbers."""
        result = string_to_dimension_level("dim1@hier2:level3")
        assert result == ("dim1", "hier2", "level3")

    def test_long_dimension_name(self):
        """Parse long dimension name."""
        result = string_to_dimension_level("very_long_dimension_name_with_many_parts")
        assert result == ("very_long_dimension_name_with_many_parts", None, None)

    def test_special_characters_in_middle_raises_error(self):
        """Special characters in the middle should raise ArgumentError."""
        with pytest.raises(ArgumentError) as exc_info:
            string_to_dimension_level("date@yearly:level#1")
        assert "does not match drilldown level pattern" in str(exc_info.value)

    def test_only_at_sign_raises_error(self):
        """String with only @ should raise ArgumentError."""
        with pytest.raises(ArgumentError):
            string_to_dimension_level("@")

    def test_only_colon_raises_error(self):
        """String with only : should raise ArgumentError."""
        with pytest.raises(ArgumentError):
            string_to_dimension_level(":")

    def test_at_sign_at_end_raises_error(self):
        """String ending with @ should raise ArgumentError."""
        with pytest.raises(ArgumentError):
            string_to_dimension_level("date@")

    def test_colon_at_end_raises_error(self):
        """String ending with : should raise ArgumentError."""
        with pytest.raises(ArgumentError):
            string_to_dimension_level("date@yearly:")

    def test_double_at_sign_raises_error(self):
        """String with double @ should raise ArgumentError."""
        with pytest.raises(ArgumentError):
            string_to_dimension_level("date@@yearly")

    def test_double_colon_raises_error(self):
        """String with double : should raise ArgumentError."""
        with pytest.raises(ArgumentError):
            string_to_dimension_level("date@yearly::level")

    def test_at_after_colon_raises_error(self):
        """String with @ after : should raise ArgumentError."""
        with pytest.raises(ArgumentError):
            string_to_dimension_level("date:level@hier")

    def test_return_type_is_tuple(self):
        """Return type should be a tuple."""
        result = string_to_dimension_level("date")
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_error_message_contains_input(self):
        """Error message should contain the input string."""
        with pytest.raises(ArgumentError) as exc_info:
            string_to_dimension_level("bad@input!")
        assert "bad@input!" in str(exc_info.value)

    def test_error_message_contains_pattern_info(self):
        """Error message should mention the expected pattern."""
        with pytest.raises(ArgumentError) as exc_info:
            string_to_dimension_level("date hier")
        assert "dimension@hierarchy:level" in str(exc_info.value)

    def test_single_letter_dimension(self):
        """Parse single letter dimension."""
        result = string_to_dimension_level("d")
        assert result == ("d", None, None)

    def test_unicode_dimension_name(self):
        """Parse dimension name with unicode characters."""
        result = string_to_dimension_level("月份")
        assert result == ("月份", None, None)

    def test_mixed_case(self):
        """Parse mixed case dimension, hierarchy, and level names."""
        result = string_to_dimension_level("Date@FiscalYear:Quarter")
        assert result == ("Date", "FiscalYear", "Quarter")

    def test_all_uppercase(self):
        """Parse all uppercase names."""
        result = string_to_dimension_level("DATE@YEARLY:YEAR")
        assert result == ("DATE", "YEARLY", "YEAR")

    def test_all_lowercase(self):
        """Parse all lowercase names."""
        result = string_to_dimension_level("date@yearly:year")
        assert result == ("date", "yearly", "year")
