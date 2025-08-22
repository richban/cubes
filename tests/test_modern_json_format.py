"""
Test the modern array-based JSON format for dimensions.

This focuses on Pattern 2: Array-based structure as the standard going forward.
"""

import unittest

from cubes.metadata_v2.dimension import Dimension


class ModernJsonFormatTestCase(unittest.TestCase):
    """Test case for modern array-based JSON format."""

    def test_modern_date_dimension(self):
        """Test parsing a date dimension with modern array-based format."""
        modern_date_json = {
            "name": "date",
            "label": "Date",
            "levels": [
                {
                    "name": "year",
                    "label": "Year",
                    "key": "year",
                    "attributes": ["year"],
                },
                {
                    "name": "month",
                    "label": "Month",
                    "key": "month",
                    "label_attribute": "month_name",
                    "attributes": ["month", "month_name", "month_sname"],
                },
                {
                    "name": "day",
                    "label": "Day",
                    "key": "day",
                    "attributes": ["day", "weekday_name", "weekday_number"],
                },
            ],
            "hierarchies": [
                {"name": "default", "levels": ["year", "month", "day"]},
                {"name": "ym", "levels": ["year", "month"]},
            ],
        }

        dimension = Dimension.model_validate(modern_date_json)

        # Validate basic properties
        self.assertEqual("date", dimension.name)
        self.assertEqual("Date", dimension.label)

        # Validate levels
        self.assertEqual(3, len(dimension.levels))
        self.assertEqual(
            ["year", "month", "day"], [level.name for level in dimension.levels]
        )

        # Validate hierarchies
        self.assertEqual(2, len(dimension.hierarchies))

        default_hierarchy = dimension.hierarchy("default")
        self.assertEqual(["year", "month", "day"], default_hierarchy.level_names)

        ym_hierarchy = dimension.hierarchy("ym")
        self.assertEqual(["year", "month"], ym_hierarchy.level_names)

    def test_modern_geography_dimension(self):
        """Test parsing a geography dimension with modern format."""
        modern_geo_json = {
            "name": "geography",
            "label": "Geography",
            "levels": [
                {
                    "name": "region",
                    "label": "Region",
                    "attributes": ["region_code", "region_name"],
                },
                {
                    "name": "county",
                    "label": "County",
                    "attributes": ["county_code", "county_name"],
                },
            ],
            "hierarchies": [{"name": "default", "levels": ["region", "county"]}],
        }

        dimension = Dimension.model_validate(modern_geo_json)

        # Validate basic properties
        self.assertEqual("geography", dimension.name)
        self.assertEqual("Geography", dimension.label)

        # Validate levels
        self.assertEqual(2, len(dimension.levels))
        level_names = [level.name for level in dimension.levels]
        self.assertEqual(["region", "county"], level_names)

        # Test hierarchy
        default_hierarchy = dimension.hierarchy("default")
        self.assertEqual(["region", "county"], default_hierarchy.level_names)

    def test_simple_dimension_with_string_levels(self):
        """Test creating dimensions with simple string levels."""
        simple_json = {
            "name": "product",
            "levels": ["category", "subcategory", "product"],
            "hierarchies": [
                {"name": "default", "levels": ["category", "subcategory", "product"]}
            ],
        }

        dimension = Dimension.model_validate(simple_json)

        self.assertEqual("product", dimension.name)
        self.assertEqual(3, len(dimension.levels))

        # String levels should get default attributes
        category_level = dimension.level("category")
        self.assertEqual("category", category_level.key)
        self.assertEqual("category", category_level.label_attribute)

    def test_mixed_level_formats(self):
        """Test mixing strings and detailed level definitions."""
        mixed_json = {
            "name": "customer",
            "levels": [
                "segment",  # Simple string
                {
                    "name": "customer",
                    "key": "customer_id",
                    "label_attribute": "customer_name",
                    "attributes": ["customer_id", "customer_name", "email", "phone"],
                },
            ],
            "hierarchies": [{"name": "default", "levels": ["segment", "customer"]}],
        }

        dimension = Dimension.model_validate(mixed_json)

        # Simple string level
        segment_level = dimension.level("segment")
        self.assertEqual("segment", segment_level.key)
        self.assertEqual(1, len(segment_level.attributes))

        # Detailed level
        customer_level = dimension.level("customer")
        self.assertEqual("customer_id", customer_level.key)
        self.assertEqual("customer_name", customer_level.label_attribute)
        self.assertEqual(4, len(customer_level.attributes))


if __name__ == "__main__":
    unittest.main()
