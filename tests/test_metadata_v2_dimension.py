"""
Tests for the Pydantic-based dimension classes.
"""

import unittest

from pydantic import ValidationError

from cubes.errors import HierarchyError
from cubes.metadata_v2.dimension import Dimension, Hierarchy, Level


class LevelTestCase(unittest.TestCase):
    """Test case for the Level class."""

    def test_level_creation_from_string(self):
        level = Level(name="product")
        self.assertEqual("product", level.name)
        self.assertEqual(1, len(level.attributes))
        self.assertEqual("product", level.attributes[0].name)
        self.assertEqual("product", level.key)
        self.assertEqual("product", level.label_attribute)

    def test_level_creation_from_dict(self):
        level = Level.model_validate(
            {"name": "product", "attributes": ["id", "name", "description"]}
        )
        self.assertEqual("product", level.name)
        self.assertEqual(3, len(level.attributes))
        self.assertEqual("id", level.key)
        self.assertEqual("name", level.label_attribute)

    def test_level_creation_errors(self):
        with self.assertRaises(ValidationError):
            Level.model_validate(123)  # Invalid metadata type
        with self.assertRaises(ValidationError):
            Level.model_validate({})  # Missing name


class HierarchyTestCase(unittest.TestCase):
    """Test case for the Hierarchy class."""

    def setUp(self):
        self.levels = [
            Level(name="year"),
            Level(name="month"),
            Level(name="day"),
        ]
        self.hierarchy = Hierarchy(name="ymd", levels=self.levels)

    def test_hierarchy_creation(self):
        self.assertEqual("ymd", self.hierarchy.name)
        self.assertEqual(3, len(self.hierarchy.levels))

    def test_hierarchy_from_metadata(self):
        h = Hierarchy.model_validate({"name": "ym", "levels": ["year", "month"]})
        self.assertEqual("ym", h.name)
        self.assertEqual(2, len(h.levels))
        self.assertEqual("year", h.levels[0].name)

    def test_hierarchy_from_metadata_errors(self):
        with self.assertRaises(ValidationError):
            Hierarchy.model_validate(123)
        with self.assertRaises(ValidationError):
            Hierarchy.model_validate({"name": "test"})  # No levels

    def test_len_getitem_contains(self):
        self.assertEqual(3, len(self.hierarchy))
        self.assertEqual(self.levels[0], self.hierarchy[0])
        self.assertEqual(self.levels[1], self.hierarchy["month"])
        self.assertTrue("year" in self.hierarchy)
        self.assertTrue(self.levels[2] in self.hierarchy)
        self.assertFalse("week" in self.hierarchy)

    def test_level_names(self):
        self.assertEqual(["year", "month", "day"], self.hierarchy.level_names)

    def test_level_access(self):
        self.assertEqual(self.levels[0], self.hierarchy.level("year"))
        self.assertEqual(self.levels[1], self.hierarchy.level(self.levels[1]))
        with self.assertRaises(HierarchyError):
            self.hierarchy.level("nonexistent")

    def test_level_index(self):
        self.assertEqual(0, self.hierarchy.level_index("year"))
        self.assertEqual(2, self.hierarchy.level_index(self.levels[2]))
        with self.assertRaises(HierarchyError):
            self.hierarchy.level_index("nonexistent")

    def test_next_previous_level(self):
        self.assertEqual(self.levels[0], self.hierarchy.next_level(None))
        self.assertEqual(self.levels[1], self.hierarchy.next_level("year"))
        self.assertIsNone(self.hierarchy.next_level("day"))
        self.assertIsNone(self.hierarchy.previous_level(None))
        self.assertEqual(self.levels[1], self.hierarchy.previous_level("day"))
        self.assertIsNone(self.hierarchy.previous_level("year"))

    def test_is_last(self):
        self.assertFalse(self.hierarchy.is_last("year"))
        self.assertTrue(self.hierarchy.is_last("day"))

    def test_levels_for_depth(self):
        self.assertEqual(1, len(self.hierarchy.levels_for_depth(1)))
        self.assertEqual(self.levels[0], self.hierarchy.levels_for_depth(1)[0])
        self.assertEqual(2, len(self.hierarchy.levels_for_depth(1, drilldown=True)))
        with self.assertRaises(HierarchyError):
            self.hierarchy.levels_for_depth(4)

    def test_levels_for_path(self):
        path = ["year", "month"]
        self.assertEqual(2, len(self.hierarchy.levels_for_path(path)))
        self.assertEqual(3, len(self.hierarchy.levels_for_path(path, drilldown=True)))

    def test_rollup(self):
        path = ["year", "month", "day"]
        self.assertEqual(["year", "month"], self.hierarchy.rollup(path, "month"))
        self.assertEqual(["year", "month"], self.hierarchy.rollup(path, self.levels[1]))
        self.assertEqual(["year", "month"], self.hierarchy.rollup(path))
        self.assertEqual([], self.hierarchy.rollup(["year"]))
        with self.assertRaises(HierarchyError):
            self.hierarchy.rollup(["year"], "day")

    def test_path_is_base(self):
        self.assertFalse(self.hierarchy.path_is_base(["year"]))
        self.assertTrue(self.hierarchy.path_is_base(["year", "month", "day"]))

    def test_key_attributes(self):
        keys = self.hierarchy.key_attributes
        self.assertEqual(3, len(keys))
        self.assertEqual("year", keys[0].name)

    def test_all_attributes(self):
        all_attrs = self.hierarchy.all_attributes
        self.assertEqual(3, len(all_attrs))
        self.assertEqual("year", all_attrs[0].name)


class DimensionTestCase(unittest.TestCase):
    """Test case for the modernized Dimension class."""

    def test_explicit_creation(self):
        """Test creating a dimension with explicit parameters."""
        dim = Dimension(name="date", levels=["year", "month"])
        self.assertEqual("date", dim.name)
        self.assertEqual(2, len(dim.levels))
        self.assertEqual("year", dim.levels[0].name)
        self.assertEqual("month", dim.levels[1].name)
        self.assertFalse(dim.is_flat)

    def test_default_hierarchy_creation(self):
        """Test that a default hierarchy is created if none is provided."""
        levels_meta = ["year", "month", "day"]
        dim = Dimension.model_validate({"name": "date", "levels": levels_meta})
        self.assertEqual(1, len(dim.hierarchies))
        self.assertEqual("default", dim.hierarchies[0].name)
        self.assertEqual("default", dim.default_hierarchy_name)
        self.assertEqual(3, len(dim.hierarchies[0].levels))

    def test_explicit_hierarchy(self):
        """Test creating a dimension with explicit hierarchies."""
        dim_meta = {
            "name": "date",
            "levels": ["year", "month"],
            "hierarchies": [{"name": "ym", "levels": ["year", "month"]}],
        }
        dim = Dimension.model_validate(dim_meta)
        self.assertEqual(1, len(dim.hierarchies))
        self.assertEqual("ym", dim.hierarchies[0].name)

    def test_attribute_ownership_and_ref(self):
        """Test that attributes are owned and their refs are correct."""
        dim_meta = {
            "name": "product",
            "levels": [{"name": "category", "attributes": ["id", "name"]}],
        }
        dim = Dimension.model_validate(dim_meta)
        attr = dim.levels[0].attributes[0]
        self.assertEqual(dim, attr.dimension)
        self.assertEqual("product.id", attr.ref)

    def test_get_hierarchy(self):
        """Test retrieving a hierarchy by name or default."""
        dim_meta = {
            "name": "date",
            "levels": ["year", "month"],
            "hierarchies": [
                {"name": "ym", "levels": ["year", "month"]},
                {"name": "y", "levels": ["year"]},
            ],
            "default_hierarchy_name": "ym",
        }
        dim = Dimension.model_validate(dim_meta)

        self.assertEqual("ym", dim.hierarchy().name)
        self.assertEqual("y", dim.hierarchy("y").name)

        with self.assertRaises(HierarchyError):
            dim.hierarchy("dne")

    def test_dimension_creation_errors(self):
        """Test error handling during dimension creation."""
        with self.assertRaises(ValidationError):
            Dimension.model_validate(123)  # Invalid metadata type

        with self.assertRaises(ValidationError):
            Dimension.model_validate({"levels": ["year"]})  # Missing name

        # Test that string input no longer works
        with self.assertRaises(TypeError):
            Dimension("date")  # String input not supported


if __name__ == "__main__":
    unittest.main()
