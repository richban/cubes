"""
Tests for modern Cell and Cut implementations equivalent to legacy tests.
"""

import unittest

import pytest

from cubes.errors import ArgumentError, HierarchyError, ModelError
from cubes.metadata_v2 import Attribute, Cube, Dimension, Hierarchy, Level
from cubes.query_v2.cells import (
    Cell,
    PointCut,
    RangeCut,
    SetCut,
)


class CutsTestCase(unittest.TestCase):
    """Test modern cut implementations equivalent to legacy CutsTestCase."""

    def setUp(self):
        """Setup test data with modern metadata_v2 models."""
        # Create dimension with date hierarchy
        key_attr = Attribute(name="id", label="ID")
        label_attr = Attribute(name="name", label="Name")

        # Date dimension levels
        year_level = Level(name="year", attributes=[key_attr, label_attr], key="id")
        month_level = Level(name="month", attributes=[key_attr, label_attr], key="id")
        day_level = Level(name="day", attributes=[key_attr, label_attr], key="id")

        date_hierarchy = Hierarchy(
            name="default", levels=[year_level, month_level, day_level]
        )
        self.dim_date = Dimension(
            name="date",
            levels=[year_level, month_level, day_level],
            hierarchies=[date_hierarchy],
        )

        # Create cube
        self.cube = Cube(name="transactions", dimensions=[self.dim_date], measures=[])

    def test_cut_depth(self):
        """Test level depth calculation for different cuts."""
        self.assertEqual(
            1, PointCut(dimension=self.dim_date, path=[1]).level_depth()
        )
        self.assertEqual(
            3,
            PointCut(dimension=self.dim_date, path=[1, 1, 1]).level_depth(),
        )
        self.assertEqual(
            1,
            RangeCut(
                dimension=self.dim_date, from_path=[1], to_path=[1]
            ).level_depth(),
        )
        self.assertEqual(
            3,
            RangeCut(
                dimension=self.dim_date, from_path=["1", "1", "1"], to_path=["1"]
            ).level_depth(),
        )
        self.assertEqual(
            1,
            SetCut(dimension=self.dim_date, paths=[["1"], ["1"]]).level_depth(),
        )
        self.assertEqual(
            3,
            SetCut(
                dimension=self.dim_date, paths=[["1"], ["1"], ["1", "1", "1"]]
            ).level_depth(),
        )

    def test_cut_to_dict(self):
        """Test converting cuts to dictionary representation."""
        # Point cut
        point_cut = PointCut(
            dimension=self.dim_date,
            path=[2010],
            hierarchy="default",
            invert=False,
            hidden=False,
        )
        expected_point = {
            "type": "point",
            "dimension": "date",
            "path": ["2010"],
            "level_depth": 1,
            "hierarchy": "default",
            "invert": False,
            "hidden": False,
        }
        self.assertEqual(expected_point, point_cut.to_dict())

        # Range cut
        range_cut = RangeCut(
            dimension=self.dim_date,
            from_path=[2010],
            to_path=[2012, 10],
            hierarchy="default",
            invert=False,
            hidden=False,
        )
        expected_range = {
            "type": "range",
            "dimension": "date",
            "from": ["2010"],
            "to": ["2012", "10"],
            "level_depth": 2,
            "hierarchy": "default",
            "invert": False,
            "hidden": False,
        }
        self.assertEqual(expected_range, range_cut.to_dict())

        # Set cut
        set_cut = SetCut(
            dimension=self.dim_date,
            paths=[[2010], [2012, 10]],
            hierarchy="default",
            invert=False,
            hidden=False,
        )
        expected_set = {
            "type": "set",
            "dimension": "date",
            "paths": [["2010"], ["2012", "10"]],
            "level_depth": 2,
            "hierarchy": "default",
            "invert": False,
            "hidden": False,
        }
        self.assertEqual(expected_set, set_cut.to_dict())

    def test_cut_validation(self):
        """Test cut validation with invalid hierarchy."""
        with pytest.raises((ValueError, ModelError, HierarchyError)):
            PointCut(
                dimension=self.dim_date,
                path=[2010],
                hierarchy="invalid_hierarchy",
            )

    def test_cut_invert(self):
        """Test invert functionality."""
        cut = PointCut(dimension=self.dim_date, path=[2010], invert=True)
        self.assertTrue(cut.invert)

        cut_dict = cut.to_dict()
        self.assertTrue(cut_dict["invert"])


def create_modern_cut_from_dict(desc, cube):
    """
    Create modern cut from dictionary description.

    This is the modern equivalent of cut_from_dict().
    """
    cut_type = desc["type"]
    dimension_name = desc["dimension"]
    dimension = cube.dimension(dimension_name)

    hierarchy = desc.get("hierarchy")
    invert = desc.get("invert", False)
    hidden = desc.get("hidden", False)

    if cut_type == "point":
        return PointCut(
            dimension=dimension,
            path=desc["path"],
            hierarchy=hierarchy,
            invert=invert,
            hidden=hidden,
        )
    elif cut_type == "range":
        return RangeCut(
            dimension=dimension,
            from_path=desc.get("from"),
            to_path=desc.get("to"),
            hierarchy=hierarchy,
            invert=invert,
            hidden=hidden,
        )
    elif cut_type == "set":
        return SetCut(
            dimension=dimension,
            paths=desc["paths"],
            hierarchy=hierarchy,
            invert=invert,
            hidden=hidden,
        )
    else:
        raise ArgumentError(f"Unknown cut type: {cut_type}")


class CutFromDictTestCase(unittest.TestCase):
    """Test modern cut creation from dictionaries."""

    def setUp(self):
        """Setup test data."""
        key_attr = Attribute(name="id", label="ID")
        label_attr = Attribute(name="name", label="Name")

        year_level = Level(name="year", attributes=[key_attr, label_attr], key="id")
        month_level = Level(name="month", attributes=[key_attr, label_attr], key="id")

        date_hierarchy = Hierarchy(name="default", levels=[year_level, month_level])
        self.dim_date = Dimension(
            name="date",
            levels=[year_level, month_level],
            hierarchies=[date_hierarchy],
        )

        self.cube = Cube(name="transactions", dimensions=[self.dim_date], measures=[])

    def test_cut_from_dict(self):
        """Test creating cuts from dictionary descriptions."""
        # Point cut
        d = {
            "type": "point",
            "path": ["2010"],
            "dimension": "date",
            "level_depth": 1,
            "hierarchy": None,
            "invert": False,
            "hidden": False,
        }

        cut = create_modern_cut_from_dict(d, self.cube)
        expected_cut = PointCut(dimension=self.dim_date, path=[2010])

        self.assertEqual(expected_cut.dimension.name, cut.dimension.name)
        self.assertEqual(expected_cut.path, cut.path)
        self.assertEqual(expected_cut.hierarchy, cut.hierarchy)
        self.assertEqual(expected_cut.invert, cut.invert)
        self.assertEqual(expected_cut.hidden, cut.hidden)

        # Verify to_dict roundtrip
        cut_dict = cut.to_dict()
        self.assertEqual(d["type"], cut_dict["type"])
        self.assertEqual(d["path"], cut_dict["path"])
        self.assertEqual(d["dimension"], cut_dict["dimension"])
        self.assertEqual(d["invert"], cut_dict["invert"])
        self.assertEqual(d["hidden"], cut_dict["hidden"])

        # Range cut
        d = {
            "type": "range",
            "from": ["2010"],
            "to": ["2012", "10"],
            "dimension": "date",
            "level_depth": 2,
            "hierarchy": None,
            "invert": False,
            "hidden": False,
        }
        cut = create_modern_cut_from_dict(d, self.cube)
        expected_cut = RangeCut(
            dimension=self.dim_date, from_path=[2010], to_path=[2012, 10]
        )

        self.assertEqual(expected_cut.dimension.name, cut.dimension.name)
        self.assertEqual(expected_cut.from_path, cut.from_path)
        self.assertEqual(expected_cut.to_path, cut.to_path)

        # Set cut
        d = {
            "type": "set",
            "paths": [["2010"], ["2012", "10"]],
            "dimension": "date",
            "level_depth": 2,
            "hierarchy": None,
            "invert": False,
            "hidden": False,
        }
        cut = create_modern_cut_from_dict(d, self.cube)
        expected_cut = SetCut(
            dimension=self.dim_date, paths=[[2010], [2012, 10]]
        )

        self.assertEqual(expected_cut.dimension.name, cut.dimension.name)
        self.assertEqual(expected_cut.paths, cut.paths)

        # Invalid cut type - include required dimension key
        self.assertRaises(
            ArgumentError,
            create_modern_cut_from_dict,
            {"type": "xxx", "dimension": "date"},
            self.cube,
        )


class CellInteractiveSlicingTestCase(unittest.TestCase):
    """Test modern cell slicing operations equivalent to legacy tests."""

    def setUp(self):
        """Setup test data."""
        # Create dimensions
        key_attr = Attribute(name="id", label="ID")
        label_attr = Attribute(name="name", label="Name")

        # Date dimension with multiple levels for drilldown
        year_level = Level(name="year", attributes=[key_attr, label_attr], key="id")
        month_level = Level(name="month", attributes=[key_attr, label_attr], key="id")
        date_hierarchy = Hierarchy(name="default", levels=[year_level, month_level])
        date_dim = Dimension(
            name="date", levels=[year_level, month_level], hierarchies=[date_hierarchy]
        )

        # Supplier dimension
        supplier_level = Level(
            name="supplier", attributes=[key_attr, label_attr], key="id"
        )
        supplier_hierarchy = Hierarchy(name="default", levels=[supplier_level])
        supplier_dim = Dimension(
            name="supplier", levels=[supplier_level], hierarchies=[supplier_hierarchy]
        )

        # CPV dimension with multiple levels
        division_level = Level(
            name="division", attributes=[key_attr, label_attr], key="id"
        )
        group_level = Level(name="group", attributes=[key_attr, label_attr], key="id")
        class_level = Level(name="class", attributes=[key_attr, label_attr], key="id")
        category_level = Level(
            name="category", attributes=[key_attr, label_attr], key="id"
        )

        cpv_hierarchy = Hierarchy(
            name="default",
            levels=[division_level, group_level, class_level, category_level],
        )
        cpv_dim = Dimension(
            name="cpv",
            levels=[division_level, group_level, class_level, category_level],
            hierarchies=[cpv_hierarchy],
        )

        self.cube = Cube(
            name="contracts", dimensions=[date_dim, supplier_dim, cpv_dim], measures=[]
        )

    def test_cutting(self):
        """Test cell cutting operations."""
        full_cube = Cell(cube=self.cube, cuts=[])
        self.assertEqual(self.cube, full_cube.cube)
        self.assertEqual(0, len(full_cube.cuts))

        # Add first cut
        date_dim = self.cube.dimension("date")
        cell = full_cube.slice(PointCut(dimension=date_dim, path=[2010]))
        self.assertEqual(1, len(cell.cuts))

        # Add more cuts
        supplier_dim = self.cube.dimension("supplier")
        cpv_dim = self.cube.dimension("cpv")

        cell = cell.slice(PointCut(dimension=supplier_dim, path=["1234"]))
        cell = cell.slice(PointCut(dimension=cpv_dim, path=["50", "20"]))
        self.assertEqual(3, len(cell.cuts))
        self.assertEqual(self.cube, cell.cube)

        # Adding existing slice should replace it
        cell = cell.slice(PointCut(dimension=date_dim, path=[2011]))
        self.assertEqual(3, len(cell.cuts))

        # Verify the date cut was updated
        date_cut = cell.cut_for_dimension("date")
        self.assertEqual(["2011"], date_cut.path)

    def test_multi_slice(self):
        """Test multi-slice operations."""
        full_cube = Cell(cube=self.cube, cuts=[])

        date_dim = self.cube.dimension("date")
        cpv_dim = self.cube.dimension("cpv")
        supplier_dim = self.cube.dimension("supplier")

        cuts_list = [
            PointCut(dimension=date_dim, path=[2010]),
            PointCut(dimension=cpv_dim, path=["50", "20"]),
            PointCut(dimension=supplier_dim, path=["1234"]),
        ]

        cell_list = full_cube.multi_slice(cuts_list)
        self.assertEqual(3, len(cell_list.cuts))

    def test_get_cell_dimension_cut(self):
        """Test getting cuts by dimension."""
        full_cube = Cell(cube=self.cube, cuts=[])

        date_dim = self.cube.dimension("date")
        supplier_dim = self.cube.dimension("supplier")

        cell = full_cube.slice(PointCut(dimension=date_dim, path=[2010]))
        cell = cell.slice(PointCut(dimension=supplier_dim, path=["1234"]))

        # Get existing cuts
        cut = cell.cut_for_dimension("date")
        self.assertEqual(cut.dimension.name, "date")

        # Get non-existent cut
        cut = cell.cut_for_dimension("cpv")
        self.assertEqual(cut, None)

        # Test invalid dimension - should return None since we don't validate dimension names
        cut = cell.cut_for_dimension("someunknown")
        self.assertEqual(cut, None)

    def test_slice_drilldown(self):
        """Test cell drilldown operations."""
        date_dim = self.cube.dimension("date")

        # Start with empty path
        cut = PointCut(dimension=date_dim, path=[])
        original_cell = Cell(cube=self.cube, cuts=[cut])

        # Drill down to 2010
        cell = original_cell.drilldown("date", "2010")
        self.assertEqual(["2010"], cell.cut_for_dimension("date").path)

        # Drill down to month 1
        cell = cell.drilldown("date", "1")
        self.assertEqual(["2010", "1"], cell.cut_for_dimension("date").path)


class ModernStringConversionsTestCase(unittest.TestCase):
    """Test string conversion functionality for modern cuts."""

    def setUp(self):
        """Setup test data."""
        key_attr = Attribute(name="id", label="ID")
        label_attr = Attribute(name="name", label="Name")

        # Create multiple levels for hierarchy to support multi-level paths
        level1 = Level(name="level1", attributes=[key_attr, label_attr], key="id")
        level2 = Level(name="level2", attributes=[key_attr, label_attr], key="id")
        level3 = Level(name="level3", attributes=[key_attr, label_attr], key="id")

        hierarchy = Hierarchy(name="default", levels=[level1, level2, level3])

        self.dim_foo = Dimension(
            name="foo", levels=[level1, level2, level3], hierarchies=[hierarchy]
        )

        self.cube = Cube(name="test", dimensions=[self.dim_foo], measures=[])

    def test_cut_string_conversions(self):
        """Test cut string representations."""
        # Basic point cut
        cut = PointCut(dimension=self.dim_foo, path=["10"])
        self.assertEqual("foo:10", str(cut))

        # Multi-level path
        cut = PointCut(dimension=self.dim_foo, path=["123_abc_", "10", "_"])
        self.assertEqual("foo:123_abc_,10,_", str(cut))

        # Path with spaces
        cut = PointCut(dimension=self.dim_foo, path=["123_ abc_"])
        self.assertEqual("foo:123_ abc_", str(cut))

    def test_range_cut_string(self):
        """Test range cut string representation."""
        # Basic range
        cut = RangeCut(
            dimension=self.dim_foo, from_path=[2010], to_path=[2011]
        )
        self.assertEqual("foo:2010-2011", str(cut))

        # Open-ended ranges
        cut = RangeCut(dimension=self.dim_foo, from_path=[2010], to_path=None)
        self.assertEqual("foo:2010-", str(cut))

        cut = RangeCut(dimension=self.dim_foo, from_path=None, to_path=[2010])
        self.assertEqual("foo:-2010", str(cut))

        # Multi-level range
        cut = RangeCut(
            dimension=self.dim_foo,
            from_path=["2010", "11", "12"],
            to_path=["2011", "2", "3"],
        )
        self.assertEqual("foo:2010,11,12-2011,2,3", str(cut))

    def test_set_cut_string(self):
        """Test set cut string representation."""
        cut = SetCut(
            dimension=self.dim_foo, paths=[["1"], ["2", "3"], ["qwe", "asd", "100"]]
        )
        self.assertEqual("foo:1;2,3;qwe,asd,100", str(cut))

    def test_hierarchy_cut(self):
        """Test cuts with explicit hierarchy."""
        # Create dimension with named hierarchy
        key_attr = Attribute(name="id", label="ID")
        level = Level(name="default", attributes=[key_attr], key="id")
        hierarchy = Hierarchy(name="dqmy", levels=[level])

        date_dim = Dimension(name="date", levels=[level], hierarchies=[hierarchy])

        cut = PointCut(dimension=date_dim, path=["10"], hierarchy="dqmy")
        self.assertEqual("date@dqmy:10", str(cut))


if __name__ == "__main__":
    unittest.main()
