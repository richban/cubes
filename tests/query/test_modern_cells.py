"""
Tests for modern Cell and Cut implementations.
"""

import pytest
from pydantic import ValidationError

from cubes.errors import ArgumentError
from cubes.metadata_v2 import Attribute, Cube, Dimension, Hierarchy, Level
from cubes.query_v2.cells import (
    Cell,
    PointCut,
    RangeCut,
    SetCut,
)


class TestCuts:
    """Test modern cut implementations."""

    def setup_method(self):
        """Setup test data."""
        # Create a simple dimension for testing
        key_attr = Attribute(name="id", label="ID")
        label_attr = Attribute(name="name", label="Name")

        level1 = Level(name="category", attributes=[key_attr, label_attr], key="id")
        level2 = Level(name="subcategory", attributes=[key_attr, label_attr], key="id")

        hierarchy = Hierarchy(name="default", levels=[level1, level2])
        self.dimension = Dimension(
            name="product", levels=[level1, level2], hierarchies=[hierarchy]
        )

        # Create a simple cube
        self.cube = Cube(name="sales", dimensions=[self.dimension], measures=[])

    def test_modern_point_cut_creation(self):
        """Test creating a modern point cut."""
        cut = PointCut(
            dimension=self.dimension, path=["electronics"], hierarchy="default"
        )

        assert cut.dimension == self.dimension
        assert cut.path == ["electronics"]
        assert cut.hierarchy == "default"
        assert cut.level_depth() == 1
        assert not cut.invert
        assert not cut.hidden

    def test_modern_point_cut_validation(self):
        """Test point cut validation."""
        # Valid path should work
        cut = PointCut(
            dimension=self.dimension,
            path=["electronics", "laptops"],
            hierarchy="default",
        )
        assert cut.level_depth() == 2

        # Invalid hierarchy should raise error during creation
        from cubes.errors import HierarchyError

        with pytest.raises((ValidationError, ValueError, HierarchyError)):
            PointCut(
                dimension=self.dimension,
                path=["electronics"],
                hierarchy="invalid_hierarchy",
            )

    def test_modern_range_cut_creation(self):
        """Test creating a modern range cut."""
        cut = RangeCut(
            dimension=self.dimension,
            from_path=["electronics"],
            to_path=["furniture"],
            hierarchy="default",
        )

        assert cut.from_path == ["electronics"]
        assert cut.to_path == ["furniture"]
        assert cut.level_depth() == 1

    def test_modern_range_cut_open_ranges(self):
        """Test range cuts with open ranges."""
        # Open start
        cut = RangeCut(
            dimension=self.dimension,
            from_path=None,
            to_path=["furniture"],
            hierarchy="default",
        )
        assert cut.level_depth() == 1

        # Open end
        cut = RangeCut(
            dimension=self.dimension,
            from_path=["electronics"],
            to_path=None,
            hierarchy="default",
        )
        assert cut.level_depth() == 1

        # Both open
        cut = RangeCut(
            dimension=self.dimension, from_path=None, to_path=None, hierarchy="default"
        )
        assert cut.level_depth() == 0

    def test_modern_set_cut_creation(self):
        """Test creating a modern set cut."""
        cut = SetCut(
            dimension=self.dimension,
            paths=[["electronics"], ["furniture"], ["books"]],
            hierarchy="default",
        )

        assert len(cut.paths) == 3
        assert cut.level_depth() == 1

    def test_modern_set_cut_validation(self):
        """Test set cut validation."""
        # Valid paths should work
        cut = SetCut(
            dimension=self.dimension,
            paths=[["electronics", "laptops"], ["furniture", "chairs"]],
            hierarchy="default",
        )
        assert cut.level_depth() == 2

        # Empty paths should work
        cut = SetCut(dimension=self.dimension, paths=[], hierarchy="default")
        assert cut.level_depth() == 0

    def test_cut_to_dict(self):
        """Test converting cuts to dictionary."""
        cut = PointCut(
            dimension=self.dimension,
            path=["electronics"],
            hierarchy="default",
            invert=True,
            hidden=True,
        )

        d = cut.to_dict()
        assert d["type"] == "point"
        assert d["dimension"] == "product"
        assert d["path"] == ["electronics"]
        assert d["hierarchy"] == "default"
        assert d["invert"] is True
        assert d["hidden"] is True
        assert d["level_depth"] == 1

    def test_cut_string_representation(self):
        """Test string representation of cuts."""
        cut = PointCut(
            dimension=self.dimension,
            path=["electronics", "laptops"],
            hierarchy="default",
        )

        str_repr = str(cut)
        assert "product" in str_repr
        assert "electronics" in str_repr
        assert "laptops" in str_repr


class TestCell:
    """Test modern cell implementation."""

    def setup_method(self):
        """Setup test data."""
        # Create dimensions for testing
        key_attr = Attribute(name="id", label="ID")
        label_attr = Attribute(name="name", label="Name")

        level1 = Level(name="category", attributes=[key_attr, label_attr], key="id")
        level2 = Level(name="subcategory", attributes=[key_attr, label_attr], key="id")

        hierarchy = Hierarchy(name="default", levels=[level1, level2])
        self.product_dim = Dimension(
            name="product", levels=[level1, level2], hierarchies=[hierarchy]
        )
        self.date_dim = Dimension(
            name="date", levels=[level1, level2], hierarchies=[hierarchy]
        )

        # Create a cube with these dimensions
        self.cube = Cube(
            name="sales", dimensions=[self.product_dim, self.date_dim], measures=[]
        )

    def test_modern_cell_creation(self):
        """Test creating a modern cell."""
        cut1 = PointCut(dimension=self.product_dim, path=["electronics"])
        cut2 = PointCut(dimension=self.date_dim, path=["2023"])

        cell = Cell(cube=self.cube, cuts=[cut1, cut2])

        assert cell.cube == self.cube
        assert len(cell.cuts) == 2
        assert cell.cuts[0] == cut1
        assert cell.cuts[1] == cut2

    def test_cell_validation(self):
        """Test cell validation."""
        # Valid cuts should work
        cut = PointCut(dimension=self.product_dim, path=["electronics"])
        cell = Cell(cube=self.cube, cuts=[cut])
        assert len(cell.cuts) == 1

        # Cut with dimension not in cube should fail validation
        other_dimension = Dimension(name="other", hierarchies=[])
        invalid_cut = PointCut(dimension=other_dimension, path=["test"])

        from cubes.errors import NoSuchDimensionError

        with pytest.raises((ValidationError, ValueError, NoSuchDimensionError)):
            Cell(cube=self.cube, cuts=[invalid_cut])

    def test_cell_slicing(self):
        """Test cell slicing operations."""
        cut1 = PointCut(dimension=self.product_dim, path=["electronics"])
        cell = Cell(cube=self.cube, cuts=[cut1])

        # Add a new cut
        cut2 = PointCut(dimension=self.date_dim, path=["2023"])
        new_cell = cell.slice(cut2)

        assert len(new_cell.cuts) == 2
        assert new_cell.cuts[0] == cut1
        assert new_cell.cuts[1] == cut2

        # Replace existing cut
        cut3 = PointCut(dimension=self.product_dim, path=["furniture"])
        newer_cell = new_cell.slice(cut3)

        assert len(newer_cell.cuts) == 2
        assert newer_cell.cuts[0] == cut3  # Replaced
        assert newer_cell.cuts[1] == cut2  # Unchanged

    def test_cell_cut_lookup(self):
        """Test finding cuts for dimensions."""
        cut1 = PointCut(dimension=self.product_dim, path=["electronics"])
        cut2 = RangeCut(
            dimension=self.date_dim, from_path=["2023"], to_path=["2024"]
        )

        cell = Cell(cube=self.cube, cuts=[cut1, cut2])

        # Find cuts by dimension name
        product_cut = cell.cut_for_dimension("product")
        assert product_cut == cut1

        date_cut = cell.cut_for_dimension("date")
        assert date_cut == cut2

        nonexistent_cut = cell.cut_for_dimension("nonexistent")
        assert nonexistent_cut is None

        # Find point cuts specifically
        point_cut = cell.point_cut_for_dimension("product")
        assert point_cut == cut1

        point_cut_date = cell.point_cut_for_dimension("date")
        assert point_cut_date is None  # It's a range cut, not point cut

    def test_cell_drilldown(self):
        """Test cell drilldown operations."""
        cut = PointCut(dimension=self.product_dim, path=["electronics"])
        cell = Cell(cube=self.cube, cuts=[cut])

        # Drill down to next level
        drilled_cell = cell.drilldown("product", "laptops")

        drilled_cut = drilled_cell.point_cut_for_dimension("product")
        assert drilled_cut.path == ["electronics", "laptops"]

        # Drill down dimension that doesn't have a cut yet
        drilled_cell2 = cell.drilldown("date", "2023")

        date_cut = drilled_cell2.point_cut_for_dimension("date")
        assert date_cut.path == ["2023"]

    def test_cell_rollup(self):
        """Test cell rollup operations."""
        cut = PointCut(
            dimension=self.product_dim, path=["electronics", "laptops"]
        )
        cell = Cell(cube=self.cube, cuts=[cut])

        # Roll up one level
        rolled_cell = cell.rollup_dim("product")

        rolled_cut = rolled_cell.point_cut_for_dimension("product")
        assert rolled_cut.path == ["electronics"]

        # Roll up to top level (should remove the cut)
        rolled_again = rolled_cell.rollup_dim("product")
        final_cut = rolled_again.point_cut_for_dimension("product")
        # Should be None or empty path depending on implementation
        assert final_cut is None or len(final_cut.path) == 0

    def test_cell_multi_slice(self):
        """Test applying multiple cuts at once."""
        cell = Cell(cube=self.cube, cuts=[])

        cuts = [
            PointCut(dimension=self.product_dim, path=["electronics"]),
            PointCut(dimension=self.date_dim, path=["2023"]),
        ]

        multi_sliced = cell.multi_slice(cuts)

        assert len(multi_sliced.cuts) == 2
        assert multi_sliced.point_cut_for_dimension("product").path == ["electronics"]
        assert multi_sliced.point_cut_for_dimension("date").path == ["2023"]

    def test_cell_level_depths(self):
        """Test level depth calculation."""
        cuts = [
            PointCut(dimension=self.product_dim, path=["electronics", "laptops"]),
            PointCut(dimension=self.date_dim, path=["2023"]),
        ]

        cell = Cell(cube=self.cube, cuts=cuts)
        depths = cell.level_depths()

        assert depths["product"] == 2
        assert depths["date"] == 1

    def test_cell_dimension_cuts(self):
        """Test getting cuts for specific dimensions."""
        cuts = [
            PointCut(dimension=self.product_dim, path=["electronics"]),
            RangeCut(dimension=self.product_dim, from_path=["a"], to_path=["z"]),
            PointCut(dimension=self.date_dim, path=["2023"]),
        ]

        cell = Cell(cube=self.cube, cuts=cuts)

        # Get cuts for product dimension
        product_cuts = cell.dimension_cuts("product")
        assert len(product_cuts) == 2

        # Get cuts excluding product dimension
        non_product_cuts = cell.dimension_cuts("product", exclude=True)
        assert len(non_product_cuts) == 1
        assert non_product_cuts[0].dimension.name == "date"

    def test_cell_public_cell(self):
        """Test filtering hidden cuts."""
        cuts = [
            PointCut(
                dimension=self.product_dim, path=["electronics"], hidden=False
            ),
            PointCut(dimension=self.date_dim, path=["2023"], hidden=True),
        ]

        cell = Cell(cube=self.cube, cuts=cuts)
        public_cell = cell.public_cell()

        assert len(public_cell.cuts) == 1
        assert public_cell.cuts[0].dimension.name == "product"

    def test_cell_to_dict(self):
        """Test converting cell to dictionary."""
        cut = PointCut(dimension=self.product_dim, path=["electronics"])
        cell = Cell(cube=self.cube, cuts=[cut])

        d = cell.to_dict()
        assert d["cube"] == "sales"
        assert len(d["cuts"]) == 1
        assert d["cuts"][0]["type"] == "point"
        assert d["cuts"][0]["dimension"] == "product"

    def test_cell_boolean_behavior(self):
        """Test cell boolean behavior."""
        empty_cell = Cell(cube=self.cube, cuts=[])
        assert not empty_cell

        cut = PointCut(dimension=self.product_dim, path=["electronics"])
        non_empty_cell = Cell(cube=self.cube, cuts=[cut])
        assert non_empty_cell

    def test_cell_combination(self):
        """Test combining cells."""
        cut1 = PointCut(dimension=self.product_dim, path=["electronics"])
        cell1 = Cell(cube=self.cube, cuts=[cut1])

        cut2 = PointCut(dimension=self.date_dim, path=["2023"])
        cell2 = Cell(cube=self.cube, cuts=[cut2])

        # Combine cells from same cube
        combined = cell1 & cell2
        assert len(combined.cuts) == 2

        # Try to combine cells from different cubes
        other_cube = Cube(name="other", dimensions=[], measures=[])
        other_cell = Cell(cube=other_cube, cuts=[])

        with pytest.raises(ArgumentError):
            cell1 & other_cell
