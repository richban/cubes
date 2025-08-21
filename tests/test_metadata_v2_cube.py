"""
Tests for modern Cube implementation.
"""

import unittest

from cubes.errors import (
    ArgumentError,
    ModelError,
    NoSuchAttributeError,
    NoSuchDimensionError,
)
from cubes.metadata_v2.attributes import Attribute, Measure, MeasureAggregate
from cubes.metadata_v2.cube import Cube, CubeType, create_star_schema_cube, merge_cubes
from cubes.metadata_v2.dimension import Dimension, Level


class CubeTestCase(unittest.TestCase):
    """Test the modern Cube class"""

    def setUp(self):
        """Set up test data"""
        # Create dimensions
        self.date_dim = Dimension(
            name="date",
            levels=[
                Level(name="year"),
                Level(
                    name="month",
                    attributes=[Attribute(name="month"), Attribute(name="month_name")],
                ),
                Level(name="day"),
            ],
        )

        self.product_dim = Dimension(
            name="product",
            levels=[
                Level(
                    name="category",
                    attributes=[
                        Attribute(name="category_id"),
                        Attribute(name="category_name"),
                    ],
                ),
                Level(
                    name="product",
                    attributes=[
                        Attribute(name="product_id"),
                        Attribute(name="product_name"),
                        Attribute(name="description"),
                    ],
                ),
            ],
        )

        # Create measures
        self.measures = [
            Measure(name="amount", aggregates=["sum", "avg"]),
            Measure(name="quantity", aggregates=["sum", "min", "max"]),
            Measure(name="discount", aggregates=["sum", "avg"]),
        ]

        # Create basic cube
        self.cube = Cube(
            name="sales",
            dimensions=[self.date_dim, self.product_dim],
            measures=self.measures,
            fact_table="sales_fact",
        )

    def test_simple_cube_creation(self):
        """Test creating a simple cube"""
        cube = Cube(name="simple")

        self.assertEqual("simple", cube.name)
        self.assertEqual(CubeType.FACT, cube.cube_type)
        self.assertEqual([], cube.dimensions)
        self.assertEqual([], cube.measures)
        self.assertEqual(1, len(cube.aggregates))  # Auto-generated fact count
        self.assertEqual("fact_count", cube.aggregates[0].name)

    def test_cube_with_dimensions_and_measures(self):
        """Test cube with full configuration"""
        self.assertEqual("sales", self.cube.name)
        self.assertEqual("sales_fact", self.cube.fact_table)
        self.assertEqual(2, len(self.cube.dimensions))
        self.assertEqual(3, len(self.cube.measures))
        self.assertTrue(len(self.cube.aggregates) > 0)  # Auto-generated

    def test_auto_generate_aggregates(self):
        """Test automatic aggregate generation"""
        cube = Cube(
            name="test",
            measures=[
                Measure(name="sales", aggregates=["sum", "avg"]),
                Measure(name="profit", aggregates=["sum"]),
            ],
            auto_generate_aggregates=True,
        )

        # Should have auto-generated aggregates
        self.assertEqual(3, len(cube.aggregates))  # sales_sum, sales_avg, profit_sum
        agg_names = [agg.name for agg in cube.aggregates]
        self.assertIn("sales_sum", agg_names)
        self.assertIn("sales_avg", agg_names)
        self.assertIn("profit_sum", agg_names)

    def test_cube_type_validation(self):
        """Test cube type validation"""
        cube = Cube(name="test", cube_type=CubeType.AGGREGATE)
        self.assertEqual(CubeType.AGGREGATE, cube.cube_type)

        cube2 = Cube(name="test2", cube_type="virtual")
        self.assertEqual(CubeType.VIRTUAL, cube2.cube_type)

    def test_nonadditive_dimensions(self):
        """Test non-additive dimension handling"""
        cube = Cube(
            name="test",
            dimensions=[self.date_dim],
            measures=[Measure(name="balance")],
            nonadditive_dimensions={"date"},
        )

        self.assertEqual({"date"}, cube.nonadditive_dimensions)
        # Measure should inherit nonadditive property
        self.assertIsNotNone(cube.measures[0].nonadditive)

    def test_dimension_names_property(self):
        """Test dimension names computed property"""
        names = self.cube.dimension_names
        self.assertEqual(["date", "product"], names)

    def test_measure_names_property(self):
        """Test measure names computed property"""
        names = self.cube.measure_names
        self.assertEqual(["amount", "quantity", "discount"], names)

    def test_get_dimension(self):
        """Test getting dimension by name"""
        date_dim = self.cube.get_dimension("date")
        self.assertEqual("date", date_dim.name)
        self.assertEqual(self.date_dim, date_dim)

        with self.assertRaises(NoSuchDimensionError):
            self.cube.get_dimension("nonexistent")

    def test_get_measure(self):
        """Test getting measure by name"""
        amount = self.cube.get_measure("amount")
        self.assertEqual("amount", amount.name)

        with self.assertRaises(NoSuchAttributeError):
            self.cube.get_measure("nonexistent")

    def test_get_aggregate(self):
        """Test getting aggregate by name"""
        # Should have auto-generated aggregates
        amount_sum = self.cube.get_aggregate("amount_sum")
        self.assertEqual("amount_sum", amount_sum.name)
        self.assertEqual("amount", amount_sum.measure)

        with self.assertRaises(NoSuchAttributeError):
            self.cube.get_aggregate("nonexistent")

    def test_get_attribute_by_reference(self):
        """Test getting attributes by various reference formats"""
        # Direct measure reference
        amount = self.cube.get_attribute("amount")
        self.assertEqual("amount", amount.name)

        # Dimension attribute reference
        year_attr = self.cube.get_attribute("date.year")
        self.assertEqual("year", year_attr.name)

        # Aggregate reference
        amount_sum = self.cube.get_attribute("amount_sum")
        self.assertEqual("amount_sum", amount_sum.name)

        with self.assertRaises(NoSuchAttributeError):
            self.cube.get_attribute("nonexistent.attr")

    def test_get_attributes_multiple(self):
        """Test getting multiple attributes"""
        attrs = self.cube.get_attributes(["amount", "date.year", "quantity_sum"])
        self.assertEqual(3, len(attrs))
        self.assertEqual(["amount", "year", "quantity_sum"], [a.name for a in attrs])

    def test_all_attributes_property(self):
        """Test all attributes computed property"""
        all_attrs = self.cube.all_attributes

        # Should include dimension attributes, measures, aggregates
        attr_names = [attr.name for attr in all_attrs]

        # Check we have dimension attributes
        self.assertIn("year", attr_names)
        self.assertIn("month", attr_names)
        self.assertIn("category_id", attr_names)

        # Check we have measures
        self.assertIn("amount", attr_names)
        self.assertIn("quantity", attr_names)

        # Check we have aggregates
        agg_names = [agg.name for agg in self.cube.aggregates]
        for agg_name in agg_names:
            self.assertIn(agg_name, attr_names)

    def test_has_dimension_measure_aggregate(self):
        """Test has_* methods"""
        self.assertTrue(self.cube.has_dimension("date"))
        self.assertTrue(self.cube.has_dimension("product"))
        self.assertFalse(self.cube.has_dimension("nonexistent"))

        self.assertTrue(self.cube.has_measure("amount"))
        self.assertTrue(self.cube.has_measure("quantity"))
        self.assertFalse(self.cube.has_measure("nonexistent"))

        self.assertTrue(self.cube.has_aggregate("amount_sum"))
        self.assertFalse(self.cube.has_aggregate("nonexistent"))

    def test_add_dimension(self):
        """Test adding dimension to cube"""
        new_dim = Dimension(
            name="location",
            levels=[Level(name="country", attributes=[Attribute(name="country")])],
        )

        self.cube.add_dimension(new_dim)
        self.assertTrue(self.cube.has_dimension("location"))
        self.assertEqual(3, len(self.cube.dimensions))

        # Test duplicate dimension
        with self.assertRaises(ModelError):
            self.cube.add_dimension(new_dim)

    def test_add_measure(self):
        """Test adding measure to cube"""
        new_measure = Measure(name="profit", aggregates=["sum", "avg"])

        self.cube.add_measure(new_measure)
        self.assertTrue(self.cube.has_measure("profit"))
        self.assertEqual(4, len(self.cube.measures))

        # Check auto-generated aggregates
        self.assertTrue(self.cube.has_aggregate("profit_sum"))
        self.assertTrue(self.cube.has_aggregate("profit_avg"))

        # Test duplicate measure
        with self.assertRaises(ModelError):
            self.cube.add_measure(new_measure)

    def test_remove_dimension(self):
        """Test removing dimension from cube"""
        self.cube.remove_dimension("date")
        self.assertFalse(self.cube.has_dimension("date"))
        self.assertEqual(1, len(self.cube.dimensions))

        with self.assertRaises(NoSuchDimensionError):
            self.cube.remove_dimension("nonexistent")

    def test_remove_measure(self):
        """Test removing measure from cube"""
        self.cube.remove_measure("amount")
        self.assertFalse(self.cube.has_measure("amount"))
        self.assertEqual(2, len(self.cube.measures))

        with self.assertRaises(NoSuchAttributeError):
            self.cube.remove_measure("nonexistent")

    def test_validate_references(self):
        """Test reference validation"""
        # Valid cube should have no errors
        errors = self.cube.validate_references()
        self.assertEqual([], errors)

        # Add invalid aggregate reference
        invalid_agg = MeasureAggregate(
            name="invalid", measure="nonexistent", function="sum"
        )
        self.cube.aggregates.append(invalid_agg)

        errors = self.cube.validate_references()
        self.assertEqual(1, len(errors))
        self.assertIn("nonexistent", errors[0])

    def test_cube_with_explicit_aggregates(self):
        """Test cube with explicitly defined aggregates"""
        cube = Cube(
            name="test",
            measures=[Measure(name="sales")],
            aggregates=[
                MeasureAggregate(name="total_sales", measure="sales", function="sum"),
                MeasureAggregate(name="avg_sales", measure="sales", function="avg"),
            ],
            auto_generate_aggregates=False,
        )

        self.assertEqual(2, len(cube.aggregates))
        self.assertEqual(["total_sales", "avg_sales"], cube.aggregate_names)

    def test_cube_with_details(self):
        """Test cube with detail attributes"""
        details = [Attribute(name="transaction_id"), Attribute(name="notes")]

        cube = Cube(name="detailed", measures=[Measure(name="amount")], details=details)

        self.assertEqual(2, len(cube.details))
        self.assertIn("transaction_id", [d.name for d in cube.details])


class CubeUtilitiesTestCase(unittest.TestCase):
    """Test cube utility functions"""

    def test_merge_cubes(self):
        """Test merging multiple cubes"""
        cube1 = Cube(
            name="sales",
            dimensions=[Dimension(name="date")],
            measures=[Measure(name="sales_amount")],
        )

        cube2 = Cube(
            name="inventory",
            dimensions=[Dimension(name="product")],
            measures=[Measure(name="stock_level")],
        )

        merged = merge_cubes(cube1, cube2, name="merged")

        self.assertEqual("merged", merged.name)
        self.assertEqual(CubeType.VIRTUAL, merged.cube_type)
        self.assertEqual(2, len(merged.dimensions))
        self.assertEqual(2, len(merged.measures))
        self.assertIn("date", merged.dimension_names)
        self.assertIn("product", merged.dimension_names)

        # Test empty merge
        with self.assertRaises(ArgumentError):
            merge_cubes(name="empty")

    def test_create_star_schema_cube(self):
        """Test star schema cube creation"""
        cube = create_star_schema_cube(
            name="sales_star",
            fact_table="fact_sales",
            dimensions=["date", "product", "customer"],
            measures=[
                {"name": "amount", "aggregates": ["sum", "avg"]},
                {"name": "quantity", "aggregates": ["sum"]},
            ],
        )

        self.assertEqual("sales_star", cube.name)
        self.assertEqual("fact_sales", cube.fact_table)
        self.assertEqual(CubeType.FACT, cube.cube_type)
        self.assertEqual(3, len(cube.dimensions))
        self.assertEqual(2, len(cube.measures))


if __name__ == "__main__":
    unittest.main()
