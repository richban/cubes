"""
Tests for the Pydantic-based attribute classes.
"""

import unittest
from cubes.metadata_v2.attributes import (
    AttributeBase, Attribute, Measure, MeasureAggregate, 
    OrderType, NonAdditiveType
)
from cubes.errors import ModelError, ArgumentError, ExpressionError
from pydantic import ValidationError


class AttributeBaseTestCase(unittest.TestCase):
    """Test case for AttributeBase class"""
    
    def test_basic_initialization(self):
        """Test basic attribute creation"""
        attr = AttributeBase(name="test_attr", label="Test Attribute")
        self.assertEqual("test_attr", attr.name)
        self.assertEqual("Test Attribute", attr.label)
        self.assertEqual("test_attr", attr.ref)
        self.assertTrue(attr.is_base)
        self.assertEqual(set(), attr.dependencies)
    
    def test_order_validation(self):
        """Test order field validation and normalization"""
        attr = AttributeBase(name="test", order="asc")
        self.assertEqual(OrderType.ASC, attr.order)
        
        with self.assertRaises(ValidationError):
            AttributeBase(name="test", order="invalid")
    
    def test_expression_dependencies(self):
        """Test expression dependency detection"""
        attr = AttributeBase(
            name="computed", 
            expression="amount * 2 + discount - tax"
        )
        self.assertFalse(attr.is_base)
        self.assertEqual({"amount", "discount", "tax"}, attr.dependencies)
    
    def test_model_dump(self):
        """Test dictionary conversion using model_dump"""
        attr = AttributeBase(
            name="test",
            label="Test Label",
            format="%.2f",
            order="asc",
        )
        
        result = attr.model_dump(exclude_none=True)
        self.assertEqual("test", result["name"])
        self.assertEqual("Test Label", result["label"])
        self.assertEqual("%.2f", result["format"])
        self.assertEqual("asc", result["order"])


class AttributeTestCase(unittest.TestCase):
    """Test case for Attribute class"""
    
    def test_dimension_property_and_ref(self):
        """Test dimension property and dynamic ref"""
        attr = Attribute(name="category")
        self.assertEqual("category", attr.ref)

        class MockDimension:
            name = "product"
        
        dim = MockDimension()
        attr.dimension = dim
        
        self.assertEqual(dim, attr.dimension)
        self.assertEqual("product.category", attr.ref)
        
        attr.dimension = None
        self.assertEqual("category", attr.ref)


class MeasureTestCase(unittest.TestCase):
    """Test case for Measure class"""
    
    def test_default_aggregates(self):
        """Test default aggregate generation"""
        measure = Measure(name="amount", aggregates=["sum", "min"])
        aggs = measure.default_aggregates()
        
        self.assertEqual(2, len(aggs))
        self.assertEqual("amount_sum", aggs[0].name)
        self.assertEqual("amount_min", aggs[1].name)

    def test_formula(self):
        """Test that formula is stored correctly"""
        measure = Measure(name="revenue", formula="price * quantity")
        self.assertEqual("price * quantity", measure.formula)

    def test_nonadditive(self):
        """Test nonadditive validation"""
        measure_time = Measure(name="inventory", nonadditive="time")
        self.assertEqual(NonAdditiveType.TIME, measure_time.nonadditive)

        measure_any = Measure(name="headcount", nonadditive="any")
        self.assertEqual(NonAdditiveType.ANY, measure_any.nonadditive)

        with self.assertRaises(ValidationError):
            Measure(name="invalid", nonadditive="something")

    def test_window_size(self):
        """Test window_size validation"""
        measure = Measure(name="sales_avg", window_size=7)
        self.assertEqual(7, measure.window_size)

        with self.assertRaises(ValidationError):
            Measure(name="invalid", window_size=0)

        with self.assertRaises(ValidationError):
            Measure(name="invalid", window_size=-1)


class MeasureAggregateTestCase(unittest.TestCase):
    """Test case for MeasureAggregate class"""
    
    def test_dependencies(self):
        """Test dependency calculation"""
        agg_measure = MeasureAggregate(name="amount_sum", measure="amount")
        self.assertEqual({"amount"}, agg_measure.dependencies)
        
        agg_expr = MeasureAggregate(name="computed", expression="x + y")
        self.assertEqual({"x", "y"}, agg_expr.dependencies)


if __name__ == '__main__':
    unittest.main()