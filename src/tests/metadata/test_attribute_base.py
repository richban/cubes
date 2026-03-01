"""Tests for AttributeBase domain model."""

from dataclasses import asdict

import pytest

from cubes.metadata.models.attributes import AttributeBase, inspect_variables
from cubes.metadata.models.base import ModelError


class TestInspectVariables:
    """Test the inspect_variables utility function."""

    def test_empty_expression(self):
        """Empty expression returns empty set."""
        assert inspect_variables("") == set()

    def test_simple_variables(self):
        """Extract simple variable names."""
        result = inspect_variables("a + b")
        assert result == {"a", "b"}

    def test_expression_with_operators(self):
        """Extract variables from expression with operators."""
        result = inspect_variables("amount * price / quantity")
        assert result == {"amount", "price", "quantity"}

    def test_expression_with_numbers(self):
        """Numbers should not be included."""
        result = inspect_variables("a + 2 * b - 3.5")
        assert result == {"a", "b"}

    def test_dotted_variables(self):
        """Handle dotted attribute access like dimension.attribute."""
        result = inspect_variables("product.price * quantity")
        assert result == {"product.price", "quantity"}

    def test_python_keywords_excluded(self):
        """Python keywords should not be included."""
        result = inspect_variables("a and b or c")
        assert result == {"a", "b", "c"}

    def test_true_false_none_excluded(self):
        """True, False, None should not be included."""
        result = inspect_variables("value is None or flag is True")
        assert result == {"value", "flag"}

    def test_complex_expression(self):
        """Handle complex arithmetic expression."""
        result = inspect_variables("(revenue - cost) / revenue * 100")
        assert result == {"revenue", "cost"}

    def test_function_call_expression(self):
        """Handle function calls in expression."""
        result = inspect_variables("sum(amount) / count(*)")
        # sum and count might be keywords or just variables depending on context
        # but the important thing is that amount is captured
        assert "amount" in result

    def test_whitespace_variations(self):
        """Handle various whitespace patterns."""
        result1 = inspect_variables("a+b")
        result2 = inspect_variables("a + b")
        result3 = inspect_variables("a   +   b")
        assert result1 == result2 == result3 == {"a", "b"}


class TestAttributeBaseBasics:
    """Test basic AttributeBase creation and properties."""

    def test_minimal_attribute(self):
        """Create attribute with just name."""
        attr = AttributeBase(name="test")
        assert attr.name == "test"
        assert attr.ref == "test"
        assert attr.label is None
        assert attr.description is None
        assert attr.order is None
        assert attr.info == {}
        assert attr.format is None
        assert attr.missing_value is None
        assert attr.expression is None

    def test_full_attribute(self):
        """Create attribute with all properties."""
        attr = AttributeBase(
            name="amount",
            label="Total Amount",
            description="The total amount",
            order="desc",
            info={"key": "value"},
            format="#,##0.00",
            missing_value=0,
            expression="price * quantity",
        )
        assert attr.name == "amount"
        assert attr.ref == "amount"
        assert attr.label == "Total Amount"
        assert attr.description == "The total amount"
        assert attr.order == "desc"
        assert attr.info == {"key": "value"}
        assert attr.format == "#,##0.00"
        assert attr.missing_value == 0
        assert attr.expression == "price * quantity"

    def test_order_normalization(self):
        """Order should be normalized to lowercase."""
        attr1 = AttributeBase(name="a", order="ASC")
        attr2 = AttributeBase(name="b", order="Desc")
        attr3 = AttributeBase(name="c", order="asc")
        attr4 = AttributeBase(name="d", order="DESC")

        assert attr1.order == "asc"
        assert attr2.order == "desc"
        assert attr3.order == "asc"
        assert attr4.order == "desc"

    def test_invalid_order_raises(self):
        """Invalid order value should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            AttributeBase(name="test", order="invalid")
        assert "Unknown ordering" in str(exc_info.value)
        assert "test" in str(exc_info.value)

    def test_ref_is_name(self):
        """ref should default to name."""
        attr = AttributeBase(name="my_attribute")
        assert attr.ref == "my_attribute"

    def test_info_default_is_empty_dict(self):
        """info should default to empty dict."""
        attr1 = AttributeBase(name="a")
        attr2 = AttributeBase(name="b")
        assert attr1.info == {}
        assert attr2.info == {}
        # Modifying one should not affect the other
        attr1.info["key"] = "value"
        assert attr2.info == {}


class TestAttributeBaseEquality:
    """Test AttributeBase equality and hashing."""

    def test_equal_same_name(self):
        """Attributes with same name are equal."""
        attr1 = AttributeBase(name="amount")
        attr2 = AttributeBase(name="amount")
        assert attr1 == attr2

    def test_not_equal_different_name(self):
        """Attributes with different names are not equal."""
        attr1 = AttributeBase(name="amount")
        attr2 = AttributeBase(name="total")
        assert attr1 != attr2

    def test_not_equal_different_label(self):
        """Attributes with different labels are not equal."""
        attr1 = AttributeBase(name="amount", label="Amount")
        attr2 = AttributeBase(name="amount", label="Total")
        assert attr1 != attr2

    def test_not_equal_different_description(self):
        """Attributes with different descriptions are not equal."""
        attr1 = AttributeBase(name="amount", description="First")
        attr2 = AttributeBase(name="amount", description="Second")
        assert attr1 != attr2

    def test_not_equal_different_info(self):
        """Attributes with different info are not equal."""
        attr1 = AttributeBase(name="amount", info={"a": 1})
        attr2 = AttributeBase(name="amount", info={"b": 2})
        assert attr1 != attr2

    def test_not_equal_different_format(self):
        """Attributes with different format are not equal."""
        attr1 = AttributeBase(name="amount", format="0.00")
        attr2 = AttributeBase(name="amount", format="0.0")
        assert attr1 != attr2

    def test_not_equal_different_expression(self):
        """Attributes with different expressions are not equal."""
        attr1 = AttributeBase(name="amount", expression="a + b")
        attr2 = AttributeBase(name="amount", expression="a * b")
        assert attr1 != attr2

    def test_not_equal_different_missing_value(self):
        """Attributes with different missing_value are not equal."""
        attr1 = AttributeBase(name="amount", missing_value=0)
        attr2 = AttributeBase(name="amount", missing_value=None)
        assert attr1 != attr2

    def test_not_equal_to_non_attribute(self):
        """Attribute is not equal to non-Attribute objects."""
        attr = AttributeBase(name="amount")
        assert attr != "amount"
        assert attr != 123
        assert attr != {"name": "amount"}

    def test_hash_based_on_ref(self):
        """Hash should be based on ref."""
        attr1 = AttributeBase(name="amount")
        attr2 = AttributeBase(name="amount")
        assert hash(attr1) == hash(attr2)

    def test_can_use_in_set(self):
        """Attributes can be used in sets."""
        attr1 = AttributeBase(name="amount")
        attr2 = AttributeBase(name="amount")
        attr3 = AttributeBase(name="total")

        attr_set = {attr1, attr2, attr3}
        assert len(attr_set) == 2

    def test_can_use_as_dict_key(self):
        """Attributes can be used as dict keys."""
        attr1 = AttributeBase(name="amount")
        attr2 = AttributeBase(name="total")

        d = {attr1: "value1", attr2: "value2"}
        assert d[attr1] == "value1"
        assert d[attr2] == "value2"


class TestAttributeBaseIsBase:
    """Test the is_base property."""

    def test_no_expression_is_base(self):
        """Attribute without expression is base."""
        attr = AttributeBase(name="amount")
        assert attr.is_base is True

    def test_none_expression_is_base(self):
        """Attribute with None expression is base."""
        attr = AttributeBase(name="amount", expression=None)
        assert attr.is_base is True

    def test_empty_string_expression_is_base(self):
        """Attribute with empty string expression is base."""
        attr = AttributeBase(name="amount", expression="")
        assert attr.is_base is True

    def test_with_expression_is_not_base(self):
        """Attribute with expression is not base."""
        attr = AttributeBase(name="amount", expression="price * quantity")
        assert attr.is_base is False


class TestAttributeBaseDependencies:
    """Test the dependencies property."""

    def test_base_attribute_no_dependencies(self):
        """Base attribute has no dependencies."""
        attr = AttributeBase(name="amount")
        assert attr.dependencies == set()

    def test_expression_dependencies(self):
        """Attribute with expression has dependencies."""
        attr = AttributeBase(name="total", expression="price * quantity")
        assert attr.dependencies == {"price", "quantity"}

    def test_complex_expression_dependencies(self):
        """Handle complex expression dependencies."""
        attr = AttributeBase(
            name="margin", expression="(revenue - cost) / revenue * 100"
        )
        assert attr.dependencies == {"revenue", "cost"}


class TestAttributeBaseStringRepresentation:
    """Test string representation methods."""

    def test_str_returns_ref(self):
        """str() should return ref."""
        attr = AttributeBase(name="amount")
        assert str(attr) == "amount"

    def test_repr_contains_class_name(self):
        """repr() should contain class name."""
        attr = AttributeBase(name="amount")
        repr_str = repr(attr)
        assert "AttributeBase" in repr_str


class TestAttributeBaseToDict:
    """Test the to_dict method."""

    def test_minimal_to_dict(self):
        """Convert minimal attribute to dict."""
        attr = AttributeBase(name="amount")
        result = attr.to_dict()

        assert result["name"] == "amount"
        assert result["ref"] == "amount"
        assert result["label"] is None
        assert result["description"] is None
        assert result["order"] is None
        assert result["info"] == {}
        assert result["format"] is None
        assert result["missing_value"] is None
        assert result["expression"] is None

    def test_full_to_dict(self):
        """Convert full attribute to dict."""
        attr = AttributeBase(
            name="amount",
            label="Total Amount",
            description="The total",
            order="desc",
            info={"key": "value"},
            format="#,##0.00",
            missing_value=0,
            expression="price * quantity",
        )
        result = attr.to_dict()

        assert result["name"] == "amount"
        assert result["ref"] == "amount"
        assert result["label"] == "Total Amount"
        assert result["description"] == "The total"
        assert result["order"] == "desc"
        assert result["info"] == {"key": "value"}
        assert result["format"] == "#,##0.00"
        assert result["missing_value"] == 0
        assert result["expression"] == "price * quantity"

    def test_to_dict_defensive_copy(self):
        """to_dict should return defensive copy of info."""
        attr = AttributeBase(name="amount", info={"key": "value"})
        result = attr.to_dict()

        # Modify the result
        result["info"]["new_key"] = "new_value"

        # Original should be unchanged
        assert attr.info == {"key": "value"}

    def test_to_dict_with_create_label(self):
        """to_dict with create_label=True should generate label."""
        attr = AttributeBase(name="total_amount")
        result = attr.to_dict(create_label=True)

        assert result["label"] == "Total amount"

    def test_to_dict_without_create_label(self):
        """to_dict without create_label should not generate label."""
        attr = AttributeBase(name="total_amount")
        result = attr.to_dict()

        assert result["label"] is None


class TestAttributeBaseEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_unicode_name(self):
        """Handle unicode in attribute name."""
        attr = AttributeBase(name="měsíc")
        assert attr.name == "měsíc"
        assert attr.ref == "měsíc"

    def test_special_chars_in_expression(self):
        """Handle special characters in expression."""
        attr = AttributeBase(name="calc", expression="a**2 + b**0.5")
        assert attr.dependencies == {"a", "b"}

    def test_missing_value_various_types(self):
        """missing_value can be various types."""
        attr1 = AttributeBase(name="a", missing_value=0)
        attr2 = AttributeBase(name="b", missing_value="N/A")
        attr3 = AttributeBase(name="c", missing_value=None)
        attr4 = AttributeBase(name="d", missing_value=[1, 2, 3])

        assert attr1.missing_value == 0
        assert attr2.missing_value == "N/A"
        assert attr3.missing_value is None
        assert attr4.missing_value == [1, 2, 3]

    def test_order_case_variations(self):
        """Handle various case combinations in order."""
        attr1 = AttributeBase(name="a", order="ASC")
        attr2 = AttributeBase(name="b", order="asc")
        attr3 = AttributeBase(name="c", order="Asc")
        attr4 = AttributeBase(name="d", order="DESC")
        attr5 = AttributeBase(name="e", order="desc")
        attr6 = AttributeBase(name="f", order="Desc")

        assert attr1.order == "asc"
        assert attr2.order == "asc"
        assert attr3.order == "asc"
        assert attr4.order == "desc"
        assert attr5.order == "desc"
        assert attr6.order == "desc"

    def test_expression_with_only_keywords(self):
        """Expression with only Python keywords has no dependencies."""
        attr = AttributeBase(name="flag", expression="True and False")
        assert attr.dependencies == set()
