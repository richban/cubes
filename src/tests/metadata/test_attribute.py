"""Tests for the Attribute domain dataclass."""

import copy
from typing import Any

import pytest

from cubes.metadata.models.attributes import Attribute, AttributeBase
from cubes.metadata.models.dimension import Dimension, Level


class TestAttributeBasics:
    """Test basic Attribute creation and properties."""

    def test_create_simple_attribute(self) -> None:
        """Attribute creation with just a name."""
        attr = Attribute(name="foo")
        assert attr.name == "foo"
        assert attr.ref == "foo"
        assert str(attr) == "foo"
        assert attr.dimension is None

    def test_create_attribute_with_label(self) -> None:
        """Attribute creation with label."""
        attr = Attribute(name="foo", label="Foo Label")
        assert attr.name == "foo"
        assert attr.label == "Foo Label"
        assert attr.ref == "foo"

    def test_create_attribute_with_description(self) -> None:
        """Attribute creation with description."""
        attr = Attribute(name="foo", description="A test attribute")
        assert attr.description == "A test attribute"

    def test_create_attribute_with_order(self) -> None:
        """Attribute creation with order."""
        attr = Attribute(name="foo", order="asc")
        assert attr.order == "asc"

    def test_create_attribute_with_info(self) -> None:
        """Attribute creation with info dict."""
        info = {"key": "value"}
        attr = Attribute(name="foo", info=info)
        assert attr.info == info

    def test_create_attribute_with_format(self) -> None:
        """Attribute creation with format."""
        attr = Attribute(name="amount", format="%.2f")
        assert attr.format == "%.2f"

    def test_create_attribute_with_missing_value(self) -> None:
        """Attribute creation with missing value."""
        attr = Attribute(name="foo", missing_value=0)
        assert attr.missing_value == 0

    def test_create_attribute_with_expression(self) -> None:
        """Attribute creation with expression."""
        attr = Attribute(name="total", expression="price * quantity")
        assert attr.expression == "price * quantity"
        assert not attr.is_base


class TestAttributeDimensionOwnership:
    """Test attribute dimension ownership and ref updates."""

    def test_attribute_without_dimension(self) -> None:
        """Attribute without dimension has simple ref."""
        attr = Attribute(name="name")
        assert attr.ref == "name"
        assert attr.dimension is None

    def test_attribute_with_dimension(self) -> None:
        """Attribute with dimension has prefixed ref."""
        attr = Attribute(name="name")
        dim = Dimension(name="customer", levels=[Level(name="customer", attributes=[Attribute(name="id")])])
        attr.dimension = dim
        assert attr.ref == "customer.name"
        assert attr.dimension == dim

    def test_attribute_dimension_setter_updates_ref(self) -> None:
        """Setting dimension updates ref to include prefix."""
        attr = Attribute(name="id")
        assert attr.ref == "id"

        dim = Dimension(name="product", levels=[Level(name="product", attributes=[Attribute(name="id")])])
        attr.dimension = dim
        assert attr.ref == "product.id"

    def test_attribute_dimension_removal_clears_ref(self) -> None:
        """Removing dimension reverts ref to simple name."""
        attr = Attribute(name="id")
        dim = Dimension(name="product", levels=[Level(name="product", attributes=[Attribute(name="id")])])
        attr.dimension = dim
        assert attr.ref == "product.id"

        attr.dimension = None
        assert attr.ref == "id"
        assert attr.dimension is None

    def test_attribute_dimension_reassignment(self) -> None:
        """Reassigning dimension updates ref to new dimension."""
        attr = Attribute(name="id")
        dim1 = Dimension(name="product", levels=[Level(name="product", attributes=[Attribute(name="id")])])
        dim2 = Dimension(name="category", levels=[Level(name="category", attributes=[Attribute(name="id")])])

        attr.dimension = dim1
        assert attr.ref == "product.id"

        attr.dimension = dim2
        assert attr.ref == "category.id"


class TestAttributeDeepCopy:
    """Test Attribute deep copy behavior."""

    def test_deepcopy_disowns_dimension(self) -> None:
        """Deep copy disowns attribute from dimension."""
        attr = Attribute(name="id")
        dim = Dimension(name="product", levels=[Level(name="product", attributes=[Attribute(name="id")])])
        attr.dimension = dim
        assert attr.ref == "product.id"

        copied = copy.deepcopy(attr)
        assert copied.dimension is None
        assert copied.ref == "id"
        assert copied.name == "id"

    def test_deepcopy_preserves_all_properties(self) -> None:
        """Deep copy preserves all attribute properties."""
        attr = Attribute(
            name="amount",
            label="Amount",
            description="The amount",
            order="desc",
            info={"unit": "USD"},
            format="%.2f",
            missing_value=0,
            expression="price * quantity",
        )

        copied = copy.deepcopy(attr)

        assert copied.name == "amount"
        assert copied.label == "Amount"
        assert copied.description == "The amount"
        assert copied.order == "desc"
        assert copied.info == {"unit": "USD"}
        assert copied.format == "%.2f"
        assert copied.missing_value == 0
        assert copied.expression == "price * quantity"

    def test_deepcopy_creates_independent_copy(self) -> None:
        """Deep copy creates independent instance."""
        attr = Attribute(name="id", info={"key": "value"})
        copied = copy.deepcopy(attr)

        # Modify original
        attr.info["key"] = "modified"

        # Copy should be unchanged
        assert copied.info["key"] == "value"


class TestAttributeEquality:
    """Test Attribute equality and hashing."""

    def test_equal_attributes(self) -> None:
        """Two attributes with same properties are equal."""
        attr1 = Attribute(name="foo", label="Foo")
        attr2 = Attribute(name="foo", label="Foo")
        assert attr1 == attr2

    def test_unequal_names(self) -> None:
        """Attributes with different names are not equal."""
        attr1 = Attribute(name="foo")
        attr2 = Attribute(name="bar")
        assert attr1 != attr2

    def test_unequal_labels(self) -> None:
        """Attributes with different labels are not equal."""
        attr1 = Attribute(name="foo", label="Foo 1")
        attr2 = Attribute(name="foo", label="Foo 2")
        assert attr1 != attr2

    def test_unequal_descriptions(self) -> None:
        """Attributes with different descriptions are not equal."""
        attr1 = Attribute(name="foo", description="Desc 1")
        attr2 = Attribute(name="foo", description="Desc 2")
        assert attr1 != attr2

    def test_unequal_info(self) -> None:
        """Attributes with different info are not equal."""
        attr1 = Attribute(name="foo", info={"a": 1})
        attr2 = Attribute(name="foo", info={"b": 2})
        assert attr1 != attr2

    def test_unequal_format(self) -> None:
        """Attributes with different format are not equal."""
        attr1 = Attribute(name="foo", format="%.1f")
        attr2 = Attribute(name="foo", format="%.2f")
        assert attr1 != attr2

    def test_unequal_expression(self) -> None:
        """Attributes with different expressions are not equal."""
        attr1 = Attribute(name="foo", expression="a + b")
        attr2 = Attribute(name="foo", expression="a * b")
        assert attr1 != attr2

    def test_unequal_missing_value(self) -> None:
        """Attributes with different missing_value are not equal."""
        attr1 = Attribute(name="foo", missing_value=0)
        attr2 = Attribute(name="foo", missing_value=None)
        assert attr1 != attr2

    def test_not_equal_to_non_attribute(self) -> None:
        """Attribute is not equal to non-Attribute objects."""
        attr = Attribute(name="foo")
        assert attr != "foo"
        assert attr != {"name": "foo"}
        assert attr != 123

    def test_equality_ignores_dimension(self) -> None:
        """Equality comparison ignores dimension ownership."""
        attr1 = Attribute(name="id")
        attr2 = Attribute(name="id")

        dim = Dimension(name="product", levels=[Level(name="product", attributes=[Attribute(name="id")])])
        attr1.dimension = dim

        # Should still be equal despite different dimension ownership
        assert attr1 == attr2

    def test_hash_based_on_ref(self) -> None:
        """Hash is based on ref for use in sets and dicts."""
        attr = Attribute(name="foo")
        assert hash(attr) == hash("foo")

        dim = Dimension(name="bar", levels=[Level(name="bar", attributes=[Attribute(name="id")])])
        attr.dimension = dim
        assert hash(attr) == hash("bar.foo")

    def test_can_use_in_set(self) -> None:
        """Attributes can be used in sets."""
        attr1 = Attribute(name="foo")
        attr2 = Attribute(name="bar")

        attr_set = {attr1, attr2}
        assert len(attr_set) == 2
        assert attr1 in attr_set
        assert attr2 in attr_set

    def test_can_use_as_dict_key(self) -> None:
        """Attributes can be used as dict keys."""
        attr = Attribute(name="foo")
        d = {attr: "value"}
        assert d[attr] == "value"


class TestAttributeInheritance:
    """Test that Attribute properly inherits from AttributeBase."""

    def test_attribute_is_attribute_base(self) -> None:
        """Attribute is an instance of AttributeBase."""
        attr = Attribute(name="foo")
        assert isinstance(attr, AttributeBase)

    def test_attribute_inherits_is_base(self) -> None:
        """Attribute inherits is_base property."""
        attr1 = Attribute(name="base_attr")
        attr2 = Attribute(name="derived", expression="a + b")

        assert attr1.is_base
        assert not attr2.is_base

    def test_attribute_inherits_dependencies(self) -> None:
        """Attribute inherits dependencies property."""
        attr = Attribute(name="total", expression="price * quantity")
        deps = attr.dependencies
        assert "price" in deps
        assert "quantity" in deps

    def test_attribute_inherits_to_dict(self) -> None:
        """Attribute inherits to_dict method."""
        attr = Attribute(name="foo", label="Foo")
        d = attr.to_dict()
        assert d["name"] == "foo"
        assert d["label"] == "Foo"


class TestAttributeStringRepresentation:
    """Test string representation of attributes."""

    def test_str_returns_ref(self) -> None:
        """str(attr) returns attr.ref."""
        attr = Attribute(name="foo")
        assert str(attr) == "foo"

        attr.dimension = Dimension(name="bar", levels=[Level(name="bar", attributes=[Attribute(name="id")])])
        assert str(attr) == "bar.foo"

    def test_repr_contains_class_name(self) -> None:
        """repr contains class name."""
        attr = Attribute(name="foo")
        repr_str = repr(attr)
        assert "Attribute" in repr_str
        assert "foo" in repr_str


class TestAttributeEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_info_dict(self) -> None:
        """Empty info dict is handled correctly."""
        attr = Attribute(name="foo")
        assert attr.info == {}
        assert isinstance(attr.info, dict)

    def test_none_values(self) -> None:
        """None values are handled correctly."""
        attr = Attribute(
            name="foo",
            label=None,
            description=None,
            order=None,
            format=None,
            missing_value=None,
            expression=None,
        )
        assert attr.label is None
        assert attr.description is None
        assert attr.order is None
        assert attr.format is None
        assert attr.missing_value is None
        assert attr.expression is None

    def test_unicode_names(self) -> None:
        """Unicode attribute names are handled."""
        attr = Attribute(name="Ümläuts")
        assert attr.name == "Ümläuts"
        assert attr.ref == "Ümläuts"

    def test_special_characters_in_name(self) -> None:
        """Names with special characters are handled."""
        attr = Attribute(name="test_123")
        assert attr.name == "test_123"

    def test_underscore_name_label_generation(self) -> None:
        """to_dict with create_label converts underscores."""
        attr = Attribute(name="test_name")
        d = attr.to_dict(create_label=True)
        assert d["label"] == "Test name"

    def test_order_normalization(self) -> None:
        """Order is normalized to lowercase."""
        attr = Attribute(name="foo", order="ASC")
        assert attr.order == "asc"

        attr2 = Attribute(name="bar", order="DESC")
        assert attr2.order == "desc"

    def test_invalid_order_raises(self) -> None:
        """Invalid order value raises ValueError."""
        with pytest.raises(ValueError, match="Unknown ordering"):
            Attribute(name="foo", order="invalid")


class TestAttributeWithRealDimension:
    """Test attribute behavior with actual dimension instances."""

    def test_multiple_attributes_same_dimension(self) -> None:
        """Multiple attributes can belong to same dimension."""
        dim = Dimension(name="customer", levels=[Level(name="customer", attributes=[Attribute(name="id")])])

        attr1 = Attribute(name="id")
        attr2 = Attribute(name="name")

        attr1.dimension = dim
        attr2.dimension = dim

        assert attr1.ref == "customer.id"
        assert attr2.ref == "customer.name"
        assert attr1.dimension == attr2.dimension

    def test_attribute_in_different_dimensions(self) -> None:
        """Same attribute can be assigned to different dimensions."""
        attr = Attribute(name="id")

        dim1 = Dimension(name="customer", levels=[Level(name="customer", attributes=[Attribute(name="id")])])
        dim2 = Dimension(name="product", levels=[Level(name="product", attributes=[Attribute(name="id")])])

        attr.dimension = dim1
        assert attr.ref == "customer.id"

        attr.dimension = dim2
        assert attr.ref == "product.id"

    def test_attribute_refs_in_set(self) -> None:
        """Attributes with different refs can coexist in set."""
        dim = Dimension(name="customer", levels=[Level(name="customer", attributes=[Attribute(name="id")])])

        attr1 = Attribute(name="id")
        attr2 = Attribute(name="name")

        attr1.dimension = dim
        attr2.dimension = dim

        attr_set = {attr1, attr2}
        assert len(attr_set) == 2

        # Hash should be different because refs are different
        assert hash(attr1) != hash(attr2)
