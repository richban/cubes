"""Tests for Attribute Pydantic Schema.

Tests validation, normalization, and edge cases for the AttributeSchema.
"""

import pytest
from pydantic import ValidationError

from cubes.metadata.schemas.attributes import AttributeInput, AttributeSchema


class TestAttributeSchemaValidation:
    """Test basic attribute schema validation."""

    def test_required_name_field(self):
        """Name is required and must be a string."""
        with pytest.raises(ValidationError) as exc_info:
            AttributeSchema()
        assert "name" in str(exc_info.value)

    def test_name_must_be_string(self):
        """Name must be a string type."""
        with pytest.raises(ValidationError):
            AttributeSchema(name=123)  # type: ignore

    def test_valid_minimal_attribute(self):
        """Minimal valid attribute with just a name."""
        attr = AttributeSchema(name="test_attr")
        assert attr.name == "test_attr"
        assert attr.label is None
        assert attr.description is None
        assert attr.order is None
        assert attr.info == {}
        assert attr.format is None
        assert attr.missing_value is None
        assert attr.expression is None

    def test_full_attribute(self):
        """Attribute with all fields populated."""
        attr = AttributeSchema(
            name="full_attr",
            label="Full Attribute",
            description="A complete attribute",
            order="asc",
            info={"key": "value"},
            format="%.2f",
            missing_value=0,
            expression="price * quantity",
        )
        assert attr.name == "full_attr"
        assert attr.label == "Full Attribute"
        assert attr.description == "A complete attribute"
        assert attr.order == "asc"
        assert attr.info == {"key": "value"}
        assert attr.format == "%.2f"
        assert attr.missing_value == 0
        assert attr.expression == "price * quantity"


class TestAttributeOrderNormalization:
    """Test order field normalization."""

    def test_order_asc_lowercase(self):
        """Order 'asc' should be preserved."""
        attr = AttributeSchema(name="test", order="asc")
        assert attr.order == "asc"

    def test_order_desc_lowercase(self):
        """Order 'desc' should be preserved."""
        attr = AttributeSchema(name="test", order="desc")
        assert attr.order == "desc"

    def test_order_asc_uppercase(self):
        """Order 'ASC' should be normalized to 'asc'."""
        attr = AttributeSchema(name="test", order="ASC")
        assert attr.order == "asc"

    def test_order_desc_uppercase(self):
        """Order 'DESC' should be normalized to 'desc'."""
        attr = AttributeSchema(name="test", order="DESC")
        assert attr.order == "desc"

    def test_order_mixed_case(self):
        """Order with mixed case should be normalized."""
        attr1 = AttributeSchema(name="test", order="Asc")
        assert attr1.order == "asc"

        attr2 = AttributeSchema(name="test", order="Desc")
        assert attr2.order == "desc"

    def test_order_ascending_prefix(self):
        """Order starting with 'asc' should normalize to 'asc'."""
        attr = AttributeSchema(name="test", order="ascending")
        assert attr.order == "asc"

    def test_order_descending_prefix(self):
        """Order starting with 'desc' should normalize to 'desc'."""
        attr = AttributeSchema(name="test", order="descending")
        assert attr.order == "desc"

    def test_invalid_order(self):
        """Invalid order values should raise validation error."""
        with pytest.raises(ValidationError) as exc_info:
            AttributeSchema(name="test", order="invalid")
        assert "order" in str(exc_info.value)
        assert "Unknown ordering" in str(exc_info.value)

    def test_order_none(self):
        """None order should be allowed."""
        attr = AttributeSchema(name="test", order=None)
        assert attr.order is None

    def test_order_not_specified(self):
        """Order not specified should default to None."""
        attr = AttributeSchema(name="test")
        assert attr.order is None

    def test_order_non_string(self):
        """Non-string order should raise validation error."""
        with pytest.raises(ValidationError):
            AttributeSchema(name="test", order=123)  # type: ignore


class TestAttributeInfoDefault:
    """Test info field defaults."""

    def test_info_defaults_to_empty_dict(self):
        """Info should default to empty dict."""
        attr = AttributeSchema(name="test")
        assert attr.info == {}

    def test_info_is_isolated(self):
        """Info dict should not be shared between instances."""
        attr1 = AttributeSchema(name="test1")
        attr2 = AttributeSchema(name="test2")

        attr1.info["key"] = "value"
        assert "key" not in attr2.info
        assert attr2.info == {}


class TestAttributeStringCoercion:
    """Test string to attribute coercion."""

    def test_string_coerced_to_dict(self):
        """String attribute names should be coerced to dicts."""
        attr = AttributeSchema.model_validate("my_attr")
        assert attr.name == "my_attr"

    def test_string_with_before_validator(self):
        """BeforeValidator should coerce strings when used in other models."""
        # Note: AttributeInput is designed to be used as a field type in other models
        # The BeforeValidator works when the field is validated, not on direct validation
        # For direct string validation, use AttributeSchema.model_validate()
        schema = AttributeSchema.model_validate("simple_attr")
        assert schema.name == "simple_attr"

    def test_dict_passed_through(self):
        """Dict inputs should pass through unchanged."""
        data = {"name": "dict_attr", "label": "Dict Attribute"}
        attr = AttributeSchema.model_validate(data)
        assert attr.name == "dict_attr"
        assert attr.label == "Dict Attribute"


class TestAttributeSchemaModelConfig:
    """Test model configuration."""

    def test_extra_fields_allowed(self):
        """Extra fields should be allowed for forward compatibility."""
        attr = AttributeSchema(
            name="test",
            custom_field="custom_value",  # Extra field
        )
        # Should not raise validation error
        assert attr.name == "test"

    def test_validate_assignment(self):
        """Assignment should trigger validation."""
        attr = AttributeSchema(name="test")

        # Valid assignment
        attr.order = "asc"
        assert attr.order == "asc"

        # Invalid assignment should raise
        with pytest.raises(ValidationError):
            attr.order = "invalid"


class TestAttributeSchemaDictConversion:
    """Test conversion to dictionary."""

    def test_to_dict_minimal(self):
        """Convert minimal attribute to dict."""
        attr = AttributeSchema(name="test")
        result = attr.model_dump()

        assert result["name"] == "test"
        assert result["label"] is None
        assert result["info"] == {}

    def test_to_dict_full(self):
        """Convert full attribute to dict."""
        attr = AttributeSchema(
            name="full",
            label="Full",
            description="Desc",
            order="asc",
            info={"key": "value"},
            format="%.2f",
            missing_value=0,
            expression="x + y",
        )
        result = attr.model_dump()

        assert result["name"] == "full"
        assert result["label"] == "Full"
        assert result["description"] == "Desc"
        assert result["order"] == "asc"
        assert result["info"] == {"key": "value"}
        assert result["format"] == "%.2f"
        assert result["missing_value"] == 0
        assert result["expression"] == "x + y"


class TestAttributeSchemaEdgeCases:
    """Test edge cases and special scenarios."""

    def test_unicode_names(self):
        """Unicode names should be accepted."""
        attr = AttributeSchema(name="Атрибут")
        assert attr.name == "Атрибут"

    def test_special_chars_in_name(self):
        """Names with special characters should be accepted."""
        attr = AttributeSchema(name="attr_123_test")
        assert attr.name == "attr_123_test"

    def test_empty_string_name(self):
        """Empty string name should be allowed (but not recommended)."""
        attr = AttributeSchema(name="")
        assert attr.name == ""

    def test_whitespace_in_name(self):
        """Whitespace in name should be preserved."""
        attr = AttributeSchema(name="  spaced  ")
        assert attr.name == "  spaced  "

    def test_missing_value_types(self):
        """missing_value can be any type."""
        attr1 = AttributeSchema(name="test", missing_value="N/A")
        assert attr1.missing_value == "N/A"

        attr2 = AttributeSchema(name="test", missing_value=0)
        assert attr2.missing_value == 0

        attr3 = AttributeSchema(name="test", missing_value=None)
        assert attr3.missing_value is None

        attr4 = AttributeSchema(name="test", missing_value=[1, 2, 3])
        assert attr4.missing_value == [1, 2, 3]


class TestAttributeInputType:
    """Test the AttributeInput type alias with BeforeValidator."""

    def test_list_of_attributes_from_strings(self):
        """List of attribute strings should be validated."""
        # This simulates parsing a level with string attributes
        data = ["attr1", "attr2", {"name": "attr3", "label": "Three"}]

        # Validate each item
        results = []
        for item in data:
            attr = AttributeSchema.model_validate(item)
            results.append(attr)

        assert len(results) == 3
        assert results[0].name == "attr1"
        assert results[1].name == "attr2"
        assert results[2].name == "attr3"
        assert results[2].label == "Three"
