"""Tests for Level Pydantic Schema.

Tests validation, default attribute resolution, string coercion,
and error handling for LevelSchema.
"""

import pytest
from pydantic import ValidationError

from cubes.metadata.schemas.dimension import LevelSchema


class TestLevelSchemaBasics:
    """Test basic LevelSchema instantiation and properties."""

    def test_basic_level_with_single_attribute(self):
        """Level with single attribute uses it for key, label, and order."""
        level = LevelSchema.model_validate(
            {"name": "year", "attributes": [{"name": "year"}]}
        )

        assert level.name == "year"
        assert len(level.attributes) == 1
        assert level.key == "year"
        assert level.label_attribute == "year"
        assert level.order_attribute == "year"

    def test_basic_level_with_multiple_attributes(self):
        """Level with multiple attributes uses defaults correctly."""
        level = LevelSchema.model_validate(
            {
                "name": "customer",
                "attributes": [{"name": "customer_id"}, {"name": "customer_name"}],
            }
        )

        assert level.name == "customer"
        assert len(level.attributes) == 2
        # key defaults to first attribute
        assert level.key == "customer_id"
        # label_attribute defaults to second attribute
        assert level.label_attribute == "customer_name"
        # order_attribute defaults to first attribute
        assert level.order_attribute == "customer_id"

    def test_level_with_explicit_key(self):
        """Level with explicit key attribute."""
        level = LevelSchema.model_validate(
            {
                "name": "product",
                "attributes": [{"name": "info"}, {"name": "code"}, {"name": "name"}],
                "key": "code",
            }
        )

        assert level.key == "code"
        # label_attribute still defaults to second
        assert level.label_attribute == "code"
        assert level.order_attribute == "info"

    def test_level_with_explicit_label_attribute(self):
        """Level with explicit label_attribute."""
        level = LevelSchema.model_validate(
            {
                "name": "product",
                "attributes": [{"name": "code"}, {"name": "name"}],
                "label_attribute": "name",
            }
        )

        assert level.key == "code"  # defaults to first
        assert level.label_attribute == "name"  # explicit
        assert level.order_attribute == "code"  # defaults to first

    def test_level_with_explicit_order_attribute(self):
        """Level with explicit order_attribute."""
        level = LevelSchema.model_validate(
            {
                "name": "product",
                "attributes": [
                    {"name": "code"},
                    {"name": "name"},
                    {"name": "sort_order"},
                ],
                "order_attribute": "sort_order",
            }
        )

        assert level.key == "code"
        assert level.label_attribute == "name"
        assert level.order_attribute == "sort_order"

    def test_level_with_all_attributes_explicit(self):
        """Level with all special attributes explicitly set."""
        level = LevelSchema.model_validate(
            {
                "name": "product",
                "attributes": [
                    {"name": "info"},
                    {"name": "code"},
                    {"name": "name"},
                    {"name": "sort_key"},
                ],
                "key": "code",
                "label_attribute": "name",
                "order_attribute": "sort_key",
            }
        )

        assert level.key == "code"
        assert level.label_attribute == "name"
        assert level.order_attribute == "sort_key"


class TestLevelSchemaStringCoercion:
    """Test string coercion for level names."""

    def test_string_level_name(self):
        """String input creates level with single attribute."""
        level = LevelSchema.model_validate("year")

        assert level.name == "year"
        assert len(level.attributes) == 1
        assert level.attributes[0].name == "year"
        assert level.key == "year"
        assert level.label_attribute == "year"
        assert level.order_attribute == "year"

    def test_string_level_name_with_spaces(self):
        """String input with spaces works correctly."""
        level = LevelSchema.model_validate("customer_name")

        assert level.name == "customer_name"
        assert len(level.attributes) == 1
        assert level.key == "customer_name"


class TestLevelSchemaAttributeCoercion:
    """Test string coercion within attribute lists."""

    def test_string_attributes(self):
        """Attribute list can contain strings."""
        level = LevelSchema.model_validate(
            {"name": "customer", "attributes": ["customer_id", "customer_name"]}
        )

        assert len(level.attributes) == 2
        assert level.attributes[0].name == "customer_id"
        assert level.attributes[1].name == "customer_name"

    def test_mixed_string_and_dict_attributes(self):
        """Attribute list can mix strings and dicts."""
        level = LevelSchema.model_validate(
            {
                "name": "product",
                "attributes": ["code", {"name": "name", "label": "Product Name"}],
            }
        )

        assert len(level.attributes) == 2
        assert level.attributes[0].name == "code"
        assert level.attributes[1].name == "name"
        assert level.attributes[1].label == "Product Name"


class TestLevelSchemaOptionalFields:
    """Test optional fields like label, description, info, etc."""

    def test_level_with_label(self):
        """Level with explicit label."""
        level = LevelSchema.model_validate(
            {"name": "year", "attributes": [{"name": "year"}], "label": "Calendar Year"}
        )

        assert level.label == "Calendar Year"

    def test_level_with_description(self):
        """Level with description."""
        level = LevelSchema.model_validate(
            {
                "name": "year",
                "attributes": [{"name": "year"}],
                "description": "The calendar year dimension level",
            }
        )

        assert level.description == "The calendar year dimension level"

    def test_level_with_info(self):
        """Level with info dictionary."""
        level = LevelSchema.model_validate(
            {
                "name": "year",
                "attributes": [{"name": "year"}],
                "info": {"source": "database", "table": "dim_date"},
            }
        )

        assert level.info == {"source": "database", "table": "dim_date"}

    def test_level_with_cardinality(self):
        """Level with cardinality."""
        level = LevelSchema.model_validate(
            {"name": "year", "attributes": [{"name": "year"}], "cardinality": "low"}
        )

        assert level.cardinality == "low"

    def test_level_with_role(self):
        """Level with role."""
        level = LevelSchema.model_validate(
            {"name": "year", "attributes": [{"name": "year"}], "role": "time"}
        )

        assert level.role == "time"


class TestLevelSchemaOrderNormalization:
    """Test order field normalization."""

    def test_order_asc_lowercase(self):
        """Order 'asc' is preserved."""
        level = LevelSchema.model_validate(
            {"name": "year", "attributes": [{"name": "year"}], "order": "asc"}
        )

        assert level.order == "asc"

    def test_order_desc_lowercase(self):
        """Order 'desc' is preserved."""
        level = LevelSchema.model_validate(
            {"name": "year", "attributes": [{"name": "year"}], "order": "desc"}
        )

        assert level.order == "desc"

    def test_order_asc_uppercase(self):
        """Order 'ASC' is normalized to lowercase."""
        level = LevelSchema.model_validate(
            {"name": "year", "attributes": [{"name": "year"}], "order": "ASC"}
        )

        assert level.order == "asc"

    def test_order_desc_uppercase(self):
        """Order 'DESC' is normalized to lowercase."""
        level = LevelSchema.model_validate(
            {"name": "year", "attributes": [{"name": "year"}], "order": "DESC"}
        )

        assert level.order == "desc"

    def test_order_ascending_prefix(self):
        """Order 'ascending' is normalized to 'asc'."""
        level = LevelSchema.model_validate(
            {"name": "year", "attributes": [{"name": "year"}], "order": "ascending"}
        )

        assert level.order == "asc"

    def test_order_descending_prefix(self):
        """Order 'descending' is normalized to 'desc'."""
        level = LevelSchema.model_validate(
            {"name": "year", "attributes": [{"name": "year"}], "order": "descending"}
        )

        assert level.order == "desc"

    def test_order_invalid_value(self):
        """Invalid order value raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            LevelSchema.model_validate(
                {"name": "year", "attributes": [{"name": "year"}], "order": "invalid"}
            )

        assert "unknown ordering" in str(exc_info.value).lower()


class TestLevelSchemaNonadditiveNormalization:
    """Test nonadditive field normalization."""

    def test_nonadditive_none(self):
        """Nonadditive 'none' becomes None."""
        level = LevelSchema.model_validate(
            {"name": "year", "attributes": [{"name": "year"}], "nonadditive": "none"}
        )

        assert level.nonadditive is None

    def test_nonadditive_all(self):
        """Nonadditive 'all' is preserved."""
        level = LevelSchema.model_validate(
            {"name": "year", "attributes": [{"name": "year"}], "nonadditive": "all"}
        )

        assert level.nonadditive == "all"

    def test_nonadditive_any_normalized_to_all(self):
        """Nonadditive 'any' is normalized to 'all'."""
        level = LevelSchema.model_validate(
            {"name": "year", "attributes": [{"name": "year"}], "nonadditive": "any"}
        )

        assert level.nonadditive == "all"

    def test_nonadditive_time(self):
        """Nonadditive 'time' is preserved."""
        level = LevelSchema.model_validate(
            {"name": "year", "attributes": [{"name": "year"}], "nonadditive": "time"}
        )

        assert level.nonadditive == "time"

    def test_nonadditive_uppercase_normalized(self):
        """Nonadditive 'NONE' is normalized."""
        level = LevelSchema.model_validate(
            {"name": "year", "attributes": [{"name": "year"}], "nonadditive": "NONE"}
        )

        assert level.nonadditive is None

    def test_nonadditive_invalid_value(self):
        """Invalid nonadditive value raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            LevelSchema.model_validate(
                {
                    "name": "year",
                    "attributes": [{"name": "year"}],
                    "nonadditive": "invalid",
                }
            )

        assert "unknown non-additive" in str(exc_info.value).lower()


class TestLevelSchemaValidationErrors:
    """Test validation error cases."""

    def test_empty_attributes_list(self):
        """Empty attributes list raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            LevelSchema.model_validate({"name": "year", "attributes": []})

        assert "must have at least one attribute" in str(exc_info.value).lower()

    def test_missing_name(self):
        """Missing name raises validation error."""
        with pytest.raises(ValidationError):
            LevelSchema.model_validate({"attributes": [{"name": "year"}]})

    def test_missing_attributes(self):
        """Missing attributes raises validation error."""
        with pytest.raises(ValidationError):
            LevelSchema.model_validate({"name": "year"})

    def test_invalid_key_reference(self):
        """Key referencing non-existent attribute raises error."""
        with pytest.raises(ValidationError) as exc_info:
            LevelSchema.model_validate(
                {
                    "name": "product",
                    "attributes": [{"name": "code"}],
                    "key": "nonexistent",
                }
            )

        assert "does not reference an existing attribute" in str(exc_info.value)

    def test_invalid_label_attribute_reference(self):
        """Label_attribute referencing non-existent attribute raises error."""
        with pytest.raises(ValidationError) as exc_info:
            LevelSchema.model_validate(
                {
                    "name": "product",
                    "attributes": [{"name": "code"}],
                    "label_attribute": "nonexistent",
                }
            )

        assert "does not reference an existing attribute" in str(exc_info.value)

    def test_invalid_order_attribute_reference(self):
        """Order_attribute referencing non-existent attribute raises error."""
        with pytest.raises(ValidationError) as exc_info:
            LevelSchema.model_validate(
                {
                    "name": "product",
                    "attributes": [{"name": "code"}],
                    "order_attribute": "nonexistent",
                }
            )

        assert "does not reference an existing attribute" in str(exc_info.value)


class TestLevelSchemaAttributeInheritance:
    """Test that attribute properties are inherited from level."""

    def test_attribute_inherits_label_from_level(self):
        """When level has label and attribute doesn't, attribute inherits label."""
        level = LevelSchema.model_validate(
            {
                "name": "product_type",
                "label": "Product Type",
                "attributes": ["product_type"],  # String attribute
            }
        )

        assert level.label == "Product Type"
        assert level.attributes[0].label is None  # String attribute doesn't inherit

    def test_attribute_with_explicit_label(self):
        """Attribute with explicit label preserves it."""
        level = LevelSchema.model_validate(
            {
                "name": "product",
                "label": "Product",
                "attributes": [{"name": "code", "label": "Product Code"}],
            }
        )

        assert level.attributes[0].label == "Product Code"


class TestLevelSchemaExtraFields:
    """Test that extra fields are allowed for forward compatibility."""

    def test_extra_fields_allowed(self):
        """Extra fields are preserved for forward compatibility."""
        level = LevelSchema.model_validate(
            {
                "name": "year",
                "attributes": [{"name": "year"}],
                "custom_field": "custom_value",
                "another_field": 123,
            }
        )

        # Extra fields should be accessible via model_extra
        assert level.model_extra is not None
        assert level.model_extra.get("custom_field") == "custom_value"
        assert level.model_extra.get("another_field") == 123


class TestLevelSchemaComplexScenarios:
    """Test complex real-world scenarios."""

    def test_time_dimension_level(self):
        """Time dimension level with all fields."""
        level = LevelSchema.model_validate(
            {
                "name": "month",
                "attributes": [
                    {"name": "month", "label": "Month Number"},
                    {"name": "month_name", "label": "Month Name"},
                    {"name": "month_sname", "label": "Short Name"},
                ],
                "label": "Month",
                "description": "Calendar month level",
                "role": "time",
                "order": "asc",
                "cardinality": "low",
                "key": "month",
                "label_attribute": "month_name",
            }
        )

        assert level.name == "month"
        assert len(level.attributes) == 3
        assert level.key == "month"
        assert level.label_attribute == "month_name"
        assert level.order_attribute == "month"  # defaults to first
        assert level.role == "time"
        assert level.order == "asc"
        assert level.cardinality == "low"

    def test_flat_dimension_level(self):
        """Flat dimension with single attribute."""
        level = LevelSchema.model_validate(
            {"name": "status", "attributes": [{"name": "status"}], "label": "Status"}
        )

        assert level.name == "status"
        assert len(level.attributes) == 1
        # All special attributes point to the only attribute
        assert level.key == "status"
        assert level.label_attribute == "status"
        assert level.order_attribute == "status"

    def test_hierarchical_level_with_many_attributes(self):
        """Level with many attributes."""
        level = LevelSchema.model_validate(
            {
                "name": "customer",
                "attributes": [
                    "customer_id",
                    {"name": "first_name", "label": "First Name"},
                    {"name": "last_name", "label": "Last Name"},
                    {"name": "email", "label": "Email Address"},
                    {"name": "phone", "label": "Phone Number"},
                ],
                "key": "customer_id",
                "label_attribute": "last_name",
                "order_attribute": "customer_id",
            }
        )

        assert level.name == "customer"
        assert len(level.attributes) == 5
        assert level.key == "customer_id"
        assert level.label_attribute == "last_name"
        assert level.order_attribute == "customer_id"


class TestLevelSchemaSerialization:
    """Test serialization of LevelSchema."""

    def test_model_dump_basic(self):
        """Basic model_dump works correctly."""
        level = LevelSchema.model_validate(
            {"name": "year", "attributes": [{"name": "year"}]}
        )

        data = level.model_dump()

        assert data["name"] == "year"
        assert len(data["attributes"]) == 1
        assert data["key"] == "year"
        assert data["label_attribute"] == "year"

    def test_model_dump_with_optional_fields(self):
        """Model dump includes all fields."""
        level = LevelSchema.model_validate(
            {
                "name": "month",
                "attributes": [{"name": "month"}],
                "label": "Month",
                "description": "Calendar month",
                "cardinality": "low",
                "role": "time",
                "order": "asc",
                "nonadditive": "time",
            }
        )

        data = level.model_dump()

        assert data["name"] == "month"
        assert data["label"] == "Month"
        assert data["description"] == "Calendar month"
        assert data["cardinality"] == "low"
        assert data["role"] == "time"
        assert data["order"] == "asc"
        assert data["nonadditive"] == "time"
