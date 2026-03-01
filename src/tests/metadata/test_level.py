"""Tests for Level domain dataclass.

Tests the Level class implementation including:
- Initialization and validation
- Default attribute resolution
- Attribute lookup
- has_details property
- Equality and hashing
- Deep copy behavior
- to_dict serialization
- Integration with LevelSchema
"""

import copy
from typing import Any

import pytest

from cubes.metadata.models import Attribute, Level, ModelError
from cubes.metadata.schemas.dimension import LevelSchema


class TestLevelInitialization:
    """Tests for Level initialization and validation."""

    def test_basic_initialization(self):
        """Test basic level creation with single attribute."""
        attr = Attribute(name="year")
        level = Level(name="year", attributes=[attr])

        assert level.name == "year"
        assert len(level.attributes) == 1
        assert level.attributes[0] == attr

    def test_empty_attributes_raises_error(self):
        """Empty attribute list should raise ModelError."""
        with pytest.raises(ModelError, match="must have at least one attribute"):
            Level(name="empty", attributes=[])

    def test_multiple_attributes(self):
        """Level can have multiple attributes."""
        attrs = [
            Attribute(name="code"),
            Attribute(name="name"),
            Attribute(name="description"),
        ]
        level = Level(name="product", attributes=attrs)

        assert len(level.attributes) == 3
        assert level.attributes[0].name == "code"
        assert level.attributes[1].name == "name"
        assert level.attributes[2].name == "description"

    def test_default_key_attribute(self):
        """Key defaults to first attribute when not specified."""
        attrs = [Attribute(name="code"), Attribute(name="name")]
        level = Level(name="product", attributes=attrs)

        assert level.key.name == "code"
        assert level.key == attrs[0]

    def test_explicit_key_attribute(self):
        """Key can be explicitly set to an attribute."""
        attrs = [Attribute(name="code"), Attribute(name="name")]
        level = Level(name="product", attributes=attrs, key=attrs[1])

        assert level.key.name == "name"
        assert level.key == attrs[1]

    def test_key_by_string_reference(self):
        """Key can be specified as a string attribute name."""
        attrs = [Attribute(name="code"), Attribute(name="name")]
        level = Level(name="product", attributes=attrs, key="name")

        assert level.key.name == "name"

    def test_invalid_key_reference_raises_error(self):
        """Invalid key attribute name should raise ModelError."""
        attrs = [Attribute(name="code")]
        with pytest.raises(ModelError, match="has no attribute 'invalid'"):
            Level(name="product", attributes=attrs, key="invalid")


class TestLevelLabelAttribute:
    """Tests for label_attribute default resolution."""

    def test_default_label_single_attribute(self):
        """With single attribute, label_attribute defaults to key."""
        attrs = [Attribute(name="year")]
        level = Level(name="year", attributes=attrs)

        assert level.label_attribute.name == "year"
        assert level.label_attribute == level.key

    def test_default_label_multiple_attributes(self):
        """With multiple attributes, label_attribute defaults to second."""
        attrs = [Attribute(name="code"), Attribute(name="name")]
        level = Level(name="product", attributes=attrs)

        assert level.label_attribute.name == "name"
        assert level.label_attribute == attrs[1]

    def test_explicit_label_attribute(self):
        """Label attribute can be explicitly set."""
        attrs = [
            Attribute(name="id"),
            Attribute(name="code"),
            Attribute(name="name"),
        ]
        level = Level(name="product", attributes=attrs, label_attribute=attrs[2])

        assert level.label_attribute.name == "name"

    def test_label_by_string_reference(self):
        """Label attribute can be specified as a string attribute name."""
        attrs = [Attribute(name="code"), Attribute(name="name")]
        level = Level(name="product", attributes=attrs, label_attribute="code")

        assert level.label_attribute.name == "code"


class TestLevelOrderAttribute:
    """Tests for order_attribute default resolution."""

    def test_default_order_attribute(self):
        """Order attribute defaults to first attribute."""
        attrs = [Attribute(name="code"), Attribute(name="name")]
        level = Level(name="product", attributes=attrs)

        assert level.order_attribute.name == "code"
        assert level.order_attribute == attrs[0]

    def test_explicit_order_attribute(self):
        """Order attribute can be explicitly set."""
        attrs = [Attribute(name="code"), Attribute(name="name")]
        level = Level(name="product", attributes=attrs, order_attribute=attrs[1])

        assert level.order_attribute.name == "name"

    def test_order_by_string_reference(self):
        """Order attribute can be specified as a string attribute name."""
        attrs = [Attribute(name="code"), Attribute(name="name")]
        level = Level(name="product", attributes=attrs, order_attribute="name")

        assert level.order_attribute.name == "name"


class TestLevelNormalization:
    """Tests for value normalization."""

    def test_order_normalization_lowercase(self):
        """Order values should be normalized to lowercase."""
        attrs = [Attribute(name="code")]
        level = Level(name="test", attributes=attrs, order="ASC")
        assert level.order == "asc"

        level = Level(name="test", attributes=attrs, order="DESC")
        assert level.order == "desc"

    def test_order_normalization_mixed_case(self):
        """Order values with mixed case should be normalized."""
        attrs = [Attribute(name="code")]
        level = Level(name="test", attributes=attrs, order="Asc")
        assert level.order == "asc"

    def test_invalid_order_raises_error(self):
        """Invalid order values should raise ModelError."""
        attrs = [Attribute(name="code")]
        with pytest.raises(ModelError, match="unknown ordering"):
            Level(name="test", attributes=attrs, order="invalid")

    def test_nonadditive_normalization_none(self):
        """Nonadditive 'none' should be normalized to None."""
        attrs = [Attribute(name="code")]
        level = Level(name="test", attributes=attrs, nonadditive="none")
        assert level.nonadditive is None

        level = Level(name="test", attributes=attrs, nonadditive="NONE")
        assert level.nonadditive is None

    def test_nonadditive_normalization_all(self):
        """Nonadditive 'all' or 'any' should be normalized to 'all'."""
        attrs = [Attribute(name="code")]
        level = Level(name="test", attributes=attrs, nonadditive="all")
        assert level.nonadditive == "all"

        level = Level(name="test", attributes=attrs, nonadditive="any")
        assert level.nonadditive == "all"

        level = Level(name="test", attributes=attrs, nonadditive="ALL")
        assert level.nonadditive == "all"

    def test_nonadditive_normalization_time(self):
        """Nonadditive 'time' should be preserved."""
        attrs = [Attribute(name="code")]
        level = Level(name="test", attributes=attrs, nonadditive="time")
        assert level.nonadditive == "time"

        level = Level(name="test", attributes=attrs, nonadditive="TIME")
        assert level.nonadditive == "time"

    def test_invalid_nonadditive_raises_error(self):
        """Invalid nonadditive values should raise ModelError."""
        attrs = [Attribute(name="code")]
        with pytest.raises(ModelError, match="unknown non-additive type"):
            Level(name="test", attributes=attrs, nonadditive="invalid")


class TestLevelHasDetails:
    """Tests for has_details property."""

    def test_single_attribute_no_details(self):
        """Level with single attribute has no details."""
        attrs = [Attribute(name="year")]
        level = Level(name="year", attributes=attrs)

        assert not level.has_details

    def test_multiple_attributes_has_details(self):
        """Level with multiple attributes has details."""
        attrs = [Attribute(name="code"), Attribute(name="name")]
        level = Level(name="product", attributes=attrs)

        assert level.has_details

    def test_three_attributes_has_details(self):
        """Level with three attributes has details."""
        attrs = [
            Attribute(name="id"),
            Attribute(name="code"),
            Attribute(name="name"),
        ]
        level = Level(name="product", attributes=attrs)

        assert level.has_details


class TestLevelAttributeLookup:
    """Tests for attribute lookup method."""

    def test_find_existing_attribute(self):
        """Should return attribute by name."""
        attrs = [Attribute(name="code"), Attribute(name="name")]
        level = Level(name="product", attributes=attrs)

        found = level.attribute("code")
        assert found.name == "code"
        assert found == attrs[0]

    def test_find_nonexistent_attribute_raises(self):
        """Should raise ModelError for non-existent attribute."""
        attrs = [Attribute(name="code")]
        level = Level(name="product", attributes=attrs)

        with pytest.raises(ModelError, match="has no attribute 'missing'"):
            level.attribute("missing")

    def test_attribute_error_includes_available(self):
        """Error message should include available attributes."""
        attrs = [Attribute(name="code"), Attribute(name="name")]
        level = Level(name="product", attributes=attrs)

        with pytest.raises(ModelError) as exc_info:
            level.attribute("missing")

        assert "code" in str(exc_info.value)
        assert "name" in str(exc_info.value)


class TestLevelStringConversion:
    """Tests for string representation."""

    def test_str_returns_name(self):
        """str(level) should return level name."""
        attrs = [Attribute(name="year")]
        level = Level(name="date", attributes=attrs)

        assert str(level) == "date"

    def test_repr_includes_dict(self):
        """repr(level) should include to_dict output."""
        attrs = [Attribute(name="year")]
        level = Level(name="date", attributes=attrs)

        repr_str = repr(level)
        assert repr_str.startswith("Level(")
        assert "date" in repr_str


class TestLevelEquality:
    """Tests for equality comparison."""

    def test_equal_levels(self):
        """Levels with same properties are equal."""
        attrs1 = [Attribute(name="code"), Attribute(name="name")]
        attrs2 = [Attribute(name="code"), Attribute(name="name")]

        level1 = Level(name="product", attributes=attrs1)
        level2 = Level(name="product", attributes=attrs2)

        assert level1 == level2

    def test_unequal_different_names(self):
        """Levels with different names are not equal."""
        attrs = [Attribute(name="code")]
        level1 = Level(name="product1", attributes=attrs)
        level2 = Level(name="product2", attributes=attrs)

        assert level1 != level2

    def test_unequal_different_attributes(self):
        """Levels with different attributes are not equal."""
        attrs1 = [Attribute(name="code")]
        attrs2 = [Attribute(name="id")]

        level1 = Level(name="product", attributes=attrs1)
        level2 = Level(name="product", attributes=attrs2)

        assert level1 != level2

    def test_unequal_different_labels(self):
        """Levels with different labels are not equal."""
        attrs = [Attribute(name="code")]
        level1 = Level(name="product", attributes=attrs, label="Product")
        level2 = Level(name="product", attributes=attrs, label="Item")

        assert level1 != level2

    def test_unequal_different_keys(self):
        """Levels with different keys are not equal."""
        attrs = [Attribute(name="code"), Attribute(name="name")]
        level1 = Level(name="product", attributes=attrs, key="code")
        level2 = Level(name="product", attributes=attrs, key="name")

        assert level1 != level2

    def test_not_equal_to_non_level(self):
        """Level should not equal non-Level objects."""
        attrs = [Attribute(name="code")]
        level = Level(name="product", attributes=attrs)

        assert level != "product"
        assert level != 123
        assert level != None

    def test_hash_based_on_name(self):
        """Hash should be based on level name."""
        attrs = [Attribute(name="code")]
        level = Level(name="product", attributes=attrs)

        assert hash(level) == hash("product")


class TestLevelDeepCopy:
    """Tests for deep copy behavior."""

    def test_deepcopy_creates_new_instance(self):
        """Deep copy should create a new Level instance."""
        attrs = [Attribute(name="code"), Attribute(name="name")]
        level = Level(name="product", attributes=attrs)

        copied = copy.deepcopy(level)

        assert copied is not level
        assert copied.name == level.name

    def test_deepcopy_copies_attributes(self):
        """Deep copy should copy attributes, not share them."""
        attrs = [Attribute(name="code")]
        level = Level(name="product", attributes=attrs)

        copied = copy.deepcopy(level)

        assert copied.attributes[0] is not level.attributes[0]
        assert copied.attributes[0].name == level.attributes[0].name

    def test_deepcopy_preserves_key(self):
        """Deep copy should preserve key attribute reference."""
        attrs = [Attribute(name="code"), Attribute(name="name")]
        level = Level(name="product", attributes=attrs, key="name")

        copied = copy.deepcopy(level)

        assert copied.key.name == "name"

    def test_deepcopy_preserves_info(self):
        """Deep copy should preserve and copy info dict."""
        attrs = [Attribute(name="code")]
        level = Level(
            name="product",
            attributes=attrs,
            info={"custom": "value", "nested": {"key": "val"}},
        )

        copied = copy.deepcopy(level)

        assert copied.info == level.info
        assert copied.info is not level.info


class TestLevelToDict:
    """Tests for to_dict serialization."""

    def test_basic_to_dict(self):
        """Basic to_dict output."""
        attrs = [Attribute(name="code")]
        level = Level(name="product", attributes=attrs)

        result = level.to_dict()

        assert result["name"] == "product"
        assert result["key"] == "code"
        assert result["label_attribute"] == "code"
        assert result["order_attribute"] == "code"
        assert len(result["attributes"]) == 1

    def test_to_dict_with_label_generation(self):
        """to_dict with create_label=True should generate labels."""
        attrs = [Attribute(name="product_code")]
        level = Level(name="product", attributes=attrs)

        result = level.to_dict(create_label=True)

        assert result["label"] == "Product"
        assert result["attributes"][0]["label"] == "Product code"

    def test_to_dict_full_attribute_names(self):
        """to_dict with full_attribute_names=True should use refs."""
        attrs = [Attribute(name="code")]
        level = Level(name="product", attributes=attrs)

        result = level.to_dict(full_attribute_names=True)

        assert result["key"] == "code"  # ref is same as name without dimension

    def test_to_dict_includes_all_fields(self):
        """to_dict should include all level fields."""
        attrs = [Attribute(name="code")]
        level = Level(
            name="product",
            attributes=attrs,
            label="Product Level",
            description="A product level",
            order="asc",
            cardinality="medium",
            role="test_role",
            nonadditive="all",
            info={"custom": "data"},
        )

        result = level.to_dict()

        assert result["name"] == "product"
        assert result["label"] == "Product Level"
        assert result["description"] == "A product level"
        assert result["order"] == "asc"
        assert result["cardinality"] == "medium"
        assert result["role"] == "test_role"
        assert result["nonadditive"] == "all"
        assert result["info"] == {"custom": "data"}

    def test_to_dict_preserves_none_values(self):
        """to_dict should preserve None values appropriately."""
        attrs = [Attribute(name="code")]
        level = Level(name="product", attributes=attrs)

        result = level.to_dict()

        assert result["label"] is None
        assert result["description"] is None
        assert result["order"] is None
        assert result["cardinality"] is None


class TestLevelFromSchema:
    """Tests for Level.from_schema factory method."""

    def test_from_schema_basic(self):
        """Create Level from basic LevelSchema."""
        schema = LevelSchema.model_validate({"name": "year", "attributes": ["year"]})

        level = Level.from_schema(schema)

        assert level.name == "year"
        assert len(level.attributes) == 1
        assert level.attributes[0].name == "year"

    def test_from_schema_with_dict_attributes(self):
        """Create Level from schema with dict attributes."""
        schema = LevelSchema.model_validate(
            {"name": "month", "attributes": [{"name": "month"}, {"name": "month_name"}]}
        )

        level = Level.from_schema(schema)

        assert level.name == "month"
        assert len(level.attributes) == 2
        assert level.attributes[0].name == "month"
        assert level.attributes[1].name == "month_name"

    def test_from_schema_preserves_defaults(self):
        """Level from schema should have correct default attributes."""
        schema = LevelSchema.model_validate(
            {"name": "product", "attributes": ["code", "name"]}
        )

        level = Level.from_schema(schema)

        assert level.key.name == "code"  # First attribute
        assert level.label_attribute.name == "name"  # Second attribute
        assert level.order_attribute.name == "code"  # First attribute

    def test_from_schema_with_explicit_key(self):
        """Level from schema respects explicit key."""
        schema = LevelSchema.model_validate(
            {"name": "product", "attributes": ["id", "code", "name"], "key": "code"}
        )

        level = Level.from_schema(schema)

        assert level.key.name == "code"

    def test_from_schema_with_all_options(self):
        """Level from schema with all options set."""
        schema = LevelSchema.model_validate(
            {
                "name": "date",
                "attributes": ["year", "year_name"],
                "label": "Year Level",
                "description": "A year level",
                "order": "asc",
                "cardinality": "low",
                "role": "time",
                "nonadditive": "time",
                "info": {"custom": "value"},
                "key": "year",
                "label_attribute": "year_name",
                "order_attribute": "year",
            }
        )

        level = Level.from_schema(schema)

        assert level.name == "date"
        assert level.label == "Year Level"
        assert level.description == "A year level"
        assert level.order == "asc"
        assert level.cardinality == "low"
        assert level.role == "time"
        assert level.nonadditive == "time"
        assert level.info == {"custom": "value"}
        assert level.key.name == "year"
        assert level.label_attribute.name == "year_name"
        assert level.order_attribute.name == "year"


class TestLevelIntegration:
    """Integration tests combining multiple features."""

    def test_complex_level_creation(self):
        """Test creating a complex level with all features."""
        attrs = [
            Attribute(name="product_id", label="ID"),
            Attribute(name="product_code", label="Code"),
            Attribute(name="product_name", label="Name"),
        ]

        level = Level(
            name="product",
            attributes=attrs,
            label="Product",
            description="Product hierarchy level",
            key=attrs[1],  # Use code as key
            label_attribute=attrs[2],  # Use name as label
            order_attribute=attrs[1],  # Use code for ordering
            order="asc",
            cardinality="medium",
            role="product",
            nonadditive="all",
            info={"source": "database", "table": "products"},
        )

        # Verify all properties
        assert level.name == "product"
        assert level.label == "Product"
        assert level.description == "Product hierarchy level"
        assert level.key.name == "product_code"
        assert level.label_attribute.name == "product_name"
        assert level.order_attribute.name == "product_code"
        assert level.order == "asc"
        assert level.cardinality == "medium"
        assert level.role == "product"
        assert level.nonadditive == "all"
        assert level.info == {"source": "database", "table": "products"}
        assert level.has_details is True

        # Verify attribute lookup
        assert level.attribute("product_code").label == "Code"

        # Verify serialization
        result = level.to_dict(create_label=True)
        assert result["label"] == "Product"
        assert result["key"] == "product_code"

    def test_level_roundtrip(self):
        """Test that Level can be serialized and reconstructed."""
        original_attrs = [
            Attribute(name="month"),
            Attribute(name="month_name"),
        ]
        original = Level(
            name="month",
            attributes=original_attrs,
            label="Month",
            order="desc",
            cardinality="low",
        )

        # Serialize
        data = original.to_dict()

        # Reconstruct from schema (simulating typical usage)
        schema = LevelSchema.model_validate(data)
        reconstructed = Level.from_schema(schema)

        assert reconstructed.name == original.name
        assert reconstructed.label == original.label
        assert len(reconstructed.attributes) == len(original.attributes)
        assert reconstructed.order == original.order
        assert reconstructed.cardinality == original.cardinality

    def test_level_equality_with_copies(self):
        """Test equality between original and deep copied levels."""
        attrs = [Attribute(name="code"), Attribute(name="name")]
        original = Level(name="product", attributes=attrs)
        copied = copy.deepcopy(original)

        assert original == copied
        assert hash(original) == hash(copied)
