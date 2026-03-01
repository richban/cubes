"""Tests for Dimension Pydantic Schema.

Tests the DimensionSchema class including:
- Basic field validation
- String coercion for simple dimensions
- Levels vs attributes mutual exclusion
- Default hierarchy creation
- Default hierarchy name resolution
- Nonadditive normalization
- Template inheritance field support
- Ported legacy dimension expansion tests
"""

import pytest
from pydantic import ValidationError

from cubes.metadata.schemas.dimension import (
    DimensionSchema,
    HierarchySchema,
    LevelSchema,
)


class TestDimensionSchemaBasics:
    """Test basic DimensionSchema validation."""

    def test_valid_dimension_with_levels(self):
        """Dimension with explicit levels and hierarchies."""
        data = {
            "name": "date",
            "levels": [
                {"name": "year", "attributes": [{"name": "year"}]},
                {"name": "month", "attributes": [{"name": "month"}]},
            ],
            "hierarchies": [{"name": "default", "levels": ["year", "month"]}],
        }
        dim = DimensionSchema.model_validate(data)
        assert dim.name == "date"
        assert len(dim.levels) == 2
        assert len(dim.hierarchies) == 1

    def test_valid_dimension_full_metadata(self):
        """Dimension with all metadata fields."""
        data = {
            "name": "product",
            "levels": [{"name": "category", "attributes": [{"name": "category"}]}],
            "label": "Product Dimension",
            "description": "Product hierarchy",
            "info": {"icon": "box"},
            "role": "product",
            "cardinality": "high",
            "category": "product",
            "nonadditive": "none",
        }
        dim = DimensionSchema.model_validate(data)
        assert dim.name == "product"
        assert dim.label == "Product Dimension"
        assert dim.description == "Product hierarchy"
        assert dim.info == {"icon": "box"}
        assert dim.role == "product"
        assert dim.cardinality == "high"
        assert dim.category == "product"
        assert dim.nonadditive is None  # normalized

    def test_dimension_string_coercion(self):
        """Simple string creates dimension with single level."""
        dim = DimensionSchema.model_validate("simple_dim")
        assert dim.name == "simple_dim"
        assert len(dim.levels) == 1
        assert isinstance(dim.levels[0], LevelSchema)
        assert dim.levels[0].name == "simple_dim"

    def test_dimension_minimal_dict(self):
        """Minimal dict with just name and level strings."""
        data = {
            "name": "region",
            "levels": ["country", "state", "city"],
        }
        dim = DimensionSchema.model_validate(data)
        assert dim.name == "region"
        assert len(dim.levels) == 3
        assert dim.levels[0].name == "country"
        assert dim.levels[1].name == "state"
        assert dim.levels[2].name == "city"


class TestDimensionSchemaValidation:
    """Test DimensionSchema validation rules."""

    def test_mutual_exclusion_levels_and_attributes(self):
        """Cannot have both levels and attributes."""
        data = {
            "name": "invalid",
            "levels": [{"name": "level1", "attributes": [{"name": "attr1"}]}],
            "attributes": [{"name": "attr1"}],
        }
        with pytest.raises(ValidationError) as exc_info:
            DimensionSchema.model_validate(data)
        error_msg = str(exc_info.value)
        assert "cannot have both" in error_msg.lower()
        assert "levels" in error_msg.lower()
        assert "attributes" in error_msg.lower()

    def test_missing_name_raises_error(self):
        """Name is required."""
        data = {
            "levels": [{"name": "level1", "attributes": [{"name": "attr1"}]}],
        }
        with pytest.raises(ValidationError) as exc_info:
            DimensionSchema.model_validate(data)
        assert "name" in str(exc_info.value)

    def test_template_field_supported(self):
        """Template field is accepted for inheritance."""
        data = {
            "name": "derived",
            "template": "base_template",
            "levels": [{"name": "level1", "attributes": [{"name": "attr1"}]}],
        }
        dim = DimensionSchema.model_validate(data)
        assert dim.template == "base_template"


class TestDimensionSchemaDefaultHierarchy:
    """Test default hierarchy creation."""

    def test_default_hierarchy_created_from_levels(self):
        """If no hierarchies, create default from levels."""
        data = {
            "name": "date",
            "levels": [
                {"name": "year", "attributes": [{"name": "year"}]},
                {"name": "month", "attributes": [{"name": "month"}]},
            ],
        }
        dim = DimensionSchema.model_validate(data)
        assert dim.hierarchies is not None
        assert len(dim.hierarchies) == 1
        hier = dim.hierarchies[0]
        assert isinstance(hier, HierarchySchema)
        assert hier.name == "default"
        assert hier.levels == ["year", "month"]

    def test_no_default_hierarchy_without_levels(self):
        """No hierarchies created if no levels."""
        data = {
            "name": "empty",
            "attributes": [{"name": "single_attr"}],
        }
        dim = DimensionSchema.model_validate(data)
        assert dim.hierarchies is None
        assert dim.levels is None

    def test_explicit_hierarchy_not_overridden(self):
        """Explicit hierarchies are preserved."""
        data = {
            "name": "date",
            "levels": [
                {"name": "year", "attributes": [{"name": "year"}]},
                {"name": "quarter", "attributes": [{"name": "quarter"}]},
                {"name": "month", "attributes": [{"name": "month"}]},
            ],
            "hierarchies": [
                {"name": "fiscal", "levels": ["year", "quarter"]},
                {"name": "calendar", "levels": ["year", "month"]},
            ],
        }
        dim = DimensionSchema.model_validate(data)
        assert len(dim.hierarchies) == 2
        assert dim.hierarchies[0].name == "fiscal"
        assert dim.hierarchies[1].name == "calendar"


class TestDimensionSchemaDefaultHierarchyName:
    """Test default hierarchy name resolution."""

    def test_explicit_default_hierarchy_name(self):
        """Explicit default_hierarchy_name is preserved."""
        data = {
            "name": "date",
            "levels": [
                {"name": "year", "attributes": [{"name": "year"}]},
                {"name": "month", "attributes": [{"name": "month"}]},
            ],
            "hierarchies": [
                {"name": "fiscal", "levels": ["year", "month"]},
                {"name": "calendar", "levels": ["year", "month"]},
            ],
            "default_hierarchy_name": "fiscal",
        }
        dim = DimensionSchema.model_validate(data)
        assert dim.default_hierarchy_name == "fiscal"

    def test_default_hierarchy_name_from_default(self):
        """Use 'default' if hierarchy named default exists."""
        data = {
            "name": "date",
            "levels": [{"name": "year", "attributes": [{"name": "year"}]}],
        }
        dim = DimensionSchema.model_validate(data)
        assert dim.default_hierarchy_name == "default"

    def test_default_hierarchy_name_from_first(self):
        """Use first hierarchy name if no 'default'."""
        data = {
            "name": "date",
            "levels": [{"name": "year", "attributes": [{"name": "year"}]}],
            "hierarchies": [
                {"name": "custom", "levels": ["year"]},
            ],
        }
        dim = DimensionSchema.model_validate(data)
        assert dim.default_hierarchy_name == "custom"


class TestDimensionSchemaNonadditive:
    """Test nonadditive normalization."""

    def test_nonadditive_none_normalized(self):
        """'none' is normalized to None."""
        data = {
            "name": "test",
            "levels": [{"name": "level1", "attributes": [{"name": "attr1"}]}],
            "nonadditive": "none",
        }
        dim = DimensionSchema.model_validate(data)
        assert dim.nonadditive is None

    def test_nonadditive_any_normalized(self):
        """'any' is normalized to 'all'."""
        data = {
            "name": "test",
            "levels": [{"name": "level1", "attributes": [{"name": "attr1"}]}],
            "nonadditive": "any",
        }
        dim = DimensionSchema.model_validate(data)
        assert dim.nonadditive == "all"

    def test_nonadditive_time_preserved(self):
        """'time' is preserved."""
        data = {
            "name": "date",
            "levels": [{"name": "year", "attributes": [{"name": "year"}]}],
            "nonadditive": "time",
        }
        dim = DimensionSchema.model_validate(data)
        assert dim.nonadditive == "time"

    def test_nonadditive_all_preserved(self):
        """'all' is preserved."""
        data = {
            "name": "test",
            "levels": [{"name": "level1", "attributes": [{"name": "attr1"}]}],
            "nonadditive": "all",
        }
        dim = DimensionSchema.model_validate(data)
        assert dim.nonadditive == "all"

    def test_nonadditive_invalid_raises_error(self):
        """Invalid nonadditive value raises error."""
        data = {
            "name": "test",
            "levels": [{"name": "level1", "attributes": [{"name": "attr1"}]}],
            "nonadditive": "invalid",
        }
        with pytest.raises(ValidationError) as exc_info:
            DimensionSchema.model_validate(data)
        assert "non-additive" in str(exc_info.value).lower()

    def test_nonadditive_case_insensitive(self):
        """Nonadditive normalization is case-insensitive."""
        data = {
            "name": "test",
            "levels": [{"name": "level1", "attributes": [{"name": "attr1"}]}],
            "nonadditive": "NONE",
        }
        dim = DimensionSchema.model_validate(data)
        assert dim.nonadditive is None


class TestDimensionSchemaLevelCoercion:
    """Test level string/dict coercion."""

    def test_level_strings_coerced_to_schemas(self):
        """String level names are converted to LevelSchema."""
        data = {
            "name": "date",
            "levels": ["year", "month", "day"],
        }
        dim = DimensionSchema.model_validate(data)
        assert len(dim.levels) == 3
        for level in dim.levels:
            assert isinstance(level, LevelSchema)
        assert dim.levels[0].name == "year"
        assert dim.levels[1].name == "month"
        assert dim.levels[2].name == "day"

    def test_level_dicts_coerced_to_schemas(self):
        """Dict level definitions are converted to LevelSchema."""
        data = {
            "name": "product",
            "levels": [
                {"name": "category", "attributes": [{"name": "category_id"}]},
                {"name": "product", "attributes": [{"name": "product_id"}]},
            ],
        }
        dim = DimensionSchema.model_validate(data)
        assert len(dim.levels) == 2
        assert all(isinstance(l, LevelSchema) for l in dim.levels)

    def test_mixed_level_types(self):
        """Mixed strings and dicts are handled."""
        data = {
            "name": "date",
            "levels": [
                "year",
                {"name": "quarter", "attributes": [{"name": "quarter"}]},
                "month",
            ],
        }
        dim = DimensionSchema.model_validate(data)
        assert len(dim.levels) == 3
        for level in dim.levels:
            assert isinstance(level, LevelSchema)


class TestDimensionSchemaHierarchyCoercion:
    """Test hierarchy string/dict coercion."""

    def test_hierarchy_dicts_coerced_to_schemas(self):
        """Dict hierarchy definitions are converted to HierarchySchema."""
        data = {
            "name": "date",
            "levels": [
                {"name": "year", "attributes": [{"name": "year"}]},
                {"name": "month", "attributes": [{"name": "month"}]},
            ],
            "hierarchies": [
                {"name": "fiscal", "levels": ["year", "month"]},
            ],
        }
        dim = DimensionSchema.model_validate(data)
        assert len(dim.hierarchies) == 1
        assert isinstance(dim.hierarchies[0], HierarchySchema)


class TestDimensionSchemaTemplate:
    """Test template inheritance field."""

    def test_template_field_accepted(self):
        """Template field is accepted and stored."""
        data = {
            "name": "derived_dim",
            "template": "base_time",
            "levels": [{"name": "year", "attributes": [{"name": "year"}]}],
        }
        dim = DimensionSchema.model_validate(data)
        assert dim.template == "base_time"

    def test_no_template_defaults_to_none(self):
        """Template defaults to None if not provided."""
        data = {
            "name": "standalone",
            "levels": [{"name": "level1", "attributes": [{"name": "attr1"}]}],
        }
        dim = DimensionSchema.model_validate(data)
        assert dim.template is None


class TestDimensionSchemaFlatDimension:
    """Test flat dimension using attributes instead of levels."""

    def test_flat_dimension_with_attributes(self):
        """Dimension with attributes instead of levels."""
        data = {
            "name": "status",
            "attributes": [
                {"name": "status_code"},
                {"name": "status_name"},
            ],
        }
        dim = DimensionSchema.model_validate(data)
        assert dim.name == "status"
        assert dim.attributes is not None
        assert len(dim.attributes) == 2
        assert dim.levels is None
        assert dim.hierarchies is None

    def test_flat_dimension_with_string_attributes(self):
        """Flat dimension with string attribute names coerced to schemas."""
        from cubes.metadata.schemas.attributes import AttributeSchema

        data = {
            "name": "category",
            "attributes": ["code", "name"],
        }
        dim = DimensionSchema.model_validate(data)
        assert len(dim.attributes) == 2
        # String attributes are coerced to AttributeSchema objects
        assert isinstance(dim.attributes[0], AttributeSchema)
        assert dim.attributes[0].name == "code"
        assert isinstance(dim.attributes[1], AttributeSchema)
        assert dim.attributes[1].name == "name"


class TestLegacyCompatibility:
    """Test compatibility with legacy dimension expansion behavior."""

    def test_legacy_simple_dimension_string(self):
        """Legacy: simple string dimension name creates single level."""
        dim = DimensionSchema.model_validate("year")
        assert dim.name == "year"
        assert len(dim.levels) == 1
        assert dim.levels[0].name == "year"
        assert len(dim.hierarchies) == 1
        assert dim.hierarchies[0].name == "default"

    def test_legacy_levels_with_string_names(self):
        """Legacy: levels can be just string names."""
        data = {
            "name": "date",
            "levels": ["year", "month"],
        }
        dim = DimensionSchema.model_validate(data)
        assert len(dim.levels) == 2
        assert dim.levels[0].name == "year"
        assert dim.levels[1].name == "month"

    def test_legacy_default_hierarchy_from_levels(self):
        """Legacy: default hierarchy created from levels if none provided."""
        data = {
            "name": "product",
            "levels": [
                {"name": "category", "attributes": [{"name": "category"}]},
                {"name": "subcategory", "attributes": [{"name": "subcategory"}]},
            ],
        }
        dim = DimensionSchema.model_validate(data)
        assert len(dim.hierarchies) == 1
        hier = dim.hierarchies[0]
        assert hier.name == "default"
        assert hier.levels == ["category", "subcategory"]

    def test_legacy_full_dimension_metadata(self):
        """Legacy: full dimension with all fields."""
        data = {
            "name": "date",
            "label": "Date",
            "description": "Date dimension",
            "role": "time",
            "cardinality": "high",
            "category": "time",
            "nonadditive": "time",
            "levels": [
                {
                    "name": "year",
                    "label": "Year",
                    "attributes": [{"name": "year", "label": "Year"}],
                },
                {
                    "name": "month",
                    "label": "Month",
                    "attributes": [{"name": "month", "label": "Month"}],
                },
            ],
            "hierarchies": [
                {"name": "default", "label": "Default", "levels": ["year", "month"]}
            ],
            "default_hierarchy_name": "default",
            "info": {"format": "calendar"},
        }
        dim = DimensionSchema.model_validate(data)
        assert dim.name == "date"
        assert dim.label == "Date"
        assert dim.description == "Date dimension"
        assert dim.role == "time"
        assert dim.cardinality == "high"
        assert dim.category == "time"
        assert dim.nonadditive == "time"
        assert dim.info == {"format": "calendar"}
        assert dim.default_hierarchy_name == "default"


class TestDimensionSchemaExtraFields:
    """Test extra fields for forward compatibility."""

    def test_extra_fields_allowed(self):
        """Extra fields are allowed for forward compatibility."""
        data = {
            "name": "test",
            "levels": [{"name": "level1", "attributes": [{"name": "attr1"}]}],
            "custom_field": "custom_value",
            "another_extra": 123,
        }
        dim = DimensionSchema.model_validate(data)
        assert dim.name == "test"
        # Extra fields are stored in model_extra
        assert hasattr(dim, "custom_field")
        assert dim.custom_field == "custom_value"
        assert dim.another_extra == 123
