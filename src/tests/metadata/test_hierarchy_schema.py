"""Tests for Hierarchy Pydantic Schema.

Tests the HierarchySchema class including:
- Basic field validation
- Levels list validation (must not be empty)
- Level string references vs full definitions
- Label generation
- Ported legacy hierarchy expansion tests
"""

import pytest
from pydantic import ValidationError

from cubes.metadata.schemas.dimension import HierarchySchema, LevelSchema


class TestHierarchySchemaBasic:
    """Test basic HierarchySchema validation."""

    def test_valid_hierarchy_with_level_names(self):
        """Hierarchy with level name strings."""
        data = {
            "name": "default",
            "levels": ["year", "month", "day"],
        }
        hierarchy = HierarchySchema.model_validate(data)
        assert hierarchy.name == "default"
        assert hierarchy.levels == ["year", "month", "day"]
        assert hierarchy.label is None
        assert hierarchy.description is None
        assert hierarchy.info == {}

    def test_valid_hierarchy_with_label(self):
        """Hierarchy with label and description."""
        data = {
            "name": "fiscal",
            "levels": ["year", "quarter", "month"],
            "label": "Fiscal Calendar",
            "description": "Fiscal year hierarchy",
            "info": {"key": "value"},
        }
        hierarchy = HierarchySchema.model_validate(data)
        assert hierarchy.name == "fiscal"
        assert hierarchy.label == "Fiscal Calendar"
        assert hierarchy.description == "Fiscal year hierarchy"
        assert hierarchy.info == {"key": "value"}

    def test_hierarchy_with_single_level(self):
        """Hierarchy with just one level."""
        data = {
            "name": "simple",
            "levels": ["category"],
        }
        hierarchy = HierarchySchema.model_validate(data)
        assert hierarchy.name == "simple"
        assert hierarchy.levels == ["category"]


class TestHierarchySchemaValidation:
    """Test HierarchySchema validation rules."""

    def test_empty_levels_raises_error(self):
        """Hierarchy must have at least one level."""
        data = {
            "name": "empty",
            "levels": [],
        }
        with pytest.raises(ValidationError) as exc_info:
            HierarchySchema.model_validate(data)
        error_msg = str(exc_info.value)
        assert "must have at least one level" in error_msg
        assert "empty" in error_msg

    def test_missing_name_raises_error(self):
        """Name is required."""
        data = {
            "levels": ["year", "month"],
        }
        with pytest.raises(ValidationError) as exc_info:
            HierarchySchema.model_validate(data)
        assert "name" in str(exc_info.value)

    def test_missing_levels_raises_error(self):
        """Levels is required."""
        data = {
            "name": "no_levels",
        }
        with pytest.raises(ValidationError) as exc_info:
            HierarchySchema.model_validate(data)
        assert "levels" in str(exc_info.value)


class TestHierarchySchemaWithLevelDefinitions:
    """Test hierarchy with full level definitions."""

    def test_hierarchy_with_level_objects(self):
        """Hierarchy with full level definitions."""
        data = {
            "name": "default",
            "levels": [
                {
                    "name": "year",
                    "attributes": ["year"],
                },
                {
                    "name": "month",
                    "attributes": ["month", "month_name"],
                },
                {
                    "name": "day",
                    "attributes": ["day"],
                },
            ],
        }
        hierarchy = HierarchySchema.model_validate(data)
        assert hierarchy.name == "default"
        assert len(hierarchy.levels) == 3

        # Check that levels were coerced to LevelSchema instances
        for level in hierarchy.levels:
            assert isinstance(level, LevelSchema)

    def test_hierarchy_mixed_levels(self):
        """Hierarchy with both string names and level definitions."""
        data = {
            "name": "mixed",
            "levels": [
                "year",  # String reference
                {
                    "name": "month",
                    "attributes": ["month", "month_name"],
                },  # Full definition
                "day",  # String reference
            ],
        }
        hierarchy = HierarchySchema.model_validate(data)
        assert hierarchy.name == "mixed"
        assert len(hierarchy.levels) == 3
        # First and third should still be strings
        assert hierarchy.levels[0] == "year"
        assert hierarchy.levels[2] == "day"
        # Second should be LevelSchema
        assert isinstance(hierarchy.levels[1], LevelSchema)
        assert hierarchy.levels[1].name == "month"

    def test_level_default_attributes_applied(self):
        """Level schemas get their default attributes resolved."""
        data = {
            "name": "default",
            "levels": [
                {
                    "name": "month",
                    "attributes": ["month", "month_name", "month_sname"],
                }
            ],
        }
        hierarchy = HierarchySchema.model_validate(data)
        level = hierarchy.levels[0]
        assert isinstance(level, LevelSchema)
        assert level.name == "month"
        # Defaults should be applied
        assert level.key == "month"
        assert level.label_attribute == "month_name"
        assert level.order_attribute == "month"


class TestHierarchySchemaEdgeCases:
    """Test edge cases and special scenarios."""

    def test_extra_fields_allowed(self):
        """Extra fields should be allowed for forward compatibility."""
        data = {
            "name": "with_extras",
            "levels": ["a", "b"],
            "custom_field": "custom_value",
            "another_extra": 123,
        }
        hierarchy = HierarchySchema.model_validate(data)
        assert hierarchy.name == "with_extras"
        # Extra fields should be accessible via model_extra
        assert hierarchy.model_extra.get("custom_field") == "custom_value"
        assert hierarchy.model_extra.get("another_extra") == 123

    def test_hierarchy_serialization(self):
        """Hierarchy can be serialized to dict."""
        data = {
            "name": "default",
            "levels": ["year", "month"],
            "label": "Default Hierarchy",
        }
        hierarchy = HierarchySchema.model_validate(data)
        result = hierarchy.model_dump()
        assert result["name"] == "default"
        assert result["levels"] == ["year", "month"]
        assert result["label"] == "Default Hierarchy"

    def test_hierarchy_with_unicode(self):
        """Hierarchy handles unicode names and labels."""
        data = {
            "name": "hierarchy_测试",
            "levels": ["级别一", "级别二"],
            "label": "测试层级",
            "description": "这是一个测试",
        }
        hierarchy = HierarchySchema.model_validate(data)
        assert hierarchy.name == "hierarchy_测试"
        assert hierarchy.levels == ["级别一", "级别二"]
        assert hierarchy.label == "测试层级"

    def test_hierarchy_with_special_characters(self):
        """Hierarchy handles special characters in names."""
        data = {
            "name": "hierarchy_with-special.chars",
            "levels": ["level-1", "level_2"],
        }
        hierarchy = HierarchySchema.model_validate(data)
        assert hierarchy.name == "hierarchy_with-special.chars"
        assert hierarchy.levels == ["level-1", "level_2"]


class TestHierarchySchemaLegacyCompatibility:
    """Test compatibility with legacy hierarchy expansion tests."""

    def test_simple_hierarchy_expansion(self):
        """Simple hierarchy with level names expands correctly."""
        # This matches the pattern from legacy tests
        data = {
            "name": "default",
            "levels": ["year", "month", "day"],
        }
        hierarchy = HierarchySchema.model_validate(data)
        assert hierarchy.name == "default"
        assert hierarchy.levels == ["year", "month", "day"]

    def test_hierarchy_with_full_level_metadata(self):
        """Hierarchy with full level metadata validates correctly."""
        # From legacy: levels might contain full level definitions
        data = {
            "name": "detailed",
            "levels": [
                {
                    "name": "year",
                    "attributes": ["year"],
                    "label": "Year",
                },
                {
                    "name": "quarter",
                    "attributes": ["quarter", "quarter_name"],
                    "label": "Quarter",
                    "key": "quarter",
                    "label_attribute": "quarter_name",
                },
            ],
        }
        hierarchy = HierarchySchema.model_validate(data)
        assert len(hierarchy.levels) == 2

        year_level = hierarchy.levels[0]
        assert isinstance(year_level, LevelSchema)
        assert year_level.name == "year"
        assert year_level.label == "Year"

        quarter_level = hierarchy.levels[1]
        assert isinstance(quarter_level, LevelSchema)
        assert quarter_level.name == "quarter"
        assert quarter_level.key == "quarter"
        assert quarter_level.label_attribute == "quarter_name"

    def test_multiple_hierarchies_pattern(self):
        """Pattern for dimensions with multiple hierarchies."""
        # Common pattern: date dimension with multiple hierarchies
        default_hierarchy = {
            "name": "default",
            "levels": ["year", "quarter", "month", "day"],
        }
        fiscal_hierarchy = {
            "name": "fiscal",
            "levels": ["fiscal_year", "fiscal_quarter", "month", "day"],
            "label": "Fiscal Calendar",
        }

        h1 = HierarchySchema.model_validate(default_hierarchy)
        h2 = HierarchySchema.model_validate(fiscal_hierarchy)

        assert h1.name == "default"
        assert h1.levels == ["year", "quarter", "month", "day"]

        assert h2.name == "fiscal"
        assert h2.levels == ["fiscal_year", "fiscal_quarter", "month", "day"]
        assert h2.label == "Fiscal Calendar"


class TestHierarchySchemaTypeCoercion:
    """Test type coercion and input flexibility."""

    def test_from_dict_with_nested_levels(self):
        """Can create hierarchy from dict with nested level dicts."""
        data = {
            "name": "nested",
            "levels": [
                {"name": "country", "attributes": ["country", "country_name"]},
                {"name": "city", "attributes": ["city", "city_name"]},
            ],
        }
        hierarchy = HierarchySchema.model_validate(data)
        assert hierarchy.name == "nested"
        assert len(hierarchy.levels) == 2
        assert all(isinstance(l, LevelSchema) for l in hierarchy.levels)

    def test_level_dict_without_attributes_fails(self):
        """Level dict without attributes should fail validation."""
        data = {
            "name": "invalid",
            "levels": [
                {"name": "level_without_attrs"},  # Missing attributes
            ],
        }
        # This should fail because LevelSchema requires attributes
        with pytest.raises(ValidationError):
            HierarchySchema.model_validate(data)


class TestHierarchySchemaErrorMessages:
    """Test that error messages are descriptive."""

    def test_empty_levels_error_message(self):
        """Empty levels error includes hierarchy name."""
        data = {
            "name": "my_hierarchy",
            "levels": [],
        }
        with pytest.raises(ValidationError) as exc_info:
            HierarchySchema.model_validate(data)
        error_msg = str(exc_info.value)
        assert "my_hierarchy" in error_msg
        assert "at least one level" in error_msg

    def test_missing_required_field_errors(self):
        """Missing required fields have clear errors."""
        with pytest.raises(ValidationError) as exc_info:
            HierarchySchema.model_validate({})
        error_msg = str(exc_info.value)
        # Should mention missing fields
        assert "name" in error_msg or "levels" in error_msg
