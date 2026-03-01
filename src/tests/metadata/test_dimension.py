"""Tests for the Dimension domain dataclass.

This module tests the Dimension class including:
- Initialization and __post_init__
- Property getters
- Lookup methods (level, hierarchy, attribute)
- Clone functionality
- to_dict() serialization
- from_schema() factory method
- Template inheritance
"""

import copy
from typing import Any

import pytest

from cubes.metadata.models.attributes import Attribute
from cubes.metadata.models.base import ModelError
from cubes.metadata.models.dimension import Dimension, Hierarchy, HierarchyError, Level
from cubes.metadata.schemas.dimension import DimensionSchema


class TestDimensionInitialization:
    """Test Dimension initialization and basic properties."""

    def test_basic_dimension_creation(self) -> None:
        """Test creating a basic dimension with levels."""
        dim = Dimension(
            name="date",
            levels=[
                Level(name="year", attributes=[Attribute(name="year")]),
                Level(name="month", attributes=[Attribute(name="month")]),
            ],
        )

        assert dim.name == "date"
        assert len(dim.levels) == 2
        assert len(dim.hierarchies) == 1  # Default hierarchy created
        assert dim.hierarchies[0].name == "default"
        assert dim.default_hierarchy_name == "default"

    def test_empty_levels_raises_error(self) -> None:
        """Test that empty levels list raises ModelError."""
        with pytest.raises(ModelError, match="must have at least one level"):
            Dimension(name="empty", levels=[])

    def test_flat_dimension(self) -> None:
        """Test flat dimension with single level."""
        dim = Dimension(
            name="flag",
            levels=[Level(name="flag", attributes=[Attribute(name="flag")])],
        )

        assert dim.is_flat is True
        assert dim.has_details is False
        assert len(dim.levels) == 1

    def test_dimension_with_multiple_attributes(self) -> None:
        """Test dimension where levels have multiple attributes."""
        dim = Dimension(
            name="product",
            levels=[
                Level(
                    name="product",
                    attributes=[
                        Attribute(name="id"),
                        Attribute(name="name"),
                        Attribute(name="category"),
                    ],
                )
            ],
        )

        assert dim.has_details is True
        assert len(dim.attributes) == 3

    def test_dimension_ownership_of_attributes(self) -> None:
        """Test that dimension claims ownership of level attributes."""
        attr = Attribute(name="year")
        dim = Dimension(name="date", levels=[Level(name="year", attributes=[attr])])

        assert attr.dimension is dim
        assert attr.ref == "date.year"

    def test_duplicate_level_names_raises_error(self) -> None:
        """Test that duplicate level names raise ModelError."""
        with pytest.raises(ModelError, match="Duplicate level"):
            Dimension(
                name="bad",
                levels=[
                    Level(name="same", attributes=[Attribute(name="a")]),
                    Level(name="same", attributes=[Attribute(name="b")]),
                ],
            )

    def test_duplicate_hierarchy_names_raises_error(self) -> None:
        """Test that duplicate hierarchy names raise ModelError."""
        with pytest.raises(ModelError, match="Duplicate hierarchy"):
            Dimension(
                name="bad",
                levels=[Level(name="level", attributes=[Attribute(name="a")])],
                hierarchies=[
                    Hierarchy(
                        name="same",
                        levels=[Level(name="level", attributes=[Attribute(name="a")])],
                    ),
                    Hierarchy(
                        name="same",
                        levels=[Level(name="level", attributes=[Attribute(name="a")])],
                    ),
                ],
            )

    def test_attribute_owned_by_other_dimension_raises_error(self) -> None:
        """Test that claiming an attribute owned by another dimension raises error."""
        attr = Attribute(name="shared")
        dim1 = Dimension(name="dim1", levels=[Level(name="level", attributes=[attr])])

        # attr is now owned by dim1
        with pytest.raises(ModelError, match="cannot claim attribute"):
            Dimension(name="dim2", levels=[Level(name="level", attributes=[attr])])


class TestDimensionProperties:
    """Test Dimension property getters."""

    @pytest.fixture
    def date_dimension(self) -> Dimension:
        """Create a date dimension for testing."""
        return Dimension(
            name="date",
            levels=[
                Level(name="year", attributes=[Attribute(name="year")]),
                Level(name="month", attributes=[Attribute(name="month")]),
                Level(name="day", attributes=[Attribute(name="day")]),
            ],
            hierarchies=[
                Hierarchy(
                    name="default",
                    levels=[
                        Level(name="year", attributes=[Attribute(name="year")]),
                        Level(name="month", attributes=[Attribute(name="month")]),
                        Level(name="day", attributes=[Attribute(name="day")]),
                    ],
                ),
                Hierarchy(
                    name="weekly",
                    levels=[
                        Level(name="year", attributes=[Attribute(name="year")]),
                        Level(name="week", attributes=[Attribute(name="week")]),
                    ],
                ),
            ],
        )

    def test_level_names_property(self, date_dimension: Dimension) -> None:
        """Test level_names property."""
        names = date_dimension.level_names
        assert "year" in names
        assert "month" in names
        assert "day" in names

    def test_is_flat_property(self, date_dimension: Dimension) -> None:
        """Test is_flat property."""
        assert date_dimension.is_flat is False

        flat_dim = Dimension(
            name="flag",
            levels=[Level(name="flag", attributes=[Attribute(name="flag")])],
        )
        assert flat_dim.is_flat is True

    def test_has_details_property(self, date_dimension: Dimension) -> None:
        """Test has_details property."""
        assert date_dimension.has_details is False

        detailed_dim = Dimension(
            name="product",
            levels=[
                Level(
                    name="product",
                    attributes=[
                        Attribute(name="id"),
                        Attribute(name="name"),
                    ],
                )
            ],
        )
        assert detailed_dim.has_details is True

    def test_key_attributes_property(self, date_dimension: Dimension) -> None:
        """Test key_attributes property."""
        keys = date_dimension.key_attributes
        assert len(keys) == 3
        key_names = [k.name for k in keys]
        assert "year" in key_names
        assert "month" in key_names
        assert "day" in key_names

    def test_attributes_property(self, date_dimension: Dimension) -> None:
        """Test attributes property."""
        attrs = date_dimension.attributes
        assert len(attrs) == 3

    def test_levels_property(self, date_dimension: Dimension) -> None:
        """Test levels property."""
        levels = date_dimension.levels
        assert len(levels) == 3

    def test_hierarchies_property(self, date_dimension: Dimension) -> None:
        """Test hierarchies property."""
        hierarchies = date_dimension.hierarchies
        assert len(hierarchies) == 2


class TestDimensionLookupMethods:
    """Test Dimension lookup methods."""

    @pytest.fixture
    def date_dimension(self) -> Dimension:
        """Create a date dimension for testing."""
        year_level = Level(name="year", attributes=[Attribute(name="year")])
        month_level = Level(name="month", attributes=[Attribute(name="month")])
        day_level = Level(name="day", attributes=[Attribute(name="day")])

        return Dimension(
            name="date",
            levels=[year_level, month_level, day_level],
            hierarchies=[
                Hierarchy(name="default", levels=[year_level, month_level, day_level])
            ],
        )

    def test_level_by_name(self, date_dimension: Dimension) -> None:
        """Test level() lookup by name."""
        level = date_dimension.level("month")
        assert level.name == "month"

    def test_level_by_object(self, date_dimension: Dimension) -> None:
        """Test level() lookup by Level object."""
        month_level = date_dimension.level("month")
        result = date_dimension.level(month_level)
        assert result is month_level

    def test_level_not_found_raises_keyerror(self, date_dimension: Dimension) -> None:
        """Test level() with unknown name raises KeyError."""
        with pytest.raises(KeyError, match="No level"):
            date_dimension.level("century")

    def test_level_invalid_type_raises_valueerror(
        self, date_dimension: Dimension
    ) -> None:
        """Test level() with invalid type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown level object"):
            date_dimension.level(123)  # type: ignore[arg-type]

    def test_hierarchy_by_name(self, date_dimension: Dimension) -> None:
        """Test hierarchy() lookup by name."""
        hier = date_dimension.hierarchy("default")
        assert hier.name == "default"

    def test_hierarchy_by_object(self, date_dimension: Dimension) -> None:
        """Test hierarchy() lookup by Hierarchy object."""
        default_hier = date_dimension.hierarchy("default")
        result = date_dimension.hierarchy(default_hier)
        assert result is default_hier

    def test_hierarchy_default(self, date_dimension: Dimension) -> None:
        """Test hierarchy() returns default when no argument."""
        hier = date_dimension.hierarchy()
        assert hier.name == "default"

    def test_hierarchy_not_found_raises_error(self, date_dimension: Dimension) -> None:
        """Test hierarchy() with unknown name raises ModelError."""
        with pytest.raises(ModelError, match="No hierarchy"):
            date_dimension.hierarchy("nonexistent")

    def test_hierarchy_invalid_type_raises_valueerror(
        self, date_dimension: Dimension
    ) -> None:
        """Test hierarchy() with invalid type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown hierarchy object"):
            date_dimension.hierarchy(123)  # type: ignore[arg-type]

    def test_attribute_by_name(self, date_dimension: Dimension) -> None:
        """Test attribute() lookup by name."""
        attr = date_dimension.attribute("month")
        assert attr.name == "month"

    def test_attribute_by_ref(self, date_dimension: Dimension) -> None:
        """Test attribute() lookup by reference."""
        attr = date_dimension.attribute("date.month", by_ref=True)
        assert attr.name == "month"

    def test_attribute_not_found_raises_keyerror(
        self, date_dimension: Dimension
    ) -> None:
        """Test attribute() with unknown name raises KeyError."""
        with pytest.raises(KeyError, match="Unknown attribute"):
            date_dimension.attribute("century")


class TestDimensionClone:
    """Test Dimension clone functionality."""

    @pytest.fixture
    def date_dimension(self) -> Dimension:
        """Create a date dimension for testing."""
        year_level = Level(name="year", attributes=[Attribute(name="year")])
        month_level = Level(name="month", attributes=[Attribute(name="month")])
        day_level = Level(name="day", attributes=[Attribute(name="day")])
        week_level = Level(name="week", attributes=[Attribute(name="week")])

        return Dimension(
            name="date",
            levels=[year_level, month_level, day_level, week_level],
            hierarchies=[
                Hierarchy(name="default", levels=[year_level, month_level, day_level]),
                Hierarchy(name="weekly", levels=[year_level, week_level]),
            ],
        )

    def test_clone_basic(self, date_dimension: Dimension) -> None:
        """Test basic clone functionality."""
        clone = date_dimension.clone()

        assert clone.name == "date"
        assert clone.master is date_dimension
        assert (
            len(clone.levels) == 4
        )  # year, month, day, week (all used in hierarchies)
        assert len(clone.hierarchies) == 2

    def test_clone_with_alias(self, date_dimension: Dimension) -> None:
        """Test clone with alias (new name)."""
        clone = date_dimension.clone(alias="another_date")

        assert clone.name == "another_date"
        assert clone.master is date_dimension

    def test_clone_with_hierarchy_limit(self, date_dimension: Dimension) -> None:
        """Test clone with hierarchies parameter."""
        clone = date_dimension.clone(hierarchies=["default"])

        assert len(clone.hierarchies) == 1
        assert clone.hierarchies[0].name == "default"
        # Should have updated default_hierarchy_name
        assert clone.default_hierarchy_name == "default"

    def test_clone_with_exclude_hierarchies(self, date_dimension: Dimension) -> None:
        """Test clone with exclude_hierarchies parameter."""
        clone = date_dimension.clone(exclude_hierarchies=["weekly"])

        assert len(clone.hierarchies) == 1
        assert clone.hierarchies[0].name == "default"

    def test_clone_empty_hierarchies_raises_error(
        self, date_dimension: Dimension
    ) -> None:
        """Test that clone with empty hierarchies list raises error."""
        with pytest.raises(ModelError, match="Cannot remove all hierarchies"):
            date_dimension.clone(hierarchies=[])

    def test_clone_disowns_attributes(self, date_dimension: Dimension) -> None:
        """Test that clone disowns level attributes from original."""
        clone = date_dimension.clone()

        # Get an attribute from the original
        orig_attr = date_dimension.attribute("year")
        assert orig_attr.dimension is date_dimension

        # Get an attribute from the clone
        clone_attr = clone.attribute("year")
        assert clone_attr.dimension is clone

    def test_clone_with_nonadditive_override(self, date_dimension: Dimension) -> None:
        """Test clone with nonadditive override."""
        clone = date_dimension.clone(nonadditive="time")

        assert clone.nonadditive == "time"

    def test_clone_with_cardinality_override(self, date_dimension: Dimension) -> None:
        """Test clone with cardinality override."""
        clone = date_dimension.clone(cardinality="high")

        assert clone.cardinality == "high"

    def test_clone_levels_filtered_by_hierarchy(
        self, date_dimension: Dimension
    ) -> None:
        """Test that clone only includes levels used in selected hierarchies."""
        # weekly hierarchy only uses year and week, not month and day
        clone = date_dimension.clone(hierarchies=["weekly"])

        level_names = [l.name for l in clone.levels]
        assert "year" in level_names
        assert "week" in level_names
        assert "month" not in level_names
        assert "day" not in level_names


class TestDimensionToDict:
    """Test Dimension to_dict functionality."""

    @pytest.fixture
    def date_dimension(self) -> Dimension:
        """Create a date dimension for testing."""
        return Dimension(
            name="date",
            levels=[
                Level(name="year", attributes=[Attribute(name="year")]),
                Level(name="month", attributes=[Attribute(name="month")]),
            ],
            hierarchies=[
                Hierarchy(
                    name="default",
                    levels=[
                        Level(name="year", attributes=[Attribute(name="year")]),
                        Level(name="month", attributes=[Attribute(name="month")]),
                    ],
                )
            ],
            default_hierarchy_name="default",
            label="Date",
            description="Date dimension",
            role="time",
        )

    def test_to_dict_basic(self, date_dimension: Dimension) -> None:
        """Test basic to_dict functionality."""
        result = date_dimension.to_dict()

        assert result["name"] == "date"
        assert result["label"] == "Date"
        assert result["description"] == "Date dimension"
        assert result["role"] == "time"
        assert result["default_hierarchy_name"] == "default"
        assert "levels" in result
        assert "hierarchies" in result
        assert result["is_flat"] is False
        assert result["has_details"] is False

    def test_to_dict_with_create_label(self, date_dimension: Dimension) -> None:
        """Test to_dict with create_label=True."""
        dim = Dimension(
            name="my_dimension",
            levels=[Level(name="level", attributes=[Attribute(name="attr")])],
        )

        result = dim.to_dict(create_label=True)
        assert result["label"] == "My dimension"

    def test_to_dict_with_hierarchy_limits(self, date_dimension: Dimension) -> None:
        """Test to_dict with hierarchy_limits."""
        result = date_dimension.to_dict(hierarchy_limits={"default": "year"})

        # Should truncate default hierarchy to only include "year"
        assert len(result["hierarchies"]) == 1
        assert result["hierarchies"][0]["levels"] == ["year"]

    def test_to_dict_excludes_hierarchy(self, date_dimension: Dimension) -> None:
        """Test to_dict with hierarchy_limits excluding a hierarchy."""
        dim = Dimension(
            name="date",
            levels=[
                Level(name="year", attributes=[Attribute(name="year")]),
            ],
            hierarchies=[
                Hierarchy(
                    name="default",
                    levels=[Level(name="year", attributes=[Attribute(name="year")])],
                ),
                Hierarchy(
                    name="other",
                    levels=[Level(name="year", attributes=[Attribute(name="year")])],
                ),
            ],
        )

        result = dim.to_dict(hierarchy_limits={"other": None})

        # Should exclude "other" hierarchy
        assert len(result["hierarchies"]) == 1
        assert result["hierarchies"][0]["name"] == "default"


class TestDimensionFromSchema:
    """Test Dimension from_schema factory method."""

    def test_from_schema_basic(self) -> None:
        """Test creating Dimension from basic schema."""
        schema = DimensionSchema.model_validate(
            {
                "name": "date",
                "levels": [
                    {"name": "year", "attributes": ["year"]},
                    {"name": "month", "attributes": ["month"]},
                ],
            }
        )

        dim = Dimension.from_schema(schema)

        assert dim.name == "date"
        assert len(dim.levels) == 2
        assert len(dim.hierarchies) == 1

    def test_from_schema_with_hierarchies(self) -> None:
        """Test creating Dimension from schema with explicit hierarchies."""
        schema = DimensionSchema.model_validate(
            {
                "name": "date",
                "levels": [
                    {"name": "year", "attributes": ["year"]},
                    {"name": "month", "attributes": ["month"]},
                    {"name": "day", "attributes": ["day"]},
                ],
                "hierarchies": [
                    {"name": "ymd", "levels": ["year", "month", "day"]},
                    {"name": "ym", "levels": ["year", "month"]},
                ],
            }
        )

        dim = Dimension.from_schema(schema)

        assert len(dim.hierarchies) == 2
        assert dim.hierarchy("ymd") is not None
        assert dim.hierarchy("ym") is not None

    def test_from_schema_with_properties(self) -> None:
        """Test creating Dimension from schema with all properties."""
        schema = DimensionSchema.model_validate(
            {
                "name": "date",
                "levels": [{"name": "year", "attributes": ["year"]}],
                "label": "Date Dimension",
                "description": "A date dimension",
                "role": "time",
                "cardinality": "high",
                "category": "temporal",
                "nonadditive": "time",
            }
        )

        dim = Dimension.from_schema(schema)

        assert dim.label == "Date Dimension"
        assert dim.description == "A date dimension"
        assert dim.role == "time"
        assert dim.cardinality == "high"
        assert dim.category == "temporal"
        assert dim.nonadditive == "time"


class TestDimensionTemplateInheritance:
    """Test Dimension template inheritance via from_schema."""

    @pytest.fixture
    def template_dimension(self) -> Dimension:
        """Create a template dimension."""
        return Dimension(
            name="time",
            levels=[
                Level(name="year", attributes=[Attribute(name="year")]),
                Level(name="month", attributes=[Attribute(name="month")]),
                Level(name="day", attributes=[Attribute(name="day")]),
                Level(name="hour", attributes=[Attribute(name="hour")]),
            ],
            hierarchies=[
                Hierarchy(
                    name="full",
                    levels=[
                        Level(name="year", attributes=[Attribute(name="year")]),
                        Level(name="month", attributes=[Attribute(name="month")]),
                        Level(name="day", attributes=[Attribute(name="day")]),
                        Level(name="hour", attributes=[Attribute(name="hour")]),
                    ],
                ),
                Hierarchy(
                    name="ymd",
                    levels=[
                        Level(name="year", attributes=[Attribute(name="year")]),
                        Level(name="month", attributes=[Attribute(name="month")]),
                        Level(name="day", attributes=[Attribute(name="day")]),
                    ],
                ),
            ],
            default_hierarchy_name="full",
        )

    def test_from_schema_with_template(self, template_dimension: Dimension) -> None:
        """Test creating Dimension from schema with template."""
        schema = DimensionSchema.model_validate(
            {
                "name": "date",
                "template": "time",
            }
        )

        templates = {"time": template_dimension}
        dim = Dimension.from_schema(schema, templates)

        assert dim.name == "date"
        assert len(dim.levels) == 4  # Inherited from template
        assert len(dim.hierarchies) == 2  # Inherited from template
        assert dim.default_hierarchy_name == "full"

    def test_from_schema_template_with_hierarchy_filter(
        self, template_dimension: Dimension
    ) -> None:
        """Test template inheritance with hierarchy filtering."""
        schema = DimensionSchema.model_validate(
            {
                "name": "date",
                "template": "time",
                "hierarchies": ["ymd"],  # Only inherit ymd hierarchy
            }
        )

        templates = {"time": template_dimension}
        dim = Dimension.from_schema(schema, templates)

        assert len(dim.hierarchies) == 1
        assert dim.hierarchies[0].name == "ymd"
        # Note: Levels are inherited from template then filtered to hierarchy usage
        # All 4 template levels are inherited, but only 3 are used in ymd
        # The __post_init__ filters to only levels actually used in hierarchies
        level_names = {l.name for l in dim.levels}
        assert level_names == {"year", "month", "day"}
        assert len(dim.levels) == 3

    def test_from_schema_template_not_found_raises_error(self) -> None:
        """Test that missing template raises ModelError."""
        schema = DimensionSchema.model_validate(
            {
                "name": "date",
                "template": "nonexistent",
            }
        )

        with pytest.raises(
            ModelError, match="Template dimension 'nonexistent' not found"
        ):
            Dimension.from_schema(schema, {})

    def test_from_schema_level_name_creates_default_level(self) -> None:
        """Test that level name as string creates a default level."""
        # Note: DimensionSchema automatically converts string level names
        # to full level definitions with same-named attribute
        schema = DimensionSchema.model_validate(
            {
                "name": "date",
                "levels": ["year"],  # String level name
            }
        )

        dim = Dimension.from_schema(schema)

        assert dim.name == "date"
        assert len(dim.levels) == 1
        assert dim.levels[0].name == "year"
        assert dim.levels[0].attributes[0].name == "year"

    def test_from_schema_hierarchy_name_without_template_raises_error(self) -> None:
        """Test that hierarchy name without template raises error."""
        schema = DimensionSchema.model_validate(
            {
                "name": "date",
                "levels": [{"name": "year", "attributes": ["year"]}],
                "hierarchies": ["default"],  # String hierarchy name without template
            }
        )

        with pytest.raises(ModelError, match="Cannot specify hierarchy name"):
            Dimension.from_schema(schema)

    def test_from_schema_template_level_info_merge(
        self, template_dimension: Dimension
    ) -> None:
        """Test that level info is merged from template."""
        # Add info to template level
        template_dimension.level("year").info["format"] = "YYYY"

        schema = DimensionSchema.model_validate(
            {
                "name": "date",
                "template": "time",
                "levels": [
                    {
                        "name": "year",
                        "attributes": ["year"],
                        "info": {"units": "years"},
                    },  # Override info
                ],
            }
        )

        templates = {"time": template_dimension}
        dim = Dimension.from_schema(schema, templates)

        year_level = dim.level("year")
        assert year_level.info["units"] == "years"  # From schema
        assert year_level.info["format"] == "YYYY"  # From template


class TestDimensionEquality:
    """Test Dimension equality comparison."""

    def test_equal_dimensions(self) -> None:
        """Test that identical dimensions are equal."""
        dim1 = Dimension(
            name="date",
            levels=[
                Level(name="year", attributes=[Attribute(name="year")]),
            ],
        )
        dim2 = Dimension(
            name="date",
            levels=[
                Level(name="year", attributes=[Attribute(name="year")]),
            ],
        )

        assert dim1 == dim2

    def test_unequal_dimensions(self) -> None:
        """Test that different dimensions are not equal."""
        dim1 = Dimension(
            name="date",
            levels=[
                Level(name="year", attributes=[Attribute(name="year")]),
            ],
        )
        dim2 = Dimension(
            name="product",
            levels=[
                Level(name="product", attributes=[Attribute(name="product")]),
            ],
        )

        assert dim1 != dim2

    def test_dimension_not_equal_to_none(self) -> None:
        """Test that dimension is not equal to None."""
        dim = Dimension(
            name="date",
            levels=[Level(name="year", attributes=[Attribute(name="year")])],
        )

        assert dim != None  # noqa: E711

    def test_dimension_not_equal_to_other_type(self) -> None:
        """Test that dimension is not equal to other types."""
        dim = Dimension(
            name="date",
            levels=[Level(name="year", attributes=[Attribute(name="year")])],
        )

        assert dim != "date"
        assert dim != 123


class TestDimensionNonadditiveNormalization:
    """Test nonadditive value normalization."""

    def test_nonadditive_none_normalized_to_none(self) -> None:
        """Test that 'none' is normalized to None."""
        dim = Dimension(
            name="date",
            levels=[Level(name="year", attributes=[Attribute(name="year")])],
            nonadditive="none",
        )

        assert dim.nonadditive is None

    def test_nonadditive_all_normalized(self) -> None:
        """Test that 'all' is normalized to 'all'."""
        dim = Dimension(
            name="date",
            levels=[Level(name="year", attributes=[Attribute(name="year")])],
            nonadditive="all",
        )

        assert dim.nonadditive == "all"

    def test_nonadditive_any_normalized_to_all(self) -> None:
        """Test that 'any' is normalized to 'all'."""
        dim = Dimension(
            name="date",
            levels=[Level(name="year", attributes=[Attribute(name="year")])],
            nonadditive="any",
        )

        assert dim.nonadditive == "all"

    def test_nonadditive_time_preserved(self) -> None:
        """Test that 'time' is preserved."""
        dim = Dimension(
            name="date",
            levels=[Level(name="year", attributes=[Attribute(name="year")])],
            nonadditive="time",
        )

        assert dim.nonadditive == "time"

    def test_nonadditive_unknown_raises_error(self) -> None:
        """Test that unknown nonadditive value raises error."""
        with pytest.raises(ModelError, match="unknown non-additive type"):
            Dimension(
                name="date",
                levels=[Level(name="year", attributes=[Attribute(name="year")])],
                nonadditive="invalid",
            )


class TestDimensionTimeRole:
    """Test time dimension role and default level roles."""

    def test_time_dimension_default_roles(self) -> None:
        """Test that time dimension sets default level roles."""
        dim = Dimension(
            name="date",
            levels=[
                Level(name="year", attributes=[Attribute(name="year")]),
                Level(name="month", attributes=[Attribute(name="month")]),
                Level(name="day", attributes=[Attribute(name="day")]),
            ],
            role="time",
        )

        assert dim.level("year").role == "year"
        assert dim.level("month").role == "month"
        assert dim.level("day").role == "day"

    def test_non_time_dimension_no_default_roles(self) -> None:
        """Test that non-time dimension doesn't set default level roles."""
        dim = Dimension(
            name="product",
            levels=[
                Level(name="category", attributes=[Attribute(name="category")]),
                Level(name="product", attributes=[Attribute(name="product")]),
            ],
        )

        assert dim.level("category").role is None
        assert dim.level("product").role is None

    def test_unknown_time_role_levels_not_affected(self) -> None:
        """Test that levels not in _DEFAULT_LEVEL_ROLES are not affected."""
        dim = Dimension(
            name="date",
            levels=[
                Level(name="year", attributes=[Attribute(name="year")]),
                Level(name="custom", attributes=[Attribute(name="custom")]),
            ],
            role="time",
        )

        assert dim.level("year").role == "year"
        assert dim.level("custom").role is None


class TestDimensionStringRepresentation:
    """Test Dimension string representations."""

    def test_str_representation(self) -> None:
        """Test __str__ returns dimension name."""
        dim = Dimension(
            name="date",
            levels=[Level(name="year", attributes=[Attribute(name="year")])],
        )

        assert str(dim) == "date"

    def test_repr_representation(self) -> None:
        """Test __repr__ returns detailed representation."""
        dim = Dimension(
            name="date",
            levels=[Level(name="year", attributes=[Attribute(name="year")])],
        )

        repr_str = repr(dim)
        assert "dimension:" in repr_str
        assert "date" in repr_str
        assert "year" in repr_str
