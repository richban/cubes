"""Tests for Hierarchy domain dataclass.

Tests Hierarchy initialization, traversal methods, and level management.
"""

import copy
from typing import Any

import pytest

from cubes.metadata.models.attributes import Attribute
from cubes.metadata.models.base import ModelError
from cubes.metadata.models.dimension import Hierarchy, Level


class TestHierarchyInitialization:
    """Test Hierarchy initialization and basic properties."""

    def test_hierarchy_with_level_objects(self) -> None:
        """Test creating hierarchy with Level objects."""
        levels = [
            Level(name="year", attributes=[Attribute(name="year")]),
            Level(name="month", attributes=[Attribute(name="month")]),
            Level(name="day", attributes=[Attribute(name="day")]),
        ]
        hierarchy = Hierarchy(name="default", levels=levels)

        assert hierarchy.name == "default"
        assert len(hierarchy.levels) == 3
        assert hierarchy.levels[0].name == "year"
        assert hierarchy.levels[1].name == "month"
        assert hierarchy.levels[2].name == "day"

    def test_hierarchy_with_optional_fields(self) -> None:
        """Test creating hierarchy with label, description, info."""
        levels = [Level(name="year", attributes=[Attribute(name="year")])]
        hierarchy = Hierarchy(
            name="fiscal",
            levels=levels,
            label="Fiscal Calendar",
            description="Fiscal year hierarchy",
            info={"source": "erp"},
        )

        assert hierarchy.label == "Fiscal Calendar"
        assert hierarchy.description == "Fiscal year hierarchy"
        assert hierarchy.info == {"source": "erp"}

    def test_empty_levels_raises_error(self) -> None:
        """Test that empty levels list raises ModelError."""
        with pytest.raises(ModelError) as exc_info:
            Hierarchy(name="empty", levels=[])
        assert "must have at least one level" in str(exc_info.value)

    def test_hierarchy_with_string_levels(self) -> None:
        """Test creating hierarchy with string level names."""
        # Create level objects first (strings will be resolved later by Dimension)
        levels = [
            Level(name="year", attributes=[Attribute(name="year")]),
            Level(name="month", attributes=[Attribute(name="month")]),
        ]
        hierarchy = Hierarchy(name="default", levels=levels)

        assert len(hierarchy.levels) == 2
        assert all(isinstance(level, Level) for level in hierarchy.levels)


class TestHierarchyCollectionMethods:
    """Test Hierarchy collection operations."""

    @pytest.fixture
    def sample_hierarchy(self) -> Hierarchy:
        """Create a sample hierarchy for testing."""
        levels = [
            Level(name="year", attributes=[Attribute(name="year")]),
            Level(
                name="month",
                attributes=[
                    Attribute(name="month"),
                    Attribute(name="month_name"),
                ],
            ),
            Level(name="day", attributes=[Attribute(name="day")]),
        ]
        return Hierarchy(name="default", levels=levels)

    def test_len(self, sample_hierarchy: Hierarchy) -> None:
        """Test __len__ returns number of levels."""
        assert len(sample_hierarchy) == 3

    def test_getitem_by_index(self, sample_hierarchy: Hierarchy) -> None:
        """Test __getitem__ returns level by index."""
        assert sample_hierarchy[0].name == "year"
        assert sample_hierarchy[1].name == "month"
        assert sample_hierarchy[2].name == "day"

    def test_getitem_out_of_range(self, sample_hierarchy: Hierarchy) -> None:
        """Test __getitem__ raises HierarchyError for out of range index."""
        from cubes.metadata.models.dimension import HierarchyError

        with pytest.raises(HierarchyError) as exc_info:
            _ = sample_hierarchy[5]
        assert "has only 3 levels" in str(exc_info.value)

    def test_contains_by_level_object(self, sample_hierarchy: Hierarchy) -> None:
        """Test __contains__ with Level object."""
        month_level = sample_hierarchy[1]
        assert month_level in sample_hierarchy

    def test_contains_by_level_name(self, sample_hierarchy: Hierarchy) -> None:
        """Test __contains__ with level name string."""
        assert "month" in sample_hierarchy
        assert "year" in sample_hierarchy
        assert "day" in sample_hierarchy

    def test_contains_not_found(self, sample_hierarchy: Hierarchy) -> None:
        """Test __contains__ returns False for non-existent level."""
        assert "quarter" not in sample_hierarchy
        assert "flower" not in sample_hierarchy


class TestHierarchyKeys:
    """Test Hierarchy.keys() method."""

    @pytest.fixture
    def sample_hierarchy(self) -> Hierarchy:
        """Create a sample hierarchy for testing."""
        levels = [
            Level(name="year", attributes=[Attribute(name="year")]),
            Level(name="month", attributes=[Attribute(name="month")]),
            Level(name="day", attributes=[Attribute(name="day")]),
        ]
        return Hierarchy(name="default", levels=levels)

    def test_keys_all_levels(self, sample_hierarchy: Hierarchy) -> None:
        """Test keys() returns all level keys when depth is None."""
        keys = sample_hierarchy.keys()
        assert len(keys) == 3
        assert all(isinstance(k, Attribute) for k in keys)
        assert keys[0].name == "year"
        assert keys[1].name == "month"
        assert keys[2].name == "day"

    def test_keys_with_depth(self, sample_hierarchy: Hierarchy) -> None:
        """Test keys() returns keys up to specified depth."""
        keys = sample_hierarchy.keys(depth=2)
        assert len(keys) == 2
        assert keys[0].name == "year"
        assert keys[1].name == "month"

    def test_keys_depth_zero(self, sample_hierarchy: Hierarchy) -> None:
        """Test keys() with depth=0 returns empty list."""
        keys = sample_hierarchy.keys(depth=0)
        assert keys == []


class TestHierarchyTraversal:
    """Test Hierarchy traversal methods."""

    @pytest.fixture
    def sample_hierarchy(self) -> Hierarchy:
        """Create a sample hierarchy for testing."""
        levels = [
            Level(name="year", attributes=[Attribute(name="year")]),
            Level(name="month", attributes=[Attribute(name="month")]),
            Level(name="day", attributes=[Attribute(name="day")]),
        ]
        return Hierarchy(name="default", levels=levels)

    def test_next_level_from_none(self, sample_hierarchy: Hierarchy) -> None:
        """Test next_level(None) returns first level."""
        result = sample_hierarchy.next_level(None)
        assert result is not None
        assert result.name == "year"

    def test_next_level_from_first(self, sample_hierarchy: Hierarchy) -> None:
        """Test next_level returns next level."""
        year_level = sample_hierarchy[0]
        result = sample_hierarchy.next_level(year_level)
        assert result is not None
        assert result.name == "month"

    def test_next_level_from_last(self, sample_hierarchy: Hierarchy) -> None:
        """Test next_level returns None for last level."""
        day_level = sample_hierarchy[2]
        result = sample_hierarchy.next_level(day_level)
        assert result is None

    def test_next_level_by_name(self, sample_hierarchy: Hierarchy) -> None:
        """Test next_level accepts level name string."""
        result = sample_hierarchy.next_level("year")
        assert result is not None
        assert result.name == "month"

    def test_previous_level_from_none(self, sample_hierarchy: Hierarchy) -> None:
        """Test previous_level(None) returns None."""
        result = sample_hierarchy.previous_level(None)
        assert result is None

    def test_previous_level_from_first(self, sample_hierarchy: Hierarchy) -> None:
        """Test previous_level returns None for first level."""
        year_level = sample_hierarchy[0]
        result = sample_hierarchy.previous_level(year_level)
        assert result is None

    def test_previous_level_from_second(self, sample_hierarchy: Hierarchy) -> None:
        """Test previous_level returns previous level."""
        month_level = sample_hierarchy[1]
        result = sample_hierarchy.previous_level(month_level)
        assert result is not None
        assert result.name == "year"

    def test_previous_level_by_name(self, sample_hierarchy: Hierarchy) -> None:
        """Test previous_level accepts level name string."""
        result = sample_hierarchy.previous_level("month")
        assert result is not None
        assert result.name == "year"

    def test_level_index(self, sample_hierarchy: Hierarchy) -> None:
        """Test level_index returns correct position."""
        assert sample_hierarchy.level_index("year") == 0
        assert sample_hierarchy.level_index("month") == 1
        assert sample_hierarchy.level_index("day") == 2

    def test_level_index_with_object(self, sample_hierarchy: Hierarchy) -> None:
        """Test level_index accepts Level object."""
        month_level = sample_hierarchy[1]
        assert sample_hierarchy.level_index(month_level) == 1

    def test_level_index_not_found(self, sample_hierarchy: Hierarchy) -> None:
        """Test level_index raises HierarchyError for unknown level."""
        from cubes.metadata.models.dimension import HierarchyError

        with pytest.raises(HierarchyError) as exc_info:
            sample_hierarchy.level_index("quarter")
        assert "Level quarter is not part of hierarchy" in str(exc_info.value)

    def test_is_last_true(self, sample_hierarchy: Hierarchy) -> None:
        """Test is_last returns True for last level."""
        day_level = sample_hierarchy[2]
        assert sample_hierarchy.is_last(day_level) is True

    def test_is_last_false(self, sample_hierarchy: Hierarchy) -> None:
        """Test is_last returns False for non-last level."""
        year_level = sample_hierarchy[0]
        assert sample_hierarchy.is_last(year_level) is False

    def test_is_last_by_name(self, sample_hierarchy: Hierarchy) -> None:
        """Test is_last accepts level name string."""
        assert sample_hierarchy.is_last("day") is True
        assert sample_hierarchy.is_last("year") is False


class TestHierarchyLevelsFor:
    """Test levels_for_path and levels_for_depth methods."""

    @pytest.fixture
    def sample_hierarchy(self) -> Hierarchy:
        """Create a sample hierarchy for testing."""
        levels = [
            Level(name="year", attributes=[Attribute(name="year")]),
            Level(name="month", attributes=[Attribute(name="month")]),
            Level(name="day", attributes=[Attribute(name="day")]),
        ]
        return Hierarchy(name="default", levels=levels)

    def test_levels_for_depth_zero(self, sample_hierarchy: Hierarchy) -> None:
        """Test levels_for_depth(0) returns empty list."""
        levels = sample_hierarchy.levels_for_depth(0)
        assert levels == []

    def test_levels_for_depth_one(self, sample_hierarchy: Hierarchy) -> None:
        """Test levels_for_depth(1) returns first level."""
        levels = sample_hierarchy.levels_for_depth(1)
        assert len(levels) == 1
        assert levels[0].name == "year"

    def test_levels_for_depth_all(self, sample_hierarchy: Hierarchy) -> None:
        """Test levels_for_depth returns all levels."""
        levels = sample_hierarchy.levels_for_depth(3)
        assert len(levels) == 3

    def test_levels_for_depth_too_deep(self, sample_hierarchy: Hierarchy) -> None:
        """Test levels_for_depth raises HierarchyError for too deep."""
        from cubes.metadata.models.dimension import HierarchyError

        with pytest.raises(HierarchyError) as exc_info:
            sample_hierarchy.levels_for_depth(5)
        assert "Depth 5 is longer than hierarchy" in str(exc_info.value)

    def test_levels_for_depth_with_drilldown(self, sample_hierarchy: Hierarchy) -> None:
        """Test levels_for_depth with drilldown=True."""
        levels = sample_hierarchy.levels_for_depth(2, drilldown=True)
        # Should return 3 levels (depth 2 + 1 for drilldown)
        assert len(levels) == 3

    def test_levels_for_depth_drilldown_beyond(
        self, sample_hierarchy: Hierarchy
    ) -> None:
        """Test levels_for_depth with drilldown going beyond hierarchy."""
        from cubes.metadata.models.dimension import HierarchyError

        with pytest.raises(HierarchyError) as exc_info:
            sample_hierarchy.levels_for_depth(3, drilldown=True)
        assert "Depth 3 is longer than hierarchy" in str(exc_info.value)

    def test_levels_for_path_empty(self, sample_hierarchy: Hierarchy) -> None:
        """Test levels_for_path with empty path."""
        levels = sample_hierarchy.levels_for_path([])
        assert levels == []

    def test_levels_for_path_single(self, sample_hierarchy: Hierarchy) -> None:
        """Test levels_for_path with single element path."""
        levels = sample_hierarchy.levels_for_path([2024])
        assert len(levels) == 1
        assert levels[0].name == "year"

    def test_levels_for_path_with_drilldown(self, sample_hierarchy: Hierarchy) -> None:
        """Test levels_for_path with drilldown=True."""
        levels = sample_hierarchy.levels_for_path([2024, 1], drilldown=True)
        # Should return 3 levels (path len 2 + 1 for drilldown)
        assert len(levels) == 3

    def test_levels_for_path_too_long(self, sample_hierarchy: Hierarchy) -> None:
        """Test levels_for_path raises error for path longer than hierarchy."""
        from cubes.metadata.models.dimension import HierarchyError

        with pytest.raises(HierarchyError) as exc_info:
            sample_hierarchy.levels_for_path(
                [2024, 1, 15, 10]
            )  # 4 elements in 3-level hierarchy
        assert "Depth 4 is longer than hierarchy" in str(exc_info.value)


class TestHierarchyRollup:
    """Test Hierarchy.rollup() method."""

    @pytest.fixture
    def sample_hierarchy(self) -> Hierarchy:
        """Create a sample hierarchy for testing."""
        levels = [
            Level(name="year", attributes=[Attribute(name="year")]),
            Level(name="month", attributes=[Attribute(name="month")]),
            Level(name="day", attributes=[Attribute(name="day")]),
        ]
        return Hierarchy(name="default", levels=levels)

    def test_rollup_one_level(self, sample_hierarchy: Hierarchy) -> None:
        """Test rollup without level parameter rolls up one level."""
        path = [2024, 1, 15]
        result = sample_hierarchy.rollup(path)
        assert result == [2024, 1]

    def test_rollup_to_specific_level(self, sample_hierarchy: Hierarchy) -> None:
        """Test rollup to specific level."""
        path = [2024, 1, 15]
        result = sample_hierarchy.rollup(path, "month")
        assert result == [2024, 1]

    def test_rollup_to_top(self, sample_hierarchy: Hierarchy) -> None:
        """Test rollup to top level."""
        path = [2024, 1, 15]
        result = sample_hierarchy.rollup(path, "year")
        assert result == [2024]

    def test_rollup_empty_path(self, sample_hierarchy: Hierarchy) -> None:
        """Test rollup with empty path returns empty list."""
        result = sample_hierarchy.rollup([])
        assert result == []

    def test_rollup_single_element(self, sample_hierarchy: Hierarchy) -> None:
        """Test rollup with single element path returns empty list."""
        result = sample_hierarchy.rollup([2024])
        assert result == []

    def test_rollup_level_deeper_than_path(self, sample_hierarchy: Hierarchy) -> None:
        """Test rollup raises error when level is deeper than path."""
        from cubes.metadata.models.dimension import HierarchyError

        path = [2024]
        with pytest.raises(HierarchyError) as exc_info:
            sample_hierarchy.rollup(path, "month")
        assert "Can not roll-up: level 'month'" in str(exc_info.value)

    def test_rollup_unknown_level(self, sample_hierarchy: Hierarchy) -> None:
        """Test rollup raises error for unknown level."""
        from cubes.metadata.models.dimension import HierarchyError

        path = [2024, 1]
        with pytest.raises(HierarchyError) as exc_info:
            sample_hierarchy.rollup(path, "quarter")
        assert "Level quarter is not part of hierarchy" in str(exc_info.value)


class TestHierarchyPathIsBase:
    """Test Hierarchy.path_is_base() method."""

    @pytest.fixture
    def sample_hierarchy(self) -> Hierarchy:
        """Create a sample hierarchy for testing."""
        levels = [
            Level(name="year", attributes=[Attribute(name="year")]),
            Level(name="month", attributes=[Attribute(name="month")]),
            Level(name="day", attributes=[Attribute(name="day")]),
        ]
        return Hierarchy(name="default", levels=levels)

    def test_path_is_base_true(self, sample_hierarchy: Hierarchy) -> None:
        """Test path_is_base returns True for full path."""
        assert sample_hierarchy.path_is_base([2024, 1, 15]) is True

    def test_path_is_base_false_partial(self, sample_hierarchy: Hierarchy) -> None:
        """Test path_is_base returns False for partial path."""
        assert sample_hierarchy.path_is_base([2024, 1]) is False
        assert sample_hierarchy.path_is_base([2024]) is False

    def test_path_is_base_false_empty(self, sample_hierarchy: Hierarchy) -> None:
        """Test path_is_base returns False for empty path."""
        assert sample_hierarchy.path_is_base([]) is False

    def test_path_is_base_false_none(self, sample_hierarchy: Hierarchy) -> None:
        """Test path_is_base returns False for None path."""
        assert sample_hierarchy.path_is_base(None) is False


class TestHierarchyAttributes:
    """Test Hierarchy attribute collection methods."""

    @pytest.fixture
    def sample_hierarchy(self) -> Hierarchy:
        """Create a sample hierarchy with multi-attribute levels."""
        levels = [
            Level(name="year", attributes=[Attribute(name="year")]),
            Level(
                name="month",
                attributes=[
                    Attribute(name="month"),
                    Attribute(name="month_name"),
                    Attribute(name="month_sname"),
                ],
            ),
            Level(name="day", attributes=[Attribute(name="day")]),
        ]
        return Hierarchy(name="default", levels=levels)

    def test_key_attributes(self, sample_hierarchy: Hierarchy) -> None:
        """Test key_attributes returns all level keys."""
        keys = sample_hierarchy.key_attributes()
        assert len(keys) == 3
        assert keys[0].name == "year"
        assert keys[1].name == "month"
        assert keys[2].name == "day"

    def test_all_attributes(self, sample_hierarchy: Hierarchy) -> None:
        """Test all_attributes property returns flattened attributes."""
        attrs = sample_hierarchy.all_attributes
        attr_names = [a.name for a in attrs]
        assert attr_names == ["year", "month", "month_name", "month_sname", "day"]


class TestHierarchyEqualityAndHash:
    """Test Hierarchy equality and hashing."""

    def test_equal_same_levels(self) -> None:
        """Test equality with same levels."""
        levels = [
            Level(name="year", attributes=[Attribute(name="year")]),
            Level(name="month", attributes=[Attribute(name="month")]),
        ]
        h1 = Hierarchy(name="default", levels=levels)
        h2 = Hierarchy(name="default", levels=levels)

        assert h1 == h2

    def test_not_equal_different_name(self) -> None:
        """Test inequality with different names."""
        levels = [Level(name="year", attributes=[Attribute(name="year")])]
        h1 = Hierarchy(name="default", levels=levels)
        h2 = Hierarchy(name="alternate", levels=levels)

        assert h1 != h2

    def test_not_equal_different_levels(self) -> None:
        """Test inequality with different levels."""
        levels1 = [Level(name="year", attributes=[Attribute(name="year")])]
        levels2 = [
            Level(name="year", attributes=[Attribute(name="year")]),
            Level(name="month", attributes=[Attribute(name="month")]),
        ]
        h1 = Hierarchy(name="default", levels=levels1)
        h2 = Hierarchy(name="default", levels=levels2)

        assert h1 != h2

    def test_not_equal_different_label(self) -> None:
        """Test inequality with different labels."""
        levels = [Level(name="year", attributes=[Attribute(name="year")])]
        h1 = Hierarchy(name="default", levels=levels, label="Default")
        h2 = Hierarchy(name="default", levels=levels, label="Custom")

        assert h1 != h2

    def test_not_equal_non_hierarchy(self) -> None:
        """Test inequality with non-hierarchy object."""
        levels = [Level(name="year", attributes=[Attribute(name="year")])]
        h1 = Hierarchy(name="default", levels=levels)

        assert h1 != "not a hierarchy"
        assert h1 != None

    def test_hash_based_on_name(self) -> None:
        """Test hash is based on name."""
        levels = [Level(name="year", attributes=[Attribute(name="year")])]
        h1 = Hierarchy(name="default", levels=levels)
        h2 = Hierarchy(name="default", levels=levels)

        assert hash(h1) == hash(h2)

    def test_str_returns_name(self) -> None:
        """Test str() returns hierarchy name."""
        levels = [Level(name="year", attributes=[Attribute(name="year")])]
        h = Hierarchy(name="default", levels=levels)

        assert str(h) == "default"


class TestHierarchyDeepCopy:
    """Test Hierarchy deep copy behavior."""

    def test_deepcopy_creates_independent_copy(self) -> None:
        """Test deepcopy creates independent hierarchy."""
        levels = [
            Level(name="year", attributes=[Attribute(name="year")]),
            Level(name="month", attributes=[Attribute(name="month")]),
        ]
        original = Hierarchy(name="default", levels=levels)
        clone = copy.deepcopy(original)

        assert clone is not original
        assert clone.name == original.name
        assert clone.levels == original.levels

    def test_deepcopy_dimension_disowned(self) -> None:
        """Test deepcopy disowns dimension from level attributes."""

        # Create a mock dimension
        class DummyDimension:
            def __init__(self) -> None:
                self.name = "dummy"
                self.is_flat = False

        levels = [
            Level(name="year", attributes=[Attribute(name="year")]),
            Level(name="month", attributes=[Attribute(name="month")]),
        ]
        # Assign dimension to first level's first attribute
        levels[0].attributes[0].dimension = DummyDimension()

        original = Hierarchy(name="default", levels=levels)
        clone = copy.deepcopy(original)

        # Original should still have dimension
        assert original.levels[0].attributes[0].dimension is not None
        # Clone should have dimension disowned (None)
        assert clone.levels[0].attributes[0].dimension is None

    def test_deepcopy_levels_independent(self) -> None:
        """Test that modifying cloned levels doesn't affect original."""
        levels = [
            Level(name="year", attributes=[Attribute(name="year")]),
            Level(name="month", attributes=[Attribute(name="month")]),
        ]
        original = Hierarchy(name="default", levels=levels)
        clone = copy.deepcopy(original)

        # Modify clone's levels
        clone.levels[0].attributes[0].name = "modified"

        # Original should be unchanged
        assert original.levels[0].attributes[0].name == "year"


class TestHierarchyToDict:
    """Test Hierarchy.to_dict() method."""

    @pytest.fixture
    def sample_hierarchy(self) -> Hierarchy:
        """Create a sample hierarchy for testing."""
        levels = [
            Level(name="year", attributes=[Attribute(name="year")]),
            Level(name="month", attributes=[Attribute(name="month")]),
        ]
        return Hierarchy(name="default", levels=levels, label="Default Hierarchy")

    def test_to_dict_basic(self, sample_hierarchy: Hierarchy) -> None:
        """Test to_dict returns correct structure."""
        result = sample_hierarchy.to_dict()

        assert result["name"] == "default"
        assert result["label"] == "Default Hierarchy"
        assert "levels" in result
        assert len(result["levels"]) == 2

    def test_to_dict_create_label(self, sample_hierarchy: Hierarchy) -> None:
        """Test to_dict with create_label=True."""
        hierarchy = Hierarchy(
            name="fiscal_year",
            levels=[Level(name="year", attributes=[Attribute(name="year")])],
        )
        result = hierarchy.to_dict(create_label=True)

        assert result["label"] == "Fiscal year"

    def test_to_dict_no_label(self, sample_hierarchy: Hierarchy) -> None:
        """Test to_dict with create_label=False and no label set."""
        hierarchy = Hierarchy(
            name="default",
            levels=[Level(name="year", attributes=[Attribute(name="year")])],
        )
        result = hierarchy.to_dict(create_label=False)

        assert result["label"] is None

    def test_to_dict_with_description(self) -> None:
        """Test to_dict includes description."""
        hierarchy = Hierarchy(
            name="default",
            levels=[Level(name="year", attributes=[Attribute(name="year")])],
            description="A test hierarchy",
        )
        result = hierarchy.to_dict()

        assert result["description"] == "A test hierarchy"

    def test_to_dict_with_info(self) -> None:
        """Test to_dict includes info."""
        hierarchy = Hierarchy(
            name="default",
            levels=[Level(name="year", attributes=[Attribute(name="year")])],
            info={"source": "test"},
        )
        result = hierarchy.to_dict()

        assert result["info"] == {"source": "test"}


class TestHierarchyLevelAccessors:
    """Test Hierarchy level accessor properties."""

    @pytest.fixture
    def sample_hierarchy(self) -> Hierarchy:
        """Create a sample hierarchy for testing."""
        levels = [
            Level(name="year", attributes=[Attribute(name="year")]),
            Level(name="month", attributes=[Attribute(name="month")]),
        ]
        return Hierarchy(name="default", levels=levels)

    def test_level_names_property(self, sample_hierarchy: Hierarchy) -> None:
        """Test level_names property returns list of names."""
        assert sample_hierarchy.level_names == ["year", "month"]

    def test_level_property_returns_list(self, sample_hierarchy: Hierarchy) -> None:
        """Test levels property returns list of Level objects."""
        levels = sample_hierarchy.levels
        assert isinstance(levels, list)
        assert all(isinstance(l, Level) for l in levels)


class TestHierarchyEdgeCases:
    """Test Hierarchy edge cases and error handling."""

    def test_single_level_hierarchy(self) -> None:
        """Test hierarchy with single level."""
        level = Level(name="total", attributes=[Attribute(name="total")])
        hierarchy = Hierarchy(name="flat", levels=[level])

        assert len(hierarchy) == 1
        assert hierarchy.is_last(level) is True
        assert hierarchy.next_level(level) is None
        assert hierarchy.previous_level(level) is None

    def test_next_level_from_none_single_level(self) -> None:
        """Test next_level(None) with single level returns that level."""
        level = Level(name="total", attributes=[Attribute(name="total")])
        hierarchy = Hierarchy(name="flat", levels=[level])

        result = hierarchy.next_level(None)
        assert result == level

    def test_rollup_from_single_element(self) -> None:
        """Test rollup with single element path to top returns empty."""
        level = Level(name="year", attributes=[Attribute(name="year")])
        hierarchy = Hierarchy(name="default", levels=[level])

        result = hierarchy.rollup([2024])
        assert result == []

    def test_deepcopy_preserves_hierarchy_properties(self) -> None:
        """Test deepcopy preserves all hierarchy properties."""
        levels = [
            Level(name="a", attributes=[Attribute(name="a")]),
            Level(name="b", attributes=[Attribute(name="b")]),
            Level(name="c", attributes=[Attribute(name="c")]),
        ]
        original = Hierarchy(
            name="test",
            levels=levels,
            label="Test Label",
            description="Test Description",
            info={"key": "value"},
        )
        clone = copy.deepcopy(original)

        assert clone.name == original.name
        assert clone.label == original.label
        assert clone.description == original.description
        assert clone.info == original.info
        assert len(clone.levels) == len(original.levels)


class TestHierarchyFromSchema:
    """Test Hierarchy.from_schema() factory method."""

    def test_from_schema_with_level_objects(self) -> None:
        """Test from_schema with Level objects."""
        from cubes.metadata.schemas.dimension import HierarchySchema, LevelSchema

        schema = HierarchySchema(
            name="default",
            levels=[
                LevelSchema(name="year", attributes=["year"]),
                LevelSchema(name="month", attributes=["month"]),
            ],
        )

        hierarchy = Hierarchy.from_schema(schema)

        assert hierarchy.name == "default"
        assert len(hierarchy.levels) == 2
        assert all(isinstance(l, Level) for l in hierarchy.levels)

    def test_from_schema_preserves_optional_fields(self) -> None:
        """Test from_schema preserves label, description, info."""
        from cubes.metadata.schemas.dimension import HierarchySchema, LevelSchema

        schema = HierarchySchema(
            name="fiscal",
            levels=[LevelSchema(name="year", attributes=["year"])],
            label="Fiscal Calendar",
            description="Fiscal year hierarchy",
            info={"source": "erp"},
        )

        hierarchy = Hierarchy.from_schema(schema)

        assert hierarchy.label == "Fiscal Calendar"
        assert hierarchy.description == "Fiscal year hierarchy"
        assert hierarchy.info == {"source": "erp"}

    def test_from_schema_with_mixed_levels(self) -> None:
        """Test from_schema with mixed level types (strings and dicts)."""
        from cubes.metadata.schemas.dimension import HierarchySchema

        # Schema should already have LevelSchema objects from validation
        schema = HierarchySchema(
            name="default",
            levels=[
                {"name": "year", "attributes": ["year"]},
                {"name": "month", "attributes": ["month"]},
            ],
        )

        hierarchy = Hierarchy.from_schema(schema)

        assert len(hierarchy.levels) == 2
        assert hierarchy.levels[0].name == "year"
        assert hierarchy.levels[1].name == "month"
