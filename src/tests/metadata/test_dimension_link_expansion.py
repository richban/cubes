"""Tests for dimension link expansion utilities.

Tests for expand_dimension_links() and expand_cube_metadata() functions
that handle preprocessing and normalization of cube metadata.
"""

import pytest
import warnings

from cubes.metadata.schemas.cube import (
    expand_cube_metadata,
    expand_dimension_links,
)


class TestExpandDimensionLinks:
    """Tests for expand_dimension_links() function."""

    def test_expand_single_string(self) -> None:
        """Test expanding a single string dimension name."""
        result = expand_dimension_links(["date"])
        assert result == [{"name": "date"}]

    def test_expand_multiple_strings(self) -> None:
        """Test expanding multiple string dimension names."""
        result = expand_dimension_links(["date", "product", "region"])
        assert result == [
            {"name": "date"},
            {"name": "product"},
            {"name": "region"},
        ]

    def test_expand_empty_list(self) -> None:
        """Test expanding an empty list."""
        result = expand_dimension_links([])
        assert result == []

    def test_expand_dict_with_name(self) -> None:
        """Test expanding a dict with just a name field."""
        result = expand_dimension_links([{"name": "date"}])
        assert result == [{"name": "date"}]

    def test_expand_dict_with_hierarchies(self) -> None:
        """Test expanding a dict with hierarchies field."""
        result = expand_dimension_links(
            [{"name": "date", "hierarchies": ["ymd", "fiscal"]}]
        )
        assert result == [{"name": "date", "hierarchies": ["ymd", "fiscal"]}]

    def test_expand_dict_with_exclude_hierarchies(self) -> None:
        """Test expanding a dict with exclude_hierarchies field."""
        result = expand_dimension_links(
            [{"name": "date", "exclude_hierarchies": ["fiscal"]}]
        )
        assert result == [{"name": "date", "exclude_hierarchies": ["fiscal"]}]

    def test_expand_dict_with_default_hierarchy_name(self) -> None:
        """Test expanding a dict with default_hierarchy_name field."""
        result = expand_dimension_links(
            [{"name": "date", "default_hierarchy_name": "ymd"}]
        )
        assert result == [{"name": "date", "default_hierarchy_name": "ymd"}]

    def test_expand_dict_with_nonadditive(self) -> None:
        """Test expanding a dict with nonadditive field."""
        result = expand_dimension_links([{"name": "date", "nonadditive": "time"}])
        assert result == [{"name": "date", "nonadditive": "time"}]

    def test_expand_dict_with_cardinality(self) -> None:
        """Test expanding a dict with cardinality field."""
        result = expand_dimension_links([{"name": "date", "cardinality": "high"}])
        assert result == [{"name": "date", "cardinality": "high"}]

    def test_expand_dict_with_alias(self) -> None:
        """Test expanding a dict with alias field."""
        result = expand_dimension_links([{"name": "date", "alias": "transaction_date"}])
        assert result == [{"name": "date", "alias": "transaction_date"}]

    def test_expand_mixed_strings_and_dicts(self) -> None:
        """Test expanding a mix of strings and dicts."""
        result = expand_dimension_links(
            [
                "date",
                {"name": "product", "hierarchies": ["category"]},
                "region",
            ]
        )
        assert result == [
            {"name": "date"},
            {"name": "product", "hierarchies": ["category"]},
            {"name": "region"},
        ]

    def test_expand_preserves_extra_fields(self) -> None:
        """Test that extra fields in dict links are preserved."""
        result = expand_dimension_links(
            [
                {
                    "name": "date",
                    "custom_field": "custom_value",
                    "another": 123,
                }
            ]
        )
        assert result[0]["name"] == "date"
        assert result[0]["custom_field"] == "custom_value"
        assert result[0]["another"] == 123

    def test_expand_dict_without_name_raises_error(self) -> None:
        """Test that a dict without 'name' field raises ValueError."""
        with pytest.raises(ValueError, match="Dimension link has no name"):
            expand_dimension_links([{"hierarchies": ["ymd"]}])

    def test_expand_invalid_type_raises_error(self) -> None:
        """Test that invalid types raise TypeError."""
        with pytest.raises(TypeError, match="Dimension link must be string or dict"):
            expand_dimension_links([123])  # type: ignore[list-item]

    def test_expand_none_raises_error(self) -> None:
        """Test that None raises TypeError."""
        with pytest.raises(TypeError, match="Dimension link must be string or dict"):
            expand_dimension_links([None])  # type: ignore[list-item]

    def test_expand_does_not_mutate_original(self) -> None:
        """Test that original dicts are not mutated."""
        original = {"name": "date", "hierarchies": ["ymd"]}
        expand_dimension_links([original])
        # Original should be unchanged
        assert original == {"name": "date", "hierarchies": ["ymd"]}


class TestExpandCubeMetadata:
    """Tests for expand_cube_metadata() function."""

    def test_expand_basic_cube(self) -> None:
        """Test expanding a basic cube with dimensions."""
        metadata = {
            "name": "sales",
            "dimensions": ["date", "product"],
        }
        result = expand_cube_metadata(metadata)
        assert result["name"] == "sales"
        assert result["dimensions"] == [
            {"name": "date"},
            {"name": "product"},
        ]

    def test_expand_cube_without_dimensions(self) -> None:
        """Test expanding a cube without dimensions."""
        metadata = {"name": "sales"}
        result = expand_cube_metadata(metadata)
        assert result["name"] == "sales"
        assert "dimensions" not in result

    def test_expand_cube_with_empty_dimensions(self) -> None:
        """Test expanding a cube with empty dimensions list."""
        metadata = {"name": "sales", "dimensions": []}
        result = expand_cube_metadata(metadata)
        assert result["name"] == "sales"
        assert result["dimensions"] == []

    def test_expand_cube_no_name_raises_error(self) -> None:
        """Test that cube without name raises ValueError."""
        with pytest.raises(ValueError, match="Cube has no name"):
            expand_cube_metadata({"dimensions": ["date"]})

    def test_expand_preserves_other_fields(self) -> None:
        """Test that non-dimension fields are preserved."""
        metadata = {
            "name": "sales",
            "dimensions": ["date"],
            "measures": [{"name": "amount"}],
            "aggregates": [{"name": "total", "function": "sum"}],
            "label": "Sales Cube",
            "description": "A sales cube",
            "category": "financial",
        }
        result = expand_cube_metadata(metadata)
        assert result["name"] == "sales"
        assert result["label"] == "Sales Cube"
        assert result["description"] == "A sales cube"
        assert result["category"] == "financial"
        assert result["measures"] == [{"name": "amount"}]
        assert result["aggregates"] == [{"name": "total", "function": "sum"}]

    def test_expand_with_legacy_hierarchies_deprecation_warning(self) -> None:
        """Test that legacy hierarchies format triggers deprecation warning."""
        metadata = {
            "name": "sales",
            "dimensions": ["date", "product"],
            "hierarchies": {
                "date": ["ymd", "fiscal"],
            },
        }
        with pytest.warns(DeprecationWarning, match="hierarchies.*deprecated"):
            result = expand_cube_metadata(metadata)

        # Hierarchies should be moved to dimension links
        date_link = next(
            link for link in result["dimensions"] if link["name"] == "date"
        )
        assert date_link["hierarchies"] == ["ymd", "fiscal"]

    def test_expand_legacy_hierarchies_for_nonlinked_raises_error(self) -> None:
        """Test that hierarchies for non-linked dimensions raises error."""
        metadata = {
            "name": "sales",
            "dimensions": ["date"],
            "hierarchies": {
                "date": ["ymd"],
                "product": ["category"],  # product not in dimensions
            },
        }
        with pytest.warns(DeprecationWarning):
            with pytest.raises(
                ValueError, match="hierarchies specified for non-linked"
            ):
                expand_cube_metadata(metadata)

    def test_expand_legacy_hierarchies_removed_from_result(self) -> None:
        """Test that legacy hierarchies key is removed from result."""
        metadata = {
            "name": "sales",
            "dimensions": ["date"],
            "hierarchies": {
                "date": ["ymd"],
            },
        }
        with pytest.warns(DeprecationWarning):
            result = expand_cube_metadata(metadata)

        # Hierarchies key should still be in metadata (it's not removed)
        assert "hierarchies" in result

    def test_expand_nonadditive_propagation_to_measures(self) -> None:
        """Test that nonadditive is propagated from cube to measures."""
        metadata = {
            "name": "sales",
            "dimensions": ["date"],
            "nonadditive": "time",
            "measures": [
                {"name": "amount"},
                {"name": "quantity", "nonadditive": "all"},
            ],
        }
        result = expand_cube_metadata(metadata)

        # First measure should inherit nonadditive
        assert result["measures"][0]["nonadditive"] == "time"
        # Second measure should keep its own nonadditive
        assert result["measures"][1]["nonadditive"] == "all"

    def test_expand_nonadditive_not_propagated_if_no_measures(self) -> None:
        """Test that nonadditive is not added if no measures."""
        metadata = {
            "name": "sales",
            "dimensions": ["date"],
            "nonadditive": "time",
        }
        result = expand_cube_metadata(metadata)
        assert "measures" not in result

    def test_expand_nonadditive_removed_from_metadata(self) -> None:
        """Test that nonadditive is removed from top-level metadata."""
        metadata = {
            "name": "sales",
            "dimensions": ["date"],
            "nonadditive": "time",
            "measures": [{"name": "amount"}],
        }
        result = expand_cube_metadata(metadata)
        # nonadditive should be popped from metadata
        assert "nonadditive" not in result

    def test_expand_does_not_mutate_original(self) -> None:
        """Test that original metadata dict is not mutated."""
        original = {
            "name": "sales",
            "dimensions": ["date", "product"],
            "measures": [{"name": "amount"}],
        }
        original_copy = {
            "name": "sales",
            "dimensions": ["date", "product"],
            "measures": [{"name": "amount"}],
        }
        expand_cube_metadata(original)
        # Original should be unchanged
        assert original == original_copy

    def test_expand_complex_cube(self) -> None:
        """Test expanding a complex cube with all features."""
        metadata = {
            "name": "sales",
            "label": "Sales Analytics",
            "description": "Sales data cube",
            "category": "sales",
            "dimensions": [
                "date",
                {"name": "product", "hierarchies": ["category", "brand"]},
                {"name": "region", "alias": "sales_region"},
            ],
            "measures": [
                {"name": "amount"},
                {"name": "quantity"},
            ],
            "aggregates": [
                {"name": "total_amount", "measure": "amount", "function": "sum"},
            ],
            "mappings": {
                "amount": "sales.amount",
            },
            "fact": "sales_fact",
        }
        result = expand_cube_metadata(metadata)

        assert result["name"] == "sales"
        assert result["label"] == "Sales Analytics"
        assert result["dimensions"] == [
            {"name": "date"},
            {"name": "product", "hierarchies": ["category", "brand"]},
            {"name": "region", "alias": "sales_region"},
        ]
        assert len(result["measures"]) == 2
        assert result["fact"] == "sales_fact"

    def test_expand_with_all_link_options(self) -> None:
        """Test expanding dimensions with all link options."""
        metadata = {
            "name": "sales",
            "dimensions": [
                {
                    "name": "date",
                    "hierarchies": ["ymd"],
                    "exclude_hierarchies": ["fiscal"],
                    "default_hierarchy_name": "ymd",
                    "nonadditive": "time",
                    "cardinality": "high",
                    "alias": "transaction_date",
                }
            ],
        }
        result = expand_cube_metadata(metadata)

        link = result["dimensions"][0]
        assert link["name"] == "date"
        assert link["hierarchies"] == ["ymd"]
        assert link["exclude_hierarchies"] == ["fiscal"]
        assert link["default_hierarchy_name"] == "ymd"
        assert link["nonadditive"] == "time"
        assert link["cardinality"] == "high"
        assert link["alias"] == "transaction_date"
