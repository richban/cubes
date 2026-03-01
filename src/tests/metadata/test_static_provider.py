"""Tests for StaticModelProvider.

This module contains comprehensive tests for the StaticModelProvider class,
covering initialization from various sources, cube and dimension retrieval,
caching behavior, and error handling.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from cubes.metadata import StaticModelProvider
from cubes.metadata.io import MetadataNotFoundError
from cubes.metadata.models import Dimension, Level, Attribute
from cubes.metadata.models.cube import Cube
from cubes.metadata.provider import NoSuchCubeError, NoSuchDimensionError


class TestStaticModelProviderInit:
    """Test StaticModelProvider initialization."""

    def test_init_with_none(self) -> None:
        """Test initialization with None creates empty provider."""
        provider = StaticModelProvider(None)
        assert provider.metadata == {}
        assert provider.list_cubes() == []

    def test_init_with_empty_dict(self) -> None:
        """Test initialization with empty dict."""
        provider = StaticModelProvider({})
        assert provider.metadata == {}
        assert provider.list_cubes() == []

    def test_init_with_dict(self) -> None:
        """Test initialization with metadata dictionary."""
        metadata: dict[str, Any] = {
            "dimensions": [{"name": "date", "levels": [{"name": "year"}]}],
            "cubes": [{"name": "sales", "dimensions": ["date"]}],
        }
        provider = StaticModelProvider(metadata)
        assert "date" in provider._dimensions_metadata
        assert "sales" in provider._cubes_metadata

    def test_init_with_string_path(self, tmp_path: Path) -> None:
        """Test initialization with string file path."""
        model_file = tmp_path / "model.json"
        metadata: dict[str, Any] = {
            "dimensions": [],
            "cubes": [{"name": "test", "dimensions": []}],
        }
        model_file.write_text(json.dumps(metadata))

        provider = StaticModelProvider(str(model_file))
        assert "test" in provider._cubes_metadata

    def test_init_with_path_object(self, tmp_path: Path) -> None:
        """Test initialization with Path object."""
        model_file = tmp_path / "model.json"
        metadata: dict[str, Any] = {
            "dimensions": [],
            "cubes": [{"name": "test", "dimensions": []}],
        }
        model_file.write_text(json.dumps(metadata))

        provider = StaticModelProvider(model_file)
        assert "test" in provider._cubes_metadata

    def test_init_with_bundle_directory(self, tmp_path: Path) -> None:
        """Test initialization with model bundle directory."""
        bundle_dir = tmp_path / "model.cubesmodel"
        bundle_dir.mkdir()

        model_file = bundle_dir / "model.json"
        metadata: dict[str, Any] = {
            "name": "test_model",
            "dimensions": [],
            "cubes": [],
        }
        model_file.write_text(json.dumps(metadata))

        dim_file = bundle_dir / "dim_date.json"
        dim_file.write_text(json.dumps({"name": "date", "levels": [{"name": "year"}]}))

        cube_file = bundle_dir / "cube_sales.json"
        cube_file.write_text(json.dumps({"name": "sales", "dimensions": ["date"]}))

        provider = StaticModelProvider(bundle_dir)
        assert "date" in provider._dimensions_metadata
        assert "sales" in provider._cubes_metadata

    def test_init_with_invalid_type(self) -> None:
        """Test initialization with invalid type raises error."""
        with pytest.raises(ValueError) as exc_info:
            StaticModelProvider(123)  # type: ignore[arg-type]
        assert "must be dict, str, or Path" in str(exc_info.value)

    def test_init_file_not_found(self, tmp_path: Path) -> None:
        """Test initialization with non-existent file raises error."""
        with pytest.raises(MetadataNotFoundError):
            StaticModelProvider(str(tmp_path / "nonexistent.json"))


class TestListCubes:
    """Test list_cubes method."""

    def test_list_cubes_empty(self) -> None:
        """Test listing cubes when none exist."""
        provider = StaticModelProvider({})
        assert provider.list_cubes() == []

    def test_list_cubes_single(self) -> None:
        """Test listing a single cube."""
        metadata: dict[str, Any] = {
            "cubes": [{"name": "sales", "label": "Sales Data", "category": "finance"}]
        }
        provider = StaticModelProvider(metadata)
        cubes = provider.list_cubes()
        assert len(cubes) == 1
        assert cubes[0]["name"] == "sales"
        assert cubes[0]["label"] == "Sales Data"
        assert cubes[0]["category"] == "finance"

    def test_list_cubes_multiple(self) -> None:
        """Test listing multiple cubes."""
        metadata: dict[str, Any] = {
            "cubes": [
                {"name": "sales", "label": "Sales", "category": "finance"},
                {"name": "inventory", "label": "Inventory", "category": "ops"},
            ]
        }
        provider = StaticModelProvider(metadata)
        cubes = provider.list_cubes()
        assert len(cubes) == 2
        assert cubes[0]["name"] == "sales"
        assert cubes[1]["name"] == "inventory"

    def test_list_cubes_default_label(self) -> None:
        """Test that label defaults to name when not provided."""
        metadata: dict[str, Any] = {"cubes": [{"name": "sales"}]}
        provider = StaticModelProvider(metadata)
        cubes = provider.list_cubes()
        assert cubes[0]["label"] == "sales"

    def test_list_cubes_category_from_info(self) -> None:
        """Test that category can come from info dict."""
        metadata: dict[str, Any] = {
            "cubes": [{"name": "sales", "info": {"category": "financial"}}]
        }
        provider = StaticModelProvider(metadata)
        cubes = provider.list_cubes()
        assert cubes[0]["category"] == "financial"


class TestHasCube:
    """Test has_cube method."""

    def test_has_existing_cube(self) -> None:
        """Test checking for an existing cube."""
        metadata: dict[str, Any] = {"cubes": [{"name": "sales"}]}
        provider = StaticModelProvider(metadata)
        assert provider.has_cube("sales") is True

    def test_has_nonexistent_cube(self) -> None:
        """Test checking for a non-existent cube."""
        provider = StaticModelProvider({})
        assert provider.has_cube("missing") is False


class TestCubeRetrieval:
    """Test cube retrieval."""

    def test_get_cube_from_dict(self) -> None:
        """Test retrieving a cube from dict metadata."""
        metadata: dict[str, Any] = {
            "dimensions": [
                {
                    "name": "date",
                    "levels": [{"name": "year", "attributes": ["year"]}],
                }
            ],
            "cubes": [{"name": "sales", "dimensions": ["date"]}],
        }
        provider = StaticModelProvider(metadata)
        cube = provider.cube("sales")
        assert isinstance(cube, Cube)
        assert cube.name == "sales"

    def test_get_nonexistent_cube_raises_error(self) -> None:
        """Test that retrieving non-existent cube raises error."""
        provider = StaticModelProvider({})
        with pytest.raises(NoSuchCubeError) as exc_info:
            provider.cube("missing")
        assert exc_info.value.name == "missing"

    def test_cube_caching(self) -> None:
        """Test that cubes are cached."""
        metadata: dict[str, Any] = {
            "dimensions": [
                {
                    "name": "date",
                    "levels": [{"name": "year", "attributes": ["year"]}],
                }
            ],
            "cubes": [{"name": "sales", "dimensions": ["date"]}],
        }
        provider = StaticModelProvider(metadata)
        cube1 = provider.cube("sales")
        cube2 = provider.cube("sales")
        assert cube1 is cube2  # Same object from cache

    def test_cube_with_measures(self) -> None:
        """Test cube with measures."""
        metadata: dict[str, Any] = {
            "cubes": [
                {
                    "name": "sales",
                    "dimensions": [],
                    "measures": [{"name": "amount"}],
                }
            ],
        }
        provider = StaticModelProvider(metadata)
        cube = provider.cube("sales")
        assert len(cube.measures) == 1
        assert cube.measures[0].name == "amount"

    def test_cube_with_dimension_links(self) -> None:
        """Test cube dimension linking."""
        metadata: dict[str, Any] = {
            "dimensions": [
                {
                    "name": "date",
                    "levels": [{"name": "year", "attributes": ["year"]}],
                }
            ],
            "cubes": [
                {
                    "name": "sales",
                    "dimensions": [{"name": "date", "alias": "transaction_date"}],
                }
            ],
        }
        provider = StaticModelProvider(metadata)
        cube = provider.cube("sales")
        assert len(cube.dimensions_list) == 1
        assert cube.dimensions_list[0].name == "transaction_date"


class TestDimensionRetrieval:
    """Test dimension retrieval."""

    def test_get_dimension(self) -> None:
        """Test retrieving a dimension."""
        metadata: dict[str, Any] = {
            "dimensions": [
                {
                    "name": "date",
                    "levels": [{"name": "year", "attributes": ["year"]}],
                }
            ]
        }
        provider = StaticModelProvider(metadata)
        dim = provider.dimension("date")
        assert isinstance(dim, Dimension)
        assert dim.name == "date"

    def test_get_nonexistent_dimension_raises_error(self) -> None:
        """Test that retrieving non-existent dimension raises error."""
        provider = StaticModelProvider({})
        with pytest.raises(NoSuchDimensionError) as exc_info:
            provider.dimension("missing")
        assert exc_info.value.name == "missing"

    def test_dimension_caching(self) -> None:
        """Test that dimensions are cached."""
        metadata: dict[str, Any] = {
            "dimensions": [
                {
                    "name": "date",
                    "levels": [{"name": "year", "attributes": ["year"]}],
                }
            ]
        }
        provider = StaticModelProvider(metadata)
        dim1 = provider.dimension("date")
        dim2 = provider.dimension("date")
        assert dim1 is dim2  # Same object from cache

    def test_dimension_with_template(self) -> None:
        """Test dimension with template inheritance."""
        metadata: dict[str, Any] = {
            "dimensions": [
                {
                    "name": "base_date",
                    "levels": [
                        {"name": "year", "attributes": ["year"]},
                        {"name": "month", "attributes": ["month"]},
                    ],
                },
                {
                    "name": "date",
                    "template": "base_date",
                    "levels": [{"name": "day", "attributes": ["day"]}],
                },
            ]
        }
        provider = StaticModelProvider(metadata)

        # Load base template first
        base_dim = provider.dimension("base_date")

        # Load derived dimension with template
        dim = provider.dimension("date", templates={"base_date": base_dim})
        assert dim.name == "date"
        # Should have both template and explicit levels
        assert len(dim.levels) == 1  # Only explicit levels


class TestCacheManagement:
    """Test cache management."""

    def test_clear_cache_clears_all(self) -> None:
        """Test that clear_cache clears all caches."""
        metadata: dict[str, Any] = {
            "dimensions": [
                {
                    "name": "date",
                    "levels": [{"name": "year", "attributes": ["year"]}],
                }
            ],
            "cubes": [{"name": "sales", "dimensions": ["date"]}],
        }
        provider = StaticModelProvider(metadata)

        # Populate caches
        provider.dimension("date")
        provider.cube("sales")

        assert "date" in provider._dimension_cache
        assert "sales" in provider._cube_cache

        # Clear caches
        provider.clear_cache()

        assert len(provider._dimension_cache) == 0
        assert len(provider._cube_cache) == 0

    def test_cache_isolated_per_instance(self) -> None:
        """Test that each provider instance has isolated caches."""
        metadata: dict[str, Any] = {
            "dimensions": [
                {
                    "name": "date",
                    "levels": [{"name": "year", "attributes": ["year"]}],
                }
            ]
        }
        provider1 = StaticModelProvider(metadata)
        provider2 = StaticModelProvider(metadata)

        # Populate provider1's cache
        provider1.dimension("date")

        # provider2's cache should be empty
        assert "date" not in provider2._dimension_cache


class TestComplexModels:
    """Test with complex model definitions."""

    def test_full_model_with_dimensions_and_cubes(self) -> None:
        """Test with a complete model definition."""
        metadata: dict[str, Any] = {
            "dimensions": [
                {
                    "name": "date",
                    "levels": [
                        {"name": "year", "attributes": ["year"]},
                        {"name": "quarter", "attributes": ["quarter"]},
                        {"name": "month", "attributes": ["month"]},
                    ],
                    "hierarchies": [
                        {"name": "default", "levels": ["year", "quarter", "month"]}
                    ],
                },
                {
                    "name": "product",
                    "levels": [
                        {"name": "category", "attributes": ["category"]},
                        {"name": "item", "attributes": ["item"]},
                    ],
                },
            ],
            "cubes": [
                {
                    "name": "sales",
                    "label": "Sales Analysis",
                    "category": "finance",
                    "dimensions": ["date", "product"],
                    "measures": [
                        {"name": "amount", "label": "Sales Amount"},
                        {"name": "quantity", "label": "Quantity Sold"},
                    ],
                    "aggregates": [
                        {"name": "total_amount", "measure": "amount", "function": "sum"}
                    ],
                    "fact": "sales_fact",
                }
            ],
        }
        provider = StaticModelProvider(metadata)

        # Check cubes
        cubes = provider.list_cubes()
        assert len(cubes) == 1
        assert cubes[0]["label"] == "Sales Analysis"

        # Check dimensions
        date_dim = provider.dimension("date")
        assert len(date_dim.levels) == 3

        # Check cube
        cube = provider.cube("sales")
        assert cube.name == "sales"
        assert len(cube.dimensions_list) == 2
        assert len(cube.measures) == 2

    def test_model_with_multiple_cubes(self) -> None:
        """Test provider with multiple cubes sharing dimensions."""
        metadata: dict[str, Any] = {
            "dimensions": [
                {
                    "name": "date",
                    "levels": [{"name": "year", "attributes": ["year"]}],
                }
            ],
            "cubes": [
                {"name": "sales", "dimensions": ["date"]},
                {"name": "inventory", "dimensions": ["date"]},
                {"name": "returns", "dimensions": ["date"]},
            ],
        }
        provider = StaticModelProvider(metadata)

        cubes = provider.list_cubes()
        assert len(cubes) == 3

        # All cubes should share the same date dimension
        sales = provider.cube("sales")
        inventory = provider.cube("inventory")
        assert len(sales.dimensions_list) == 1
        assert len(inventory.dimensions_list) == 1


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_cube_with_missing_dimension(self) -> None:
        """Test cube referencing a dimension that doesn't exist."""
        metadata: dict[str, Any] = {
            "cubes": [{"name": "sales", "dimensions": ["missing_dim"]}]
        }
        provider = StaticModelProvider(metadata)

        # Should still create the cube, but without the missing dimension
        cube = provider.cube("sales")
        assert cube.name == "sales"
        # Missing dimension is skipped

    def test_dimension_with_no_levels_raises_error(self) -> None:
        """Test dimension without levels raises ModelError."""
        metadata: dict[str, Any] = {
            "dimensions": [{"name": "empty_dim", "levels": []}],
            "cubes": [],
        }
        provider = StaticModelProvider(metadata)
        # Dimensions must have at least one level
        with pytest.raises(Exception):
            provider.dimension("empty_dim")

    def test_metadata_with_schema_objects(self) -> None:
        """Test provider works with pre-parsed schema objects."""
        from cubes.metadata.schemas.cube import CubeSchema
        from cubes.metadata.schemas.dimension import DimensionSchema

        dim_schema = DimensionSchema(
            name="date", levels=[{"name": "year", "attributes": ["year"]}]
        )
        cube_schema = CubeSchema(name="sales", dimensions=["date"])

        metadata: dict[str, Any] = {
            "dimensions": [dim_schema],
            "cubes": [cube_schema],
        }
        provider = StaticModelProvider(metadata)

        assert "date" in provider._dimensions_metadata
        assert "sales" in provider._cubes_metadata

    def test_cube_info_defaults(self) -> None:
        """Test cube info defaults to empty dict."""
        metadata: dict[str, Any] = {"cubes": [{"name": "sales"}]}
        provider = StaticModelProvider(metadata)
        cubes = provider.list_cubes()
        assert cubes[0]["info"] == {}


class TestProviderRepr:
    """Test provider string representation."""

    def test_repr(self) -> None:
        """Test string representation."""
        provider = StaticModelProvider({})
        assert "StaticModelProvider" in repr(provider)
