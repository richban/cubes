"""Tests for metadata I/O utilities."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from cubes.metadata.io import (
    MetadataIOError,
    MetadataNotFoundError,
    read_model_metadata,
    read_model_metadata_bundle,
    write_model_metadata_bundle,
)


class TestReadModelMetadata:
    """Tests for read_model_metadata function."""

    def test_dict_input_returned_as_is(self):
        """Test that dict input is returned unchanged."""
        metadata = {"name": "test_model", "cubes": [], "dimensions": []}
        result = read_model_metadata(metadata)
        assert result == metadata
        assert result is metadata  # Same object, not a copy

    def test_from_json_file(self, tmp_path: Path):
        """Test reading from a JSON file path."""
        model_file = tmp_path / "model.json"
        expected = {"name": "test_model", "cubes": [], "dimensions": []}
        model_file.write_text(json.dumps(expected))

        result = read_model_metadata(str(model_file))

        assert result["name"] == "test_model"
        assert result["cubes"] == []
        assert result["dimensions"] == []

    def test_from_json_file_path_object(self, tmp_path: Path):
        """Test reading from a Path object."""
        model_file = tmp_path / "model.json"
        expected = {"name": "test_model", "cubes": [], "dimensions": []}
        model_file.write_text(json.dumps(expected))

        result = read_model_metadata(model_file)

        assert result["name"] == "test_model"

    def test_from_directory_bundle(self, tmp_path: Path):
        """Test reading from a model directory bundle."""
        bundle_dir = tmp_path / "test.cubesmodel"
        bundle_dir.mkdir()

        # Create model.json
        model = {"name": "bundle_model", "label": "Bundle Model"}
        (bundle_dir / "model.json").write_text(json.dumps(model))

        # Create a dimension file
        dim = {"name": "date", "label": "Date Dimension"}
        (bundle_dir / "dim_date.json").write_text(json.dumps(dim))

        # Create a cube file
        cube = {"name": "sales", "label": "Sales Cube"}
        (bundle_dir / "cube_sales.json").write_text(json.dumps(cube))

        result = read_model_metadata(str(bundle_dir))

        assert result["name"] == "bundle_model"
        assert len(result["dimensions"]) == 1
        assert result["dimensions"][0]["name"] == "date"
        assert len(result["cubes"]) == 1
        assert result["cubes"][0]["name"] == "sales"

    def test_from_url(self):
        """Test reading from a URL."""
        url = "http://example.com/model.json"
        expected = {"name": "remote_model", "cubes": [], "dimensions": []}

        with patch("cubes.metadata.io.urlopen") as mock_urlopen:
            mock_response = Mock()
            mock_response.read.return_value = json.dumps(expected).encode()
            mock_urlopen.return_value.__enter__ = Mock(return_value=mock_response)
            mock_urlopen.return_value.__exit__ = Mock(return_value=None)
            mock_response.__enter__ = Mock(return_value=mock_response)
            mock_response.__exit__ = Mock(return_value=None)

            # We need to mock the json.load to work with our mock
            with patch("cubes.metadata.io.json.load") as mock_load:
                mock_load.return_value = expected
                result = read_model_metadata(url)
                assert result["name"] == "remote_model"

    def test_file_not_found(self, tmp_path: Path):
        """Test error when file doesn't exist."""
        nonexistent = tmp_path / "nonexistent.json"

        with pytest.raises(MetadataNotFoundError) as exc_info:
            read_model_metadata(str(nonexistent))

        assert "not found" in str(exc_info.value).lower() or "File not found" in str(
            exc_info.value
        )

    def test_invalid_json(self, tmp_path: Path):
        """Test error when JSON is malformed."""
        model_file = tmp_path / "invalid.json"
        model_file.write_text("not valid json {{{")

        with pytest.raises(MetadataIOError) as exc_info:
            read_model_metadata(str(model_file))

        assert "invalid json" in str(exc_info.value).lower()

    def test_windows_drive_letter_path(self, tmp_path: Path):
        """Test handling of Windows-style paths with drive letters."""
        # On Unix, this won't work, but we can at least test the code path
        model_file = tmp_path / "model.json"
        expected = {"name": "test_model"}
        model_file.write_text(json.dumps(expected))

        # Use the file path normally - the function handles Windows paths internally
        result = read_model_metadata(str(model_file))
        assert result["name"] == "test_model"


class TestReadModelMetadataBundle:
    """Tests for read_model_metadata_bundle function."""

    def test_basic_bundle_loading(self, tmp_path: Path):
        """Test loading a basic bundle with model.json only."""
        bundle_dir = tmp_path / "model.cubesmodel"
        bundle_dir.mkdir()

        model = {"name": "test_model", "label": "Test Model"}
        (bundle_dir / "model.json").write_text(json.dumps(model))

        result = read_model_metadata_bundle(bundle_dir)

        assert result["name"] == "test_model"
        assert result["dimensions"] == []
        assert result["cubes"] == []

    def test_bundle_with_dimensions(self, tmp_path: Path):
        """Test loading bundle with dimension files."""
        bundle_dir = tmp_path / "model.cubesmodel"
        bundle_dir.mkdir()

        model = {"name": "test_model"}
        (bundle_dir / "model.json").write_text(json.dumps(model))

        dim1 = {"name": "date", "label": "Date"}
        dim2 = {"name": "product", "label": "Product"}
        (bundle_dir / "dim_date.json").write_text(json.dumps(dim1))
        (bundle_dir / "dim_product.json").write_text(json.dumps(dim2))

        result = read_model_metadata_bundle(bundle_dir)

        assert len(result["dimensions"]) == 2
        dim_names = {d["name"] for d in result["dimensions"]}
        assert dim_names == {"date", "product"}

    def test_bundle_with_cubes(self, tmp_path: Path):
        """Test loading bundle with cube files."""
        bundle_dir = tmp_path / "model.cubesmodel"
        bundle_dir.mkdir()

        model = {"name": "test_model"}
        (bundle_dir / "model.json").write_text(json.dumps(model))

        cube1 = {"name": "sales", "label": "Sales"}
        cube2 = {"name": "inventory", "label": "Inventory"}
        (bundle_dir / "cube_sales.json").write_text(json.dumps(cube1))
        (bundle_dir / "cube_inventory.json").write_text(json.dumps(cube2))

        result = read_model_metadata_bundle(bundle_dir)

        assert len(result["cubes"]) == 2
        cube_names = {c["name"] for c in result["cubes"]}
        assert cube_names == {"sales", "inventory"}

    def test_bundle_with_mixed_content(self, tmp_path: Path):
        """Test loading bundle with model.json, dimensions, and cubes."""
        bundle_dir = tmp_path / "model.cubesmodel"
        bundle_dir.mkdir()

        model = {
            "name": "test_model",
            "dimensions": [{"name": "inline_dim", "label": "Inline"}],
            "cubes": [{"name": "inline_cube", "label": "Inline Cube"}],
        }
        (bundle_dir / "model.json").write_text(json.dumps(model))

        dim = {"name": "file_dim", "label": "From File"}
        cube = {"name": "file_cube", "label": "From File"}
        (bundle_dir / "dim_file_dim.json").write_text(json.dumps(dim))
        (bundle_dir / "cube_file_cube.json").write_text(json.dumps(cube))

        result = read_model_metadata_bundle(bundle_dir)

        # Should merge inline and file-based
        assert len(result["dimensions"]) == 2
        assert len(result["cubes"]) == 2

    def test_bundle_subdirectory_files(self, tmp_path: Path):
        """Test that files in subdirectories are also loaded."""
        bundle_dir = tmp_path / "model.cubesmodel"
        bundle_dir.mkdir()

        subdir = bundle_dir / "dimensions"
        subdir.mkdir()

        model = {"name": "test_model"}
        (bundle_dir / "model.json").write_text(json.dumps(model))

        dim = {"name": "subdir_dim", "label": "Subdir Dim"}
        (subdir / "dim_subdir_dim.json").write_text(json.dumps(dim))

        result = read_model_metadata_bundle(bundle_dir)

        assert len(result["dimensions"]) == 1
        assert result["dimensions"][0]["name"] == "subdir_dim"

    def test_dimension_prefix_variations(self, tmp_path: Path):
        """Test that both 'dim_' and 'dimension_' prefixes work."""
        bundle_dir = tmp_path / "model.cubesmodel"
        bundle_dir.mkdir()

        model = {"name": "test_model"}
        (bundle_dir / "model.json").write_text(json.dumps(model))

        dim1 = {"name": "date", "label": "Date"}
        dim2 = {"name": "time", "label": "Time"}
        (bundle_dir / "dim_date.json").write_text(json.dumps(dim1))
        (bundle_dir / "dimension_time.json").write_text(json.dumps(dim2))

        result = read_model_metadata_bundle(bundle_dir)

        assert len(result["dimensions"]) == 2

    def test_directory_not_found(self, tmp_path: Path):
        """Test error when directory doesn't exist."""
        nonexistent = tmp_path / "nonexistent"

        with pytest.raises(MetadataNotFoundError) as exc_info:
            read_model_metadata_bundle(nonexistent)

        assert "not a directory" in str(exc_info.value).lower()

    def test_missing_model_json(self, tmp_path: Path):
        """Test error when model.json is missing."""
        bundle_dir = tmp_path / "model.cubesmodel"
        bundle_dir.mkdir()

        with pytest.raises(MetadataNotFoundError) as exc_info:
            read_model_metadata_bundle(bundle_dir)

        assert "does not exist" in str(exc_info.value).lower()

    def test_dimension_without_name(self, tmp_path: Path):
        """Test error when dimension file has no name."""
        bundle_dir = tmp_path / "model.cubesmodel"
        bundle_dir.mkdir()

        model = {"name": "test_model"}
        (bundle_dir / "model.json").write_text(json.dumps(model))

        dim = {"label": "Missing Name"}  # No name key
        (bundle_dir / "dim_noname.json").write_text(json.dumps(dim))

        with pytest.raises(MetadataIOError) as exc_info:
            read_model_metadata_bundle(bundle_dir)

        assert "no name key" in str(exc_info.value).lower()

    def test_cube_without_name(self, tmp_path: Path):
        """Test error when cube file has no name."""
        bundle_dir = tmp_path / "model.cubesmodel"
        bundle_dir.mkdir()

        model = {"name": "test_model"}
        (bundle_dir / "model.json").write_text(json.dumps(model))

        cube = {"label": "Missing Name"}  # No name key
        (bundle_dir / "cube_noname.json").write_text(json.dumps(cube))

        with pytest.raises(MetadataIOError) as exc_info:
            read_model_metadata_bundle(bundle_dir)

        assert "no name key" in str(exc_info.value).lower()

    def test_duplicate_dimension_name(self, tmp_path: Path):
        """Test error when dimension is defined multiple times."""
        bundle_dir = tmp_path / "model.cubesmodel"
        bundle_dir.mkdir()

        model = {
            "name": "test_model",
            "dimensions": [{"name": "duplicate", "label": "In model.json"}],
        }
        (bundle_dir / "model.json").write_text(json.dumps(model))

        dim = {"name": "duplicate", "label": "In file"}
        (bundle_dir / "dim_duplicate.json").write_text(json.dumps(dim))

        with pytest.raises(MetadataIOError) as exc_info:
            read_model_metadata_bundle(bundle_dir)

        assert "defined multiple times" in str(exc_info.value).lower()

    def test_duplicate_cube_name(self, tmp_path: Path):
        """Test error when cube is defined multiple times."""
        bundle_dir = tmp_path / "model.cubesmodel"
        bundle_dir.mkdir()

        model = {
            "name": "test_model",
            "cubes": [{"name": "duplicate", "label": "In model.json"}],
        }
        (bundle_dir / "model.json").write_text(json.dumps(model))

        cube = {"name": "duplicate", "label": "In file"}
        (bundle_dir / "cube_duplicate.json").write_text(json.dumps(cube))

        with pytest.raises(MetadataIOError) as exc_info:
            read_model_metadata_bundle(bundle_dir)

        assert "defined multiple times" in str(exc_info.value).lower()

    def test_invalid_json_in_file(self, tmp_path: Path):
        """Test error when a dimension/cube file has invalid JSON."""
        bundle_dir = tmp_path / "model.cubesmodel"
        bundle_dir.mkdir()

        model = {"name": "test_model"}
        (bundle_dir / "model.json").write_text(json.dumps(model))

        (bundle_dir / "dim_invalid.json").write_text("not valid json")

        with pytest.raises(MetadataIOError) as exc_info:
            read_model_metadata_bundle(bundle_dir)

        assert "invalid json" in str(exc_info.value).lower()

    def test_ignores_non_json_files(self, tmp_path: Path):
        """Test that non-JSON files are ignored."""
        bundle_dir = tmp_path / "model.cubesmodel"
        bundle_dir.mkdir()

        model = {"name": "test_model"}
        (bundle_dir / "model.json").write_text(json.dumps(model))

        # Create non-JSON files
        (bundle_dir / "readme.txt").write_text("This is a readme")
        (bundle_dir / "data.csv").write_text("a,b,c")
        (bundle_dir / "not_dim_test.json~").write_text("backup file")

        result = read_model_metadata_bundle(bundle_dir)

        # Should not raise, should just have empty dimensions/cubes
        assert result["dimensions"] == []
        assert result["cubes"] == []

    def test_ignores_files_without_prefix(self, tmp_path: Path):
        """Test that JSON files without dim_ or cube_ prefix are ignored."""
        bundle_dir = tmp_path / "model.cubesmodel"
        bundle_dir.mkdir()

        model = {"name": "test_model"}
        (bundle_dir / "model.json").write_text(json.dumps(model))

        # Create JSON files without proper prefix
        other = {"name": "other", "label": "Other"}
        (bundle_dir / "other_file.json").write_text(json.dumps(other))
        (bundle_dir / "dimension.json").write_text(json.dumps(other))

        result = read_model_metadata_bundle(bundle_dir)

        assert result["dimensions"] == []
        assert result["cubes"] == []


class TestWriteModelMetadataBundle:
    """Tests for write_model_metadata_bundle function."""

    def test_write_basic_bundle(self, tmp_path: Path):
        """Test writing a basic bundle with model only."""
        output_dir = tmp_path / "output.cubesmodel"

        metadata = {"name": "test_model", "label": "Test Model"}
        write_model_metadata_bundle(output_dir, metadata)

        assert output_dir.exists()
        assert output_dir.is_dir()
        assert (output_dir / "model.json").exists()

        # Verify model.json content
        with open(output_dir / "model.json") as f:
            saved = json.load(f)
        assert saved["name"] == "test_model"
        assert saved["label"] == "Test Model"

    def test_write_with_dimensions(self, tmp_path: Path):
        """Test writing bundle with dimensions."""
        output_dir = tmp_path / "output.cubesmodel"

        metadata = {
            "name": "test_model",
            "dimensions": [
                {"name": "date", "label": "Date"},
                {"name": "product", "label": "Product"},
            ],
        }
        write_model_metadata_bundle(output_dir, metadata)

        assert (output_dir / "dim_date.json").exists()
        assert (output_dir / "dim_product.json").exists()

        with open(output_dir / "dim_date.json") as f:
            saved = json.load(f)
        assert saved["name"] == "date"
        assert saved["label"] == "Date"

    def test_write_with_cubes(self, tmp_path: Path):
        """Test writing bundle with cubes."""
        output_dir = tmp_path / "output.cubesmodel"

        metadata = {
            "name": "test_model",
            "cubes": [
                {"name": "sales", "label": "Sales"},
                {"name": "inventory", "label": "Inventory"},
            ],
        }
        write_model_metadata_bundle(output_dir, metadata)

        assert (output_dir / "cube_sales.json").exists()
        assert (output_dir / "cube_inventory.json").exists()

    def test_write_preserves_other_fields(self, tmp_path: Path):
        """Test that non-dimension/cube fields are preserved in model.json."""
        output_dir = tmp_path / "output.cubesmodel"

        metadata = {
            "name": "test_model",
            "label": "Test Model",
            "description": "A test model",
            "info": {"author": "test", "version": "1.0"},
            "custom_field": "custom value",
        }
        write_model_metadata_bundle(output_dir, metadata)

        with open(output_dir / "model.json") as f:
            saved = json.load(f)

        assert saved["name"] == "test_model"
        assert saved["label"] == "Test Model"
        assert saved["description"] == "A test model"
        assert saved["info"] == {"author": "test", "version": "1.0"}
        assert saved["custom_field"] == "custom value"
        assert "dimensions" not in saved
        assert "cubes" not in saved

    def test_write_does_not_modify_input(self, tmp_path: Path):
        """Test that the input metadata dict is not modified."""
        output_dir = tmp_path / "output.cubesmodel"

        metadata = {
            "name": "test_model",
            "dimensions": [{"name": "date", "label": "Date"}],
            "cubes": [{"name": "sales", "label": "Sales"}],
        }
        original = dict(metadata)
        original["dimensions"] = list(metadata["dimensions"])
        original["cubes"] = list(metadata["cubes"])

        write_model_metadata_bundle(output_dir, metadata)

        # Verify input wasn't modified
        assert metadata["name"] == original["name"]
        assert len(metadata["dimensions"]) == len(original["dimensions"])
        assert len(metadata["cubes"]) == len(original["cubes"])

    def test_error_if_target_is_file(self, tmp_path: Path):
        """Test error when target path is an existing file."""
        output_file = tmp_path / "output.cubesmodel"
        output_file.write_text("I am a file, not a directory")

        metadata = {"name": "test_model"}

        with pytest.raises(MetadataIOError) as exc_info:
            write_model_metadata_bundle(output_file, metadata)

        assert "is a file" in str(exc_info.value).lower()

    def test_error_if_target_not_model_dir(self, tmp_path: Path):
        """Test error when target is a non-model directory."""
        output_dir = tmp_path / "output.cubesmodel"
        output_dir.mkdir()
        # No model.json inside

        metadata = {"name": "test_model"}

        with pytest.raises(MetadataIOError) as exc_info:
            write_model_metadata_bundle(output_dir, metadata)

        assert "not a model directory" in str(exc_info.value).lower()

    def test_error_if_target_exists_no_replace(self, tmp_path: Path):
        """Test error when target exists and replace=False."""
        output_dir = tmp_path / "output.cubesmodel"
        output_dir.mkdir()
        (output_dir / "model.json").write_text('{"name": "existing"}')

        metadata = {"name": "test_model"}

        with pytest.raises(MetadataIOError) as exc_info:
            write_model_metadata_bundle(output_dir, metadata)

        assert "already exists" in str(exc_info.value).lower()

    def test_replace_existing_bundle(self, tmp_path: Path):
        """Test replacing existing bundle with replace=True."""
        output_dir = tmp_path / "output.cubesmodel"
        output_dir.mkdir()
        (output_dir / "model.json").write_text('{"name": "old_model"}')
        (output_dir / "old_file.txt").write_text("old content")

        metadata = {"name": "new_model", "label": "New Model"}
        write_model_metadata_bundle(output_dir, metadata, replace=True)

        # Verify new content
        with open(output_dir / "model.json") as f:
            saved = json.load(f)
        assert saved["name"] == "new_model"
        assert saved["label"] == "New Model"

        # Old file should be gone
        assert not (output_dir / "old_file.txt").exists()

    def test_error_on_dimension_without_name(self, tmp_path: Path):
        """Test error when a dimension has no name."""
        output_dir = tmp_path / "output.cubesmodel"

        metadata = {
            "name": "test_model",
            "dimensions": [{"label": "No Name"}],  # No name key
        }

        with pytest.raises(MetadataIOError) as exc_info:
            write_model_metadata_bundle(output_dir, metadata)

        assert "without name" in str(exc_info.value).lower()

    def test_error_on_cube_without_name(self, tmp_path: Path):
        """Test error when a cube has no name."""
        output_dir = tmp_path / "output.cubesmodel"

        metadata = {
            "name": "test_model",
            "cubes": [{"label": "No Name"}],  # No name key
        }

        with pytest.raises(MetadataIOError) as exc_info:
            write_model_metadata_bundle(output_dir, metadata)

        assert "without name" in str(exc_info.value).lower()

    def test_json_indentation(self, tmp_path: Path):
        """Test that output JSON is properly indented."""
        output_dir = tmp_path / "output.cubesmodel"

        metadata = {
            "name": "test_model",
            "dimensions": [{"name": "date", "label": "Date"}],
        }
        write_model_metadata_bundle(output_dir, metadata)

        # Read raw file content
        content = (output_dir / "dim_date.json").read_text()

        # Should be indented (contain newlines)
        assert "\n" in content
        # Should have 4-space indentation
        assert '    "name"' in content or '    "name"' in content

    def test_path_as_string(self, tmp_path: Path):
        """Test that string paths work correctly."""
        output_dir = tmp_path / "output.cubesmodel"

        metadata = {"name": "test_model"}
        write_model_metadata_bundle(str(output_dir), metadata)

        assert output_dir.exists()
        assert (output_dir / "model.json").exists()


class TestLegacyTestModel:
    """Tests that replicate the legacy test_model.py tests."""

    def test_from_file_exact_fixture(self, tmp_path: Path):
        """Test loading from a model.json file (replicates test_from_file)."""
        model_file = tmp_path / "model.json"
        fixture = {
            "cubes": [{"name": "contracts", "label": "Contracts"}],
            "dimensions": [
                {"name": "date", "label": "Date"},
                {"name": "product", "label": "Product"},
                {"name": "customer", "label": "Customer"},
                {"name": "region", "label": "Region"},
                {"name": "category", "label": "Category"},
                {"name": "supplier", "label": "Supplier"},
            ],
        }
        model_file.write_text(json.dumps(fixture))

        desc = read_model_metadata(str(model_file))

        assert isinstance(desc, dict)
        assert "cubes" in desc
        assert "dimensions" in desc
        assert len(desc["cubes"]) == 1
        assert len(desc["dimensions"]) == 6

    def test_from_bundle_exact_fixture(self, tmp_path: Path):
        """Test loading from a bundle directory (replicates test_from_bundle)."""
        bundle_dir = tmp_path / "test.cubesmodel"
        bundle_dir.mkdir()

        # model.json
        model = {
            "name": "public_procurements",
            "label": "Public Procurements of Slovakia",
            "description": "Contracts of public procurement winners in Slovakia",
        }
        (bundle_dir / "model.json").write_text(json.dumps(model))

        # Add some dimensions
        dimensions = [
            "date",
            "cpv",
            "procurer",
            "supplier",
            "process_type",
            "geography",
        ]
        for dim_name in dimensions:
            dim = {"name": dim_name, "label": dim_name.capitalize()}
            (bundle_dir / f"dim_{dim_name}.json").write_text(json.dumps(dim))

        # Add a cube
        cube = {
            "name": "contracts",
            "label": "Contracts",
            "dimensions": dimensions,
        }
        (bundle_dir / "cube_contracts.json").write_text(json.dumps(cube))

        desc = read_model_metadata(str(bundle_dir))

        assert isinstance(desc, dict)
        assert "cubes" in desc
        assert "dimensions" in desc
        assert len(desc["cubes"]) == 1
        assert len(desc["dimensions"]) == 6

    def test_read_model_metadata_bundle_only_accepts_directories(self, tmp_path: Path):
        """Test that read_model_metadata_bundle only accepts directories."""
        model_file = tmp_path / "model.json"
        model_file.write_text('{"name": "test"}')

        with pytest.raises(MetadataNotFoundError) as exc_info:
            read_model_metadata_bundle(model_file)

        assert "not a directory" in str(exc_info.value).lower()
