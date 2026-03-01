"""Metadata I/O utilities for reading and writing model metadata.

This module provides functions for loading model metadata from various sources
(file paths, URLs, directories) and exporting models to directory bundles.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse
from urllib.request import urlopen


class MetadataIOError(Exception):
    """Error related to metadata I/O operations."""

    pass


class MetadataNotFoundError(MetadataIOError):
    """Raised when a metadata file or directory is not found."""

    pass


def _is_url(source: str) -> bool:
    """Check if source is a URL.

    Args:
        source: String to check.

    Returns:
        True if source is a URL, False otherwise.
    """
    parsed = urlparse(source)
    # Empty scheme means local file path
    # Single character scheme is Windows drive letter (e.g., 'C:')
    if parsed.scheme == "" or parsed.scheme == "file":
        return False
    if len(parsed.scheme) == 1:
        return False  # Windows drive letter edge case
    return True


def _json_from_url(url: str) -> dict[str, Any]:
    """Load JSON from a URL or file path.

    Args:
        url: URL or file path to load JSON from.

    Returns:
        Parsed JSON as a dictionary.

    Raises:
        MetadataNotFoundError: If the URL cannot be accessed.
        MetadataIOError: If the JSON is malformed.
    """
    parsed = urlparse(url)

    if parsed.scheme in ("", "file"):
        # Local file
        path = Path(parsed.path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                return cast(dict[str, Any], json.load(f))
        except FileNotFoundError:
            raise MetadataNotFoundError(f"File not found: {url}")
        except json.JSONDecodeError as e:
            raise MetadataIOError(f"Invalid JSON in {url}: {e}")
    elif len(parsed.scheme) == 1:
        # Windows drive letter edge case (e.g., 'C:\path')
        try:
            with open(url, "r", encoding="utf-8") as f:
                return cast(dict[str, Any], json.load(f))
        except FileNotFoundError:
            raise MetadataNotFoundError(f"File not found: {url}")
        except json.JSONDecodeError as e:
            raise MetadataIOError(f"Invalid JSON in {url}: {e}")
    else:
        # Remote URL
        try:
            with urlopen(url) as response:
                return cast(dict[str, Any], json.load(response))
        except Exception as e:
            raise MetadataIOError(f"Failed to load URL {url}: {e}")


def read_model_metadata(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Read model metadata from various sources.

    Supports:
    - File paths (absolute or relative)
    - URLs (http/https)
    - Directory paths (model.cubesmodel/ bundles)
    - Dict objects (returned as-is)

    Args:
        source: Source to read from. Can be:
            - Path to a JSON file
            - Path to a model directory
            - URL to a JSON file
            - Dictionary (returned as-is)

    Returns:
        Model metadata dictionary.

    Raises:
        MetadataNotFoundError: If the source cannot be found.
        MetadataIOError: If there's an error reading or parsing.

    Example:
        >>> metadata = read_model_metadata("path/to/model.json")
        >>> metadata = read_model_metadata("https://example.com/model.json")
        >>> metadata = read_model_metadata("path/to/model.cubesmodel/")
    """
    # If already a dict, return as-is
    if isinstance(source, dict):
        return source

    # Convert to string if Path
    source_str = str(source)

    # Check if it's a directory
    if not _is_url(source_str):
        path = Path(source_str)
        if path.is_dir():
            return read_model_metadata_bundle(path)

    # Otherwise, treat as file or URL
    return _json_from_url(source_str)


def read_model_metadata_bundle(path: str | Path) -> dict[str, Any]:
    """Load a model metadata bundle from a directory.

    Model directory bundles have the following structure:

        model.cubesmodel/
            model.json          # Main model file
            dim_*.json          # Dimension definitions
            cube_*.json         # Cube definitions

    The dimensions and cubes lists in model.json are concatenated with
    dimensions and cubes from the separate files.

    Args:
        path: Path to the model directory.

    Returns:
        Model metadata dictionary with merged dimensions and cubes.

    Raises:
        MetadataNotFoundError: If the directory or model.json doesn't exist.
        MetadataIOError: If there's an error reading or parsing files.

    Example:
        >>> metadata = read_model_metadata_bundle("path/to/model.cubesmodel")
    """
    path = Path(path)

    if not path.is_dir():
        raise MetadataNotFoundError(f"Path '{path}' is not a directory.")

    model_file = path / "model.json"
    if not model_file.exists():
        raise MetadataNotFoundError(f"Main model info {model_file} does not exist")

    # Load main model file
    with open(model_file, "r", encoding="utf-8") as f:
        model = cast(dict[str, Any], json.load(f))

    # Initialize dimensions and cubes lists
    if "dimensions" not in model:
        model["dimensions"] = []
    if "cubes" not in model:
        model["cubes"] = []

    # Track names to detect duplicates
    dim_names = {d["name"] for d in model["dimensions"] if "name" in d}
    cube_names = {c["name"] for c in model["cubes"] if "name" in c}

    # Find and load dim_*.json and cube_*.json files
    for file_path in path.rglob("*.json"):
        # Skip the main model.json
        if file_path.name == "model.json":
            continue

        match = re.match(r"^(dim|dimension|cube)_(.+?)\.json$", file_path.name)
        if not match:
            continue

        prefix = match.group(1)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                desc = cast(dict[str, Any], json.load(f))
        except json.JSONDecodeError as e:
            raise MetadataIOError(f"Invalid JSON in {file_path}: {e}")

        # Check for name
        if "name" not in desc:
            obj_type = "Dimension" if prefix in ("dim", "dimension") else "Cube"
            raise MetadataIOError(f"{obj_type} file '{file_path}' has no name key")

        name = desc["name"]

        if prefix in ("dim", "dimension"):
            if name in dim_names:
                raise MetadataIOError(
                    f"Dimension '{name}' defined multiple times (in '{file_path}')"
                )
            model["dimensions"].append(desc)
            dim_names.add(name)
        elif prefix == "cube":
            if name in cube_names:
                raise MetadataIOError(
                    f"Cube '{name}' defined multiple times (in '{file_path}')"
                )
            model["cubes"].append(desc)
            cube_names.add(name)

    return model


def write_model_metadata_bundle(
    path: str | Path, metadata: dict[str, Any], replace: bool = False
) -> None:
    """Write a model metadata bundle to a directory.

    Creates a model directory with the following structure:

        model.cubesmodel/
            model.json          # Main model file (without dimensions/cubes)
            dim_*.json          # Dimension definitions
            cube_*.json         # Cube definitions

    Args:
        path: Path to the target directory.
        metadata: Model metadata dictionary.
        replace: If True, replace existing directory. If False, raise an error
            if the directory already exists.

    Raises:
        MetadataIOError: If the target exists and is not a model directory,
            or if replace=False and target exists.

    Example:
        >>> metadata = {'name': 'my_model', 'dimensions': [...], 'cubes': [...]}
        >>> write_model_metadata_bundle("path/to/model.cubesmodel", metadata)
    """
    path = Path(path)

    if path.exists():
        if not path.is_dir():
            raise MetadataIOError("Target exists and is a file, cannot replace")

        model_file = path / "model.json"
        if not model_file.exists():
            raise MetadataIOError("Target is not a model directory, cannot replace.")

        if replace:
            shutil.rmtree(path)
        else:
            raise MetadataIOError(
                "Target already exists. Remove it or force replacement."
            )

    # Create directory
    path.mkdir(parents=True, exist_ok=True)

    # Make a copy of metadata to avoid modifying the original
    metadata_copy = dict(metadata)

    # Extract dimensions and cubes
    dimensions = metadata_copy.pop("dimensions", [])
    cubes = metadata_copy.pop("cubes", [])

    # Write dimension files
    for dim in dimensions:
        name = dim.get("name")
        if not name:
            raise MetadataIOError("Dimension without name cannot be written")
        filename = path / f"dim_{name}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(dim, f, indent=4)

    # Write cube files
    for cube in cubes:
        name = cube.get("name")
        if not name:
            raise MetadataIOError("Cube without name cannot be written")
        filename = path / f"cube_{name}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(cube, f, indent=4)

    # Write main model file
    model_file = path / "model.json"
    with open(model_file, "w", encoding="utf-8") as f:
        json.dump(metadata_copy, f, indent=4)
