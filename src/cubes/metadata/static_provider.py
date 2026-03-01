"""Static model provider for loading cubes and dimensions from metadata files.

This module provides the StaticModelProvider class which loads model metadata
from static sources like JSON files or dictionaries.
"""

from pathlib import Path
from typing import Any, Optional, Union

from cubes.metadata.io import read_model_metadata, read_model_metadata_bundle
from cubes.metadata.models.cube import Cube
from cubes.metadata.models.dimension import Dimension
from cubes.metadata.provider import (
    MetadataProvider,
    NoSuchCubeError,
    NoSuchDimensionError,
)
from cubes.metadata.schemas.cube import CubeSchema
from cubes.metadata.schemas.dimension import DimensionSchema


class StaticModelProvider(MetadataProvider):
    """Static model provider for loading cubes and dimensions from metadata.

    This provider loads model definitions from static sources such as JSON
    files, dictionaries, or model bundles (directories with .cubesmodel suffix).

    The provider maintains internal caches for cubes and dimensions to avoid
    repeated parsing and initialization.

    Attributes:
        metadata: The raw metadata dictionary containing cubes and dimensions.
        _dimensions_metadata: Dictionary of dimension metadata by name.
        _cubes_metadata: Dictionary of cube metadata by name.

    Example:
        >>> # From dictionary
        >>> metadata = {
        ...     "dimensions": [{"name": "date", "levels": [{"name": "year"}]}],
        ...     "cubes": [{"name": "sales", "dimensions": ["date"]}]
        ... }
        >>> provider = StaticModelProvider(metadata)
        >>> provider.list_cubes()
        [{'name': 'sales', 'label': 'sales', 'category': None, 'info': {}}]

        >>> # From file path
        >>> provider = StaticModelProvider("model.json")

        >>> # From bundle directory
        >>> provider = StaticModelProvider("model.cubesmodel/")
    """

    def __init__(
        self,
        metadata: Optional[Union[str, Path, dict[str, Any]]] = None,
    ) -> None:
        """Initialize the static model provider.

        Args:
            metadata: Model metadata source. Can be:
                - A dictionary containing model metadata
                - A file path to a JSON model file
                - A directory path to a model bundle (.cubesmodel)
                If None, an empty model is created.

        Raises:
            MetadataNotFoundError: If the specified file or directory is not found.
            MetadataIOError: If the JSON is malformed or cannot be read.
        """
        super().__init__()

        # Load metadata from various sources
        if metadata is None:
            self.metadata: dict[str, Any] = {}
        elif isinstance(metadata, dict):
            self.metadata = metadata
        elif isinstance(metadata, (str, Path)):
            path = Path(metadata)
            if path.is_dir():
                # Load from bundle directory
                self.metadata = read_model_metadata_bundle(path)
            else:
                # Load from single file
                self.metadata = read_model_metadata(path)
        else:
            raise ValueError(
                f"metadata must be dict, str, or Path, got {type(metadata).__name__}"
            )

        # Build lookup dictionaries for dimension and cube metadata
        self._dimensions_metadata: dict[str, dict[str, Any]] = {}
        for dim in self.metadata.get("dimensions", []):
            if isinstance(dim, dict):
                self._dimensions_metadata[dim["name"]] = dim
            else:
                # DimensionSchema object
                self._dimensions_metadata[dim.name] = dim

        self._cubes_metadata: dict[str, dict[str, Any]] = {}
        for cube in self.metadata.get("cubes", []):
            if isinstance(cube, dict):
                self._cubes_metadata[cube["name"]] = cube
            else:
                # CubeSchema object
                self._cubes_metadata[cube.name] = cube

    def list_cubes(self) -> list[dict[str, Any]]:
        """Get a list of metadata for cubes in the workspace.

        Returns:
            List of dictionaries with keys: name, label, category, info.
        """
        cubes = []
        for cube in self.metadata.get("cubes", []):
            if isinstance(cube, dict):
                info = {
                    "name": cube["name"],
                    "label": cube.get("label", cube["name"]),
                    "category": cube.get("category")
                    or cube.get("info", {}).get("category"),
                    "info": cube.get("info", {}),
                }
            else:
                # CubeSchema object
                info = {
                    "name": cube.name,
                    "label": cube.label or cube.name,
                    "category": cube.category
                    or (cube.info.get("category") if cube.info else None),
                    "info": dict(cube.info) if cube.info else {},
                }
            cubes.append(info)
        return cubes

    def has_cube(self, name: str) -> bool:
        """Check if the provider has a cube with the given name.

        Args:
            name: Name of the cube to check.

        Returns:
            True if the cube exists, False otherwise.
        """
        return name in self._cubes_metadata

    def cube(self, name: str) -> Cube:
        """Get a cube by name.

        Args:
            name: Name of the cube to retrieve.

        Returns:
            The Cube object.

        Raises:
            NoSuchCubeError: If the cube is not found.
        """
        # Check cache first
        if name in self._cube_cache:
            return self._cube_cache[name]

        # Get cube metadata
        if name not in self._cubes_metadata:
            raise NoSuchCubeError(f"Cube '{name}' not found", name=name)

        cube_metadata = self._cubes_metadata[name]

        # Convert to CubeSchema if needed
        if isinstance(cube_metadata, dict):
            cube_schema = CubeSchema.model_validate(cube_metadata)
        else:
            cube_schema = cube_metadata

        # Build dimensions dictionary from available dimensions
        dimensions: dict[str, Dimension] = {}
        for dim_name in self._dimensions_metadata:
            try:
                dim = self.dimension(dim_name)
                dimensions[dim_name] = dim
            except NoSuchDimensionError:
                pass  # Skip dimensions that can't be loaded

        # Create cube from schema
        cube = Cube.from_schema(cube_schema, dimensions=dimensions)

        # Cache and return
        self._cube_cache[name] = cube
        return cube

    def dimension(
        self, name: str, templates: Optional[dict[str, Dimension]] = None
    ) -> Dimension:
        """Get a dimension by name, with optional template resolution.

        Args:
            name: Name of the dimension to retrieve.
            templates: Dictionary of dimension templates that can be used
                for inheritance. Keys are template names, values are Dimension objects.

        Returns:
            The Dimension object.

        Raises:
            NoSuchDimensionError: If the dimension is not found.
            TemplateRequiredError: If a required template is missing.
        """
        # Check cache first
        if name in self._dimension_cache:
            return self._dimension_cache[name]

        # Get dimension metadata
        if name not in self._dimensions_metadata:
            raise NoSuchDimensionError(f"Dimension '{name}' not found", name=name)

        dim_metadata = self._dimensions_metadata[name]

        # Convert to DimensionSchema if needed
        if isinstance(dim_metadata, dict):
            dim_schema = DimensionSchema.model_validate(dim_metadata)
        else:
            dim_schema = dim_metadata

        # Create dimension from schema with template resolution
        dimension = Dimension.from_schema(dim_schema, templates=templates)

        # Cache and return
        self._dimension_cache[name] = dimension
        return dimension
