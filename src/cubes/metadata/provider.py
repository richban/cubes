"""Metadata provider base class for loading cubes and dimensions."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from cubes.metadata.models.base import ModelError
from cubes.metadata.models.cube import Cube
from cubes.metadata.models.dimension import Dimension


class NoSuchCubeError(ModelError):
    """Raised when a cube is not found."""

    def __init__(self, message: str, name: Optional[str] = None) -> None:
        super().__init__(message)
        self.name = name


class NoSuchDimensionError(ModelError):
    """Raised when a dimension is not found."""

    def __init__(self, message: str, name: Optional[str] = None) -> None:
        super().__init__(message)
        self.name = name


class TemplateRequiredError(ModelError):
    """Raised when a dimension requires a template that is not available."""

    def __init__(self, template: str, message: Optional[str] = None) -> None:
        if message is None:
            message = f"Dimension template '{template}' is required but not available"
        super().__init__(message)
        self.template = template


class MetadataProvider(ABC):
    """Abstract base class for metadata providers.

    Metadata providers are responsible for loading cube and dimension definitions
    from various sources (files, databases, APIs, etc.).

    Subclasses must implement:
        - list_cubes()
        - cube(name)
        - dimension(name, templates)
    """

    def __init__(self) -> None:
        """Initialize the provider with empty caches."""
        self._cube_cache: dict[str, Cube] = {}
        self._dimension_cache: dict[str, Dimension] = {}
        self._template_cache: dict[str, Dimension] = {}

    @abstractmethod
    def list_cubes(self) -> list[dict[str, Any]]:
        """Get a list of metadata for cubes in the workspace.

        Returns:
            List of dictionaries with keys: name, label, category, info.
        """
        ...

    @abstractmethod
    def has_cube(self, name: str) -> bool:
        """Check if the provider has a cube with the given name.

        Args:
            name: Name of the cube to check.

        Returns:
            True if the cube exists, False otherwise.
        """
        ...

    @abstractmethod
    def cube(self, name: str) -> Cube:
        """Get a cube by name.

        Args:
            name: Name of the cube to retrieve.

        Returns:
            The Cube object.

        Raises:
            NoSuchCubeError: If the cube is not found.
        """
        ...

    @abstractmethod
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
        ...

    def _resolve_dimension_with_templates(
        self,
        name: str,
        templates: Optional[dict[str, Dimension]] = None,
        _pending: Optional[set[str]] = None,
    ) -> Dimension:
        """Resolve a dimension with template inheritance and cycle detection.

        This method handles the recursive template resolution logic with
        cycle detection to prevent infinite loops.

        Args:
            name: Name of the dimension to resolve.
            templates: Dictionary of available dimension templates.
            _pending: Internal set used for cycle detection (tracks dimensions
                currently being resolved).

        Returns:
            The resolved Dimension object.

        Raises:
            NoSuchDimensionError: If the dimension is not found.
            TemplateRequiredError: If a required template is missing or if
                there is a template cycle.
        """
        if _pending is None:
            _pending = set()

        # Check if already cached
        if name in self._dimension_cache:
            return self._dimension_cache[name]

        # Check for cycles
        if name in _pending:
            raise TemplateRequiredError(
                name, f"Dimension template cycle detected for '{name}'"
            )

        _pending.add(name)

        try:
            # Try to get the dimension directly
            try:
                dimension = self.dimension(name, templates)
                self._dimension_cache[name] = dimension
                return dimension
            except TemplateRequiredError as e:
                # The dimension requires a template
                template_name = e.template

                if templates and template_name in templates:
                    raise ModelError(
                        f"Model provider didn't make use of dimension template "
                        f"'{template_name}' for '{name}'"
                    )

                if template_name in _pending:
                    raise TemplateRequiredError(
                        template_name, f"Dimension templates cycle in '{template_name}'"
                    )

                # Recursively resolve the template first
                if templates is None:
                    templates = {}

                resolved_template = self._resolve_dimension_with_templates(
                    template_name, templates, _pending
                )
                templates[template_name] = resolved_template

                # Now try to get the dimension again with the resolved template
                dimension = self.dimension(name, templates)
                self._dimension_cache[name] = dimension
                return dimension

        finally:
            _pending.discard(name)

    def clear_cache(self) -> None:
        """Clear all internal caches."""
        self._cube_cache.clear()
        self._dimension_cache.clear()
        self._template_cache.clear()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
