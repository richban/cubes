"""Tests for MetadataProvider base class."""

from typing import Any, Optional

import pytest

from cubes.metadata.models import Dimension, Level, Attribute
from cubes.metadata.provider import (
    MetadataProvider,
    NoSuchCubeError,
    NoSuchDimensionError,
    TemplateRequiredError,
)


class ConcreteProvider(MetadataProvider):
    """Concrete implementation for testing."""

    def __init__(
        self,
        cubes: Optional[list[dict[str, Any]]] = None,
        dimensions: Optional[dict[str, Dimension]] = None,
        raise_template_error: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._cubes = cubes or []
        self._dimensions = dimensions or {}
        self._raise_template_error = raise_template_error

    def list_cubes(self) -> list[dict[str, Any]]:
        return list(self._cubes)

    def has_cube(self, name: str) -> bool:
        return any(c["name"] == name for c in self._cubes)

    def cube(self, name: str) -> Any:
        for c in self._cubes:
            if c["name"] == name:
                return c
        raise NoSuchCubeError(f"Cube '{name}' not found", name=name)

    def dimension(
        self, name: str, templates: Optional[dict[str, Dimension]] = None
    ) -> Dimension:
        if self._raise_template_error and name == self._raise_template_error:
            raise TemplateRequiredError("template_dim")

        if name in self._dimensions:
            return self._dimensions[name]
        raise NoSuchDimensionError(f"Dimension '{name}' not found", name=name)


class TestExceptionClasses:
    """Test exception classes."""

    def test_no_such_cube_error(self) -> None:
        """Test NoSuchCubeError exception."""
        error = NoSuchCubeError("Cube not found", name="test_cube")
        assert str(error) == "Cube not found"
        assert error.name == "test_cube"

    def test_no_such_cube_error_without_name(self) -> None:
        """Test NoSuchCubeError without name."""
        error = NoSuchCubeError("Cube not found")
        assert str(error) == "Cube not found"
        assert error.name is None

    def test_no_such_dimension_error(self) -> None:
        """Test NoSuchDimensionError exception."""
        error = NoSuchDimensionError("Dimension not found", name="test_dim")
        assert str(error) == "Dimension not found"
        assert error.name == "test_dim"

    def test_no_such_dimension_error_without_name(self) -> None:
        """Test NoSuchDimensionError without name."""
        error = NoSuchDimensionError("Dimension not found")
        assert str(error) == "Dimension not found"
        assert error.name is None

    def test_template_required_error(self) -> None:
        """Test TemplateRequiredError exception."""
        error = TemplateRequiredError("base_template")
        assert error.template == "base_template"
        assert "base_template" in str(error)

    def test_template_required_error_with_message(self) -> None:
        """Test TemplateRequiredError with custom message."""
        error = TemplateRequiredError("base", message="Custom message")
        assert error.template == "base"
        assert str(error) == "Custom message"


class TestMetadataProviderAbstract:
    """Test abstract base class behavior."""

    def test_cannot_instantiate_abstract(self) -> None:
        """Test that MetadataProvider cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            MetadataProvider()
        assert "abstract" in str(exc_info.value).lower()

    def test_concrete_provider_can_be_instantiated(self) -> None:
        """Test that concrete implementations can be instantiated."""
        provider = ConcreteProvider()
        assert isinstance(provider, MetadataProvider)

    def test_provider_repr(self) -> None:
        """Test provider string representation."""
        provider = ConcreteProvider()
        assert "ConcreteProvider" in repr(provider)


class TestListCubes:
    """Test list_cubes method."""

    def test_empty_list(self) -> None:
        """Test listing cubes when none exist."""
        provider = ConcreteProvider()
        assert provider.list_cubes() == []

    def test_list_cubes(self) -> None:
        """Test listing multiple cubes."""
        cubes = [
            {"name": "sales", "label": "Sales", "category": "finance", "info": {}},
            {"name": "inventory", "label": "Inventory", "category": "ops", "info": {}},
        ]
        provider = ConcreteProvider(cubes=cubes)
        result = provider.list_cubes()
        assert len(result) == 2
        assert result[0]["name"] == "sales"
        assert result[1]["name"] == "inventory"


class TestHasCube:
    """Test has_cube method."""

    def test_has_existing_cube(self) -> None:
        """Test checking for an existing cube."""
        cubes = [{"name": "sales", "label": "Sales", "category": "finance", "info": {}}]
        provider = ConcreteProvider(cubes=cubes)
        assert provider.has_cube("sales") is True

    def test_has_nonexistent_cube(self) -> None:
        """Test checking for a non-existent cube."""
        provider = ConcreteProvider()
        assert provider.has_cube("missing") is False


class TestCube:
    """Test cube retrieval."""

    def test_get_existing_cube(self) -> None:
        """Test retrieving an existing cube."""
        cube_data = {"name": "sales", "label": "Sales"}
        provider = ConcreteProvider(cubes=[cube_data])
        result = provider.cube("sales")
        assert result["name"] == "sales"

    def test_get_nonexistent_cube_raises_error(self) -> None:
        """Test that retrieving a non-existent cube raises error."""
        provider = ConcreteProvider()
        with pytest.raises(NoSuchCubeError) as exc_info:
            provider.cube("missing")
        assert exc_info.value.name == "missing"
        assert "missing" in str(exc_info.value)


class TestDimension:
    """Test dimension retrieval."""

    def test_get_existing_dimension(self) -> None:
        """Test retrieving an existing dimension."""
        attr = Attribute(name="id")
        level = Level(name="base", attributes=[attr])
        dim = Dimension(
            name="date",
            levels=[level],
            hierarchies=[],
        )
        provider = ConcreteProvider(dimensions={"date": dim})
        result = provider.dimension("date")
        assert result.name == "date"

    def test_get_nonexistent_dimension_raises_error(self) -> None:
        """Test that retrieving a non-existent dimension raises error."""
        provider = ConcreteProvider()
        with pytest.raises(NoSuchDimensionError) as exc_info:
            provider.dimension("missing")
        assert exc_info.value.name == "missing"
        assert "missing" in str(exc_info.value)


class TestTemplateResolution:
    """Test template resolution with cycle detection."""

    def test_resolve_dimension_from_cache(self) -> None:
        """Test that cached dimensions are returned directly."""
        attr = Attribute(name="id")
        level = Level(name="base", attributes=[attr])
        dim = Dimension(
            name="date",
            levels=[level],
            hierarchies=[],
        )
        provider = ConcreteProvider(dimensions={"date": dim})

        # First call populates cache
        result1 = provider._resolve_dimension_with_templates("date")
        assert result1.name == "date"

        # Second call should use cache
        result2 = provider._resolve_dimension_with_templates("date")
        assert result2 is result1  # Same object from cache

    def test_resolve_with_template(self) -> None:
        """Test resolving a dimension that requires a template."""
        # Create template dimension
        attr = Attribute(name="id")
        level = Level(name="base", attributes=[attr])
        template_dim = Dimension(
            name="template_dim",
            levels=[level],
            hierarchies=[],
        )

        # Provider that requires template for "derived_dim"
        provider = ConcreteProvider(
            dimensions={
                "template_dim": template_dim,
            },
            raise_template_error="derived_dim",
        )

        # This should resolve the template and raise error about not using it
        with pytest.raises(Exception):
            provider._resolve_dimension_with_templates("derived_dim")

    def test_template_cycle_detection(self) -> None:
        """Test that template cycles are detected."""

        # Create a mock provider that simulates cycle
        class CycleProvider(ConcreteProvider):
            def __init__(self) -> None:
                super().__init__()
                self._call_count = 0

            def dimension(
                self, name: str, templates: Optional[dict[str, Dimension]] = None
            ) -> Dimension:
                self._call_count += 1
                if self._call_count == 1:
                    raise TemplateRequiredError("dim_b")
                elif self._call_count == 2:
                    raise TemplateRequiredError("dim_a")
                raise NoSuchDimensionError("Not found")

        provider = CycleProvider()

        with pytest.raises(TemplateRequiredError) as exc_info:
            provider._resolve_dimension_with_templates("dim_a")
        assert "cycle" in str(exc_info.value).lower()

    def test_self_referencing_cycle(self) -> None:
        """Test detection of self-referencing cycle."""

        class SelfRefProvider(ConcreteProvider):
            def dimension(
                self, name: str, templates: Optional[dict[str, Dimension]] = None
            ) -> Dimension:
                # Always requires itself as template
                raise TemplateRequiredError(name)

        provider = SelfRefProvider()

        with pytest.raises(TemplateRequiredError) as exc_info:
            provider._resolve_dimension_with_templates("dim_a")
        assert "cycle" in str(exc_info.value).lower()
        assert exc_info.value.template == "dim_a"


class TestCacheManagement:
    """Test cache management methods."""

    def test_clear_cache(self) -> None:
        """Test clearing all caches."""
        attr = Attribute(name="id")
        level = Level(name="base", attributes=[attr])
        dim = Dimension(
            name="date",
            levels=[level],
            hierarchies=[],
        )
        provider = ConcreteProvider(dimensions={"date": dim})

        # Populate cache
        provider._resolve_dimension_with_templates("date")
        assert "date" in provider._dimension_cache

        # Clear cache
        provider.clear_cache()
        assert len(provider._dimension_cache) == 0
        assert len(provider._cube_cache) == 0
        assert len(provider._template_cache) == 0

    def test_cache_isolated_per_instance(self) -> None:
        """Test that each provider instance has its own cache."""
        provider1 = ConcreteProvider()
        provider2 = ConcreteProvider()

        # Add to provider1's cache
        attr = Attribute(name="id")
        level = Level(name="base", attributes=[attr])
        dim = Dimension(name="date", levels=[level], hierarchies=[])
        provider1._dimension_cache["date"] = dim

        # provider2's cache should be empty
        assert "date" not in provider2._dimension_cache


class TestPendingTracking:
    """Test the _pending tracking for cycle detection."""

    def test_pending_is_cleaned_up_on_success(self) -> None:
        """Test that pending set is cleaned up after successful resolution."""
        attr = Attribute(name="id")
        level = Level(name="base", attributes=[attr])
        dim = Dimension(name="date", levels=[level], hierarchies=[])
        provider = ConcreteProvider(dimensions={"date": dim})

        # Resolve successfully
        provider._resolve_dimension_with_templates("date")

        # Pending should be empty (would need to track internally)
        # This is more of an integration test that no cycles are falsely detected

    def test_pending_is_cleaned_up_on_error(self) -> None:
        """Test that pending set is cleaned up even on error."""
        provider = ConcreteProvider()

        with pytest.raises(NoSuchDimensionError):
            provider._resolve_dimension_with_templates("missing")

        # Next call should work normally (no false cycle detection)
        # Since provider has no dimensions, it will still fail
        with pytest.raises(NoSuchDimensionError):
            provider._resolve_dimension_with_templates("another_missing")
