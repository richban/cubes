"""Tests for ModelObject base class."""

import pytest
from dataclasses import dataclass, field
from typing import Any, Optional

from cubes.metadata.models.base import ModelObject, to_label


class TestToLabel:
    """Tests for to_label utility function."""

    def test_basic_conversion(self) -> None:
        """Test basic underscore to space conversion."""
        assert to_label("test_name") == "Test name"
        assert to_label("hello_world") == "Hello world"

    def test_no_capitalize(self) -> None:
        """Test conversion without capitalization."""
        assert to_label("test_name", capitalize=False) == "test name"

    def test_no_underscores(self) -> None:
        """Test string without underscores."""
        assert to_label("name") == "Name"
        assert to_label("name", capitalize=False) == "name"

    def test_multiple_underscores(self) -> None:
        """Test string with multiple underscores."""
        assert to_label("first_second_third") == "First second third"


class TestModelObjectBasics:
    """Tests for ModelObject initialization and basic properties."""

    def test_basic_creation(self) -> None:
        """Test creating ModelObject with minimal arguments."""
        obj = ModelObject(name="test")
        assert obj.name == "test"
        assert obj.label is None
        assert obj.description is None
        assert obj.info == {}

    def test_full_creation(self) -> None:
        """Test creating ModelObject with all arguments."""
        obj = ModelObject(
            name="test",
            label="Test Label",
            description="A test object",
            info={"key": "value"},
        )
        assert obj.name == "test"
        assert obj.label == "Test Label"
        assert obj.description == "A test object"
        assert obj.info == {"key": "value"}

    def test_info_defaults_to_empty_dict(self) -> None:
        """Test that info defaults to empty dict, not shared reference."""
        obj1 = ModelObject(name="test1")
        obj2 = ModelObject(name="test2")

        obj1.info["key"] = "value"
        assert obj1.info == {"key": "value"}
        assert obj2.info == {}  # Should not share the dict

    def test_mutable_info(self) -> None:
        """Test that info can be modified after creation."""
        obj = ModelObject(name="test", info={"initial": True})
        obj.info["added"] = "later"
        assert obj.info == {"initial": True, "added": "later"}


class TestModelObjectToDict:
    """Tests for ModelObject.to_dict method."""

    def test_basic_to_dict(self) -> None:
        """Test basic dictionary conversion."""
        obj = ModelObject(name="test")
        result = obj.to_dict()

        assert result == {"name": "test", "info": {}}

    def test_full_to_dict(self) -> None:
        """Test conversion with all fields set."""
        obj = ModelObject(
            name="test", label="Test Label", description="A test", info={"key": "value"}
        )
        result = obj.to_dict()

        assert result == {
            "name": "test",
            "label": "Test Label",
            "description": "A test",
            "info": {"key": "value"},
        }

    def test_to_dict_skips_none_values(self) -> None:
        """Test that None values are excluded from output."""
        obj = ModelObject(name="test")
        result = obj.to_dict()

        assert "label" not in result
        assert "description" not in result

    def test_to_dict_with_create_label(self) -> None:
        """Test creating label from name when create_label=True."""
        obj = ModelObject(name="test_name")
        result = obj.to_dict(create_label=True)

        assert result["label"] == "Test name"

    def test_to_dict_with_create_label_preserves_existing(self) -> None:
        """Test that existing label is preserved when create_label=True."""
        obj = ModelObject(name="test_name", label="Custom Label")
        result = obj.to_dict(create_label=True)

        assert result["label"] == "Custom Label"

    def test_to_dict_ignores_options(self) -> None:
        """Test that extra options are ignored (for compatibility)."""
        obj = ModelObject(name="test")
        result = obj.to_dict(some_option=True, another=123)

        assert "some_option" not in result
        assert "another" not in result


class TestModelObjectInheritance:
    """Tests for ModelObject subclassing."""

    def test_can_subclass(self) -> None:
        """Test that ModelObject can be subclassed."""

        @dataclass
        class CustomObject(ModelObject):
            custom_field: str = "default"

        obj = CustomObject(name="test", custom_field="value")

        assert obj.name == "test"
        assert obj.custom_field == "value"
        assert obj.label is None

    def test_subclass_to_dict(self) -> None:
        """Test that subclasses work with to_dict."""

        @dataclass
        class CustomObject(ModelObject):
            custom_field: str = "default"

        obj = CustomObject(name="test", custom_field="value")
        result = obj.to_dict()

        assert result["name"] == "test"
        # Note: to_dict only returns base fields, not custom fields
        # This matches legacy behavior
