"""Tests for ModelObject base class."""

import pytest
from cubes.metadata.models import ModelObject


class TestModelObject:
    """Tests for ModelObject dataclass."""

    def test_basic_initialization(self):
        """Test basic ModelObject initialization."""
        obj = ModelObject(name="test_object")
        assert obj.name == "test_object"
        assert obj.label is None
        assert obj.description is None
        assert obj.info == {}

    def test_full_initialization(self):
        """Test ModelObject with all properties."""
        obj = ModelObject(
            name="test_object",
            label="Test Object Label",
            description="A test object description",
            info={"key": "value", "count": 42},
        )
        assert obj.name == "test_object"
        assert obj.label == "Test Object Label"
        assert obj.description == "A test object description"
        assert obj.info == {"key": "value", "count": 42}

    def test_info_defaults_to_empty_dict(self):
        """Test that info defaults to an empty dictionary."""
        obj1 = ModelObject(name="obj1")
        obj2 = ModelObject(name="obj2")

        # Ensure they have separate dict instances
        assert obj1.info is not obj2.info
        assert obj1.info == {}
        assert obj2.info == {}

    def test_to_dict_basic(self):
        """Test to_dict with default options."""
        obj = ModelObject(
            name="test_object",
            label="Test Label",
            description="Test Description",
            info={"key": "value"},
        )

        result = obj.to_dict()

        assert result == {
            "name": "test_object",
            "label": "Test Label",
            "description": "Test Description",
            "info": {"key": "value"},
        }

    def test_to_dict_with_none_values(self):
        """Test to_dict preserves None values."""
        obj = ModelObject(name="test_object")

        result = obj.to_dict()

        assert result["name"] == "test_object"
        assert result["label"] is None
        assert result["description"] is None
        assert result["info"] == {}

    def test_to_dict_create_label_false(self):
        """Test to_dict with create_label=False (default)."""
        obj = ModelObject(name="test_object")

        result = obj.to_dict(create_label=False)

        assert result["label"] is None

    def test_to_dict_create_label_true(self):
        """Test to_dict with create_label=True generates label from name."""
        obj = ModelObject(name="test_object")

        result = obj.to_dict(create_label=True)

        assert result["label"] == "Test object"

    def test_to_dict_create_label_with_existing_label(self):
        """Test to_dict with create_label=True doesn't override existing label."""
        obj = ModelObject(name="test_object", label="Custom Label")

        result = obj.to_dict(create_label=True)

        assert result["label"] == "Custom Label"

    def test_to_dict_create_label_camelcase(self):
        """Test label generation handles camelCase names (only first letter capitalized)."""
        obj = ModelObject(name="camelCaseName")

        result = obj.to_dict(create_label=True)

        # capitalize() only capitalizes first letter, rest are lowercase
        assert result["label"] == "Camelcasename"

    def test_to_dict_create_label_multiple_underscores(self):
        """Test label generation replaces all underscores."""
        obj = ModelObject(name="multi_word_name_with_many_parts")

        result = obj.to_dict(create_label=True)

        assert result["label"] == "Multi word name with many parts"

    def test_to_dict_modifying_returned_dict(self):
        """Test that modifying returned dict doesn't affect original."""
        obj = ModelObject(name="test", info={"original": "value"})

        result = obj.to_dict()
        result["info"]["new"] = "added"
        result["name"] = "changed"

        # Original should be unchanged
        assert obj.name == "test"
        assert obj.info == {"original": "value"}

    def test_equality(self):
        """Test that ModelObjects can be compared."""
        obj1 = ModelObject(name="test", label="Label")
        obj2 = ModelObject(name="test", label="Label")
        obj3 = ModelObject(name="other", label="Label")

        assert obj1 == obj2
        assert obj1 != obj3

    def test_repr(self):
        """Test the string representation."""
        obj = ModelObject(name="test_object", label="Test")

        repr_str = repr(obj)

        assert "ModelObject" in repr_str
        assert "name='test_object'" in repr_str
