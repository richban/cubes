"""Tests for object_dict utility function."""

import pytest

from cubes.metadata.models.base import ModelError, ModelObject, object_dict


class MockNamedObject:
    """Mock object with name attribute for testing."""

    def __init__(self, name: str, value: str = ""):
        self.name = name
        self.value = value

    def __repr__(self) -> str:
        return f"MockNamedObject({self.name!r})"


class MockRefObject:
    """Mock object with ref attribute for testing."""

    def __init__(self, name: str, ref: str, value: str = ""):
        self.name = name
        self.ref = ref
        self.value = value

    def __repr__(self) -> str:
        return f"MockRefObject({self.name!r}, {self.ref!r})"


class TestObjectDict:
    """Tests for object_dict function."""

    def test_empty_list_returns_empty_dict(self):
        """Empty list should return empty dictionary."""
        result = object_dict([])
        assert result == {}
        assert isinstance(result, dict)

    def test_single_object_by_name(self):
        """Single object should be keyed by name."""
        obj = MockNamedObject("test")
        result = object_dict([obj])

        assert result == {"test": obj}
        assert result["test"] is obj

    def test_multiple_objects_by_name(self):
        """Multiple objects should be keyed by their names."""
        obj1 = MockNamedObject("first", "value1")
        obj2 = MockNamedObject("second", "value2")
        obj3 = MockNamedObject("third", "value3")

        result = object_dict([obj1, obj2, obj3])

        assert result == {"first": obj1, "second": obj2, "third": obj3}
        assert result["first"] is obj1
        assert result["second"] is obj2
        assert result["third"] is obj3

    def test_preserves_insertion_order(self):
        """Dictionary should preserve insertion order (Python 3.7+)."""
        objs = [MockNamedObject(f"obj{i}") for i in range(5)]
        result = object_dict(objs)

        assert list(result.keys()) == [f"obj{i}" for i in range(5)]

    def test_duplicate_name_raises_model_error(self):
        """Duplicate names should raise ModelError."""
        obj1 = MockNamedObject("duplicate")
        obj2 = MockNamedObject("duplicate")

        with pytest.raises(ModelError) as exc_info:
            object_dict([obj1, obj2])

        assert "duplicate" in str(exc_info.value)
        assert "Duplicate key duplicate" in str(exc_info.value)

    def test_custom_error_message(self):
        """Custom error message should be used for duplicates."""
        obj1 = MockNamedObject("key")
        obj2 = MockNamedObject("key")

        with pytest.raises(ModelError) as exc_info:
            object_dict([obj1, obj2], error_message="Custom error: {key}")

        assert "Custom error: key" in str(exc_info.value)

    def test_error_message_with_context(self):
        """Error message should support context formatting."""
        obj1 = MockNamedObject("mykey")
        obj2 = MockNamedObject("mykey")

        with pytest.raises(ModelError) as exc_info:
            object_dict(
                [obj1, obj2],
                error_message="Duplicate {key} in {container}",
                error_dict={"container": "test_container"},
            )

        assert "Duplicate mykey in test_container" in str(exc_info.value)

    def test_by_ref_uses_ref_attribute(self):
        """by_ref=True should use ref attribute as key."""
        obj = MockRefObject("name1", "ref1", "value1")

        result = object_dict([obj], by_ref=True)

        assert "ref1" in result
        assert "name1" not in result
        assert result["ref1"] is obj

    def test_by_ref_multiple_objects(self):
        """Multiple objects with by_ref=True."""
        obj1 = MockRefObject("name1", "ref_a", "value1")
        obj2 = MockRefObject("name2", "ref_b", "value2")
        obj3 = MockRefObject("name3", "ref_c", "value3")

        result = object_dict([obj1, obj2, obj3], by_ref=True)

        assert result == {"ref_a": obj1, "ref_b": obj2, "ref_c": obj3}

    def test_duplicate_ref_raises_error(self):
        """Duplicate refs should raise error when by_ref=True."""
        obj1 = MockRefObject("name1", "sameref")
        obj2 = MockRefObject("name2", "sameref")

        with pytest.raises(ModelError) as exc_info:
            object_dict([obj1, obj2], by_ref=True)

        assert "sameref" in str(exc_info.value)

    def test_model_objects_by_name(self):
        """Test with actual ModelObject instances by name."""
        obj1 = ModelObject(name="dim1", label="Dimension 1")
        obj2 = ModelObject(name="dim2", label="Dimension 2")

        result = object_dict([obj1, obj2])

        assert result["dim1"] is obj1
        assert result["dim2"] is obj2
        assert result["dim1"].label == "Dimension 1"

    def test_mixed_object_types_by_name(self):
        """Dict can contain different object types as long as they have name."""
        model_obj = ModelObject(name="model")
        mock_obj = MockNamedObject("mock")

        result = object_dict([model_obj, mock_obj])

        assert result["model"] is model_obj
        assert result["mock"] is mock_obj

    def test_returns_regular_dict_not_ordereddict(self):
        """Should return regular dict, not OrderedDict."""
        objs = [MockNamedObject(f"obj{i}") for i in range(3)]
        result = object_dict(objs)

        assert type(result) is dict

    def test_error_message_defaults(self):
        """Default error message when no custom message provided."""
        obj1 = MockNamedObject("testkey")
        obj2 = MockNamedObject("testkey")

        with pytest.raises(ModelError) as exc_info:
            object_dict([obj1, obj2])

        assert str(exc_info.value) == "Duplicate key testkey"

    def test_error_dict_optional(self):
        """error_dict parameter is optional."""
        obj1 = MockNamedObject("key")
        obj2 = MockNamedObject("key")

        # Should work without error_dict
        with pytest.raises(ModelError):
            object_dict([obj1, obj2], error_message="Error: {key}")


class TestObjectDictEdgeCases:
    """Edge case tests for object_dict."""

    def test_objects_with_same_name_different_values(self):
        """Duplicate detection should be based on name, not value."""
        obj1 = MockNamedObject("same", "value1")
        obj2 = MockNamedObject("same", "value2")

        with pytest.raises(ModelError):
            object_dict([obj1, obj2])

    def test_large_number_of_objects(self):
        """Should handle large numbers of objects efficiently."""
        objs = [MockNamedObject(f"obj_{i:04d}") for i in range(1000)]

        result = object_dict(objs)

        assert len(result) == 1000
        assert all(f"obj_{i:04d}" in result for i in range(1000))

    def test_special_characters_in_names(self):
        """Should handle special characters in names."""
        obj1 = MockNamedObject("name-with-dashes")
        obj2 = MockNamedObject("name_with_underscores")
        obj3 = MockNamedObject("name.with.dots")
        obj4 = MockNamedObject("NameWithCaps123")

        result = object_dict([obj1, obj2, obj3, obj4])

        assert result["name-with-dashes"] is obj1
        assert result["name_with_underscores"] is obj2
        assert result["name.with.dots"] is obj3
        assert result["NameWithCaps123"] is obj4

    def test_unicode_names(self):
        """Should handle unicode names."""
        obj1 = MockNamedObject("名前")  # Japanese
        obj2 = MockNamedObject("имя")  # Russian
        obj3 = MockNamedObject("🔑")  # Emoji

        result = object_dict([obj1, obj2, obj3])

        assert result["名前"] is obj1
        assert result["имя"] is obj2
        assert result["🔑"] is obj3


class TestModelError:
    """Tests for ModelError exception."""

    def test_is_exception(self):
        """ModelError should be an Exception."""
        assert issubclass(ModelError, Exception)

    def test_can_be_raised_with_message(self):
        """Should be raiseable with a message."""
        with pytest.raises(ModelError) as exc_info:
            raise ModelError("test message")

        assert str(exc_info.value) == "test message"

    def test_can_be_caught_as_exception(self):
        """Should be catchable as Exception."""
        try:
            raise ModelError("test")
        except Exception as e:
            assert isinstance(e, ModelError)
            assert str(e) == "test"
