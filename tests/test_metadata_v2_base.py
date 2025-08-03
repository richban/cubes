"""
Tests for the Pydantic-based base metadata classes.
"""

import unittest
from cubes.metadata_v2.base import MetadataObject
from pydantic import ValidationError

from cubes.errors import ModelError, ArgumentError


class MockModelObject(MetadataObject):
    """Mock model object for testing"""
    pass


class MetadataObjectTestCase(unittest.TestCase):
    """Test case for MetadataObject base class"""
    
    def test_basic_initialization(self):
        """Test basic object creation"""
        obj = MockModelObject(name="test", label="Test Object")
        self.assertEqual("test", obj.name)
        self.assertEqual("Test Object", obj.label)
        self.assertIsNone(obj.description)
        self.assertEqual({}, obj.info)
    
    def test_info_defaults_to_dict(self):
        """Test that info always defaults to empty dict"""
        obj = MockModelObject(name="test")
        self.assertEqual({}, obj.info)
        
        obj = MockModelObject(name="test", info=None)
        self.assertEqual({}, obj.info)
    
    def test_name_validation(self):
        """Test name validation"""
        # Valid names
        MockModelObject(name="valid_name")
        MockModelObject(name="valid-name")
        MockModelObject(name=None)  # None is allowed
        
        # Invalid names
        with self.assertRaises(ValueError):
            MockModelObject(name="")
        
        with self.assertRaises(ValueError):
            MockModelObject(name="   ")  # Only whitespace
    
    def test_info_validation(self):
        """Test info field validation"""
        # Valid info
        obj = MockModelObject(name="test", info={"key": "value"})
        self.assertEqual({"key": "value"}, obj.info)
        
        # Invalid info type
        with self.assertRaises(ValueError):
            MockModelObject(name="test", info="not a dict")
    
    def test_to_dict(self):
        """Test to_dict method"""
        obj = MockModelObject(
            name="test",
            label="Test Label",
            description="Test description",
            info={"custom": "value"}
        )
        
        result = obj.to_dict()
        
        self.assertEqual("test", result["name"])
        self.assertEqual("Test Label", result["label"])
        self.assertEqual("Test description", result["description"])
        self.assertEqual({"custom": "value"}, result["info"])
    
    def test_get_label(self):
        """Test get_label method"""
        # With explicit label
        obj = MockModelObject(name="test_name", label="Custom Label")
        self.assertEqual("Custom Label", obj.get_label())
        
        # Without label - should generate from name  
        obj = MockModelObject(name="test_name")
        self.assertEqual("Test Name", obj.get_label())
        
        # With snake_case name
        obj = MockModelObject(name="snake_case_name")
        self.assertEqual("Snake Case Name", obj.get_label())
        
        # With kebab-case name
        obj = MockModelObject(name="kebab-case-name")
        self.assertEqual("Kebab Case Name", obj.get_label())
        
        # Without name or label
        obj = MockModelObject()
        self.assertEqual("<MockModelObject>", obj.get_label())
    
    def test_from_metadata_string(self):
        """Test creating object from string metadata"""
        obj = MockModelObject(name="test_name")
        self.assertEqual("test_name", obj.name)
        self.assertIsNone(obj.label)
        self.assertEqual({}, obj.info)
    
    def test_from_metadata_dict(self):
        """Test creating object from dictionary metadata"""
        metadata = {
            "name": "test",
            "label": "Test Label",
            "description": "Test description",
            "info": {"key": "value"}
        }
        
        obj = MockModelObject(**metadata)
        self.assertEqual("test", obj.name)
        self.assertEqual("Test Label", obj.label)
        self.assertEqual("Test description", obj.description)
        self.assertEqual({"key": "value"}, obj.info)
    
    def test_from_metadata_invalid(self):
        """Test error handling for invalid metadata"""
        with self.assertRaises(ValidationError):
            MockModelObject(name=123)  # Invalid type
    
    def test_str_and_repr(self):
        """Test string representations"""
        obj = MockModelObject(name="test")
        self.assertEqual("test", str(obj))
        self.assertEqual("MockModelObject(name='test')", repr(obj))
        
        # Test with no name
        obj = MockModelObject()
        self.assertEqual("<MockModelObject>", str(obj))
    
    def test_equality(self):
        """Test object equality"""
        obj1 = MockModelObject(name="test", label="Label")
        obj2 = MockModelObject(name="test", label="Label")
        obj3 = MockModelObject(name="test", label="Different")
        
        self.assertEqual(obj1, obj2)
        self.assertNotEqual(obj1, obj3)
        self.assertNotEqual(obj1, "not an object")
    
    def test_hash(self):
        """Test object hashing"""
        obj1 = MockModelObject(name="test")
        obj2 = MockModelObject(name="test")
        obj3 = MockModelObject(name="different")
        
        # Objects with same name should have same hash
        self.assertEqual(hash(obj1), hash(obj2))
        # Objects with different names should have different hashes
        self.assertNotEqual(hash(obj1), hash(obj3))





if __name__ == '__main__':
    unittest.main()