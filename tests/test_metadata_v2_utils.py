"""
Tests for the Pydantic-based metadata utility functions.
"""

import unittest
import os
import json
import shutil
from cubes.metadata_v2.base import MetadataObject
from cubes.metadata_v2.utils import make_object_dict, read_model_metadata, read_model_metadata_bundle, write_model_metadata_bundle
from cubes.errors import ModelError, ArgumentError, CubesError


class MockModelObject(MetadataObject):
    """Mock model object for testing"""
    pass


class ObjectDictTestCase(unittest.TestCase):
    """Test case for make_object_dict utility function"""
    
    def test_basic_object_dict(self):
        """Test basic object dictionary creation"""
        obj1 = MockModelObject(name="obj1")
        obj2 = MockModelObject(name="obj2")
        
        result = make_object_dict([obj1, obj2])
        
        self.assertIsInstance(result, dict)
        self.assertEqual(2, len(result))
        self.assertEqual(obj1, result["obj1"])
        self.assertEqual(obj2, result["obj2"])
    
    def test_duplicate_names(self):
        """Test error handling for duplicate names"""
        obj1 = MockModelObject(name="duplicate")
        obj2 = MockModelObject(name="duplicate")
        
        with self.assertRaises(ModelError):
            make_object_dict([obj1, obj2])
    
    def test_by_ref(self):
        """Test make_object_dict with by_ref option"""
        # Create objects with ref attribute
        obj1 = MockModelObject(name="obj1")
        setattr(obj1, 'ref', 'ref1')
        obj2 = MockModelObject(name="obj2")
        setattr(obj2, 'ref', 'ref2')
        
        result = make_object_dict([obj1, obj2], by_ref=True)
        
        self.assertEqual(obj1, result["ref1"])
        self.assertEqual(obj2, result["ref2"])
    
    def test_by_ref_fallback_to_name(self):
        """Test by_ref falls back to name when ref doesn't exist"""
        obj1 = MockModelObject(name="obj1")
        obj2 = MockModelObject(name="obj2")
        
        result = make_object_dict([obj1, obj2], by_ref=True)
        
        self.assertEqual(obj1, result["obj1"])
        self.assertEqual(obj2, result["obj2"])

class ModelIOTestCase(unittest.TestCase):
    def setUp(self):
        self.models_dir = os.path.join(os.path.dirname(__file__), 'models')
        self.temp_dir = 'temp_model_bundle'
        os.makedirs(self.temp_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_read_model_metadata_from_file(self):
        model_path = os.path.join(self.models_dir, 'model.json')
        read_data = read_model_metadata(model_path)
        self.assertEqual('public_procurements', read_data['name'])

    def test_read_model_metadata_bundle(self):
        bundle_path = os.path.join(self.models_dir, 'test.cubesmodel')
        bundle = read_model_metadata_bundle(bundle_path)
        self.assertEqual('public_procurements', bundle['name'])
        self.assertEqual(6, len(bundle['dimensions']))
        self.assertEqual(1, len(bundle['cubes']))

    def test_write_model_metadata_bundle(self):
        metadata = {
            'name': 'test_write_bundle',
            'dimensions': [{'name': 'dim1'}],
            'cubes': [{'name': 'cube1'}]
        }

        write_model_metadata_bundle(self.temp_dir, metadata, replace=True)

        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'model.json')))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'dim_dim1.json')))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'cube_cube1.json')))


if __name__ == '__main__':
    unittest.main()