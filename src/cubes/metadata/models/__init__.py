"""Metadata models package."""

from cubes.metadata.models.attributes import Attribute, AttributeBase, inspect_variables
from cubes.metadata.models.base import ModelError, ModelObject, object_dict

__all__ = [
    "Attribute",
    "AttributeBase",
    "inspect_variables",
    "ModelError",
    "ModelObject",
    "object_dict",
]
