"""Cubes metadata package."""

from cubes.metadata.io import (
    MetadataIOError,
    MetadataNotFoundError,
    read_model_metadata,
    read_model_metadata_bundle,
    write_model_metadata_bundle,
)
from cubes.metadata.models import ModelError, ModelObject, object_dict

__all__ = [
    # I/O functions
    "read_model_metadata",
    "read_model_metadata_bundle",
    "write_model_metadata_bundle",
    "MetadataIOError",
    "MetadataNotFoundError",
    # Model classes and utilities
    "ModelError",
    "ModelObject",
    "object_dict",
]
