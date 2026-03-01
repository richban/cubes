"""Cubes metadata package."""

from cubes.metadata.io import (
    MetadataIOError,
    MetadataNotFoundError,
    read_model_metadata,
    read_model_metadata_bundle,
    write_model_metadata_bundle,
)
from cubes.metadata.models import (
    ExpressionError,
    ModelError,
    ModelObject,
    object_dict,
)
from cubes.metadata.provider import (
    MetadataProvider,
    NoSuchCubeError,
    NoSuchDimensionError,
    TemplateRequiredError,
)
from cubes.metadata.static_provider import StaticModelProvider
from cubes.metadata.validation import (
    ValidationError,
    validate_model,
)

__all__ = [
    # I/O functions
    "read_model_metadata",
    "read_model_metadata_bundle",
    "write_model_metadata_bundle",
    "MetadataIOError",
    "MetadataNotFoundError",
    # Model classes and utilities
    "ExpressionError",
    "ModelError",
    "ModelObject",
    "object_dict",
    # Provider classes
    "MetadataProvider",
    "NoSuchCubeError",
    "NoSuchDimensionError",
    "TemplateRequiredError",
    "StaticModelProvider",
    # Validation
    "ValidationError",
    "validate_model",
]
