"""
Cubes Metadata V2 - Modern Pydantic-based metadata models

This module provides Pydantic-based implementations of the Cubes metadata classes,
offering improved type safety, validation, and developer experience while maintaining
compatibility with the existing metadata system.

Key improvements over cubes.metadata:
- Runtime type validation with Pydantic
- Declarative field validators
- Built-in JSON serialization/deserialization
- Better IDE support and error messages
- Improved performance with Pydantic V2
"""

from .base import MetadataObject

# Import other modules as they become available
__all__ = ["MetadataObject"]

try:
    from .attributes import AttributeBase, Attribute, Measure, MeasureAggregate

    __all__.extend(["AttributeBase", "Attribute", "Measure", "MeasureAggregate"])
except ImportError:
    pass

try:
    from .dimension import Level, Hierarchy, Dimension

    __all__.extend(["Level", "Hierarchy", "Dimension"])
except ImportError:
    pass

try:
    from .cube import Cube

    __all__.append("Cube")
except ImportError:
    pass

try:
    from .model import Model, merge_models, create_model_from_cubes

    __all__.extend(["Model", "merge_models", "create_model_from_cubes"])
except ImportError:
    pass

__version__ = "2.0.0"
