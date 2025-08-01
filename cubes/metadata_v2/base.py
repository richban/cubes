"""
Modern Pydantic-based base classes for Cubes metadata models.

This module provides clean, modern Python base classes using Pydantic
for validation, serialization, and type safety without backward compatibility
concerns.
"""

from collections import OrderedDict
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..common import to_label
from ..errors import ArgumentError, ModelError


class MetadataObject(BaseModel):
    """
    Modern base class for all Cubes metadata objects.

    Uses Pydantic for validation, serialization, and type safety.
    Clean, modern Python without backward compatibility baggage.
    """

    model_config = ConfigDict(
        # Validate assignments for type safety
        validate_assignment=True,
        # Allow extra fields for extensibility
        extra="allow",
        # Use enum values for serialization
        use_enum_values=True,
        # Allow arbitrary types if needed
        arbitrary_types_allowed=True,
        # Don't freeze by default - allow mutation
        frozen=False,
        # Populate by field name
        populate_by_name=True,
    )

    # Core metadata fields
    name: str | None = Field(None, description="Unique identifier")
    label: str | None = Field(None, description="Human-readable label")
    description: str | None = Field(None, description="Detailed description")
    info: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    # Localization support for subclasses
    localizable_attributes: ClassVar[list[str]] = ["label", "description"]
    localizable_lists: ClassVar[list[str]] = []

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str | None) -> str | None:
        """Validate that name is a non-empty string when provided."""
        if v is not None and (not isinstance(v, str) or not v.strip()):
            raise ValueError("name must be a non-empty string")
        return v

    @field_validator("info", mode="before")
    @classmethod
    def validate_info(cls, v: Any) -> dict[str, Any]:
        """Ensure info is always a dictionary."""
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError("info must be a dictionary")
        return v

    def get_label(self) -> str:
        """Get display label, using name as fallback."""
        if self.label:
            return self.label
        if self.name:
            return to_label(self.name)
        return f"<{self.__class__.__name__}>"

    def localized(self, context) -> "MetadataObject":
        """
        Returns a localized copy of the object.

        Args:
            context: Localization context with translation data

        Returns:
            Localized copy of the object
        """
        # Create a deep copy
        localized_copy = self.model_copy(deep=True)

        # Apply localizable attributes
        for attr in self.localizable_attributes:
            if hasattr(context, "get"):
                localized_value = context.get(attr, getattr(self, attr))
                setattr(localized_copy, attr, localized_value)

        # Apply localizable lists
        for attr in self.localizable_lists:
            if hasattr(localized_copy, attr):
                current_list = getattr(localized_copy, attr, [])
                localized_list = []

                for obj in current_list:
                    if hasattr(context, "object_localization") and hasattr(
                        obj, "localized"
                    ):
                        obj_context = context.object_localization(attr, obj.name)
                        localized_list.append(obj.localized(obj_context))
                    else:
                        localized_list.append(obj)

                setattr(localized_copy, attr, localized_list)

        return localized_copy

    @classmethod
    def from_metadata(cls, metadata, templates: dict | None = None):
        """
        Create instance from metadata.

        Args:
            metadata: String name or dictionary metadata
            templates: Optional templates dictionary for subclasses

        Returns:
            New instance of the class

        Raises:
            ArgumentError: If metadata type is invalid
            ModelError: If object creation fails
        """
        if isinstance(metadata, str):
            return cls(name=metadata)
        elif isinstance(metadata, dict):
            try:
                return cls(**metadata)
            except Exception as e:
                raise ModelError(f"Failed to create {cls.__name__}: {e}")
        else:
            raise ArgumentError(f"Invalid metadata type: {type(metadata)}")

    def __str__(self) -> str:
        """String representation using name or class name."""
        return self.name or f"<{self.__class__.__name__}>"

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"{self.__class__.__name__}(name={self.name!r})"

    def to_dict(
        self, create_label: bool | None = None, **options: Any
    ) -> dict[str, Any]:
        """
        Convert to dictionary representation using Pydantic's model_dump.

        Args:
            create_label: If True, generate label from name if not set.
            **options: Additional options passed to Pydantic's model_dump.

        Returns:
            Dictionary representation of the object.
        """
        # Use Pydantic's model_dump for serialization
        result = self.model_dump(exclude_none=True, **options)

        # Preserve the custom label creation logic
        if create_label and self.name and "label" not in result:
            result["label"] = self.get_label()

        return result

    def __hash__(self) -> int:
        """Hash based on name for use in sets/dicts."""
        return hash(self.name) if self.name else id(self)
