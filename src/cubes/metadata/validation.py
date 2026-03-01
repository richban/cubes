"""Model validation layer using Pydantic.

Provides validation for model metadata using Pydantic's native validation
instead of jsonschema. Converts validation errors to a consistent format.
"""

from typing import Any, NamedTuple, Optional

from pydantic import ValidationError as PydanticValidationError

from cubes.metadata.schemas import CubeSchema, DimensionSchema


class ValidationError(NamedTuple):
    """Validation error with structured information.

    Attributes:
        severity: Error severity - 'error' or 'default' (warning).
        scope: Validation scope - 'model', 'cube', or 'dimension'.
        object: Object name being validated (cube/dimension name or None).
        property: Property path where error occurred (dot-separated or None).
        message: Human-readable error message.
    """

    severity: str
    scope: str
    object: Optional[str]
    property: Optional[str]
    message: str


def validate_model(metadata: dict[str, Any]) -> list[ValidationError]:
    """Validate model metadata.

    Validates the entire model including cubes and dimensions.
    Returns a list of validation errors. An empty list indicates
    valid metadata.

    Args:
        metadata: Dictionary containing model metadata with optional
            'cubes' and 'dimensions' keys.

    Returns:
        List of ValidationError tuples. Empty list if validation passes.

    Example:
        >>> metadata = {
        ...     "cubes": [{"name": "sales", "dimensions": ["date"]}],
        ...     "dimensions": [{"name": "date", "levels": [{"name": "year"}]}]
        ... }
        >>> errors = validate_model(metadata)
        >>> if not errors:
        ...     print("Model is valid")
    """
    validator = ModelMetadataValidator(metadata)
    return validator.validate()


class ModelMetadataValidator:
    """Validator for model metadata.

    Validates model, cube, and dimension metadata using Pydantic schemas.
    Detects duplicate names, missing required fields, and schema violations.
    """

    def __init__(self, metadata: dict[str, Any]):
        """Initialize validator with metadata.

        Args:
            metadata: Dictionary containing model metadata.
        """
        self.metadata = metadata

    def validate(self) -> list[ValidationError]:
        """Run all validations and return collected errors.

        Returns:
            List of validation errors found during validation.
        """
        errors: list[ValidationError] = []

        # Validate model-level structure
        errors += self._validate_model_structure()

        # Validate cubes
        if "cubes" in self.metadata:
            errors += self._validate_cubes()

        # Validate dimensions
        if "dimensions" in self.metadata:
            errors += self._validate_dimensions()

        return errors

    def _validate_model_structure(self) -> list[ValidationError]:
        """Validate top-level model structure.

        Returns:
            List of validation errors.
        """
        errors: list[ValidationError] = []

        # Check for duplicate dimension names
        dims = self.metadata.get("dimensions", [])
        if isinstance(dims, list):
            dim_names: dict[str, int] = {}
            for dim in dims:
                if isinstance(dim, dict):
                    name = dim.get("name")
                    if name:
                        if name in dim_names:
                            errors.append(
                                ValidationError(
                                    severity="error",
                                    scope="model",
                                    object=None,
                                    property="dimensions",
                                    message=f"Duplicate dimension name: '{name}'",
                                )
                            )
                        dim_names[name] = dim_names.get(name, 0) + 1

        # Check for duplicate cube names
        cubes = self.metadata.get("cubes", [])
        if isinstance(cubes, list):
            cube_names: dict[str, int] = {}
            for cube in cubes:
                if isinstance(cube, dict):
                    name = cube.get("name")
                    if name:
                        if name in cube_names:
                            errors.append(
                                ValidationError(
                                    severity="error",
                                    scope="model",
                                    object=None,
                                    property="cubes",
                                    message=f"Duplicate cube name: '{name}'",
                                )
                            )
                        cube_names[name] = cube_names.get(name, 0) + 1

        return errors

    def _validate_cubes(self) -> list[ValidationError]:
        """Validate all cubes in the model.

        Returns:
            List of validation errors.
        """
        errors: list[ValidationError] = []
        cubes = self.metadata.get("cubes", [])

        if not isinstance(cubes, list):
            return errors

        for cube in cubes:
            if not isinstance(cube, dict):
                continue

            name = cube.get("name", "<unnamed>")
            errors += self._validate_cube(cube, name)

        return errors

    def _validate_cube(self, cube: dict[str, Any], name: str) -> list[ValidationError]:
        """Validate a single cube using Pydantic schema.

        Args:
            cube: Cube metadata dictionary.
            name: Cube name for error reporting.

        Returns:
            List of validation errors.
        """
        errors: list[ValidationError] = []

        try:
            CubeSchema.model_validate(cube)
        except PydanticValidationError as e:
            for error in e.errors():
                # Convert Pydantic error path to dot-separated string
                path_parts = [str(p) for p in error["loc"]]
                prop = ".".join(path_parts) if path_parts else None

                errors.append(
                    ValidationError(
                        severity="error",
                        scope="cube",
                        object=name,
                        property=prop,
                        message=error["msg"],
                    )
                )

        # Validate joins have required master and detail fields
        errors += self._validate_joins(cube, name)

        return errors

    def _validate_joins(self, cube: dict[str, Any], name: str) -> list[ValidationError]:
        """Validate that joins have required master and detail fields.

        Args:
            cube: Cube metadata dictionary.
            name: Cube name for error reporting.

        Returns:
            List of validation errors.
        """
        errors: list[ValidationError] = []
        joins = cube.get("joins", [])

        if not isinstance(joins, list):
            return errors

        for i, join in enumerate(joins):
            if not isinstance(join, dict):
                continue

            if "master" not in join:
                errors.append(
                    ValidationError(
                        severity="error",
                        scope="cube",
                        object=name,
                        property=f"joins.{i}.master",
                        message="Join is missing required 'master' field",
                    )
                )

            if "detail" not in join:
                errors.append(
                    ValidationError(
                        severity="error",
                        scope="cube",
                        object=name,
                        property=f"joins.{i}.detail",
                        message="Join is missing required 'detail' field",
                    )
                )

        return errors

    def _validate_dimensions(self) -> list[ValidationError]:
        """Validate all dimensions in the model.

        Returns:
            List of validation errors.
        """
        errors: list[ValidationError] = []
        dimensions = self.metadata.get("dimensions", [])

        if not isinstance(dimensions, list):
            return errors

        for dim in dimensions:
            if isinstance(dim, str):
                # String dimension names are warnings - they create flat dimensions
                errors.append(
                    ValidationError(
                        severity="default",
                        scope="model",
                        object=None,
                        property="dimensions",
                        message=f"Dimension '{dim}' is not described, "
                        "creating flat single-attribute dimension",
                    )
                )
                continue

            if not isinstance(dim, dict):
                continue

            name = dim.get("name", "<unnamed>")
            errors += self._validate_dimension(dim, name)

        return errors

    def _validate_dimension(
        self, dim: dict[str, Any], name: str
    ) -> list[ValidationError]:
        """Validate a single dimension using Pydantic schema.

        Args:
            dim: Dimension metadata dictionary.
            name: Dimension name for error reporting.

        Returns:
            List of validation errors.
        """
        errors: list[ValidationError] = []

        # Check mutually exclusive levels vs attributes (legacy behavior)
        has_levels = "levels" in dim
        has_attributes = "attributes" in dim

        if not has_levels and not has_attributes:
            errors.append(
                ValidationError(
                    severity="default",
                    scope="dimension",
                    object=name,
                    property=None,
                    message="Neither levels nor attributes specified, "
                    "creating flat dimension without details",
                )
            )
        elif has_levels and has_attributes:
            errors.append(
                ValidationError(
                    severity="error",
                    scope="dimension",
                    object=name,
                    property=None,
                    message="Both levels and attributes specified",
                )
            )

        # Check for default hierarchy warning (legacy behavior)
        if not has_levels and not has_attributes:
            # Flat dimension, no hierarchy needed
            pass
        elif "default_hierarchy_name" not in dim:
            errors.append(
                ValidationError(
                    severity="default",
                    scope="dimension",
                    object=name,
                    property=None,
                    message="No default hierarchy name specified, using first one",
                )
            )

        try:
            DimensionSchema.model_validate(dim)
        except PydanticValidationError as e:
            for error in e.errors():
                # Convert Pydantic error path to dot-separated string
                path_parts = [str(p) for p in error["loc"]]
                prop = ".".join(path_parts) if path_parts else None

                errors.append(
                    ValidationError(
                        severity="error",
                        scope="dimension",
                        object=name,
                        property=prop,
                        message=error["msg"],
                    )
                )

        return errors
