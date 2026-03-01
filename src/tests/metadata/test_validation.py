"""Tests for model validation layer.

Tests the validation layer that replaces jsonschema with Pydantic validation.
Covers validation errors, warnings, duplicate detection, and join validation.
"""

import pytest

from cubes.metadata.validation import (
    ModelMetadataValidator,
    ValidationError,
    validate_model,
)


class TestValidationError:
    """Tests for ValidationError named tuple."""

    def test_validation_error_creation(self) -> None:
        """Test creating a ValidationError."""
        error = ValidationError(
            severity="error",
            scope="cube",
            object="sales",
            property="name",
            message="Name is required",
        )

        assert error.severity == "error"
        assert error.scope == "cube"
        assert error.object == "sales"
        assert error.property == "name"
        assert error.message == "Name is required"

    def test_validation_error_unpacking(self) -> None:
        """Test unpacking ValidationError tuple."""
        error = ValidationError(
            severity="default",
            scope="dimension",
            object="date",
            property=None,
            message="Warning message",
        )

        severity, scope, obj, prop, message = error
        assert severity == "default"
        assert scope == "dimension"
        assert obj == "date"
        assert prop is None
        assert message == "Warning message"


class TestValidateModel:
    """Tests for validate_model() function."""

    def test_empty_metadata(self) -> None:
        """Test validating empty metadata returns no errors."""
        errors = validate_model({})
        assert errors == []

    def test_valid_model(self) -> None:
        """Test validating a valid model with cubes and dimensions."""
        metadata = {
            "cubes": [
                {
                    "name": "sales",
                    "dimensions": ["date"],
                }
            ],
            "dimensions": [
                {
                    "name": "date",
                    "levels": [{"name": "year", "attributes": ["year"]}],
                }
            ],
        }

        errors = validate_model(metadata)
        # Should have warning about missing default_hierarchy_name
        assert len(errors) == 1
        assert errors[0].severity == "default"
        assert "default hierarchy" in errors[0].message.lower()

    def test_duplicate_dimension_names(self) -> None:
        """Test detecting duplicate dimension names."""
        metadata = {
            "dimensions": [
                {"name": "date", "levels": [{"name": "year"}]},
                {"name": "date", "levels": [{"name": "month"}]},
            ]
        }

        errors = validate_model(metadata)

        dim_errors = [e for e in errors if "Duplicate dimension" in e.message]
        assert len(dim_errors) == 1
        assert dim_errors[0].severity == "error"
        assert "date" in dim_errors[0].message

    def test_duplicate_cube_names(self) -> None:
        """Test detecting duplicate cube names."""
        metadata = {
            "cubes": [
                {"name": "sales"},
                {"name": "sales"},
            ]
        }

        errors = validate_model(metadata)

        cube_errors = [e for e in errors if "Duplicate cube" in e.message]
        assert len(cube_errors) == 1
        assert cube_errors[0].severity == "error"
        assert "sales" in cube_errors[0].message


class TestDimensionValidation:
    """Tests for dimension validation."""

    def test_string_dimension_warning(self) -> None:
        """Test that string dimension names generate warnings."""
        metadata = {
            "dimensions": ["date"],
        }

        errors = validate_model(metadata)

        warnings = [e for e in errors if e.severity == "default"]
        assert len(warnings) >= 1
        assert any("not described" in e.message for e in warnings)
        assert any("flat single-attribute" in e.message for e in warnings)

    def test_missing_levels_and_attributes(self) -> None:
        """Test warning when dimension has neither levels nor attributes."""
        metadata = {
            "dimensions": [
                {"name": "empty_dim"},
            ]
        }

        errors = validate_model(metadata)

        warnings = [e for e in errors if "Neither levels nor attributes" in e.message]
        assert len(warnings) == 1
        assert warnings[0].severity == "default"
        assert warnings[0].object == "empty_dim"

    def test_both_levels_and_attributes_error(self) -> None:
        """Test error when dimension has both levels and attributes."""
        metadata = {
            "dimensions": [
                {
                    "name": "bad_dim",
                    "levels": [{"name": "year"}],
                    "attributes": [{"name": "month"}],
                }
            ]
        }

        errors = validate_model(metadata)

        level_attr_errors = [
            e for e in errors if "Both levels and attributes" in e.message
        ]
        assert len(level_attr_errors) == 1
        assert level_attr_errors[0].severity == "error"
        assert level_attr_errors[0].object == "bad_dim"

    def test_missing_default_hierarchy_warning(self) -> None:
        """Test warning when default hierarchy name is not specified."""
        metadata = {
            "dimensions": [
                {
                    "name": "date",
                    "levels": [{"name": "year", "attributes": ["year"]}],
                }
            ]
        }

        errors = validate_model(metadata)

        warnings = [e for e in errors if "default hierarchy" in e.message.lower()]
        assert len(warnings) == 1
        assert warnings[0].severity == "default"
        assert warnings[0].object == "date"

    def test_dimension_with_levels_valid(self) -> None:
        """Test valid dimension with levels."""
        metadata = {
            "dimensions": [
                {
                    "name": "date",
                    "levels": [
                        {
                            "name": "year",
                            "attributes": [{"name": "year"}],
                        }
                    ],
                    "default_hierarchy_name": "default",
                }
            ]
        }

        errors = validate_model(metadata)
        # Should only have schema validation warnings, no errors
        schema_errors = [e for e in errors if e.severity == "error"]
        assert len(schema_errors) == 0


class TestCubeValidation:
    """Tests for cube validation."""

    def test_cube_without_name(self) -> None:
        """Test error when cube is missing name field."""
        metadata = {
            "cubes": [
                {"dimensions": ["date"]},
            ]
        }

        errors = validate_model(metadata)

        name_errors = [e for e in errors if e.property == "name"]
        assert len(name_errors) >= 1
        assert all(e.scope == "cube" for e in name_errors)

    def test_cube_with_valid_dimensions(self) -> None:
        """Test valid cube with dimension links."""
        metadata = {
            "cubes": [
                {
                    "name": "sales",
                    "dimensions": ["date", "product"],
                }
            ]
        }

        errors = validate_model(metadata)
        # Should have no cube-level errors
        cube_errors = [e for e in errors if e.scope == "cube" and e.severity == "error"]
        assert len(cube_errors) == 0

    def test_cube_join_missing_master(self) -> None:
        """Test error when join is missing master field."""
        metadata = {
            "cubes": [
                {
                    "name": "sales",
                    "joins": [
                        {"detail": "product.id"},
                    ],
                }
            ]
        }

        errors = validate_model(metadata)

        master_errors = [e for e in errors if "master" in e.message.lower()]
        assert len(master_errors) == 1
        assert master_errors[0].severity == "error"
        assert "master" in master_errors[0].message

    def test_cube_join_missing_detail(self) -> None:
        """Test error when join is missing detail field."""
        metadata = {
            "cubes": [
                {
                    "name": "sales",
                    "joins": [
                        {"master": "sales.product_id"},
                    ],
                }
            ]
        }

        errors = validate_model(metadata)

        detail_errors = [e for e in errors if "detail" in e.message.lower()]
        assert len(detail_errors) == 1
        assert detail_errors[0].severity == "error"
        assert "detail" in detail_errors[0].message

    def test_cube_join_valid(self) -> None:
        """Test valid cube with complete join."""
        metadata = {
            "cubes": [
                {
                    "name": "sales",
                    "joins": [
                        {
                            "master": "sales.product_id",
                            "detail": "product.id",
                        },
                    ],
                }
            ]
        }

        errors = validate_model(metadata)

        join_errors = [e for e in errors if e.property and "join" in e.property.lower()]
        assert len(join_errors) == 0

    def test_multiple_joins_validation(self) -> None:
        """Test validation of multiple joins in a cube."""
        metadata = {
            "cubes": [
                {
                    "name": "sales",
                    "joins": [
                        {"master": "sales.product_id"},  # Missing detail
                        {"detail": "date.id"},  # Missing master
                        {
                            "master": "sales.store_id",
                            "detail": "store.id",
                        },  # Valid
                    ],
                }
            ]
        }

        errors = validate_model(metadata)

        # Should have 2 errors (one for each incomplete join)
        join_errors = [e for e in errors if e.property and "joins" in e.property]
        assert len(join_errors) == 2


class TestModelMetadataValidator:
    """Tests for ModelMetadataValidator class."""

    def test_validator_initialization(self) -> None:
        """Test creating a validator instance."""
        metadata = {"cubes": [], "dimensions": []}
        validator = ModelMetadataValidator(metadata)

        assert validator.metadata == metadata

    def test_validator_validate_method(self) -> None:
        """Test running full validation."""
        metadata = {
            "dimensions": [
                {
                    "name": "date",
                    "levels": [{"name": "year", "attributes": ["year"]}],
                }
            ]
        }

        validator = ModelMetadataValidator(metadata)
        errors = validator.validate()

        # Should have default hierarchy warning
        assert len(errors) == 1
        assert errors[0].severity == "default"


class TestPydanticErrorConversion:
    """Tests for Pydantic error conversion to ValidationError format."""

    def test_pydantic_validation_error_conversion(self) -> None:
        """Test that Pydantic errors are converted to ValidationError format."""
        metadata = {
            "cubes": [
                {
                    "name": "sales",
                    "measures": [
                        {"name": 123},  # Invalid: name should be string
                    ],
                }
            ]
        }

        errors = validate_model(metadata)

        # Should have schema validation errors
        schema_errors = [
            e for e in errors if e.scope == "cube" and e.severity == "error"
        ]
        assert len(schema_errors) >= 1

    def test_error_property_path_format(self) -> None:
        """Test that error property paths are dot-separated."""
        metadata = {
            "dimensions": [
                {
                    "name": "date",
                    "levels": [
                        {
                            "name": "year",
                            "attributes": [
                                {"name": 123}  # Invalid type
                            ],
                        }
                    ],
                }
            ]
        }

        errors = validate_model(metadata)

        # Find errors with property paths
        path_errors = [e for e in errors if e.property is not None]
        assert len(path_errors) >= 1

        # Check that paths use dot notation
        for error in path_errors:
            if error.property:
                assert "." in error.property or error.property in [
                    "levels",
                    "attributes",
                ]


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_none_metadata(self) -> None:
        """Test that None metadata is handled gracefully."""
        # This should not raise an exception
        with pytest.raises((AttributeError, TypeError)):
            validate_model(None)  # type: ignore[arg-type]

    def test_empty_lists(self) -> None:
        """Test validating metadata with empty lists."""
        metadata = {
            "cubes": [],
            "dimensions": [],
        }

        errors = validate_model(metadata)
        assert errors == []

    def test_non_dict_cubes(self) -> None:
        """Test handling non-dict items in cubes list."""
        metadata = {
            "cubes": [
                "not_a_dict",  # Invalid
                {"name": "valid_cube"},
            ]
        }

        # Should not raise exception
        errors = validate_model(metadata)
        # Should still validate the valid cube
        assert any(e.object == "valid_cube" for e in errors if e.object) or not errors

    def test_non_dict_dimensions(self) -> None:
        """Test handling non-dict items in dimensions list."""
        metadata = {
            "dimensions": [
                123,  # Invalid
                {"name": "valid_dim"},
            ]
        }

        # Should not raise exception
        errors = validate_model(metadata)

    def test_nested_validation_errors(self) -> None:
        """Test validation of deeply nested structures."""
        metadata = {
            "dimensions": [
                {
                    "name": "date",
                    "levels": [
                        {
                            "name": "year",
                            "attributes": [
                                {
                                    "name": "year",
                                    "order": "invalid_order",  # Invalid enum value
                                }
                            ],
                        }
                    ],
                }
            ]
        }

        errors = validate_model(metadata)

        # Should have errors about invalid order
        order_errors = [e for e in errors if "order" in str(e.property)]
        assert len(order_errors) >= 1

    def test_unicode_in_names(self) -> None:
        """Test handling unicode characters in names."""
        metadata = {
            "cubes": [
                {"name": "売上"},  # Japanese for "sales"
            ],
            "dimensions": [
                {"name": "日付"},  # Japanese for "date"
            ],
        }

        # Should not raise exception
        errors = validate_model(metadata)
        # Should have warnings about flat dimensions (neither levels nor attributes)
        assert any("Neither levels nor attributes" in e.message for e in errors)
