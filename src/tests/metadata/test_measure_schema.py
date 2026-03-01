"""Tests for Measure Pydantic Schema (US-007).

Tests for measure validation including:
- Field validation (formula, aggregates, nonadditive, window_size)
- Nonadditive normalization (none->None, all/any->all, time->time)
- Aggregate function validation
- String coercion support
- Integration with AttributeSchema base fields
"""

import pytest
from pydantic import ValidationError

from cubes.metadata.schemas import MeasureSchema


class TestMeasureSchemaBasic:
    """Test basic measure schema creation and fields."""

    def test_minimal_measure(self):
        """Measure with just name should use defaults."""
        measure = MeasureSchema(name="amount")
        assert measure.name == "amount"
        assert measure.label is None
        assert measure.description is None
        assert measure.aggregates == ["sum"]
        assert measure.nonadditive is None
        assert measure.window_size is None
        assert measure.formula is None

    def test_full_measure(self):
        """Measure with all fields set."""
        measure = MeasureSchema(
            name="revenue",
            label="Total Revenue",
            description="Sum of all revenue",
            order="asc",
            info={"currency": "USD"},
            format=".2f",
            missing_value=0,
            expression="price * quantity",
            formula="revenue_formula",
            aggregates=["sum", "avg", "max"],
            nonadditive="time",
            window_size=30,
        )
        assert measure.name == "revenue"
        assert measure.label == "Total Revenue"
        assert measure.description == "Sum of all revenue"
        assert measure.order == "asc"
        assert measure.info == {"currency": "USD"}
        assert measure.format == ".2f"
        assert measure.missing_value == 0
        assert measure.expression == "price * quantity"
        assert measure.formula == "revenue_formula"
        assert measure.aggregates == ["sum", "avg", "max"]
        assert measure.nonadditive == "time"
        assert measure.window_size == 30


class TestMeasureSchemaStringCoercion:
    """Test string coercion to measure dict."""

    def test_string_to_measure(self):
        """String input should be coerced to measure with name."""
        measure = MeasureSchema.model_validate("sales")
        assert measure.name == "sales"
        assert measure.aggregates == ["sum"]

    def test_dict_input_unchanged(self):
        """Dict input should work normally."""
        measure = MeasureSchema.model_validate(
            {"name": "profit", "aggregates": ["sum", "avg"]}
        )
        assert measure.name == "profit"
        assert measure.aggregates == ["sum", "avg"]


class TestMeasureSchemaNonadditiveNormalization:
    """Test nonadditive field normalization."""

    @pytest.mark.parametrize(
        "input_value,expected",
        [
            (None, None),
            ("none", None),
            ("NONE", None),
            ("None", None),
            ("all", "all"),
            ("ALL", "all"),
            ("All", "all"),
            ("any", "all"),
            ("ANY", "all"),
            ("Any", "all"),
            ("time", "time"),
            ("TIME", "time"),
            ("Time", "time"),
        ],
    )
    def test_nonadditive_normalization(self, input_value, expected):
        """Test various nonadditive input normalizations."""
        measure = MeasureSchema(name="amount", nonadditive=input_value)
        assert measure.nonadditive == expected

    def test_invalid_nonadditive(self):
        """Invalid nonadditive value should raise error."""
        with pytest.raises(ValidationError) as exc_info:
            MeasureSchema(name="amount", nonadditive="invalid")
        assert "Unknown non-additive type" in str(exc_info.value)

    def test_nonadditive_non_string(self):
        """Non-string nonadditive should raise error."""
        with pytest.raises(ValidationError) as exc_info:
            MeasureSchema(name="amount", nonadditive=123)
        assert "Nonadditive must be a string" in str(exc_info.value)


class TestMeasureSchemaAggregatesValidation:
    """Test aggregates field validation."""

    def test_default_aggregates(self):
        """Default aggregates should be ["sum"]."""
        measure = MeasureSchema(name="amount")
        assert measure.aggregates == ["sum"]

    def test_custom_aggregates(self):
        """Custom aggregates should be accepted."""
        measure = MeasureSchema(
            name="amount", aggregates=["count", "avg", "min", "max"]
        )
        assert measure.aggregates == ["count", "avg", "min", "max"]

    def test_invalid_aggregate(self):
        """Invalid aggregate function should raise error."""
        with pytest.raises(ValidationError) as exc_info:
            MeasureSchema(name="amount", aggregates=["invalid_function"])
        assert "Unsupported aggregate functions" in str(exc_info.value)
        assert "invalid_function" in str(exc_info.value)

    def test_multiple_invalid_aggregates(self):
        """Multiple invalid aggregates should all be reported."""
        with pytest.raises(ValidationError) as exc_info:
            MeasureSchema(name="amount", aggregates=["bad1", "bad2", "sum"])
        error_msg = str(exc_info.value)
        assert "bad1" in error_msg
        assert "bad2" in error_msg

    def test_all_supported_aggregates(self):
        """All supported aggregates should be accepted."""
        supported = [
            "sum",
            "count",
            "count_nonempty",
            "min",
            "max",
            "avg",
            "identity",
            "mean",
            "std",
            "var",
            "median",
        ]
        measure = MeasureSchema(name="amount", aggregates=supported)
        assert measure.aggregates == supported

    def test_empty_aggregates(self):
        """Empty aggregates list should be accepted."""
        measure = MeasureSchema(name="amount", aggregates=[])
        assert measure.aggregates == []


class TestMeasureSchemaWindowSize:
    """Test window_size field."""

    def test_window_size_integer(self):
        """Window size should accept integers."""
        measure = MeasureSchema(name="amount", window_size=30)
        assert measure.window_size == 30

    def test_window_size_none(self):
        """Window size can be None."""
        measure = MeasureSchema(name="amount", window_size=None)
        assert measure.window_size is None

    def test_window_size_zero(self):
        """Window size can be zero."""
        measure = MeasureSchema(name="amount", window_size=0)
        assert measure.window_size == 0


class TestMeasureSchemaInheritance:
    """Test inheritance from AttributeSchema."""

    def test_attribute_fields_present(self):
        """Measure should have all AttributeSchema fields."""
        measure = MeasureSchema(
            name="amount",
            label="Amount Label",
            description="Amount Description",
            order="desc",
            info={"key": "value"},
            format=".2f",
            missing_value=0,
            expression="price * quantity",
        )
        assert measure.label == "Amount Label"
        assert measure.description == "Amount Description"
        assert measure.order == "desc"
        assert measure.info == {"key": "value"}
        assert measure.format == ".2f"
        assert measure.missing_value == 0
        assert measure.expression == "price * quantity"

    def test_order_normalization(self):
        """Order should be normalized from AttributeSchema."""
        measure = MeasureSchema(name="amount", order="ASC")
        assert measure.order == "asc"

        measure = MeasureSchema(name="amount", order="DESC")
        assert measure.order == "desc"

    def test_extra_fields_allowed(self):
        """Extra fields should be allowed for forward compatibility."""
        measure = MeasureSchema(name="amount", custom_field="custom_value")
        assert measure.model_dump().get("custom_field") == "custom_value"


class TestMeasureSchemaSerialization:
    """Test measure serialization to dict."""

    def test_model_dump_basic(self):
        """Basic serialization should work."""
        measure = MeasureSchema(name="amount", formula="test_formula")
        data = measure.model_dump()
        assert data["name"] == "amount"
        assert data["formula"] == "test_formula"
        assert data["aggregates"] == ["sum"]

    def test_model_dump_excludes_none(self):
        """Serialization should include None values."""
        measure = MeasureSchema(name="amount")
        data = measure.model_dump()
        assert data["nonadditive"] is None
        assert data["window_size"] is None
        assert data["formula"] is None


class TestMeasureSchemaEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_unicode_name(self):
        """Measure name should accept unicode."""
        measure = MeasureSchema(name="売上")
        assert measure.name == "売上"

    def test_special_chars_in_formula(self):
        """Formula can contain special characters."""
        measure = MeasureSchema(name="amount", formula="complex(formula)")
        assert measure.formula == "complex(formula)"

    def test_large_window_size(self):
        """Window size can be large."""
        measure = MeasureSchema(name="amount", window_size=999999)
        assert measure.window_size == 999999

    def test_aggregates_with_identity(self):
        """Identity aggregate should be supported."""
        measure = MeasureSchema(name="amount", aggregates=["identity", "sum"])
        assert "identity" in measure.aggregates


class TestMeasureSchemaValidationAssignment:
    """Test that validation runs on attribute assignment."""

    def test_invalid_assignment_raises_error(self):
        """Setting invalid value should raise ValidationError."""
        measure = MeasureSchema(name="amount")
        with pytest.raises(ValidationError):
            measure.nonadditive = "invalid"
