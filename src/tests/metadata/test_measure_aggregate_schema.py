"""Tests for MeasureAggregate Pydantic schema."""

import pytest
from cubes.metadata.schemas.attributes import (
    MeasureAggregateSchema,
    SUPPORTED_AGGREGATE_FUNCTIONS,
)


class TestMeasureAggregateSchemaBasics:
    """Test basic MeasureAggregateSchema initialization."""

    def test_minimal_creation(self):
        """Test creating schema with just a name."""
        schema = MeasureAggregateSchema.model_validate({"name": "amount_sum"})
        assert schema.name == "amount_sum"
        assert schema.label is None
        assert schema.description is None
        assert schema.function is None
        assert schema.measure is None
        assert schema.expression is None
        assert schema.formula is None
        assert schema.nonadditive is None
        assert schema.window_size is None
        assert schema.info == {}

    def test_string_coercion(self):
        """Test that string input is coerced to dict."""
        schema = MeasureAggregateSchema.model_validate("amount_sum")
        assert schema.name == "amount_sum"
        assert schema.label is None

    def test_full_creation(self):
        """Test creating schema with all fields."""
        schema = MeasureAggregateSchema.model_validate(
            {
                "name": "total_amount",
                "label": "Total Amount",
                "description": "Sum of all amounts",
                "function": "sum",
                "measure": "amount",
                "formula": "custom_formula",
                "nonadditive": "time",
                "window_size": 30,
                "info": {"unit": "USD"},
                "format": "currency",
            }
        )
        assert schema.name == "total_amount"
        assert schema.label == "Total Amount"
        assert schema.description == "Sum of all amounts"
        assert schema.function == "sum"
        assert schema.measure == "amount"
        assert schema.formula == "custom_formula"
        assert schema.nonadditive == "time"
        assert schema.window_size == 30
        assert schema.info == {"unit": "USD"}
        assert schema.format == "currency"

    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed for forward compatibility."""
        schema = MeasureAggregateSchema.model_validate(
            {
                "name": "amount_sum",
                "custom_field": "custom_value",
            }
        )
        assert schema.name == "amount_sum"
        # Extra fields are stored in model_extra
        assert hasattr(schema, "custom_field")
        assert schema.custom_field == "custom_value"


class TestMeasureAggregateSchemaValidation:
    """Test MeasureAggregateSchema validation rules."""

    def test_measure_and_expression_mutual_exclusivity(self):
        """Test that measure and expression cannot both be set."""
        with pytest.raises(ValueError) as exc_info:
            MeasureAggregateSchema.model_validate(
                {
                    "name": "invalid_agg",
                    "measure": "amount",
                    "expression": "amount * 2",
                }
            )
        error_msg = str(exc_info.value)
        assert "cannot have both 'measure' and 'expression'" in error_msg
        assert "invalid_agg" in error_msg

    def test_measure_only_valid(self):
        """Test that measure alone is valid."""
        schema = MeasureAggregateSchema.model_validate(
            {
                "name": "amount_sum",
                "measure": "amount",
                "function": "sum",
            }
        )
        assert schema.measure == "amount"
        assert schema.function == "sum"
        assert schema.expression is None

    def test_expression_only_valid(self):
        """Test that expression alone is valid."""
        schema = MeasureAggregateSchema.model_validate(
            {
                "name": "custom_calc",
                "expression": "amount * price",
            }
        )
        assert schema.expression == "amount * price"
        assert schema.measure is None

    def test_neither_measure_nor_expression_valid(self):
        """Test that having neither measure nor expression is valid (base aggregate)."""
        schema = MeasureAggregateSchema.model_validate(
            {
                "name": "backend_agg",
            }
        )
        assert schema.measure is None
        assert schema.expression is None

    def test_supported_functions(self):
        """Test that all supported aggregation functions are accepted."""
        for func in SUPPORTED_AGGREGATE_FUNCTIONS:
            schema = MeasureAggregateSchema.model_validate(
                {
                    "name": f"amount_{func}",
                    "measure": "amount",
                    "function": func,
                }
            )
            assert schema.function == func

    def test_unsupported_function(self):
        """Test that unsupported function raises error."""
        with pytest.raises(ValueError) as exc_info:
            MeasureAggregateSchema.model_validate(
                {
                    "name": "amount_invalid",
                    "measure": "amount",
                    "function": "unsupported_function",
                }
            )
        error_msg = str(exc_info.value)
        assert "Unsupported aggregate function" in error_msg
        assert "unsupported_function" in error_msg

    def test_no_function_with_measure_valid(self):
        """Test that measure without function is valid (identity)."""
        schema = MeasureAggregateSchema.model_validate(
            {
                "name": "amount",
                "measure": "amount",
            }
        )
        assert schema.measure == "amount"
        assert schema.function is None


class TestMeasureAggregateSchemaNonadditive:
    """Test nonadditive field normalization."""

    def test_nonadditive_none_becomes_none(self):
        """Test that 'none' becomes None."""
        schema = MeasureAggregateSchema.model_validate(
            {
                "name": "amount_sum",
                "nonadditive": "none",
            }
        )
        assert schema.nonadditive is None

    def test_nonadditive_all_normalized(self):
        """Test that 'all' and 'any' become 'all'."""
        for value in ["all", "ALL", "All", "any", "ANY", "Any"]:
            schema = MeasureAggregateSchema.model_validate(
                {
                    "name": "amount_sum",
                    "nonadditive": value,
                }
            )
            assert schema.nonadditive == "all"

    def test_nonadditive_time_preserved(self):
        """Test that 'time' is preserved."""
        for value in ["time", "TIME", "Time"]:
            schema = MeasureAggregateSchema.model_validate(
                {
                    "name": "amount_sum",
                    "nonadditive": value,
                }
            )
            assert schema.nonadditive == "time"

    def test_nonadditive_invalid(self):
        """Test that invalid nonadditive value raises error."""
        with pytest.raises(ValueError) as exc_info:
            MeasureAggregateSchema.model_validate(
                {
                    "name": "amount_sum",
                    "nonadditive": "invalid",
                }
            )
        error_msg = str(exc_info.value)
        assert "Unknown non-additive type" in error_msg

    def test_nonadditive_non_string(self):
        """Test that non-string nonadditive value raises error."""
        with pytest.raises(ValueError) as exc_info:
            MeasureAggregateSchema.model_validate(
                {
                    "name": "amount_sum",
                    "nonadditive": 123,
                }
            )
        error_msg = str(exc_info.value)
        assert "Nonadditive must be a string" in error_msg


class TestMeasureAggregateSchemaOrder:
    """Test order field normalization (inherited from AttributeSchema)."""

    def test_order_asc_normalized(self):
        """Test that asc variants are normalized to 'asc'."""
        for value in ["asc", "ASC", "Asc", "ascending", "ASCENDING"]:
            schema = MeasureAggregateSchema.model_validate(
                {
                    "name": "amount_sum",
                    "order": value,
                }
            )
            assert schema.order == "asc"

    def test_order_desc_normalized(self):
        """Test that desc variants are normalized to 'desc'."""
        for value in ["desc", "DESC", "Desc", "descending", "DESCENDING"]:
            schema = MeasureAggregateSchema.model_validate(
                {
                    "name": "amount_sum",
                    "order": value,
                }
            )
            assert schema.order == "desc"

    def test_order_invalid(self):
        """Test that invalid order value raises error."""
        with pytest.raises(ValueError) as exc_info:
            MeasureAggregateSchema.model_validate(
                {
                    "name": "amount_sum",
                    "order": "invalid",
                }
            )
        error_msg = str(exc_info.value)
        assert "Unknown ordering" in error_msg


class TestMeasureAggregateSchemaSerialization:
    """Test MeasureAggregateSchema serialization."""

    def test_model_dump_basic(self):
        """Test basic model_dump output."""
        schema = MeasureAggregateSchema.model_validate(
            {
                "name": "amount_sum",
                "measure": "amount",
                "function": "sum",
            }
        )
        data = schema.model_dump()
        assert data["name"] == "amount_sum"
        assert data["measure"] == "amount"
        assert data["function"] == "sum"
        assert data["label"] is None

    def test_model_dump_excludes_defaults(self):
        """Test that None values are included in output."""
        schema = MeasureAggregateSchema.model_validate({"name": "amount_sum"})
        data = schema.model_dump()
        assert data["name"] == "amount_sum"
        assert data["function"] is None
        assert data["measure"] is None
        assert data["expression"] is None


class TestMeasureAggregateSchemaEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_info_dict(self):
        """Test that empty info dict is handled correctly."""
        schema = MeasureAggregateSchema.model_validate(
            {
                "name": "amount_sum",
                "info": {},
            }
        )
        assert schema.info == {}

    def test_nested_info_dict(self):
        """Test that nested info dict is preserved."""
        schema = MeasureAggregateSchema.model_validate(
            {
                "name": "amount_sum",
                "info": {"nested": {"key": "value"}, "list": [1, 2, 3]},
            }
        )
        assert schema.info["nested"]["key"] == "value"
        assert schema.info["list"] == [1, 2, 3]

    def test_unicode_names(self):
        """Test that unicode names are accepted."""
        schema = MeasureAggregateSchema.model_validate(
            {
                "name": "金額合計",
            }
        )
        assert schema.name == "金額合計"

    def test_special_characters_in_names(self):
        """Test names with special characters."""
        schema = MeasureAggregateSchema.model_validate(
            {
                "name": "amount_sum_v2",
            }
        )
        assert schema.name == "amount_sum_v2"

    def test_zero_window_size(self):
        """Test that window_size of 0 is valid."""
        schema = MeasureAggregateSchema.model_validate(
            {
                "name": "amount_sum",
                "window_size": 0,
            }
        )
        assert schema.window_size == 0

    def test_negative_window_size(self):
        """Test that negative window_size is valid (not restricted)."""
        schema = MeasureAggregateSchema.model_validate(
            {
                "name": "amount_sum",
                "window_size": -5,
            }
        )
        assert schema.window_size == -5


class TestMeasureAggregateSchemaIdentityPattern:
    """Test identity aggregate pattern (no function, no measure)."""

    def test_identity_aggregate_pattern(self):
        """Test schema for identity aggregate pattern."""
        schema = MeasureAggregateSchema.model_validate(
            {
                "name": "amount",
                "measure": None,
                "function": None,
            }
        )
        assert schema.name == "amount"
        assert schema.measure is None
        assert schema.function is None

    def test_identity_with_description(self):
        """Test identity aggregate with description."""
        schema = MeasureAggregateSchema.model_validate(
            {
                "name": "fact_count",
                "label": "Record Count",
                "description": "Total number of records",
                "function": "count",
            }
        )
        assert schema.name == "fact_count"
        assert schema.function == "count"
        assert schema.measure is None
