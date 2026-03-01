"""Tests for Measure and MeasureAggregate domain models."""

import copy
from typing import Any

import pytest

from cubes.metadata.models.attributes import Measure, MeasureAggregate
from cubes.metadata.models.base import ModelError


class TestMeasureBasics:
    """Test basic Measure initialization."""

    def test_measure_minimal(self):
        """Create measure with just name."""
        measure = Measure(name="amount")
        assert measure.name == "amount"
        assert measure.label is None
        assert measure.description is None
        assert measure.ref == "amount"
        assert measure.aggregates == ["sum"]
        assert measure.nonadditive is None
        assert measure.window_size is None
        assert measure.formula is None

    def test_measure_full(self):
        """Create measure with all properties."""
        measure = Measure(
            name="revenue",
            label="Total Revenue",
            description="The total revenue",
            order="desc",
            info={"unit": "USD"},
            format=".,2f",
            missing_value=0,
            expression="price * quantity",
            formula="revenue_formula",
            aggregates=["sum", "avg", "max"],
            nonadditive="time",
            window_size=30,
        )
        assert measure.name == "revenue"
        assert measure.label == "Total Revenue"
        assert measure.description == "The total revenue"
        assert measure.order == "desc"
        assert measure.info == {"unit": "USD"}
        assert measure.format == ".,2f"
        assert measure.missing_value == 0
        assert measure.expression == "price * quantity"
        assert measure.formula == "revenue_formula"
        assert measure.aggregates == ["sum", "avg", "max"]
        assert measure.nonadditive == "time"
        assert measure.window_size == 30
        assert measure.ref == "revenue"

    def test_measure_default_aggregates(self):
        """Default aggregates should be ["sum"]."""
        measure = Measure(name="amount")
        assert measure.aggregates == ["sum"]

    def test_measure_empty_aggregates_defaults_to_sum(self):
        """Empty aggregates list should default to ["sum"]."""
        measure = Measure(name="amount", aggregates=[])
        assert measure.aggregates == ["sum"]


class TestMeasureNonadditive:
    """Test nonadditive normalization."""

    def test_nonadditive_none_string(self):
        """ "none" should be normalized to None."""
        measure = Measure(name="amount", nonadditive="none")
        assert measure.nonadditive is None

    def test_nonadditive_none_none(self):
        """None should stay None."""
        measure = Measure(name="amount", nonadditive=None)
        assert measure.nonadditive is None

    def test_nonadditive_all_lowercase(self):
        """ "all" should stay "all"."""
        measure = Measure(name="amount", nonadditive="all")
        assert measure.nonadditive == "all"

    def test_nonadditive_all_uppercase(self):
        """ "ALL" should be normalized to "all"."""
        measure = Measure(name="amount", nonadditive="ALL")
        assert measure.nonadditive == "all"

    def test_nonadditive_any_normalized_to_all(self):
        """ "any" should be normalized to "all"."""
        measure = Measure(name="amount", nonadditive="any")
        assert measure.nonadditive == "all"

    def test_nonadditive_any_uppercase(self):
        """ "ANY" should be normalized to "all"."""
        measure = Measure(name="amount", nonadditive="ANY")
        assert measure.nonadditive == "all"

    def test_nonadditive_time(self):
        """ "time" should stay "time"."""
        measure = Measure(name="amount", nonadditive="time")
        assert measure.nonadditive == "time"

    def test_nonadditive_time_uppercase(self):
        """ "TIME" should be normalized to "time"."""
        measure = Measure(name="amount", nonadditive="TIME")
        assert measure.nonadditive == "time"

    def test_nonadditive_invalid(self):
        """Invalid nonadditive value should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Measure(name="amount", nonadditive="invalid")
        assert "Unknown non-additive type" in str(exc_info.value)


class TestMeasureProperties:
    """Test Measure properties inherited from AttributeBase."""

    def test_is_base_true(self):
        """Measure without expression is base."""
        measure = Measure(name="amount")
        assert measure.is_base is True

    def test_is_base_false(self):
        """Measure with expression is not base."""
        measure = Measure(name="revenue", expression="price * quantity")
        assert measure.is_base is False

    def test_dependencies_base(self):
        """Base measure has no dependencies."""
        measure = Measure(name="amount")
        assert measure.dependencies == set()

    def test_dependencies_with_expression(self):
        """Measure with expression has dependencies."""
        measure = Measure(name="revenue", expression="price * quantity")
        assert measure.dependencies == {"price", "quantity"}


class TestMeasureEquality:
    """Test Measure equality comparison."""

    def test_equal_measures(self):
        """Two measures with same properties are equal."""
        m1 = Measure(name="amount", aggregates=["sum", "min"])
        m2 = Measure(name="amount", aggregates=["sum", "min"])
        assert m1 == m2

    def test_unequal_name(self):
        """Measures with different names are not equal."""
        m1 = Measure(name="amount")
        m2 = Measure(name="total")
        assert m1 != m2

    def test_unequal_aggregates(self):
        """Measures with different aggregates are not equal."""
        m1 = Measure(name="amount", aggregates=["sum"])
        m2 = Measure(name="amount", aggregates=["avg"])
        assert m1 != m2

    def test_unequal_formula(self):
        """Measures with different formulas are not equal."""
        m1 = Measure(name="amount", formula="formula1")
        m2 = Measure(name="amount", formula="formula2")
        assert m1 != m2

    def test_unequal_nonadditive(self):
        """Measures with different nonadditive are not equal."""
        m1 = Measure(name="amount", nonadditive="time")
        m2 = Measure(name="amount", nonadditive="all")
        assert m1 != m2

    def test_unequal_window_size(self):
        """Measures with different window_size are not equal."""
        m1 = Measure(name="amount", window_size=7)
        m2 = Measure(name="amount", window_size=30)
        assert m1 != m2

    def test_not_equal_to_attribute_base(self):
        """Measure is not equal to AttributeBase."""
        from cubes.metadata.models.attributes import AttributeBase

        measure = Measure(name="amount")
        attr = AttributeBase(name="amount")
        assert measure != attr

    def test_hash_based_on_ref(self):
        """Hash is based on ref."""
        m1 = Measure(name="amount")
        m2 = Measure(name="amount")
        assert hash(m1) == hash(m2)


class TestMeasureDeepCopy:
    """Test Measure deep copy behavior."""

    def test_deepcopy_basic(self):
        """Deep copy creates equal but independent measure."""
        measure = Measure(
            name="amount", aggregates=["sum", "min"], info={"key": "value"}
        )
        copied = copy.deepcopy(measure)

        assert copied == measure
        assert copied is not measure
        assert copied.aggregates is not measure.aggregates
        assert copied.info is not measure.info

    def test_deepcopy_preserves_properties(self):
        """Deep copy preserves all properties."""
        measure = Measure(
            name="revenue",
            label="Revenue",
            description="Total revenue",
            order="desc",
            info={"unit": "USD"},
            format=".,2f",
            missing_value=0,
            expression="price * quantity",
            formula="revenue_formula",
            aggregates=["sum", "avg"],
            nonadditive="time",
            window_size=30,
        )
        copied = copy.deepcopy(measure)

        assert copied.name == measure.name
        assert copied.label == measure.label
        assert copied.description == measure.description
        assert copied.order == measure.order
        assert copied.info == measure.info
        assert copied.format == measure.format
        assert copied.missing_value == measure.missing_value
        assert copied.expression == measure.expression
        assert copied.formula == measure.formula
        assert copied.aggregates == measure.aggregates
        assert copied.nonadditive == measure.nonadditive
        assert copied.window_size == measure.window_size


class TestMeasureDefaultAggregates:
    """Test Measure.default_aggregates() method."""

    def test_default_aggregates_single_sum(self):
        """Default aggregates with single sum."""
        measure = Measure(name="amount")
        aggs = measure.default_aggregates()

        assert len(aggs) == 1
        assert aggs[0].name == "amount_sum"
        assert aggs[0].measure == "amount"
        assert aggs[0].function == "sum"
        assert aggs[0].formula is None

    def test_default_aggregates_multiple(self):
        """Default aggregates with multiple functions."""
        measure = Measure(name="amount", aggregates=["sum", "min", "max"])
        aggs = measure.default_aggregates()

        assert len(aggs) == 3
        assert aggs[0].name == "amount_sum"
        assert aggs[0].measure == "amount"
        assert aggs[0].function == "sum"

        assert aggs[1].name == "amount_min"
        assert aggs[1].measure == "amount"
        assert aggs[1].function == "min"

        assert aggs[2].name == "amount_max"
        assert aggs[2].measure == "amount"
        assert aggs[2].function == "max"

    def test_default_aggregates_identity(self):
        """Identity aggregate creates measure with no measure/function."""
        measure = Measure(name="amount", aggregates=["identity"])
        aggs = measure.default_aggregates()

        assert len(aggs) == 1
        assert aggs[0].name == "amount"
        assert aggs[0].measure is None
        assert aggs[0].function is None

    def test_default_aggregates_inherits_properties(self):
        """Default aggregates inherit properties from measure."""
        measure = Measure(
            name="revenue",
            description="Total revenue",
            order="desc",
            info={"unit": "USD"},
            format=".,2f",
            window_size=30,
            aggregates=["sum"],
        )
        aggs = measure.default_aggregates()

        assert aggs[0].description == "Total revenue"
        assert aggs[0].order == "desc"
        assert aggs[0].info == {"unit": "USD"}
        assert aggs[0].format == ".,2f"
        assert aggs[0].window_size == 30

    def test_default_aggregates_returns_list(self):
        """default_aggregates returns a list."""
        measure = Measure(name="amount")
        aggs = measure.default_aggregates()

        assert isinstance(aggs, list)
        assert all(isinstance(a, MeasureAggregate) for a in aggs)


class TestMeasureToDict:
    """Test Measure.to_dict() method."""

    def test_to_dict_minimal(self):
        """to_dict for minimal measure."""
        measure = Measure(name="amount")
        d = measure.to_dict()

        assert d["name"] == "amount"
        assert d["label"] is None
        assert d["description"] is None
        assert d["ref"] == "amount"
        assert d["formula"] is None
        assert d["aggregates"] == ["sum"]
        assert d["nonadditive"] is None
        assert d["window_size"] is None

    def test_to_dict_full(self):
        """to_dict for measure with all properties."""
        measure = Measure(
            name="revenue",
            label="Revenue",
            description="Total revenue",
            formula="revenue_formula",
            aggregates=["sum", "avg"],
            nonadditive="time",
            window_size=30,
        )
        d = measure.to_dict()

        assert d["name"] == "revenue"
        assert d["label"] == "Revenue"
        assert d["description"] == "Total revenue"
        assert d["formula"] == "revenue_formula"
        assert d["aggregates"] == ["sum", "avg"]
        assert d["nonadditive"] == "time"
        assert d["window_size"] == 30

    def test_to_dict_create_label(self):
        """to_dict with create_label generates label from name."""
        measure = Measure(name="total_amount")
        d = measure.to_dict(create_label=True)

        assert d["label"] == "Total amount"

    def test_to_dict_preserves_existing_label(self):
        """to_dict preserves existing label even with create_label=True."""
        measure = Measure(name="amount", label="Custom Label")
        d = measure.to_dict(create_label=True)

        assert d["label"] == "Custom Label"


class TestMeasureStrRepr:
    """Test Measure string representations."""

    def test_str_returns_ref(self):
        """str(measure) returns ref."""
        measure = Measure(name="amount")
        assert str(measure) == "amount"

    def test_repr_contains_class_name(self):
        """repr contains class name."""
        measure = Measure(name="amount")
        assert "Measure(" in repr(measure)


class TestMeasureAggregateBasics:
    """Test basic MeasureAggregate initialization."""

    def test_aggregate_minimal(self):
        """Create aggregate with just name."""
        agg = MeasureAggregate(name="amount_sum")
        assert agg.name == "amount_sum"
        assert agg.label is None
        assert agg.function is None
        assert agg.measure is None
        assert agg.formula is None
        assert agg.nonadditive is None
        assert agg.window_size is None

    def test_aggregate_full(self):
        """Create aggregate with all properties."""
        agg = MeasureAggregate(
            name="total_revenue",
            label="Total Revenue",
            description="Sum of revenue",
            order="desc",
            info={"unit": "USD"},
            format=".,2f",
            missing_value=0,
            expression="SUM(revenue)",
            function="sum",
            formula="total_formula",
            measure="revenue",
            nonadditive="time",
            window_size=30,
        )
        assert agg.name == "total_revenue"
        assert agg.label == "Total Revenue"
        assert agg.description == "Sum of revenue"
        assert agg.order == "desc"
        assert agg.info == {"unit": "USD"}
        assert agg.format == ".,2f"
        assert agg.missing_value == 0
        assert agg.expression == "SUM(revenue)"
        assert agg.function == "sum"
        assert agg.formula == "total_formula"
        assert agg.measure == "revenue"
        assert agg.nonadditive == "time"
        assert agg.window_size == 30


class TestMeasureAggregateIsBase:
    """Test MeasureAggregate.is_base property."""

    def test_is_base_true_no_expression_no_function(self):
        """Aggregate with no expression and no function is base."""
        agg = MeasureAggregate(name="backend_agg")
        assert agg.is_base is True

    def test_is_base_false_with_expression(self):
        """Aggregate with expression is not base."""
        agg = MeasureAggregate(name="custom", expression="a + b")
        assert agg.is_base is False

    def test_is_base_false_with_function(self):
        """Aggregate with function is not base."""
        agg = MeasureAggregate(name="amount_sum", function="sum")
        assert agg.is_base is False

    def test_is_base_false_with_both(self):
        """Aggregate with both expression and function is not base."""
        agg = MeasureAggregate(name="custom", expression="a + b", function="sum")
        assert agg.is_base is False


class TestMeasureAggregateDependencies:
    """Test MeasureAggregate.dependencies property."""

    def test_dependencies_empty_no_measure_no_expression(self):
        """No measure and no expression gives empty dependencies."""
        agg = MeasureAggregate(name="backend_agg")
        assert agg.dependencies == set()

    def test_dependencies_with_measure(self):
        """Measure gives set containing measure name."""
        agg = MeasureAggregate(name="amount_sum", measure="amount", function="sum")
        assert agg.dependencies == {"amount"}

    def test_dependencies_with_expression(self):
        """Expression gives variables from expression."""
        agg = MeasureAggregate(
            name="custom", expression="SUM(price * quantity) / COUNT(1)"
        )
        # Note: SUM, COUNT are treated as identifiers by inspect_variables
        # In practice these would be SQL functions, but for testing:
        deps = agg.dependencies
        assert "price" in deps
        assert "quantity" in deps

    def test_dependencies_error_both_measure_and_expression(self):
        """Having both measure and expression raises ModelError."""
        agg = MeasureAggregate(
            name="invalid", measure="amount", expression="SUM(amount)"
        )
        with pytest.raises(ModelError) as exc_info:
            _ = agg.dependencies
        assert "has both measure and expression set" in str(exc_info.value)


class TestMeasureAggregateEquality:
    """Test MeasureAggregate equality comparison."""

    def test_equal_aggregates(self):
        """Two aggregates with same properties are equal."""
        a1 = MeasureAggregate(name="amount_sum", measure="amount", function="sum")
        a2 = MeasureAggregate(name="amount_sum", measure="amount", function="sum")
        assert a1 == a2

    def test_unequal_function(self):
        """Aggregates with different functions are not equal."""
        a1 = MeasureAggregate(name="amount_sum", measure="amount", function="sum")
        a2 = MeasureAggregate(name="amount_sum", measure="amount", function="avg")
        assert a1 != a2

    def test_unequal_measure(self):
        """Aggregates with different measures are not equal."""
        a1 = MeasureAggregate(name="amount_sum", measure="amount", function="sum")
        a2 = MeasureAggregate(name="revenue_sum", measure="revenue", function="sum")
        assert a1 != a2

    def test_not_equal_to_measure(self):
        """Aggregate is not equal to Measure."""
        agg = MeasureAggregate(name="amount_sum", measure="amount", function="sum")
        measure = Measure(name="amount")
        assert agg != measure


class TestMeasureAggregateDeepCopy:
    """Test MeasureAggregate deep copy behavior."""

    def test_deepcopy_basic(self):
        """Deep copy creates equal but independent aggregate."""
        agg = MeasureAggregate(
            name="amount_sum",
            measure="amount",
            function="sum",
            info={"key": "value"},
        )
        copied = copy.deepcopy(agg)

        assert copied == agg
        assert copied is not agg
        assert copied.info is not agg.info


class TestMeasureAggregateToDict:
    """Test MeasureAggregate.to_dict() method."""

    def test_to_dict_minimal(self):
        """to_dict for minimal aggregate."""
        agg = MeasureAggregate(name="amount_sum")
        d = agg.to_dict()

        assert d["name"] == "amount_sum"
        assert d["function"] is None
        assert d["measure"] is None
        assert d["formula"] is None

    def test_to_dict_full(self):
        """to_dict for aggregate with all properties."""
        agg = MeasureAggregate(
            name="total",
            function="sum",
            measure="amount",
            formula="total_formula",
            nonadditive="time",
            window_size=30,
        )
        d = agg.to_dict()

        assert d["name"] == "total"
        assert d["function"] == "sum"
        assert d["measure"] == "amount"
        assert d["formula"] == "total_formula"
        assert d["nonadditive"] == "time"
        assert d["window_size"] == 30


class TestMeasureAggregateStrRepr:
    """Test MeasureAggregate string representations."""

    def test_str_returns_ref(self):
        """str(aggregate) returns ref."""
        agg = MeasureAggregate(name="amount_sum")
        assert str(agg) == "amount_sum"

    def test_repr_contains_class_name(self):
        """repr contains class name."""
        agg = MeasureAggregate(name="amount_sum")
        assert "MeasureAggregate(" in repr(agg)
