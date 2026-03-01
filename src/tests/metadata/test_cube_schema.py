"""Tests for Cube Pydantic Schema.

Tests the CubeSchema class including:
- Basic field validation
- Dimension link expansion
- Measure and aggregate handling
- Default fact_count aggregate generation
- Implicit aggregate generation from measures
- Nonadditive property propagation
- Aggregate label templates
- Ported legacy cube expansion tests
"""

import pytest
from pydantic import ValidationError

from cubes.metadata.schemas.cube import (
    CubeSchema,
    DEFAULT_FACT_COUNT_AGGREGATE,
    IMPLICIT_AGGREGATE_LABELS,
    _expand_dimension_link,
    _get_measure_aggregate_label,
)
from cubes.metadata.schemas.attributes import (
    MeasureAggregateSchema,
    MeasureSchema,
)


class TestCubeSchemaBasics:
    """Test basic CubeSchema validation."""

    def test_valid_cube_minimal(self):
        """Cube with just a name gets default fact_count aggregate."""
        data = {"name": "sales"}
        cube = CubeSchema.model_validate(data)
        assert cube.name == "sales"
        assert len(cube.aggregates) == 1
        assert cube.aggregates[0].name == "fact_count"
        assert cube.aggregates[0].function == "count"

    def test_valid_cube_full_metadata(self):
        """Cube with all metadata fields."""
        data = {
            "name": "sales",
            "label": "Sales Cube",
            "description": "Sales data cube",
            "info": {"source": "ERP"},
            "category": "financial",
        }
        cube = CubeSchema.model_validate(data)
        assert cube.name == "sales"
        assert cube.label == "Sales Cube"
        assert cube.description == "Sales data cube"
        assert cube.info == {"source": "ERP"}
        assert cube.category == "financial"

    def test_cube_without_name_raises_error(self):
        """Cube must have a name."""
        data = {}
        with pytest.raises(ValidationError) as exc_info:
            CubeSchema.model_validate(data)
        assert "name" in str(exc_info.value)


class TestDimensionLinks:
    """Test dimension link expansion."""

    def test_dimension_link_string_expansion(self):
        """String dimension names are expanded to dicts."""
        data = {
            "name": "sales",
            "dimensions": ["date", "product", "store"],
        }
        cube = CubeSchema.model_validate(data)
        assert len(cube.dimensions) == 3
        assert cube.dimensions[0] == {"name": "date"}
        assert cube.dimensions[1] == {"name": "product"}
        assert cube.dimensions[2] == {"name": "store"}

    def test_dimension_link_dict_preserved(self):
        """Dict dimension links are preserved with all fields."""
        data = {
            "name": "sales",
            "dimensions": [
                {"name": "date", "hierarchies": ["ymd"]},
                {"name": "product", "nonadditive": "time"},
            ],
        }
        cube = CubeSchema.model_validate(data)
        assert len(cube.dimensions) == 2
        assert cube.dimensions[0]["name"] == "date"
        assert cube.dimensions[0]["hierarchies"] == ["ymd"]
        assert cube.dimensions[1]["name"] == "product"
        assert cube.dimensions[1]["nonadditive"] == "time"

    def test_dimension_link_mixed(self):
        """Mixed string and dict dimension links."""
        data = {
            "name": "sales",
            "dimensions": [
                "date",
                {"name": "product", "cardinality": "high"},
            ],
        }
        cube = CubeSchema.model_validate(data)
        assert len(cube.dimensions) == 2
        assert cube.dimensions[0] == {"name": "date"}
        assert cube.dimensions[1]["name"] == "product"
        assert cube.dimensions[1]["cardinality"] == "high"

    def test_dimension_link_missing_name_raises_error(self):
        """Dimension link dict must have a name field."""
        data = {
            "name": "sales",
            "dimensions": [{"hierarchies": ["ymd"]}],  # Missing name
        }
        with pytest.raises(ValueError) as exc_info:
            CubeSchema.model_validate(data)
        assert "name" in str(exc_info.value).lower()


class TestExpandDimensionLinkFunction:
    """Test _expand_dimension_link helper function."""

    def test_expand_string_link(self):
        """String link is converted to dict."""
        result = _expand_dimension_link("date")
        assert result == {"name": "date"}

    def test_expand_dict_link(self):
        """Dict link is preserved."""
        link = {"name": "date", "hierarchies": ["ymd"]}
        result = _expand_dimension_link(link)
        assert result == {"name": "date", "hierarchies": ["ymd"]}

    def test_expand_dict_missing_name_raises_error(self):
        """Dict without name field raises error."""
        with pytest.raises(ValueError) as exc_info:
            _expand_dimension_link({"hierarchies": ["ymd"]})
        assert "name" in str(exc_info.value).lower()

    def test_expand_invalid_type_raises_error(self):
        """Non-string, non-dict raises error."""
        with pytest.raises(ValueError) as exc_info:
            _expand_dimension_link(123)
        assert "string or dict" in str(exc_info.value).lower()


class TestMeasuresAndAggregates:
    """Test measures and aggregates handling."""

    def test_cube_with_measures(self):
        """Cube with explicit measures."""
        data = {
            "name": "sales",
            "measures": [
                {"name": "amount", "aggregates": ["sum", "avg"]},
                {"name": "quantity", "aggregates": ["sum"]},
            ],
        }
        cube = CubeSchema.model_validate(data)
        assert len(cube.measures) == 2
        assert cube.measures[0].name == "amount"
        assert cube.measures[1].name == "quantity"

    def test_cube_with_aggregates(self):
        """Cube with explicit aggregates."""
        data = {
            "name": "sales",
            "aggregates": [
                {"name": "total_amount", "measure": "amount", "function": "sum"},
                {"name": "avg_amount", "measure": "amount", "function": "avg"},
            ],
        }
        cube = CubeSchema.model_validate(data)
        assert len(cube.aggregates) == 2
        assert cube.aggregates[0].name == "total_amount"
        assert cube.aggregates[1].name == "avg_amount"

    def test_cube_with_measures_no_default_aggregate(self):
        """Cube with measures should not get default fact_count."""
        data = {
            "name": "sales",
            "measures": [{"name": "amount"}],
        }
        cube = CubeSchema.model_validate(data)
        # Should not have fact_count since we have measures
        assert not any(a.name == "fact_count" for a in cube.aggregates)


class TestDefaultFactCountAggregate:
    """Test default fact_count aggregate generation."""

    def test_default_fact_count_created(self):
        """Default fact_count aggregate is created when no measures/aggregates."""
        data = {"name": "sales"}
        cube = CubeSchema.model_validate(data)
        assert len(cube.aggregates) == 1
        assert cube.aggregates[0].name == "fact_count"
        assert cube.aggregates[0].label == "Count"
        assert cube.aggregates[0].function == "count"

    def test_no_default_when_measures_present(self):
        """No default fact_count when measures are specified."""
        data = {
            "name": "sales",
            "measures": [{"name": "amount"}],
        }
        cube = CubeSchema.model_validate(data)
        assert not any(a.name == "fact_count" for a in cube.aggregates)

    def test_no_default_when_aggregates_present(self):
        """No default fact_count when aggregates are specified."""
        data = {
            "name": "sales",
            "aggregates": [{"name": "total", "function": "count"}],
        }
        cube = CubeSchema.model_validate(data)
        assert not any(a.name == "fact_count" for a in cube.aggregates)


class TestImplicitAggregates:
    """Test implicit aggregate generation from measures."""

    def test_implicit_aggregates_disabled_by_default(self):
        """Implicit aggregates are disabled by default."""
        data = {
            "name": "sales",
            "measures": [{"name": "amount", "aggregates": ["sum", "avg"]}],
        }
        cube = CubeSchema.model_validate(data)
        # Should not generate implicit aggregates
        assert len(cube.aggregates) == 0

    def test_implicit_aggregates_generation(self):
        """Implicit aggregates are generated when enabled."""
        data = {
            "name": "sales",
            "implicit_aggregates": True,
            "measures": [
                {"name": "amount", "aggregates": ["sum", "avg"]},
            ],
        }
        cube = CubeSchema.model_validate(data)
        # Should generate amount_sum and amount_avg
        aggregate_names = {a.name for a in cube.aggregates}
        assert "amount_sum" in aggregate_names
        assert "amount_avg" in aggregate_names

    def test_implicit_aggregate_identity(self):
        """Identity aggregate uses measure name."""
        data = {
            "name": "sales",
            "implicit_aggregates": True,
            "measures": [
                {"name": "amount", "aggregates": ["identity"]},
            ],
        }
        cube = CubeSchema.model_validate(data)
        aggregate_names = {a.name for a in cube.aggregates}
        assert "amount" in aggregate_names
        # Identity aggregate should have no function or measure
        amount_agg = next(a for a in cube.aggregates if a.name == "amount")
        assert amount_agg.function is None
        assert amount_agg.measure is None

    def test_implicit_aggregates_no_duplicates(self):
        """Implicit aggregates don't duplicate existing aggregates."""
        data = {
            "name": "sales",
            "implicit_aggregates": True,
            "measures": [{"name": "amount", "aggregates": ["sum"]}],
            "aggregates": [
                {"name": "amount_sum", "measure": "amount", "function": "sum"},
            ],
        }
        cube = CubeSchema.model_validate(data)
        # Should have only one amount_sum aggregate
        amount_sum_count = sum(1 for a in cube.aggregates if a.name == "amount_sum")
        assert amount_sum_count == 1

    def test_implicit_aggregates_multiple_measures(self):
        """Implicit aggregates from multiple measures."""
        data = {
            "name": "sales",
            "implicit_aggregates": True,
            "measures": [
                {"name": "amount", "aggregates": ["sum"]},
                {"name": "quantity", "aggregates": ["sum"]},
            ],
        }
        cube = CubeSchema.model_validate(data)
        aggregate_names = {a.name for a in cube.aggregates}
        assert "amount_sum" in aggregate_names
        assert "quantity_sum" in aggregate_names

    def test_implicit_aggregates_default_to_sum(self):
        """Measures without aggregates default to sum."""
        data = {
            "name": "sales",
            "implicit_aggregates": True,
            "measures": [{"name": "amount"}],  # No aggregates specified
        }
        cube = CubeSchema.model_validate(data)
        aggregate_names = {a.name for a in cube.aggregates}
        assert "amount_sum" in aggregate_names


class TestAggregateLabelTemplates:
    """Test implicit aggregate label generation."""

    def test_get_measure_aggregate_label_sum(self):
        """Sum aggregate label template."""
        label = _get_measure_aggregate_label("sum", "amount")
        assert label == "Sum of amount"

    def test_get_measure_aggregate_label_count(self):
        """Count aggregate label template."""
        label = _get_measure_aggregate_label("count", "records")
        assert label == "Record Count"

    def test_get_measure_aggregate_label_count_nonempty(self):
        """Count nonempty aggregate label template."""
        label = _get_measure_aggregate_label("count_nonempty", "amount")
        assert label == "Non-empty count of amount"

    def test_get_measure_aggregate_label_min(self):
        """Min aggregate label template."""
        label = _get_measure_aggregate_label("min", "amount")
        assert label == "amount Minimum"

    def test_get_measure_aggregate_label_max(self):
        """Max aggregate label template."""
        label = _get_measure_aggregate_label("max", "amount")
        assert label == "amount Maximum"

    def test_get_measure_aggregate_label_avg(self):
        """Avg aggregate label template."""
        label = _get_measure_aggregate_label("avg", "amount")
        assert label == "Average of amount"

    def test_get_measure_aggregate_label_unknown(self):
        """Unknown function uses measure name as template."""
        label = _get_measure_aggregate_label("custom", "amount")
        assert label == "amount"

    def test_implicit_aggregate_labels_applied(self):
        """Implicit aggregates get proper labels."""
        data = {
            "name": "sales",
            "implicit_aggregates": True,
            "measures": [{"name": "amount", "aggregates": ["sum"]}],
        }
        cube = CubeSchema.model_validate(data)
        amount_sum = next(a for a in cube.aggregates if a.name == "amount_sum")
        assert amount_sum.label == "Sum of amount"


class TestNonadditivePropagation:
    """Test nonadditive property propagation from cube to measures."""

    def test_nonadditive_propagated_to_measures(self):
        """Cube nonadditive is inherited by measures without their own."""
        data = {
            "name": "sales",
            "nonadditive": "time",
            "measures": [
                {"name": "amount"},  # No nonadditive specified
                {"name": "quantity", "nonadditive": "all"},  # Has own nonadditive
            ],
        }
        cube = CubeSchema.model_validate(data)
        # amount should inherit "time"
        amount_measure = next(m for m in cube.measures if m.name == "amount")
        assert amount_measure.nonadditive == "time"
        # quantity should keep "all"
        quantity_measure = next(m for m in cube.measures if m.name == "quantity")
        assert quantity_measure.nonadditive == "all"

    def test_nonadditive_not_overridden(self):
        """Measure's own nonadditive is not overridden."""
        data = {
            "name": "sales",
            "nonadditive": "time",
            "measures": [{"name": "amount", "nonadditive": "all"}],
        }
        cube = CubeSchema.model_validate(data)
        amount_measure = next(m for m in cube.measures if m.name == "amount")
        assert amount_measure.nonadditive == "all"

    def test_nonadditive_propagated_to_implicit_aggregates(self):
        """Nonadditive is propagated to implicit aggregates."""
        data = {
            "name": "sales",
            "nonadditive": "time",
            "implicit_aggregates": True,
            "measures": [{"name": "amount", "aggregates": ["sum"]}],
        }
        cube = CubeSchema.model_validate(data)
        amount_sum = next(a for a in cube.aggregates if a.name == "amount_sum")
        assert amount_sum.nonadditive == "time"


class TestPhysicalProperties:
    """Test physical cube properties."""

    def test_mappings(self):
        """Cube mappings field."""
        data = {
            "name": "sales",
            "mappings": {
                "amount": "fact_table.amount",
                "date_id": "fact_table.date_id",
            },
        }
        cube = CubeSchema.model_validate(data)
        assert cube.mappings["amount"] == "fact_table.amount"
        assert cube.mappings["date_id"] == "fact_table.date_id"

    def test_joins(self):
        """Cube joins field."""
        data = {
            "name": "sales",
            "joins": [
                {"master": "fact_table.date_id", "detail": "date.id"},
                {"master": "fact_table.product_id", "detail": "product.id"},
            ],
        }
        cube = CubeSchema.model_validate(data)
        assert len(cube.joins) == 2
        assert cube.joins[0]["master"] == "fact_table.date_id"

    def test_fact_table(self):
        """Cube fact table field."""
        data = {
            "name": "sales",
            "fact": "sales_fact",
        }
        cube = CubeSchema.model_validate(data)
        assert cube.fact == "sales_fact"

    def test_key(self):
        """Cube key field."""
        data = {
            "name": "sales",
            "key": "sale_id",
        }
        cube = CubeSchema.model_validate(data)
        assert cube.key == "sale_id"

    def test_browser_options(self):
        """Cube browser_options field."""
        data = {
            "name": "sales",
            "browser_options": {
                "denormalized_view": "sales_view",
                "page_size": 1000,
            },
        }
        cube = CubeSchema.model_validate(data)
        assert cube.browser_options["denormalized_view"] == "sales_view"
        assert cube.browser_options["page_size"] == 1000

    def test_store(self):
        """Cube store field."""
        data = {
            "name": "sales",
            "store": "main_db",
        }
        cube = CubeSchema.model_validate(data)
        assert cube.store == "main_db"


class TestDetails:
    """Test cube details field."""

    def test_details_list(self):
        """Cube with detail attributes."""
        data = {
            "name": "sales",
            "details": [
                {"name": "transaction_id"},
                {"name": "customer_note"},
            ],
        }
        cube = CubeSchema.model_validate(data)
        assert len(cube.details) == 2


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_dimensions_list(self):
        """Empty dimensions list is allowed."""
        data = {
            "name": "sales",
            "dimensions": [],
        }
        cube = CubeSchema.model_validate(data)
        assert cube.dimensions == []

    def test_empty_measures_list(self):
        """Empty measures list is allowed."""
        data = {
            "name": "sales",
            "measures": [],
        }
        cube = CubeSchema.model_validate(data)
        assert cube.measures == []
        # Should still get default fact_count since measures is empty and aggregates not provided
        assert len(cube.aggregates) == 1
        assert cube.aggregates[0].name == "fact_count"

    def test_empty_aggregates_list_no_default(self):
        """Explicitly empty aggregates list does not get default fact_count."""
        data = {
            "name": "sales",
            "aggregates": [],
        }
        cube = CubeSchema.model_validate(data)
        # Explicitly empty aggregates should remain empty (no default fact_count)
        assert cube.aggregates == []

    def test_measure_string_coercion(self):
        """Measures can be specified as strings."""
        data = {
            "name": "sales",
            "measures": ["amount", "quantity"],
        }
        cube = CubeSchema.model_validate(data)
        assert len(cube.measures) == 2
        assert cube.measures[0].name == "amount"
        assert cube.measures[1].name == "quantity"

    def test_aggregate_string_coercion(self):
        """Aggregates can be specified as strings."""
        data = {
            "name": "sales",
            "aggregates": ["total"],
        }
        cube = CubeSchema.model_validate(data)
        assert len(cube.aggregates) == 1
        assert cube.aggregates[0].name == "total"


class TestLegacyExpansionPatterns:
    """Test ported legacy cube expansion patterns."""

    def test_legacy_cube_with_dimensions(self):
        """Legacy cube with dimension strings."""
        data = {
            "name": "sales",
            "dimensions": ["date", "product"],
        }
        cube = CubeSchema.model_validate(data)
        assert len(cube.dimensions) == 2
        assert all(isinstance(d, dict) for d in cube.dimensions)

    def test_legacy_nonadditive_propagation(self):
        """Legacy nonadditive propagation to measures."""
        data = {
            "name": "sales",
            "nonadditive": "time",
            "measures": [
                {"name": "amount"},
            ],
        }
        cube = CubeSchema.model_validate(data)
        assert cube.measures[0].nonadditive == "time"
