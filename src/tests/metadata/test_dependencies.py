"""Tests for attribute dependency utilities."""

import pytest

from cubes.metadata.models import ExpressionError
from cubes.metadata.models.attributes import (
    Attribute,
    AttributeBase,
    collect_attributes,
    collect_dependencies,
    depsort_attributes,
)


class TestCollectAttributes:
    """Tests for collect_attributes utility function."""

    def test_empty_attributes(self):
        """Test collecting with no attributes returns empty list."""
        result = collect_attributes(None)
        assert result == []

    def test_empty_list(self):
        """Test collecting with empty list returns empty list."""
        result = collect_attributes([])
        assert result == []

    def test_single_attribute(self):
        """Test collecting single attribute."""
        attr = Attribute(name="test_attr")
        result = collect_attributes([attr])
        assert result == [attr]

    def test_multiple_attributes(self):
        """Test collecting multiple attributes."""
        attrs = [Attribute(name="a"), Attribute(name="b"), Attribute(name="c")]
        result = collect_attributes(attrs)
        assert result == attrs

    def test_with_container(self):
        """Test collecting from container with all_attributes."""
        attrs = [Attribute(name="a")]

        class MockContainer:
            all_attributes = [Attribute(name="b"), Attribute(name="c")]

        container = MockContainer()
        result = collect_attributes(attrs, container)

        assert len(result) == 3
        assert result[0].name == "a"
        assert result[1].name == "b"
        assert result[2].name == "c"

    def test_with_multiple_containers(self):
        """Test collecting from multiple containers."""
        attrs = [Attribute(name="a")]

        class Container1:
            all_attributes = [Attribute(name="b")]

        class Container2:
            all_attributes = [Attribute(name="c")]

        result = collect_attributes(attrs, Container1(), Container2())

        assert len(result) == 3
        assert [attr.name for attr in result] == ["a", "b", "c"]

    def test_with_none_container(self):
        """Test that None containers are skipped."""
        attrs = [Attribute(name="a")]
        result = collect_attributes(attrs, None)
        assert result == attrs

    def test_with_container_no_all_attributes(self):
        """Test that containers without all_attributes are skipped."""
        attrs = [Attribute(name="a")]

        class NoAttrsContainer:
            pass

        result = collect_attributes(attrs, NoAttrsContainer())
        assert result == attrs

    def test_with_empty_container(self):
        """Test that containers with empty all_attributes work."""
        attrs = [Attribute(name="a")]

        class EmptyContainer:
            all_attributes = []

        result = collect_attributes(attrs, EmptyContainer())
        assert result == attrs


class TestDepsortAttributes:
    """Tests for depsort_attributes topological sorting function."""

    def test_empty_list(self):
        """Test sorting empty list returns empty list."""
        result = depsort_attributes([], {})
        assert result == []

    def test_single_attribute_no_deps(self):
        """Test sorting single attribute with no dependencies."""
        deps = {"a": set()}
        result = depsort_attributes(["a"], deps)
        assert result == ["a"]

    def test_single_attribute_with_deps(self):
        """Test sorting single attribute that depends on another."""
        deps = {"a": set(), "b": {"a"}}
        result = depsort_attributes(["b"], deps)
        assert result == ["a", "b"]

    def test_linear_dependency_chain(self):
        """Test sorting linear chain: c -> b -> a."""
        deps = {"a": set(), "b": {"a"}, "c": {"b"}}
        result = depsort_attributes(["c", "b", "a"], deps)
        assert result == ["a", "b", "c"]

    def test_diamond_dependency(self):
        """Test sorting diamond pattern: d depends on b and c, both depend on a."""
        deps = {"a": set(), "b": {"a"}, "c": {"a"}, "d": {"b", "c"}}
        result = depsort_attributes(["d"], deps)
        # a must come before b and c, which must come before d
        assert result[0] == "a"
        assert result[-1] == "d"
        assert "b" in result[1:3]
        assert "c" in result[1:3]

    def test_unordered_input(self):
        """Test that input order doesn't matter, output is always sorted by deps."""
        deps = {"a": set(), "b": {"a"}, "c": {"b"}}
        # Try different input orders
        result1 = depsort_attributes(["c", "b", "a"], deps)
        result2 = depsort_attributes(["a", "c", "b"], deps)
        result3 = depsort_attributes(["b", "a", "c"], deps)

        assert result1 == ["a", "b", "c"]
        assert result2 == ["a", "b", "c"]
        assert result3 == ["a", "b", "c"]

    def test_multiple_base_attributes(self):
        """Test sorting with multiple independent base attributes."""
        deps = {"a": set(), "b": set(), "c": {"a"}, "d": {"b"}}
        result = depsort_attributes(["c", "d"], deps)
        # Both c and d should come after their respective bases
        assert result[0] in ["a", "b"]
        assert result[1] in ["a", "b"]
        assert result[2] in ["c", "d"]
        assert result[3] in ["c", "d"]
        assert "a" in result[:2]
        assert "b" in result[:2]

    def test_unknown_attribute(self):
        """Test that unknown attribute raises ExpressionError."""
        deps = {"a": set()}
        with pytest.raises(ExpressionError, match="Unknown attribute"):
            depsort_attributes(["unknown"], deps)

    def test_simple_circular_dependency(self):
        """Test that simple circular dependency is detected."""
        deps = {"loop1": {"loop2"}, "loop2": {"loop1"}}
        with pytest.raises(ExpressionError, match="Circular attribute reference"):
            depsort_attributes(["loop1", "loop2"], deps)

    def test_indirect_circular_dependency(self):
        """Test that indirect circular dependency is detected."""
        deps = {"a": {"b"}, "b": {"c"}, "c": {"a"}}
        with pytest.raises(ExpressionError, match="Circular attribute reference"):
            depsort_attributes(["a", "b", "c"], deps)

    def test_circular_with_intermediate(self):
        """Test circular dependency via intermediate attributes."""
        deps = {
            "indirect_loop1": {"intermediate"},
            "intermediate": {"indirect_loop1"},
            "indirect_loop2": {"indirect_loop1"},
        }
        with pytest.raises(ExpressionError, match="Circular attribute reference"):
            depsort_attributes(
                ["indirect_loop1", "intermediate", "indirect_loop2"], deps
            )

    def test_self_reference(self):
        """Test that attribute depending on itself is circular."""
        deps = {"a": {"a"}}
        with pytest.raises(ExpressionError, match="Circular attribute reference"):
            depsort_attributes(["a"], deps)

    def test_constant_expression(self):
        """Test attribute with constant expression (no dependencies)."""
        deps = {
            "a": set(),
            "b": set(),  # expression "10" has no variable dependencies
            "c": {"b"},
        }
        result = depsort_attributes(["c"], deps)
        assert result == ["b", "c"]

    def test_subset_of_attributes(self):
        """Test sorting only a subset of known attributes."""
        deps = {
            "a": set(),
            "b": {"a"},
            "c": {"b"},
            "d": set(),  # Not requested, but might be needed as dependency
        }
        # Request only c, which depends on b, which depends on a
        result = depsort_attributes(["c"], deps)
        assert result == ["a", "b", "c"]
        assert "d" not in result


class TestCollectDependencies:
    """Tests for collect_dependencies function."""

    def test_empty_list(self):
        """Test collecting dependencies of empty list."""
        result = collect_dependencies([], [])
        assert result == []

    def test_base_attribute_no_deps(self):
        """Test attribute with no expression returns just itself."""
        attr = AttributeBase(name="a")
        all_attrs = [attr]
        result = collect_dependencies([attr], all_attrs)
        assert result == ["a"]

    def test_single_dependency(self):
        """Test attribute depending on one other attribute."""
        base_attr = AttributeBase(name="base")
        derived_attr = AttributeBase(name="derived", expression="base * 2")

        result = collect_dependencies([derived_attr], [base_attr, derived_attr])
        assert result == ["base", "derived"]

    def test_chain_of_dependencies(self):
        """Test chain of attribute dependencies."""
        attr_a = AttributeBase(name="a")
        attr_b = AttributeBase(name="b", expression="a + 1")
        attr_c = AttributeBase(name="c", expression="b + 1")

        all_attrs = [attr_a, attr_b, attr_c]
        result = collect_dependencies([attr_c], all_attrs)
        assert result == ["a", "b", "c"]

    def test_multiple_attributes(self):
        """Test collecting dependencies for multiple attributes."""
        attr_a = AttributeBase(name="a")
        attr_b = AttributeBase(name="b", expression="a + 1")
        attr_c = AttributeBase(name="c", expression="a * 2")

        all_attrs = [attr_a, attr_b, attr_c]
        result = collect_dependencies([attr_b, attr_c], all_attrs)
        # a must come first, then b and c in any order
        assert result[0] == "a"
        assert "b" in result[1:]
        assert "c" in result[1:]

    def test_missing_dependency(self):
        """Test that missing dependency raises ExpressionError."""
        attr = AttributeBase(name="derived", expression="unknown + 1")

        with pytest.raises(ExpressionError, match="Unknown attribute"):
            collect_dependencies([attr], [attr])

    def test_circular_dependency(self):
        """Test that circular dependency raises ExpressionError."""
        attr_a = AttributeBase(name="loop1", expression="loop2 + 1")
        attr_b = AttributeBase(name="loop2", expression="loop1 + 1")

        with pytest.raises(ExpressionError, match="Circular"):
            collect_dependencies([attr_a, attr_b], [attr_a, attr_b])

    def test_complex_expressions(self):
        """Test dependencies extracted from complex expressions."""
        attr_price = AttributeBase(name="price")
        attr_qty = AttributeBase(name="quantity")
        attr_total = AttributeBase(
            name="total", expression="price * quantity + price * 0.1"
        )

        all_attrs = [attr_price, attr_qty, attr_total]
        result = collect_dependencies([attr_total], all_attrs)

        # Both price and quantity must come before total
        assert "price" in result[:-1]
        assert "quantity" in result[:-1]
        assert result[-1] == "total"

    def test_attributes_with_dimension_refs(self):
        """Test attributes with dimension-prefixed refs."""
        # Create attributes with dimension-prefixed refs
        attr_a = AttributeBase(name="dim.a")
        attr_b = AttributeBase(name="dim.b", expression="dim.a + 1")

        # Override refs to match names (simulating dimension ownership)
        object.__setattr__(attr_a, "ref", "dim.a")
        object.__setattr__(attr_b, "ref", "dim.b")

        all_attrs = [attr_a, attr_b]
        result = collect_dependencies([attr_b], all_attrs)

        # Should return refs
        assert "dim.a" in result
        assert "dim.b" in result

    def test_uses_ref_not_name(self):
        """Test that collect_dependencies uses ref, not name."""
        attr = AttributeBase(name="original_name")
        attr.ref = "custom_ref"

        all_attrs = [attr]
        result = collect_dependencies([attr], all_attrs)

        # Should return ref, not name
        assert result == ["custom_ref"]


class TestIntegration:
    """Integration tests combining multiple utilities."""

    def test_full_workflow(self):
        """Test complete workflow from attributes to sorted dependencies."""
        # Create attributes
        attrs = [
            Attribute(name="id"),
            Attribute(name="amount", expression="price * quantity"),
            Attribute(name="price"),
            Attribute(name="quantity"),
        ]

        # Collect all attributes
        collected = collect_attributes(attrs)
        assert len(collected) == 4

        # Collect and sort dependencies
        result = collect_dependencies([attrs[1]], collected)  # amount

        # amount depends on price and quantity
        # Should have id, price, quantity first (order may vary for independent ones)
        # Then amount last
        assert result[-1] == "amount"
        assert "price" in result[:-1]
        assert "quantity" in result[:-1]

    def test_with_mock_cube_container(self):
        """Test collect_attributes with mock cube container."""
        cube_attrs = [Attribute(name="fact_a"), Attribute(name="fact_b")]
        dim_attrs = [Attribute(name="dim_attr")]

        class MockCube:
            all_attributes = cube_attrs

        result = collect_attributes(dim_attrs, MockCube())
        assert len(result) == 3
        assert result[0] == dim_attrs[0]
        assert result[1:] == cube_attrs

    def test_depsort_matches_legacy_tests(self):
        """Test that depsort produces same results as legacy test cases."""
        # These are the exact test cases from legacy test_expressions.py
        attrs = [
            {"name": "a"},
            {"name": "b", "expression": "a + 1"},
            {"name": "c", "expression": "b + 1"},
            {"name": "d", "expression": "10"},
            {"name": "e", "expression": "unknown"},
            {"name": "loop1", "expression": "loop2 + 1"},
            {"name": "loop2", "expression": "loop1 + 1"},
            {"name": "indirect_loop1", "expression": "intermediate"},
            {"name": "intermediate", "expression": "indirect_loop1"},
            {"name": "indirect_loop2", "expression": "indirect_loop1"},
        ]

        # Create AttributeBase objects (not Attribute since we don't need dimensions)
        attr_objects = {a["name"]: AttributeBase(**a) for a in attrs}

        deps = {name: attr.dependencies for name, attr in attr_objects.items()}

        # Test from legacy: test_sorted_attributes_base
        result = depsort_attributes([], deps)
        assert result == []

        result = depsort_attributes(["a"], deps)
        assert result == ["a"]

        # Test from legacy: test_sorted_attributes
        result = depsort_attributes(["c", "b", "a"], deps)
        assert result == ["a", "b", "c"]

        # Test from legacy: test_sorted_unknown
        with pytest.raises(ExpressionError, match="Unknown"):
            depsort_attributes(["c", "b"], {})

        # Test from legacy: test_sorted_circular
        with pytest.raises(ExpressionError, match="Circular"):
            depsort_attributes(["loop1", "loop2"], deps)

        with pytest.raises(ExpressionError, match="Circular"):
            depsort_attributes(
                ["indirect_loop1", "intermediate", "indirect_loop2"], deps
            )
