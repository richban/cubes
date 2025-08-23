import pytest
from unittest.mock import Mock
from cubes.errors import ArgumentError
from cubes.query_v2.browser import DrilldownSpec, DrilldownItem
from cubes.metadata_v2 import Dimension, Level, Hierarchy

@pytest.mark.parametrize(
    "spec, expected_str",
    [
        (DrilldownSpec(dimension="geography"), "geography"),
        (DrilldownSpec(dimension="geography", level="country"), "geography:country"),
        (DrilldownSpec(dimension="geography", hierarchy="alt"), "geography@alt"),
        (
            DrilldownSpec(dimension="geography", hierarchy="alt", level="country"),
            "geography@alt:country",
        ),
        (DrilldownSpec(dimension="date", level="year"), "date:year"),
    ],
)
def test_drilldown_spec_str_representation(spec, expected_str):
    """Tests the string representation of DrilldownSpec."""
    assert str(spec) == expected_str

@pytest.mark.parametrize(
    "spec_string, expected_spec",
    [
        ("geography", DrilldownSpec(dimension="geography")),
        ("geography:country", DrilldownSpec(dimension="geography", level="country")),
        ("geography@alt", DrilldownSpec(dimension="geography", hierarchy="alt")),
        (
            "geography@alt:country",
            DrilldownSpec(dimension="geography", hierarchy="alt", level="country"),
        ),
        ("date:year", DrilldownSpec(dimension="date", level="year")),
    ],
)
def test_from_string_valid(spec_string, expected_spec):
    """Tests creating DrilldownSpec from valid string formats."""
    spec = DrilldownSpec.from_string(spec_string)
    assert spec == expected_spec

@pytest.mark.parametrize(
    "invalid_string",
    ["", ":level", "@hierarchy", "@hierarchy:level", "dim:level:another"],
)
def test_from_string_invalid(invalid_string):
    """Tests creating DrilldownSpec from invalid string formats."""
    with pytest.raises(ArgumentError):
        DrilldownSpec.from_string(invalid_string)

@pytest.mark.parametrize(
    "spec_tuple, expected_spec",
    [
        (("geography", None, None), DrilldownSpec(dimension="geography")),
        (("geography", None, "country"), DrilldownSpec(dimension="geography", level="country")),
        (("geography", "alt", None), DrilldownSpec(dimension="geography", hierarchy="alt")),
        (
            ("geography", "alt", "country"),
            DrilldownSpec(dimension="geography", hierarchy="alt", level="country"),
        ),
    ],
)
def test_from_tuple_valid(spec_tuple, expected_spec):
    """Tests creating DrilldownSpec from valid tuple formats."""
    spec = DrilldownSpec.from_tuple(spec_tuple)
    assert spec == expected_spec

@pytest.mark.parametrize(
    "invalid_tuple",
    [(), ("geography",), ("geography", "alt", "country", "extra"), (None, None, None)],
)
def test_from_tuple_invalid(invalid_tuple):
    """Tests creating DrilldownSpec from invalid tuple formats."""
    with pytest.raises(ArgumentError):
        DrilldownSpec.from_tuple(invalid_tuple)

def test_from_format_string():
    """Tests from_format with a string."""
    spec = DrilldownSpec.from_format("geography@alt:country")
    assert spec == DrilldownSpec("geography", "alt", "country")

def test_from_format_tuple():
    """Tests from_format with a tuple."""
    spec = DrilldownSpec.from_format(("geography", "alt", "country"))
    assert spec == DrilldownSpec("geography", "alt", "country")

def test_from_format_drilldown_spec():
    """Tests from_format with a DrilldownSpec instance."""
    original_spec = DrilldownSpec("geography", "alt", "country")
    spec = DrilldownSpec.from_format(original_spec)
    assert spec is original_spec

def test_from_format_drilldown_item():
    """Tests from_format with a DrilldownItem."""
    mock_dimension = Mock(spec=Dimension)
    mock_dimension.name = "geography"
    mock_hierarchy = Mock(spec=Hierarchy)
    mock_hierarchy.name = "alt"
    mock_level = Mock(spec=Level)
    mock_level.name = "country"
    mock_dimension.hierarchy.return_value = mock_hierarchy

    drilldown_item = DrilldownItem(
        dimension=mock_dimension,
        hierarchy=mock_hierarchy,
        levels=[mock_level],
        keys=["country_key"],
    )
    spec = DrilldownSpec.from_format(drilldown_item)
    assert spec == DrilldownSpec("geography", "alt", "country")

def test_from_format_dimension():
    """Tests from_format with a Dimension object."""
    mock_dimension = Mock(spec=Dimension)
    mock_dimension.name = "geography"
    mock_hierarchy = Mock(spec=Hierarchy)
    mock_level = Mock(spec=Level)
    mock_level.name = "country"
    mock_hierarchy.levels = [mock_level]
    mock_dimension.hierarchy.return_value = mock_hierarchy

    spec = DrilldownSpec.from_format(mock_dimension)
    assert spec == DrilldownSpec("geography", None, "country")

@pytest.mark.parametrize("invalid_format", [123, 12.34, {"a": 1}, b"bytes"])
def test_from_format_unsupported(invalid_format):
    """Tests from_format with unsupported types."""
    with pytest.raises(ArgumentError):
        DrilldownSpec.from_format(invalid_format)