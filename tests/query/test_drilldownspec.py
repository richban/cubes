
import pytest

from cubes.errors import ArgumentError
from cubes.query_v2.browser import DrilldownSpec


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


@pytest.mark.parametrize("invalid_format", [123, 12.34, {"a": 1}, b"bytes"])
def test_from_format_unsupported(invalid_format):
    """Tests from_format with unsupported types."""
    with pytest.raises(ArgumentError):
        DrilldownSpec.from_format(invalid_format)
