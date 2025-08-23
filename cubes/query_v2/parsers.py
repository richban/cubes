"""
Modern parsers for OLAP query parameters with legacy format compatibility.

Transforms string-based query parameters into validated Cut and Drilldown objects,
maintaining 100% compatibility with existing cubes query formats while adding
comprehensive type safety and error handling.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from cubes.errors import ArgumentError
from cubes.metadata_v2 import Cube, Dimension
from cubes.query_v2.cells import Cut, PointCut, RangeCut, SetCut


class CutParser:
    """
    Modern parser for cut strings with legacy format compatibility.

    Parses cut specifications in the legacy format:
    - Point cuts: "date:2004,1"
    - Range cuts: "date:2004-2010" or "date:2004,1-2010,12"
    - Set cuts: "date:2004;2010;2011"
    - Multiple cuts: "date:2004|class:5"
    - Hierarchy specification: "date@fiscal:2004"
    - Inversion: "!date:2004"

    Transforms these into modern Pydantic-validated Cut objects.
    """

    CUT_STRING_SEPARATOR = re.compile(r"(?<!\\)\|")
    DIMENSION_STRING_SEPARATOR = re.compile(r"(?<!\\):")
    PATH_STRING_SEPARATOR = re.compile(r"(?<!\\),")
    RANGE_CUT_SEPARATOR = re.compile(r"(?<!\\)-")
    SET_CUT_SEPARATOR = re.compile(r"(?<!\\);")

    # Pattern matching for cut types
    RE_SET = re.compile(r".*;.*")
    RE_RANGE = re.compile(r".*-.*")

    def __init__(self, cube: Cube):
        """
        Initialize parser for specific cube.

        Args:
            cube: Cube metadata for dimension validation and resolution
        """
        self.cube = cube

    def parse_cuts(
        self,
        cuts_string: str,
        member_converters: dict[str, Callable] | None = None,
        role_member_converters: dict[str, Callable] | None = None,
    ) -> list[Cut]:
        """
        Parse cuts string into modern Cut objects.

        Maintains complete compatibility with legacy cuts_from_string format.

        Args:
            cuts_string: Cut specifications like "date:2004,1|class=5"
            member_converters: Optional converters by dimension name
            role_member_converters: Optional converters by dimension role

        Returns:
            List of Cut objects (PointCut, RangeCut, SetCut)

        Raises:
            ArgumentError: If parsing fails with detailed error context
        """
        if not cuts_string:
            return []

        cuts = []
        dim_cuts = self.CUT_STRING_SEPARATOR.split(cuts_string)

        for i, dim_cut in enumerate(dim_cuts):
            dim_cut = dim_cut.strip()
            if not dim_cut:
                continue

            try:
                cut = self.parse_single_cut(
                    dim_cut, member_converters, role_member_converters
                )
                cuts.append(cut)
            except Exception as e:
                raise ArgumentError(
                    f"Failed to parse cut {i + 1} of {len(dim_cuts)}: '{dim_cut}' - {e}"
                ) from e

        return cuts

    def parse_single_cut(
        self,
        cut_string: str,
        member_converters: dict[str, Callable] | None = None,
        role_member_converters: dict[str, Callable] | None = None,
    ) -> Cut:
        """
        Parse single cut string into Cut object.

        Handles all legacy cut formats:
        - Point: "date:2004,1"
        - Range: "date:2004-2010" or "date:2004,1-2010,12"
        - Set: "date:2004;2010;2011"
        - Hierarchy: "date@fiscal:2004"
        - Inversion: "!date:2004"
        - Empty path: "date:"

        Args:
            cut_string: Single cut specification
            member_converters: Optional converters by dimension name
            role_member_converters: Optional converters by dimension role

        Returns:
            Cut object (PointCut, RangeCut, or SetCut)

        Raises:
            ArgumentError: If cut format is invalid or dimension doesn't exist
        """
        member_converters = member_converters or {}
        role_member_converters = role_member_converters or {}

        # Parse dimension specification and path
        dim_spec, path_spec = self._parse_dimension_spec(cut_string)
        dimension_name, hierarchy, invert = self._parse_dim_hierarchy_invert(dim_spec)

        # Get dimension object and validate
        try:
            dimension = self.cube.dimension(dimension_name)
        except Exception as e:
            raise ArgumentError(
                f"Unknown dimension '{dimension_name}' in cube '{self.cube.name}'"
            ) from e

        # Get converter function if available
        converter = self._get_converter(
            dimension, member_converters, role_member_converters
        )

        # Parse path based on detected cut type
        if self.RE_SET.match(path_spec):
            return self._create_set_cut(
                dimension, hierarchy, invert, path_spec, converter
            )
        elif self.RE_RANGE.match(path_spec):
            return self._create_range_cut(
                dimension, hierarchy, invert, path_spec, converter
            )
        else:
            return self._create_point_cut(
                dimension, hierarchy, invert, path_spec, converter
            )

    def _parse_dimension_spec(self, cut_string: str) -> tuple[str, str]:
        """
        Parse 'dimension:path' format with proper error handling.

        Args:
            cut_string: Cut specification to parse

        Returns:
            Tuple of (dimension_spec, path_spec)

        Raises:
            ArgumentError: If format is invalid
        """
        try:
            parts = self.DIMENSION_STRING_SEPARATOR.split(cut_string, 1)
            if len(parts) != 2:
                raise ValueError("Missing dimension separator ':'")
            return parts[0], parts[1]
        except ValueError as e:
            raise ArgumentError(f"Invalid cut format '{cut_string}': {e}") from e

    def _parse_dim_hierarchy_invert(
        self, dim_spec: str
    ) -> tuple[str, str | None, bool]:
        """
        Parse dimension specification for inversion and hierarchy.

        Supports formats:
        - dimension
        - !dimension (inverted)
        - dimension@hierarchy
        - !dimension@hierarchy (inverted with hierarchy)

        Args:
            dim_spec: Dimension part of cut specification

        Returns:
            Tuple of (dimension_name, hierarchy_name, invert_flag)

        Raises:
            ArgumentError: If dimension specification is invalid
        """
        # Regex to match: optional !, dimension name, optional @hierarchy
        dim_hier_pattern = re.compile(
            r"(?P<invert>!)?"
            r"(?P<dim>[a-zA-Z_][a-zA-Z0-9_]*)"
            r"(@(?P<hier>[a-zA-Z_][a-zA-Z0-9_]*))?"
        )

        match = dim_hier_pattern.match(dim_spec)
        if not match:
            raise ArgumentError(f"Invalid dimension specification: '{dim_spec}'")

        groups = match.groupdict()
        invert = bool(groups.get("invert"))
        dimension = groups["dim"]
        hierarchy = groups.get("hier")

        return dimension, hierarchy, invert

    def _create_point_cut(
        self,
        dimension: Dimension,
        hierarchy: str | None,
        invert: bool,
        path_spec: str,
        converter: Callable | None,
    ) -> PointCut:
        """
        Create PointCut from path specification.

        Args:
            dimension: Dimension object
            hierarchy: Optional hierarchy name
            invert: Whether cut is inverted
            path_spec: Path specification (comma-separated)
            converter: Optional path converter function

        Returns:
            PointCut with validated path
        """
        # Handle empty path case
        if path_spec == "":
            path = [""]
        else:
            path = self._parse_path(path_spec)

        # Apply converter if provided
        if converter:
            try:
                path = converter(dimension, hierarchy, path)
            except Exception as e:
                raise ArgumentError(
                    f"Path converter failed for dimension '{dimension.name}': {e}"
                ) from e

        # Create and validate PointCut
        try:
            return PointCut(
                dimension=dimension, hierarchy=hierarchy, path=path, invert=invert
            )
        except Exception as e:
            raise ArgumentError(f"Failed to create point cut: {e}") from e

    def _create_range_cut(
        self,
        dimension: Dimension,
        hierarchy: str | None,
        invert: bool,
        path_spec: str,
        converter: Callable | None,
    ) -> RangeCut:
        """
        Create RangeCut from range path specification.

        Supports formats:
        - "from-to": Full range
        - "from-": Open-ended range from value
        - "-to": Open-ended range to value
        - "-": Invalid (both sides empty)

        Args:
            dimension: Dimension object
            hierarchy: Optional hierarchy name
            invert: Whether cut is inverted
            path_spec: Range specification with '-' separator
            converter: Optional path converter function

        Returns:
            RangeCut with validated paths

        Raises:
            ArgumentError: If range format is invalid
        """
        range_parts = self.RANGE_CUT_SEPARATOR.split(path_spec, 1)
        if len(range_parts) != 2:
            raise ArgumentError(
                f"Invalid range format: '{path_spec}' - expected single '-' separator"
            )

        from_spec, to_spec = range_parts

        # Parse paths (empty strings become None)
        from_path = self._parse_path(from_spec) if from_spec else None
        to_path = self._parse_path(to_spec) if to_spec else None

        # Validate at least one side is specified
        if from_path is None and to_path is None:
            raise ArgumentError(
                f"Invalid range format: '{path_spec}' - at least one range boundary required"
            )

        # Apply converter if provided
        if converter:
            try:
                if from_path:
                    from_path = converter(dimension, hierarchy, from_path)
                if to_path:
                    to_path = converter(dimension, hierarchy, to_path)
            except Exception as e:
                raise ArgumentError(f"Path converter failed for range cut: {e}") from e

        # Create and validate RangeCut
        try:
            return RangeCut(
                dimension=dimension,
                hierarchy=hierarchy,
                from_path=from_path,
                to_path=to_path,
                invert=invert,
            )
        except Exception as e:
            raise ArgumentError(f"Failed to create range cut: {e}") from e

    def _create_set_cut(
        self,
        dimension: Dimension,
        hierarchy: str | None,
        invert: bool,
        path_spec: str,
        converter: Callable | None,
    ) -> SetCut:
        """
        Create SetCut from set path specification.

        Parses semicolon-separated paths: "path1;path2;path3"

        Args:
            dimension: Dimension object
            hierarchy: Optional hierarchy name
            invert: Whether cut is inverted
            path_spec: Set specification with ';' separators
            converter: Optional path converter function

        Returns:
            SetCut with validated paths

        Raises:
            ArgumentError: If set format is invalid or empty
        """
        path_strings = self.SET_CUT_SEPARATOR.split(path_spec)

        # Filter out empty paths and parse
        paths = []
        for path_string in path_strings:
            if path_string.strip():  # Skip empty paths
                paths.append(self._parse_path(path_string))

        if not paths:
            raise ArgumentError(f"Set cut cannot be empty: '{path_spec}'")

        # Apply converter if provided
        if converter:
            try:
                paths = [converter(dimension, hierarchy, path) for path in paths]
            except Exception as e:
                raise ArgumentError(f"Path converter failed for set cut: {e}") from e

        # Create and validate SetCut
        try:
            return SetCut(
                dimension=dimension,
                hierarchy=hierarchy,
                paths=paths,
                invert=invert,
            )
        except Exception as e:
            raise ArgumentError(f"Failed to create set cut: {e}") from e

    def _parse_path(self, path_string: str) -> list[str]:
        """
        Parse comma-separated path components.

        Args:
            path_string: Path specification like "2004,1,5"

        Returns:
            List of path components
        """
        if not path_string:
            return []

        # Split by comma and preserve empty components (legacy behavior)
        return self.PATH_STRING_SEPARATOR.split(path_string)

    def _get_converter(
        self,
        dimension: Dimension,
        member_converters: dict[str, Callable],
        role_member_converters: dict[str, Callable],
    ) -> Callable | None:
        """
        Get appropriate converter for dimension.

        Looks up converter by dimension name first, then by role if available.

        Args:
            dimension: Dimension object
            member_converters: Converters by dimension name
            role_member_converters: Converters by dimension role

        Returns:
            Converter function or None if not found
        """
        # Try dimension name first
        converter = member_converters.get(dimension.name)

        # Try role if no dimension-specific converter
        if not converter and hasattr(dimension, "role") and dimension.role:
            converter = role_member_converters.get(dimension.role)

        return converter
