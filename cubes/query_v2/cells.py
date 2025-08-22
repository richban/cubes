"""
Modern Cell and Cut implementations using Pydantic for type safety and validation.
"""

from abc import ABC, abstractmethod
from typing import Any, Union

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from cubes.errors import ArgumentError, ModelError
from cubes.metadata_v2 import Cube, Dimension, Hierarchy


class Cut(BaseModel, ABC):
    """Abstract base class for modern cut implementations using Pydantic."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="forbid"
    )

    dimension: Dimension = Field(..., description="The dimension this cut applies to")
    hierarchy: str | None = Field(
        None,
        examples=["fiscal_year"],
        description="Hierarchy name within the dimension",
    )
    invert: bool = Field(
        False, examples=[False, True], description="Whether to invert the cut logic"
    )
    hidden: bool = Field(
        False,
        examples=[False, True],
        description="Whether this cut is hidden from public view",
    )

    @field_validator("hierarchy")
    @classmethod
    def validate_hierarchy(cls, v: str | None, info) -> str | None:
        """Validate hierarchy exists in the dimension."""
        if v is not None and "dimension" in info.data:
            dimension = info.data["dimension"]
            try:
                dimension.hierarchy(v)
            except ModelError as e:
                msg = f"Invalid hierarchy '{v}' for dimension '{dimension.name}': {e}"
                raise ValueError(msg)
        return v

    @abstractmethod
    def level_depth(self) -> int:
        """Returns the deepest level index for this cut."""

    @computed_field
    @property
    def hierarchy_obj(self) -> Hierarchy:
        """Get the actual hierarchy object."""
        return self.dimension.hierarchy(self.hierarchy)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "dimension": self.dimension.name,
            "hierarchy": self.hierarchy,
            "level_depth": self.level_depth(),
            "invert": self.invert,
            "hidden": self.hidden,
        }


class PointCut(Cut):
    """Type-safe point cut using Pydantic dimension reference."""

    path: list[str] = Field(
        ...,
        examples=[["US", "California", "San Francisco"]],
        description="Dimension path for the point cut",
    )

    @field_validator("path", mode="before")
    @classmethod
    def validate_path(cls, v: list[Any]) -> list[str]:
        """Convert path elements to strings (like legacy behavior)."""
        return [str(item) for item in v]

    @field_validator("path")
    @classmethod
    def validate_path_hierarchy(cls, v: list[str], info) -> list[str]:
        """Validate path against dimension hierarchy."""
        if "dimension" in info.data:
            dimension = info.data["dimension"]
            hierarchy_name = info.data.get("hierarchy")
            try:
                hierarchy = dimension.hierarchy(hierarchy_name)
                hierarchy.levels_for_path(v)
            except Exception as e:
                msg = f"Invalid path {v} for dimension '{dimension.name}': {e}"
                raise ValueError(msg)
        return v

    def level_depth(self) -> int:
        """Returns the depth of the path."""
        return len(self.path)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        d = super().to_dict()
        d["type"] = "point"
        d["path"] = self.path
        return d

    def __str__(self) -> str:
        """String representation for URLs."""
        from cubes.query.cells import (
            DIMENSION_STRING_SEPARATOR_CHAR,
            string_from_hierarchy,
            string_from_path,
        )

        path_str = string_from_path(self.path)
        dim_str = string_from_hierarchy(self.dimension.name, self.hierarchy)
        return (
            ("!" if self.invert else "")
            + dim_str
            + DIMENSION_STRING_SEPARATOR_CHAR
            + path_str
        )


class RangeCut(Cut):
    """Type-safe range cut using Pydantic dimension reference."""

    from_path: list[str] | None = Field(
        None, examples=[["2020", "Q1"]], description="Starting path for range"
    )
    to_path: list[str] | None = Field(
        None, examples=[["2023", "Q4"]], description="Ending path for range"
    )

    @field_validator("from_path", "to_path", mode="before")
    @classmethod
    def validate_paths_convert(cls, v: list[Any] | None) -> list[str] | None:
        """Convert path elements to strings (like legacy behavior)."""
        if v is None:
            return None
        return [str(item) for item in v]

    @field_validator("from_path", "to_path")
    @classmethod
    def validate_paths_hierarchy(cls, v: list[str] | None, info) -> list[str] | None:
        """Validate paths against dimension hierarchy."""
        if v is not None and "dimension" in info.data:
            dimension = info.data["dimension"]
            hierarchy_name = info.data.get("hierarchy")
            try:
                hierarchy = dimension.hierarchy(hierarchy_name)
                hierarchy.levels_for_path(v)
            except Exception as e:
                msg = f"Invalid path {v} for dimension '{dimension.name}': {e}"
                raise ValueError(msg)
        return v

    def level_depth(self) -> int:
        """Returns the depth of the deepest path."""
        if self.from_path and not self.to_path:
            return len(self.from_path)
        if not self.from_path and self.to_path:
            return len(self.to_path)
        if self.from_path and self.to_path:
            return max(len(self.from_path), len(self.to_path))
        return 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        d = super().to_dict()
        d["type"] = "range"
        d["from"] = self.from_path
        d["to"] = self.to_path
        return d

    def __str__(self) -> str:
        """String representation for URLs."""
        from cubes.query.cells import (
            DIMENSION_STRING_SEPARATOR_CHAR,
            RANGE_CUT_SEPARATOR_CHAR,
            string_from_hierarchy,
            string_from_path,
        )

        from_path_str = string_from_path(self.from_path or [])
        to_path_str = string_from_path(self.to_path or [])
        range_str = from_path_str + RANGE_CUT_SEPARATOR_CHAR + to_path_str
        dim_str = string_from_hierarchy(self.dimension.name, self.hierarchy)

        return (
            ("!" if self.invert else "")
            + dim_str
            + DIMENSION_STRING_SEPARATOR_CHAR
            + range_str
        )


class SetCut(Cut):
    """Type-safe set cut using Pydantic dimension reference."""

    paths: list[list[str]] = Field(
        ...,
        examples=[[["Electronics"], ["Books"], ["Clothing", "Mens"]]],
        description="List of paths for the set cut",
    )

    @field_validator("paths", mode="before")
    @classmethod
    def validate_paths_convert(cls, v: list[list[Any]]) -> list[list[str]]:
        """Convert path elements to strings (like legacy behavior)."""
        return [[str(item) for item in path] for path in v]

    @field_validator("paths")
    @classmethod
    def validate_paths_hierarchy(cls, v: list[list[str]], info) -> list[list[str]]:
        """Validate all paths against dimension hierarchy."""
        if "dimension" in info.data:
            dimension = info.data["dimension"]
            hierarchy_name = info.data.get("hierarchy")
            try:
                hierarchy = dimension.hierarchy(hierarchy_name)
                for path in v:
                    hierarchy.levels_for_path(path)
            except Exception as e:
                msg = f"Invalid paths {v} for dimension '{dimension.name}': {e}"
                raise ValueError(msg)
        return v

    def level_depth(self) -> int:
        """Returns the depth of the deepest path."""
        if not self.paths:
            return 0
        return max(len(path) for path in self.paths)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        d = super().to_dict()
        d["type"] = "set"
        d["paths"] = self.paths
        return d

    def __str__(self) -> str:
        """String representation for URLs."""
        from cubes.query.cells import (
            DIMENSION_STRING_SEPARATOR_CHAR,
            SET_CUT_SEPARATOR_CHAR,
            string_from_hierarchy,
            string_from_path,
        )

        path_strings = [string_from_path(path) for path in self.paths]
        set_string = SET_CUT_SEPARATOR_CHAR.join(path_strings)
        dim_str = string_from_hierarchy(self.dimension.name, self.hierarchy)

        return (
            ("!" if self.invert else "")
            + dim_str
            + DIMENSION_STRING_SEPARATOR_CHAR
            + set_string
        )


class Cell(BaseModel):
    """Type-safe cell implementation using Pydantic."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="forbid"
    )

    cube: Cube = Field(..., description="The cube this cell applies to")
    cuts: list[Cut] = Field(
        default_factory=list, example=[], description="List of cuts defining the cell"
    )

    @field_validator("cuts")
    @classmethod
    def validate_cuts(cls, v: list[Cut], info) -> list[Cut]:
        """Validate cuts are consistent and dimensions exist in cube."""
        if "cube" in info.data:
            cube = info.data["cube"]
            for cut in v:
                try:
                    # Verify dimension exists in cube
                    cube.dimension(cut.dimension.name)
                except ModelError as e:
                    msg = f"Cut dimension '{cut.dimension.name}' not found in cube '{cube.name}': {e}"
                    raise ValueError(msg)
        return v

    def slice(self, cut: Cut) -> "Cell":
        """Returns new cell by slicing with the provided cut."""
        cuts = self.cuts[:]
        index = self._find_dimension_cut(cut.dimension.name)
        if index is not None:
            cuts[index] = cut
        else:
            cuts.append(cut)

        return Cell(cube=self.cube, cuts=cuts)

    def _find_dimension_cut(self, dimension_name: str) -> int | None:
        """Find index of cut for the given dimension."""
        for i, cut in enumerate(self.cuts):
            if cut.dimension.name == dimension_name:
                return i
        return None

    def cut_for_dimension(self, dimension_name: str) -> Cut | None:
        """Return first cut for the given dimension."""
        for cut in self.cuts:
            if cut.dimension.name == dimension_name:
                return cut
        return None

    def point_cut_for_dimension(self, dimension_name: str) -> PointCut | None:
        """Return first point cut for the given dimension."""
        for cut in self.cuts:
            if cut.dimension.name == dimension_name and isinstance(cut, PointCut):
                return cut
        return None

    def drilldown(
        self, dimension_name: str, value: str, hierarchy: str | None = None
    ) -> "Cell":
        """Create another cell by drilling down dimension to next level."""
        dimension = self.cube.dimension(dimension_name)
        dim_cut = self.cut_for_dimension(dimension_name)

        old_path = dim_cut.path if isinstance(dim_cut, PointCut) else []
        new_cut = PointCut(
            dimension=dimension, path=[*old_path, value], hierarchy=hierarchy
        )

        return self.slice(new_cut)

    def rollup_dim(
        self,
        dimension_name: str,
        level: str | None = None,
        hierarchy: str | None = None,
    ) -> "Cell":
        """Roll up dimension by one or more levels."""
        dimension = self.cube.dimension(dimension_name)
        dim_cut = self.point_cut_for_dimension(dimension_name)

        if not dim_cut:
            return Cell(cube=self.cube, cuts=self.cuts[:])

        cuts = [cut for cut in self.cuts if cut.dimension.name != dimension_name]

        hier = dimension.hierarchy(hierarchy)
        rollup_path = hier.rollup(dim_cut.path, level)

        if rollup_path:
            new_cut = PointCut(
                dimension=dimension, path=rollup_path, hierarchy=hierarchy
            )
            cuts.append(new_cut)

        return Cell(cube=self.cube, cuts=cuts)

    def multi_slice(self, cuts: list[Cut]) -> "Cell":
        """Create another cell by applying multiple cuts."""
        cell = self
        for cut in cuts:
            cell = cell.slice(cut)
        return cell

    @computed_field
    @property
    def all_attributes(self) -> list[Any]:
        """Returns all key attributes used in the cell's cuts."""
        attributes = set()

        for cut in self.cuts:
            depth = cut.level_depth()
            if depth:
                hier = cut.hierarchy_obj
                keys = [hier.levels[i].key_attribute for i in range(depth)]
                attributes.update(keys)

        return list(attributes)

    def level_depths(self) -> dict[str, int]:
        """Returns dictionary of dimension names and their level depths."""
        levels = {}

        for cut in self.cuts:
            level = cut.level_depth()
            dim_name = cut.dimension.name
            levels[dim_name] = max(level, levels.get(dim_name, 0))

        return levels

    def deepest_levels(
        self, include_empty: bool = False
    ) -> list[tuple[Dimension, Hierarchy, Any]]:
        """Returns list of tuples with deepest levels for each cut."""
        levels = []

        for cut in self.cuts:
            depth = cut.level_depth()
            hier = cut.hierarchy_obj
            if depth:
                level = hier.levels[depth - 1]
                levels.append((cut.dimension, hier, level))
            elif include_empty:
                levels.append((cut.dimension, hier, None))

        return levels

    def dimension_cuts(
        self, dimension_name: str, exclude: bool = False
    ) -> list[Cut]:
        """Return cuts for dimension, or all except that dimension if exclude=True."""
        result = []
        for cut in self.cuts:
            if (exclude and cut.dimension.name != dimension_name) or (
                not exclude and cut.dimension.name == dimension_name
            ):
                result.append(cut)
        return result

    def public_cell(self) -> "Cell":
        """Returns cell with only non-hidden cuts."""
        cuts = [cut for cut in self.cuts if not cut.hidden]
        return Cell(cube=self.cube, cuts=cuts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {"cube": self.cube.name, "cuts": [cut.to_dict() for cut in self.cuts]}

    def __str__(self) -> str:
        """String representation using cuts-to-string conversion."""

        # Convert modern cuts to legacy format for string representation
        legacy_cuts = []
        for cut in self.cuts:
            # This would need conversion logic to legacy Cut objects
            # For now, use the string representation of modern cuts
            legacy_cuts.append(str(cut))
        return "|".join(legacy_cuts)

    def __bool__(self) -> bool:
        """Returns True if the cell contains cuts."""
        return bool(self.cuts)

    def __and__(self, other: "Cell") -> "Cell":
        """Combine two cells from the same cube."""
        if self.cube.name != other.cube.name:
            msg = f"Cannot combine cells from different cubes '{self.cube.name}' and '{other.cube.name}'"
            raise ArgumentError(msg)
        cuts = self.cuts + other.cuts
        return Cell(cube=self.cube, cuts=cuts)


# Type aliases for convenience
Cut = Union[PointCut, RangeCut, SetCut]
