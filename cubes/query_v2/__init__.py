"""
Modern query module for Cubes OLAP framework.

This module provides modernized query interfaces using Pydantic-based
metadata models from cubes.metadata_v2.
"""

from .cells import Cell, PointCut, RangeCut, SetCut
from .query import AggregateSpec, QuerySpec

__all__ = [
    "Cell",
    "PointCut",
    "RangeCut",
    "SetCut",
    "QuerySpec",
]
