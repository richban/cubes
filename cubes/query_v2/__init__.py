"""
Modern query module for Cubes OLAP framework.

This module provides modernized query interfaces using Pydantic-based
metadata models from cubes.metadata_v2.
"""

from .browser import (
    AggregationBrowser,
    AggregationResult,
    Drilldown,
    DrilldownSpec,
    OrderDirection,
)
from .cells import Cell, PointCut, RangeCut, SetCut

__all__ = [
    "Cell",
    "PointCut",
    "RangeCut",
    "SetCut",
    "AggregationBrowser",
    "AggregationResult",
    "Drilldown",
    "DrilldownSpec",
    "OrderDirection",
]
