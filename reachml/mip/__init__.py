"""Mixed-integer programming (MIP) models for ReachML actions.

This package exposes a thin, solver-agnostic facade (`BaseMIP`, `EnumeratorMIP`)
that delegates to per-solver backends (CPLEX/SCIP) implementing a small
interface. Optional dependencies are imported only by the backend modules.
"""

from .base_mip import BaseMIP
from .enumerator_mip import EnumeratorMIP

__all__ = [
    "BaseMIP",
    "EnumeratorMIP",
]
