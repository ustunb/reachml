"""MIP backend interface, settings, and shared helpers.

Defines the `MIPBackend` protocol implemented by solver backends (CPLEX/SCIP),
common helpers, and the `load_backend` registry used by the public MIP facade.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Protocol, Tuple

import numpy as np

from ..action_set import ActionSet


@dataclass(frozen=True)
class MIPSettings:
    eps_min: float = 0.5
    round_precision: int = 8


class MIPBackend(Protocol):
    """Protocol each solver backend must implement."""

    solver_name: str

    # Build/configure
    def build_model(
        self, action_set: ActionSet, x: np.ndarray, actionable_indices: List[int]
    ) -> Tuple[object, object]:
        ...

    def configure(self, model: object, print_flag: bool) -> object:
        ...

    def add_constraints(self, model: object, indices: object, action_set: ActionSet, x: np.ndarray) -> Tuple[object, object]:
        ...

    # Solve/inspect
    def solve(self, model: object) -> None:
        ...

    def has_solution(self, model: object) -> bool:
        ...

    def read_vectors(self, model: object, indices: object, names: List[str]) -> Dict[str, np.ndarray]:
        ...

    def add_nogood(self, model: object, indices: object, actions: List[np.ndarray], actionable_indices: List[int], settings: MIPSettings) -> Tuple[object, object, int]:
        ...

    def stats(self, model: object) -> Dict:
        ...

    # Generic constraint operations
    def add_linear_constraint(
        self,
        model: object,
        indices: object,
        name: str,
        terms: List[tuple[str, int, float]],
        sense: str,
        rhs: float,
    ) -> None:
        ...

    def delete_constraint(self, model: object, indices: object, name: str) -> None:
        ...

    def solution_status(self, model: object) -> str:
        ...


# --------------- Common checks/helpers ---------------


def check_bounds(values: np.ndarray, lb: np.ndarray, ub: np.ndarray, tol: float = 1e-9) -> bool:
    values = np.array(values)
    lb = np.array(lb)
    ub = np.array(ub)
    return np.all(values >= lb - tol) and np.all(values <= ub + tol)


def check_abs_decomposition(a: np.ndarray, a_pos: np.ndarray, a_neg: np.ndarray, tol: float = 1e-9) -> bool:
    a = np.array(a)
    a_pos = np.array(a_pos)
    a_neg = np.array(a_neg)
    return np.allclose(a, a_pos - a_neg, atol=tol) and np.allclose(np.abs(a), a_pos + a_neg, atol=tol)


def check_nogood_dp_dn(sign: np.ndarray, dp: np.ndarray, dn: np.ndarray, eps_min: float, tol: float = 1e-9) -> bool:
    l1 = np.sum(dp + dn, axis=1)
    if not np.all(l1 >= eps_min - tol):
        return False
    if np.any(np.logical_and(dp > tol, dn > tol)):
        return False
    if not np.allclose(sign[dp > tol], 1.0, atol=tol):
        return False
    if not np.allclose(sign[dn > tol], 0.0, atol=tol):
        return False
    return True


def load_backend(name: str) -> MIPBackend:
    """Dynamically import and return the backend implementation for ``name``."""
    if name == "cplex":
        from .backends.cplex import CplexBackend

        return CplexBackend()
    if name == "scip":
        from .backends.scip import ScipBackend

        return ScipBackend()
    raise ValueError(f"unknown solver backend: {name}")

