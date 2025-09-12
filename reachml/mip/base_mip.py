"""Public BaseMIP facade class for building solver-agnostic MIPs.

Moved from ``reachml.mip.__init__`` to avoid circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from ..action_set import ActionSet
from ..utils import DEFAULT_SOLVER
from .backend_interface import MIPSettings, load_backend


@dataclass
class BaseMIP:
    """Base MIP builder used by the public API.

    This class builds the underlying solver model via a backend.
    """

    action_set: ActionSet
    x: np.ndarray
    print_flag: bool = False
    solver: str = DEFAULT_SOLVER
    settings: MIPSettings = MIPSettings()

    def __init__(self, action_set, x, print_flag: bool = False, solver: str = DEFAULT_SOLVER, **kwargs):
        assert isinstance(action_set, ActionSet)
        assert any(action_set.actionable)
        self.action_set = action_set

        assert isinstance(x, (list, np.ndarray))
        x = np.array(x, dtype=np.float64).flatten()
        assert len(x) == len(self.action_set)
        self.x = x

        self.actionable_indices: List[int] = list(range(len(self.x)))
        self.settings = MIPSettings(**(dict(MIPSettings().__dict__) | kwargs))
        self.print_flag = print_flag
        self.solver = solver

        # backend
        self._backend = load_backend(solver)

        # build, add constraints, configure
        mip, indices = self._backend.build_model(self.action_set, self.x, self.actionable_indices)
        mip, indices = self._backend.add_constraints(mip, indices, self.action_set, self.x)
        mip = self._backend.configure(mip, print_flag=self.print_flag)

        self.mip = mip
        self.indices = indices

    # Generic operations
    def add_linear_constraint(self, name: str, terms, sense: str, rhs: float) -> None:
        self._backend.add_linear_constraint(self.mip, self.indices, name, terms, sense, rhs)

    def delete_constraint(self, name: str) -> None:
        self._backend.delete_constraint(self.mip, self.indices, name)

    # Solve
    def solve_model(self):
        self._backend.solve(self.mip)

    @property
    def solution_status(self) -> str:
        return self._backend.solution_status(self.mip)

    # Solution state/info
    @property
    def solution_exists(self):
        return self._backend.has_solution(self.mip)

    @property
    def current_solution(self):
        vecs = self._backend.read_vectors(self.mip, self.indices, ["c"])
        c_vals = vecs.get("c")
        if c_vals is None:
            return None

        # For SCIP, mirror original rounding/snapping behavior to ensure parity
        if self.solver == "scip":
            snapped = list(c_vals.copy())
            tol = 1e-5
            for j in self.actionable_indices:
                feat = self.action_set[j]
                lb_c = float(self.indices.lb["c"][j])
                ub_c = float(self.indices.ub["c"][j])

                if feat.variable_type is int:
                    step = getattr(feat, "step_size", 1.0) or 1.0
                    final = np.round((self.x[j] + c_vals[j]) / step) * step
                    final = min(max(final, self.x[j] + lb_c), self.x[j] + ub_c)
                    snapped[j] = final - self.x[j]
                elif feat.variable_type is not float:
                    # Original behavior retained: treat as continuous, with warning bounds check
                    if c_vals[j] < lb_c - tol or c_vals[j] > ub_c + tol:
                        # optional: could log a warning; keep value as is rounded
                        pass
                    snapped[j] = round(c_vals[j])
            return snapped

        return c_vals

    @property
    def solution_info(self):
        return self._backend.stats(self.mip)
