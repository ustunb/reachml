"""EnumeratorMIP helper built on top of BaseMIP to iterate distinct solutions.

Moved from ``reachml.mip.__init__`` to avoid circular imports.
"""

from __future__ import annotations

import numpy as np

from ..utils import DEFAULT_SOLVER
from .base_mip import BaseMIP


class EnumeratorMIP(BaseMIP):
    """Extend BaseMIP with helpers to enumerate distinct solutions."""

    def __init__(self, action_set, x, print_flag: bool = False, solver: str = DEFAULT_SOLVER, **kwargs):
        super().__init__(action_set, x, print_flag=print_flag, solver=solver, **kwargs)
        self.n_sols = 0

    # Enumeration helpers
    def remove_actions(self, actions):
        assert isinstance(actions, list)
        self.mip, self.indices, added = self._backend.add_nogood(
            self.mip, self.indices, actions, self.actionable_indices, self.settings
        )
        # after adding nogood constraints, find next solution
        self._backend.solve(self.mip)
        self.n_sols += added

        return True

    def check_solution(self):
        # Perform basic sanity checks common across backends
        if not self.solution_exists:
            raise AssertionError("no feasible solution found")

        # Bounds and decomposition checks (when available)
        vecs = self._backend.read_vectors(self.mip, self.indices, ["a", "a_pos", "a_neg", "c"])
        a = vecs.get("a")
        a_pos = vecs.get("a_pos")
        a_neg = vecs.get("a_neg")
        c = vecs.get("c")
        if a is not None and a_pos is not None and a_neg is not None:
            assert np.allclose(a, a_pos - a_neg)
            assert np.allclose(np.abs(a), a_pos + a_neg)
        return True

