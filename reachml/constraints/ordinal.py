"""Ordinal encoding constraint ensuring valid dummy transitions."""

from itertools import product

import numpy as np

from ..utils import parse_attribute_name
from .reachability import ReachabilityConstraint


class OrdinalEncoding(ReachabilityConstraint):
    """Ensure actions preserve one-hot encoding of an ordinal attribute.

    attribute. This constraint should be specified over a subset of boolean
    features that are produced by a one-hot encoding of an ordinal attribute Z.

    Given an ordinal attribute Z with `m` levels: `z[0], z[1], .. z[m-1]`,
    the boolean features - i.e., dummies - have the form:

      x[0] := 1[Z = 0]
      x[1] := 1[Z = 1]
      ...
      x[m-1] := 1[Z = z[m-1]]

    Here z[0] ≤ z[1] ≤ ... z[m-1] denote the levels of Z in increasing order,
    and x[k] := 1[Z = k] is a dummy variable set to 1 if and only if Z == k.

    todo: Example:
    """

    def __init__(
        self, names, parent=None, exhaustive=True, step_direction=0, drop_invalid_values=True
    ):
        """Initialize ordinal encoding with names and options.

        Args:
            names: Ordinal dummy feature names.
            parent: Optional ActionSet.
            exhaustive: If True, exactly one dummy must be on; else at most one.
            step_direction: 0 free, 1 increasing only, -1 decreasing only.
            drop_invalid_values: If True, discard infeasible dummy patterns.
        """
        assert len(names) >= 2, "constraint only applies to 2 or more features"
        assert isinstance(exhaustive, bool)
        assert step_direction in (0, 1, -1)
        self._limit = 1
        self._exhaustive = bool(exhaustive)
        self._step_direction = step_direction

        # create values
        values = np.array(list(product([1, 0], repeat=len(names))))
        if drop_invalid_values:
            keep_idx = [self.check_encoding(v) for v in values]
            values = values[keep_idx, :]
            values = values[np.lexsort(values.T, axis=0), :]

        n = values.shape[0]
        reachability = np.eye(n)
        for i, p in enumerate(values):
            for j, q in enumerate(values):
                if i != j:
                    out = self.check_encoding(p) and self.check_encoding(q)
                    if step_direction > 0:
                        out = out and (i < j)
                    elif step_direction < 0:
                        out = out and (j < i)
                    reachability[i, j] = out

        super().__init__(names=names, values=values, reachability=reachability, parent=parent)
        self._parameters = self._parameters + ("exhaustive", "step_direction")

    def check_encoding(self, x):
        """Return True if `x` obeys the one-hot cardinality rule."""
        if self.exhaustive:
            out = np.sum(x) == self.limit
        else:
            out = np.less_equal(np.sum(x), self.limit)
        return out

    def check_feasibility(self, x):
        """Return True if `x` is feasible under encoding and base constraints."""
        out = self.check_encoding(x) and super().check_feasibility(x)
        return out

    @property
    def limit(self):
        """Cardinality requirement (1 for standard one-hot)."""
        return self._limit

    @property
    def exhaustive(self):
        """Whether exactly one dummy must be active (vs. at most one)."""
        return self._exhaustive

    @property
    def step_direction(self):
        """Allowed transition direction among ordinal levels."""
        return self._step_direction

    def __str__(self):
        """Human-readable description of the ordinal constraint."""
        name_list = ", ".join(f"`{n}`" for n in self.names)
        attribute_name = parse_attribute_name(self.names, default_name="ordinal attribute")
        s = f"Actions on [{name_list}] must preserve one-hot encoding of {attribute_name}."
        f"{'Exactly' if self.exhaustive else 'At most'} {self.limit} of [{name_list}] can be TRUE."
        if self.step_direction > 0:
            s = (
                f"{s}, which can only increase."
                f"Actions can only turn on higher-level dummies that are off"
                f", where {self.names[0]} is the lowest-level dummy "
                f"and {self.names[-1]} is the highest-level-dummy."
            )
        elif self.step_direction < 0:
            s = (
                f"{s}, which can only decrease."
                f"Actions can only turn off higher-level dummies that are on"
                f", where {self.names[0]} is the lowest-level dummy "
                f"and {self.names[-1]} is the highest-level-dummy."
            )
        return s
