"""One-hot encoding constraint over Boolean dummy features."""

import numpy as np

from ..utils import parse_attribute_name
from .abstract import ActionabilityConstraint


class OneHotEncoding(ActionabilityConstraint):
    """Ensure actions preserve one-hot encoding of a categorical attribute.

    attribute. This constraint should be specified over a collection of Boolean
    features produced through a one-hot encoding of an categorical attribute Z.

    Given an categorical attribute Z with `m` categories: `z[0], z[1], .. z[m-1]`,
    the boolean features - i.e., dummies - have the form:

      x[0] := 1[Z = z[0]]
      x[1] := 1[Z = z[1]]
      ...
      x[m-1] := 1[Z = z[m-1]]

    Here z[0], ... z[m-1] denote the different values that Z can take
    and x[k] := 1[Z = k] is a dummy variable set to 1 if and only if Z == k.

    todo: Example:

    """

    VALID_LIMIT_TYPES = ("equal", "max")

    def __init__(self, names, parent=None, limit=1, limit_type="equal"):
        """Initialize one-hot constraint with names and a cardinality limit.

        Args:
            names: List of Boolean dummy feature names for one categorical field.
            parent: Optional ActionSet.
            limit: Integer cardinality constraint (exact or max).
            limit_type: Either "equal" or "max".
        """
        assert isinstance(limit, int)
        assert 0 <= limit <= len(names), f"limit must be between 0 to {len(names)}"
        assert limit_type in OneHotEncoding.VALID_LIMIT_TYPES
        self._limit = limit
        self._limit_type = limit_type
        self._parameters = ("limit", "limit_type")
        super().__init__(names=names, parent=parent)

    def check_compatibility(self, action_set):
        """Return True if the constraint is compatible with the ActionSet.

        Called when attaching this constraint via `ActionSet.constraints.add`.
        """
        assert all(vtype is bool for vtype in action_set[self.names].variable_type), (
            "features must be bool"
        )
        return True

    @property
    def limit(self):
        """Required number of active dummies (exact or upper bound)."""
        return self._limit

    @property
    def limit_type(self):
        """Either "equal" or "max" (at most)."""
        return self._limit_type

    def __str__(self):
        """Human-readable message describing the one-hot constraint."""
        name_list = ", ".join(f"`{n}`" for n in self.names)
        attribute_name = parse_attribute_name(self.names, default_name="categorical attribute")
        s = f"Actions on [{name_list}] must preserve one-hot encoding of {attribute_name}."
        if self.limit_type == "equal":
            s = f"{s}. Exactly {self.limit} of [{name_list}] must be TRUE"
        elif self.limit_type == "max":
            s = f"{s}. At most {self.limit} of [{name_list}] must be TRUE"
        return s

    def check_feasibility(self, x):
        """Return True if `x` satisfies dummy domain and cardinality constraints."""
        x = np.array(x)
        # if matrix then apply this function again for each point in the matrix
        if x.ndim == 2 and x.shape[0] > 1:
            return np.apply_over_axes(self.check_feasibility, a=x, axis=0)
        v = x[self.indices]
        out = self.check_feature_vector(v) and np.isin(v, (0, 1)).all()
        if out:
            if self.limit_type == "max":
                out = np.less_equal(np.sum(v), self.limit)
            elif self.limit_type == "equal":
                out = np.equal(np.sum(v), self.limit)
        return out

    def adapt(self, x):
        """Adapt to `x` (passes through selected indices for convenience)."""
        assert self.check_feasibility(x), f"{self.__class__} is infeasible at x = {str(x)}"
        return x[self.indices]

    def add_to_cpx(self, cpx, indices, x):
        """Add one-hot cardinality constraint to CPLEX model."""
        from cplex import Cplex, SparsePair
        assert isinstance(cpx, Cplex)
        cons = cpx.linear_constraints
        x_values = self.adapt(x)
        a = [f"a[{idx}]" for idx in self.indices]
        # todo: pull constraint_id from object
        cons.add(
            names=[f"onehot_{self.id}"],
            # todo, name this using constraint id
            lin_expr=[SparsePair(ind=a, val=[1.0] * len(x_values))],
            senses="E" if self._limit_type == "equal" else "L",
            rhs=[float(self._limit - np.sum(x_values))],
        )

        return cpx, indices

    def add_to_scip(self, scip, indices, x):
        """Add one-hot cardinality constraint to a SCIP model."""
        from pyscipopt import Model, quicksum
        assert isinstance(scip, Model)

        x_values = self.adapt(x)
        # K = how many net +1s you need to hit the target one-hot count
        rhs = float(self._limit - (np.sum(x_values)))
        cname_base = f"onehot_{self.id}"

        a_vars = []
        print(f"vars: {scip.getVars()}")

        for idx in self.indices:
            a_var = indices.get_var(scip, f"a[{idx}]")
            print(f"a_var: {a_var}")
            a_vars.append(a_var)

        scip.infinity()

        if self._limit_type == "equal":
            scip.addCons(quicksum(a_vars) == rhs, name=f"{cname_base}_cons_eq")
        else:
            scip.addCons(quicksum(a_vars) <= rhs, name=f"{cname_base}_cons_leq")

        return scip, indices
