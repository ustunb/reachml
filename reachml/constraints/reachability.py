"""Generalized reachability constraints for subsets of features."""

from functools import reduce

import numpy as np

from .abstract import ActionabilityConstraint


class ReachabilityConstraint(ActionabilityConstraint):
    """Generalized reachability constraints.

    - Restrict actions over a subset of features to a finite set of values.
    - The reachability matrix indicates if we can reach value k from value j.
    """

    def __init__(self, names, values, parent=None, reachability=None):
        """Initialize with feature names, allowed values, and reachability.

        Args:
            names: Names of features.
            values: 2D array of distinct allowed values for the named features.
            parent: Optional ActionSet.
            reachability: Optional binary n×n matrix; R[j,k] = 1 if k→j.
        """
        # sport
        # sort_idx = np.argsort(names)
        # names = [names[i] for i in sort_idx]
        # values = values[:, sort_idx]
        n, d = values.shape
        assert n >= 1, "values should have at least 2 rows"
        assert len(names) == d, f"values should have len(names) = {len(names)} dimensions"
        assert len(np.unique(values, axis=0)) == len(values), "values should be unique"
        assert np.isfinite(values).all(), "values should be finite"
        if reachability is None:
            reachability = np.ones((n, n))  # assume all points are reachable
        else:
            reachability = np.array(reachability)
            assert n == reachability.shape[0]
            assert self.check_reachability_matrix(reachability), "invalid reachability matrix"
        # todo: sort by name
        # names, values, reachability = self.sort_parameters(names, values, reachability)
        self._values = values
        self._reachability = reachability
        # self._parameters = ('values', 'reachability')
        self._parameters = ("names", "values", "reachability")
        super().__init__(names, parent)

    @property
    def values(self):
        """Allowed values for the feature subset as an array."""
        return self._values

    @property
    def reachability(self):
        """Binary matrix encoding which values are mutually reachable."""
        return self._reachability

    def __str__(self):
        """Human-readable description of allowed values and reachability."""
        name_list = ", ".join(f"`{n}`" for n in self.names)
        s = f"The values of [{name_list}] must belong to one of {len(self.values)} values"
        if not np.all(self._reachability):
            s = f"{s} with custom reachability conditions."
        return s

    @staticmethod
    def sort_parameters(names, values, reachability):
        """Sort names, values and reachability by alphabetical order."""
        sort_idx = np.argsort(names)
        names = [names[i] for i in sort_idx]
        new_values = values[:, sort_idx]
        n = values.shape[0]
        new_index = {
            i: np.flatnonzero(np.all(new_values == v, axis=1))[0] for i, v in enumerate(values)
        }
        new_reachability = np.zeros_like(reachability)
        for i in range(n):
            for j in range(n):
                new_reachability[new_index[i], new_index[j]] = reachability[i, j]
        assert np.sum(new_reachability[:]) == np.sum(new_reachability[:])
        return names, new_values, new_reachability

    @staticmethod
    def check_reachability_matrix(reachability):
        """Minimal sanity checks for a reachability matrix."""
        out = (
            reachability.ndim == 2
            and reachability.shape[0] == reachability.shape[1]
            and np.all(np.diagonal(reachability) == 1)
            and np.isin(reachability, (0, 1)).all()
        )
        return out

    def check_compatibility(self, action_set):
        """Return True if the constraint is compatible with the ActionSet."""
        ub = np.max(self.values, axis=0)
        lb = np.min(self.values, axis=0)
        assert np.less_equal(ub, action_set[self.names].ub).all()
        assert np.greater_equal(lb, action_set[self.names].lb).all()
        return True

    def check_feasibility(self, x):
        """Return True if `x` is feasible with respect to allowed values."""
        x = np.array(x)
        if (
            x.ndim == 2 and x.shape[0] > 1
        ):  # if matrix then apply this function again for each point in the matrix
            return np.apply_over_axes(self.check_feasibility, a=x, axis=0)
        v = x[self.indices]
        out = np.all(self._values == v, axis=1).any()  # checks finite-ness
        return out

    def adapt(self, x):
        """Adapt constraint parameters for point `x`."""
        x = np.array(x).flatten().astype(float)
        assert self.check_feasibility(x), f"{self.__class__} is infeasible at x = {str(x)}"
        v = x[self.indices]
        idx = np.flatnonzero(np.all(self._values == v, axis=1))[0]
        reachable_points = self._reachability[idx]
        a_null = np.zeros_like(v)  # null_action
        action_values = [
            p - v if reachable else a_null
            for p, reachable in zip(self._values, reachable_points, strict=False)
        ]
        return reachable_points, action_values

    def add_to_cpx(self, cpx, indices, x):
        """Add reachability constraints to a CPLEX model."""
        from cplex import Cplex, SparsePair
        from ..mip.backends.cplex_utils import combine, get_cpx_variable_args
        assert isinstance(cpx, Cplex)
        reachable_points, action_values = self.adapt(x)
        n_points = len(reachable_points)
        point_indices = np.arange(n_points)  # indices of points from 1...n_points
        reachability_indicators = [f"r[{self.id, k}]" for k in point_indices]

        # add variables/constraints
        vars = cpx.variables
        cons = cpx.linear_constraints

        # Indicator r[id, k] = 1 if we move to values[k, :] for this constraint
        variable_args = {
            "v": get_cpx_variable_args(
                name=reachability_indicators, obj=0.0, lb=0.0, ub=reachable_points, vtype="B"
            )
        }
        vars.add(**reduce(combine, variable_args.values()))

        # assign to exactly one reachable point:
        # sum r[const_id, k] = 1
        cons.add(
            names=[f"reachability_{self.id}_limit"],
            lin_expr=[SparsePair(ind=reachability_indicators, val=[1.0] * n_points)],
            senses="E",
            rhs=[1.0],
        )

        # assign to point k, then set a[j] = action_value[k][j]
        # a[j] := sum(r[const_id, k] * action_value[k][j])
        for i, j in enumerate(self.indices):
            ind = [f"a[{j}]"] + reachability_indicators
            val = [-1.0] + [
                float(ak[i] * ek) for ak, ek in zip(action_values, reachable_points, strict=False)
            ]
            cons.add(
                names=[f"reachability_{self.id}_assignment_a[{j}]"],
                lin_expr=[SparsePair(ind=ind, val=val)],
                senses="E",
                rhs=[0.0],
            )

        # Optional: declare a type-1 SOS with weights for each reachable point.

        # append indices
        indices.append_variables(variable_args)
        indices.params.update({"Ak": [action_values], "Ek": [reachable_points]})
        return cpx, indices

    def add_to_scip(self, scip, indices, x):
        """Add reachability constraints to a SCIP model."""
        from pyscipopt import Model, quicksum
        from ..mip.backends.scip_utils import combine
        assert isinstance(scip, Model)

        # Compute reachable mask and candidate action values
        reachable_points, action_values = self.adapt(
            x
        )  # reachable_points: list[0/1], action_values: list[list[float]]
        n_points = len(reachable_points)
        point_indices = list(range(n_points))

        # Indicator variable names r[(id, k)]
        reachability_indicators = [f"r[{(self.id, k)}]" for k in point_indices]

        # Add indicator vars using the indices helper (adds to model and records).
        # Use reachability mask as upper bounds (ub=0 fixes var to 0).
        variable_args = {
            "r": {
                "names": reachability_indicators,
                "obj": [0.0] * n_points,
                "lb": [0.0] * n_points,
                "ub": [float(e) for e in reachable_points],
                "types": ["I"] * n_points,
            }
        }
        var_args = reduce(combine, variable_args.values())
        # This line ensures that the reduced variable group is in the correct form
        un_reduced = {"r": var_args}
        indices.append_variables(scip, un_reduced)

        r_vars = []
        for i in range(len(variable_args["r"]["names"])):
            var = scip.addVar(
                name=variable_args["r"]["names"][i],
                obj=variable_args["r"]["obj"][i],
                lb=variable_args["r"]["lb"][i],
                ub=variable_args["r"]["ub"][i],
                vtype=variable_args["r"]["types"][i],
            )
            r_vars.append(var)

        # --- Exactly one point selected: sum_k r_k = 1 ---
        scip.addCons(quicksum(r_vars) == 1, name=f"reachability_{self.id}_limit")

        for i, j in enumerate(self.indices):
            a_var = indices.get_var(scip, f"c[{j}]")
            # Precompute coefficients (mask by reachability)
            coeffs = [
                float(ak[i]) * float(ek)
                for ak, ek in zip(action_values, reachable_points, strict=False)
            ]
            scip.addCons(
                a_var == quicksum(coeff * r for coeff, r in zip(coeffs, r_vars, strict=False)),
                name=f"reachability_{self.id}_assignment_c[{j}]",
            )

        indices.params.update({"Ck": [action_values], "Ek": [reachable_points]})

        return scip, indices
