"""Directional linkage constraint linking source to target changes by scale."""

from functools import reduce

import numpy as np

from .abstract import ActionabilityConstraint


class DirectionalLinkage(ActionabilityConstraint):
    """Constraint to link action in a source feature to changes in target feature.

    Given a set of features `names`:
    - names[0] is the "source feature"
    - names[1:] are the "target features"

    This constraint ensures that any action in a "source feature" will induce
    a S[k]-unit change in each target feature k.
    """

    def __init__(self, names, linkage_type="E", parent=None, scales=None, keep_bounds=False):
        """Initialize linkage with names, type, scales, and bounds policy.

        Args:
            names: [source, target1, ...].
            linkage_type: One of {"E", "G", "L"} for equality/ineq sense.
            parent: Optional ActionSet.
            scales: Relative scale per feature; non-zero, normalized by source.
            keep_bounds: If True, keep target bounds unchanged; else widen by linkage.
        """
        assert len(names) >= 2
        scales = np.ones(len(names)) if scales is None else np.array(scales).flatten()
        assert len(scales) == len(names)
        assert np.count_nonzero(scales) == len(scales)
        self._parameters = ("source", "targets", "scales", "keep_bounds")
        super().__init__(names=names, parent=parent)
        self._source = self.names[0]
        self._targets = self.names[1:]
        self._scales = np.array(scales[1:]) / float(scales[0])
        self._linkage_type = linkage_type
        self._keep_bounds = keep_bounds

    @property
    def source(self):
        """Source feature name."""
        return self._source

    @property
    def targets(self):
        """List of target feature names."""
        return self._targets

    @property
    def scales(self):
        """Scale factors applied to each target relative to the source."""
        return self._scales

    @property
    def linkage_type(self):
        """Linkage sense: "E" (==), "G" (>=), or "L" (<=)."""
        return self._linkage_type

    @property
    def keep_bounds(self):
        """Whether to preserve original bounds on target changes."""
        return self._keep_bounds

    def check_compatibility(self, action_set):
        """Return True if the constraint is compatible with the ActionSet.

        Called when attaching this constraint via `ActionSet.constraints.add`.
        """
        # check for circular dependencies
        L = action_set.constraints.linkage_matrix
        source_index = action_set.get_feature_indices(self.source)
        target_indices = action_set.get_feature_indices(self.targets)
        for k, target in zip(target_indices, self.targets, strict=False):
            assert L[k, source_index] == 0, (
                f"Circular Dependency: "
                f"Cannot link actions from {self.source}->{target}."
                f"action_set already contains link from {target}->{self.source}"
            )

        # check that source is actionable
        assert action_set[self.source].actionable

        # check that scales are compatible
        target_actions = [a for a in action_set if a.name in self.targets]
        step_compatability = [
            np.mod(scale, a.step_size) == 0 if a.discrete else True
            for a, scale in zip(target_actions, self.scales, strict=False)
        ]
        assert all(step_compatability)
        return True

    def __str__(self):
        """Human-readable description of the linkage relationship."""
        if len(self._targets) == 1:
            units = (
                f"Each unit change in {self._source} leads to a "
                f"{self._scales[0]:1.2f}-unit change in {self._targets[0]}"
            )
        else:
            units = f"Each unit change in {self._source} leads to:" + ", ".join(
                [
                    f"{s:1.2f}-unit change in {n}"
                    for n, s in zip(self._targets, self._scales, strict=False)
                ]
            )
        s = f"Actions on {self._source} will induce actions on {self._targets}. "
        s += f"{units}"
        return s

    def check_feasibility(self, x):
        """Return True if `x` is realizable under these constraints.

        Args:
            x: 1D feature vector or 2D feature matrix.
        """
        x = np.array(x)
        # if matrix then apply this function again for each point in the matrix
        if x.ndim == 2 and x.shape[0] > 1:
            return np.apply_over_axes(self.check_feasibility, a=x, axis=0)
        v = x[self.indices]
        out = self.check_feature_vector(v)
        return out

    def adapt(self, x):
        """Adapt constraint parameters for feature vector `x`."""
        assert self.check_feasibility(x), f"{self.__class__} is infeasible at x = {str(x)}"
        j = self.indices[0]  # j == source index
        aj_max = self.parent[j].get_action_bound(x[j], bound_type="ub")
        aj_min = self.parent[j].get_action_bound(x[j], bound_type="lb")
        b_ub = np.maximum(self.scales * aj_max, self.scales * aj_min)
        b_lb = np.minimum(self.scales * aj_max, self.scales * aj_min)
        return b_ub, b_lb

    def add_to_cpx(self, cpx, indices, x):
        """Add constraints and variables for this linkage to a CPLEX model."""
        from cplex import Cplex, SparsePair
        from ..mip.backends.cplex_utils import combine, get_cpx_variable_args

        assert isinstance(cpx, Cplex)
        vars = cpx.variables
        cons = cpx.linear_constraints
        b_ub, b_lb = self.adapt(x)

        # get indices of source and targets
        j = self.indices[0]  # j == source index
        target_indices = self.indices[1:]

        # define variables to capture linkage effects from source
        # b[j, k]
        b = [f"b[{j},{k}]" for k in target_indices]
        variable_args = {"b": get_cpx_variable_args(obj=0.0, name=b, lb=b_lb, ub=b_ub, vtype="C")}
        vars.add(**reduce(combine, variable_args.values()))
        indices.append_variables(variable_args)

        # add constraint to set
        # b[j,k] = scale[k]*a[j]
        for bjk, sk in zip(b, self.scales, strict=False):
            cons.add(
                names=[f"set_{bjk}"],
                lin_expr=[SparsePair(ind=[bjk, f"a[{j}]"], val=[1.0, -sk])],
                senses=self.linkage_type,
                rhs=[0.0],
            )

        # add linkage effect to aggregate change variables for targets
        # c[k] = a[k] + b[j][k]
        # c[k] - a[k] -b[j][k] = 0
        c = [f"c[{k}]" for k in target_indices]
        for ck, bjk in zip(c, b, strict=False):
            cons.set_linear_components(f"set_{ck}", [[bjk], [-1.0]])

        # update bounds on aggregate change variables for targets
        if not self.keep_bounds:
            # update upper bound on c[k] for target
            c_ub = vars.get_upper_bounds(c) + b_ub
            vars.set_upper_bounds([(ck, uk) for ck, uk in zip(c, c_ub, strict=False)])

            # update lower bound on c[k] for target
            c_lb = vars.get_lower_bounds(c) + b_lb
            vars.set_lower_bounds([(ck, lk) for ck, lk in zip(c, c_lb, strict=False)])

            # update indices
            indices.ub["c"] = vars.get_upper_bounds(indices.names["c"])
            indices.lb["c"] = vars.get_lower_bounds(indices.names["c"])

        return cpx, indices

    def add_to_scip(self, scip, indices, x):
        """Add constraints and variables for this linkage to a SCIP model."""
        from pyscipopt import Model
        from ..mip.backends.scip_utils import combine

        assert isinstance(scip, Model)
        b_ub, b_lb = self.adapt(x)

        # get indices of source and targets
        j = self.indices[0]  # j == source index
        target_indices = self.indices[1:]

        # define variables to capture linkage effects from source
        # b[j, k]
        b = [f"b[{j},{k}]" for k in target_indices]
        b_dict = var_args = {
            "names": b,
            "obj": [0.0] * len(b),
            "lb": list(b_lb),
            "ub": list(b_ub),
            "types": ["C"] * len(b),
        }
        variable_args = {"b": b_dict}
        var_args = reduce(combine, variable_args.values())

        # Re-group the variables to be in the correct format
        reduced_args = {"b": var_args}
        indices.append_variables(scip, reduced_args)

        b_vars = []
        for i in range(len(variable_args["b"]["names"])):
            scip.addVar(
                name=variable_args["b"]["names"][i],
                obj=variable_args["b"]["obj"][i],
                lb=variable_args["b"]["lb"][i],
                ub=variable_args["b"]["ub"][i],
                vtype=variable_args["b"]["types"][i],
            )
            b_vars.append(indices.get_var(scip, variable_args["b"]["names"][i]))

        a_j = indices.get_var(scip, f"a[{j}]")
        # add constraint
        # b[j,k] = scale[k]*a[j]
        for bjk, sk in zip(b_vars, self.scales, strict=False):
            scip.addCons(bjk - float(sk) * a_j == 0.0, name=f"set_{bjk}]")

        # add linkage effect to aggregate change variables for targets
        # c[k] = a[k] + b[j][k]
        # c[k] - a[k] -b[j][k] = 0
        # c[k] = a[k] + b[j,k]
        for k, bjk in zip(target_indices, b_vars, strict=False):
            old = indices.get_constraint(f"set_c_[{k}]")
            scip.delCons(old)

            c_k = indices.get_var(scip, f"c[{k}]")
            a_k = indices.get_var(scip, f"a[{k}]")
            new = scip.addCons(c_k - a_k - bjk == 0.0, name=f"set_c_[{k}]")
            indices.replace_constraint(f"set_c_[{k}]", new)

        if not self.keep_bounds:
            # update upper bound on c[k] for target
            for t_idx, ub_add, lb_add in zip(target_indices, b_ub, b_lb, strict=False):
                c_name = f"c[{t_idx}]"
                c_var = indices.get_var(scip, c_name)

                # read current bounds from SCIP, then widen
                cur_ub = indices.__dict__["ub"]["c"][t_idx]
                cur_lb = indices.__dict__["lb"]["c"][t_idx]
                new_ub = float(cur_ub) + float(ub_add)
                new_lb = float(cur_lb) + float(lb_add)

                scip.chgVarUb(c_var, new_ub)
                scip.chgVarLb(c_var, new_lb)

                # keep indices metadata in sync (find the exact slot for c_name)
                c_pos = indices.names["c"].index(c_name)
                indices.ub["c"][c_pos] = new_ub
                indices.lb["c"][c_pos] = new_lb

        return scip, indices
