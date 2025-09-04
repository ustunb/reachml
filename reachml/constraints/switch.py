"""Mutability switch constraint controlling target changes based on a switch."""

from __future__ import annotations

from functools import reduce

import numpy as np

from .abstract import ActionabilityConstraint


class MutabilitySwitch(ActionabilityConstraint):
    """If the switch is on, targets cannot change; if off, targets may change.

    Example:
    - If Balance_eq_0 = 1 → [Balance_geq_20, Balance_geq_50, Balance_geq_90] are off
    - If Balance_eq_0 = 0 → those targets can change (or must change if forced)
    """

    def __init__(self, switch, targets, on_value=1, force_change_when_off=True, parent=None):
        """Initialize with a switch feature and target feature names."""
        assert isinstance(switch, str)
        if isinstance(targets, str):
            targets = [targets]
        assert switch not in targets
        assert np.isin(on_value, (0, 1))
        self._switch = str(switch)
        self._targets = targets
        self._on_value = bool(on_value)
        self._force_change_when_on = bool(force_change_when_off)
        self._parameters = ("switch", "targets", "on_value", "force_change_when_off")
        super().__init__(names=[switch] + targets, parent=parent)

    @property
    def switch(self):
        """Switch feature name."""
        return self._switch

    @property
    def targets(self):
        """List of target feature names."""
        return self._targets

    @property
    def on_value(self):
        """Value (0/1) that activates the switch."""
        return self._on_value

    @property
    def force_change_when_off(self):
        """Whether targets must change when the switch is off."""
        return self._force_change_when_on

    def __str__(self):
        """Human-readable description of the switch behavior."""
        target_names = ", ".join(f"`{n}`" for n in self._targets)
        s = f"If {self.switch}={self.on_value} then {target_names} cannot change."
        if self.force_change_when_off:
            s += f"\nIf {self.switch}={not self.on_value} then {target_names} must change."
        return s

    def check_compatibility(self, action_set):
        """Return True if the constraint is compatible with the ActionSet."""
        assert self.switch in action_set.names
        assert action_set[self.switch].actionable, (
            f"switch feature `{self.switch}` must be actionable"
        )
        assert action_set[self.switch].variable_type is bool, (
            f"switch feature `{self.switch}` must be boolean"
        )
        for n in self.targets:
            assert n in action_set.names, f"action set does not contain target feature {n}"
            assert action_set[n].actionable, f"target feature {n} must be actionable"
        return True

    def check_feasibility(self, x):
        """Return True if `x` satisfies the switch/target change conditions."""
        x = np.array(x)
        # if matrix then apply this function again for each point in the matrix
        if x.ndim == 2 and x.shape[0] > 1:
            return np.apply_over_axes(self.check_feasibility, a=x, axis=0)
        switch_idx = self.indices[0]
        target_idx = self.indices[1:]
        v = x[self.indices]
        out = self.check_feature_vector(v)
        if x[switch_idx] == self.on_value:
            out &= np.all(x[target_idx] == 0.0)
        elif self.force_change_when_off:
            out &= np.all(x[target_idx] != 0.0)
        return out

    def adapt(self, x):
        """Adapt constraint bounds and switch state for feature vector `x`."""
        assert self.check_feasibility(x), f"{self.__class__} is infeasible at x = {str(x)}"
        x_switch = x[self.indices[0]]
        a_pos_max = np.abs(self.parent.get_bounds(x, bound_type="ub")).astype(float)
        a_neg_max = np.abs(self.parent.get_bounds(x, bound_type="lb")).astype(float)
        print(f"A_pos_max: {a_pos_max}")
        print(f"A_neg_max: {a_neg_max}")
        return x_switch, a_pos_max, a_neg_max

    def add_to_cpx(self, cpx, indices, x):
        """Add switch constraints to a CPLEX model."""
        from cplex import Cplex, SparsePair
        from ..mip.backends.cplex_utils import combine, get_cpx_variable_args
        assert isinstance(cpx, Cplex)
        vars = cpx.variables
        cons = cpx.linear_constraints
        x_switch, A_pos_max, A_neg_max = self.adapt(x)
        switch_idx = self.parent.get_feature_indices(self.switch)
        target_idx = self.parent.get_feature_indices(self.targets)
        a_switch = f"a[{switch_idx}]"

        # add switching variable to CPLEX
        w = f"w[{self.id}]"
        variable_args = {"w": get_cpx_variable_args(obj=0.0, name=w, lb=0.0, ub=1.0, vtype="B")}
        vars.add(**reduce(combine, variable_args.values()))

        # add constraint to set switching variable
        # x'[j] = self.on_value => w[j] = 1
        if self.on_value == 1:
            # w[j] = x'[j] =
            # w[j] = x[j] + a[j]
            # -> w[j] - a[j] = x[j]
            cons.add(
                names=[f"set_{w}"],
                lin_expr=[SparsePair(ind=[w, a_switch], val=[1.0, -1.0])],
                senses="E",
                rhs=[x_switch],
            )
        else:
            # w[j] = 1-x'[j]
            # w[j] = 1-(x[j] + a[j])
            # -> w[j] + a[j] = 1 - x[j]
            cons.add(
                names=[f"set_{w}"],
                lin_expr=[SparsePair(ind=[w, a_switch], val=[1.0, 1.0])],
                senses="E",
                rhs=[1.0 - x_switch],
            )

        # For each, we need to add
        [f"a[{k}]" for k in target_idx]
        a_pos_targets = [f"a[{k}]_pos" for k in target_idx]
        a_neg_targets = [f"a[{k}]_neg" for k in target_idx]
        for k in target_idx:
            a_pos_k = f"a[{k}]_pos"
            a_neg_k = f"a[{k}]_neg"
            A_pos = A_pos_max[k]
            A_neg = A_neg_max[k]
            # cons.add(names = [f"switch_{self.id}_for_target_{k}_up"],
            #          lin_expr = [SparsePair(ind = [a_k, w], val = [1.0, A_pos])],
            #          senses = "L",
            #          rhs = [A_pos])
            # cons.add(names = [f"switch_{self.id}_for_target_{k}_dn"],
            #          lin_expr = [SparsePair(ind = [a_k, w], val = [1.0, -A_neg])],
            #          senses = "G",
            #          rhs = [-A_neg])
            cons.add(
                names=[f"switch_{self.id}_for_target_{k}_pos"],
                lin_expr=[SparsePair(ind=[a_pos_k, w], val=[1.0, A_pos])],
                senses="L",
                rhs=[A_pos],
            )
            cons.add(
                names=[f"switch_{self.id}_for_target_{k}_neg"],
                lin_expr=[SparsePair(ind=[a_neg_k, w], val=[1.0, A_neg])],
                senses="L",
                rhs=[A_neg],
            )

        if self.force_change_when_off:
            n_targets = len(self.targets)
            min_step_size = np.min([a.step_size for a in self.parent if a.name in self.targets])
            min_step_size = 0.99 * min_step_size
            print(f"forcing constraint - min_step_size: {min_step_size}")
            cons.add(
                names=[f"switch_{self.id}_force_change_when_off"],
                lin_expr=[
                    SparsePair(
                        ind=a_pos_targets + a_neg_targets + [w],
                        val=np.ones(2 * n_targets).tolist() + [min_step_size],
                    )
                ],
                senses="G",
                rhs=[min_step_size],
            )

        indices.append_variables(variable_args)
        return cpx, indices

    def add_to_scip(self, scip: Model, indices, x):
        """Add switch constraints to a SCIP model."""
        from pyscipopt import Model, quicksum
        from ..mip.backends.scip_utils import combine
        assert isinstance(scip, Model)

        # Adapt and resolve indices/vars
        x_switch, A_pos_max, A_neg_max = self.adapt(x)
        switch_idx = self.parent.get_feature_indices(self.switch)
        if isinstance(switch_idx, (list, tuple)):
            switch_idx = switch_idx[0]
        target_idx = self.parent.get_feature_indices(self.targets)

        a_switch = indices.get_var(scip, f"a[{switch_idx}]")

        # Add switching variable (binary)
        w_name = f"w[{self.id}]"
        w = scip.addVar(name=w_name, vtype="B", lb=0.0, ub=1.0, obj=0.0)
        variables = {"names": [w_name], "types": ["B"], "lb": [0.0], "ub": [1.0], "obj": [0.0]}
        variable_args = {"w": variables}
        reduced_args = reduce(combine, variable_args.values())
        reduced_args = {"w": reduced_args}
        indices.append_variables(scip, reduced_args)

        # Set switching variable:
        # if on_value == 1:  w - a_switch = x_switch
        # else:              w + a_switch = 1 - x_switch
        if self.on_value == 1:
            scip.addCons(w - a_switch == x_switch, name=f"set_{w_name}")
        else:
            scip.addCons(w + a_switch == 1.0 - x_switch, name=f"set_{w_name}")

        # Per-target switching bounds:
        # a_pos_k + A_pos * w <= A_pos  ==> a_pos_k <= A_pos * (1 - w)
        # a_neg_k + A_neg * w <= A_neg  ==> a_neg_k <= A_neg * (1 - w)
        a_pos_vars, a_neg_vars = [], []
        for k in target_idx:
            a_pos_k = indices.get_var(scip, f"a[{k}]_pos")
            a_neg_k = indices.get_var(scip, f"a[{k}]_neg")
            s_k = indices.get_var(scip, f"a[{k}]_sign")
            A_pos = A_pos_max[k]
            A_neg = A_neg_max[k]

            scip.addCons(a_pos_k + A_pos * w <= A_pos, name=f"switch_{self.id}_for_target_{k}_pos")
            scip.addCons(a_neg_k + A_neg * w <= A_neg, name=f"switch_{self.id}_for_target_{k}_neg")

            scip.addCons(a_pos_k <= A_pos * s_k, name=f"switch_{self.id}_sign_pos_{k}")
            scip.addCons(a_neg_k <= A_neg * (1 - s_k), name=f"switch_{self.id}_sign_neg_{k}")

            a_pos_vars.append(a_pos_k)
            a_neg_vars.append(a_neg_k)

        # Optional: force some change when "off"
        if getattr(self, "force_change_when_off", False):
            min_step_size = min(a.step_size for a in self.parent if a.name in self.targets)
            min_step_size *= 0.99
            print(f"forcing constraint - min_step_size: {min_step_size}")

            # sum(a_pos) + sum(a_neg) + min_step_size * w >= min_step_size
            # If w == 0 (off), enforce total movement >= min_step_size
            scip.addCons(
                quicksum(a_pos_vars) + quicksum(a_neg_vars) + min_step_size * w >= min_step_size,
                name=f"switch_{self.id}_force_change_when_off",
            )

        return scip, indices
