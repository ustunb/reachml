"""If-Then actionability constraint linking feature conditions."""

from __future__ import annotations

from functools import reduce

import numpy as np

from .abstract import ActionabilityConstraint


class Condition(object):
    """A simple condition over a single feature value.

    Args:
        constraint_level: Currently only action-level constraints are supported.
        sense: "E" or "G".
        value: Required threshold/value for the condition.
    """

    def __init__(self, name, sense, value):
        """Initialize a condition on feature `name` with `sense` and `value`."""
        self._name = name
        assert sense in ("E", "G")
        self._sense = sense
        self._value = float(value)

    @property
    def sense(self):
        """Return the sense, either "E" or "G"."""
        return self._sense

    @property
    def name(self):
        """Feature name this condition applies to."""
        return self._name

    @property
    def value(self):
        """Numeric value for the condition."""
        return self._value

    def __eq__(self, other):
        """Structural equality by fields."""
        out = (
            (self.name == other.name)
            and (self.sense == other.sense)
            and (self.value) == (other.value)
        )
        return out

    def __str__(self):
        """Human-readable representation of the condition."""
        sense = "=" if self.sense == "E" else ">"
        s = f"{self.name} {sense} {self.value}"
        return s


class IfThenConstraint(ActionabilityConstraint):
    """Constraint enforcing a then-condition when an if-condition holds."""
    def __init__(self, if_condition, then_condition, parent=None):
        """Initialize the constraint with `if_condition` and `then_condition`."""
        self._if_condition = if_condition
        self._then_condition = then_condition
        self._parameters = ("if_condition", "then_condition")
        super().__init__(names=[if_condition.name, then_condition.name], parent=parent)

    @property
    def if_condition(self):
        """The antecedent condition."""
        return self._if_condition

    @property
    def then_condition(self):
        """The consequent condition applied when the antecedent holds."""
        return self._then_condition

    def __str__(self):
        """Human-readable representation of the if-then rule."""
        s = f"If {self.if_condition}, then {self.then_condition}"
        return s

    def check_compatibility(self, action_set):
        """Return True if the constraint is compatible with the ActionSet.

        Called when attaching this constraint via `ActionSet.constraints.add`.
        """
        # check that values are within upper and lower bound
        values = [self.if_condition.value, self.then_condition.value]
        assert np.greater_equal(values, action_set[self.names].lb).all()
        assert np.less_equal(values, action_set[self.names].ub).all()
        return True

    def check_feasibility(self, x):
        """Placeholder feasibility check (always True for this constraint)."""
        return True

    def adapt(self, x):
        """Compute big-M bound for the antecedent feature at `x`."""
        a_ub = self.parent.get_bounds(x, bound_type="ub")
        if_idx = self.parent.get_feature_indices([self.if_condition.name])[0]
        if_val_max = a_ub[if_idx]
        return if_val_max

    # TODO: bug for not allowing 0 action for if feature
    def add_to_cpx(self, cpx, indices, x):
        """Add if-then constraints to a CPLEX model using a big-M formulation."""
        from cplex import Cplex, SparsePair
        from ..mip.backends.cplex_utils import combine, get_cpx_variable_args
        assert isinstance(cpx, Cplex)
        vars = cpx.variables
        cons = cpx.linear_constraints
        if_val_max = self.adapt(x)
        if_idx = self.parent.get_feature_indices([self.if_condition.name])[0]
        if_val = self.if_condition.value
        then_idx = self.parent.get_feature_indices([self.then_condition.name])[0]
        then_val = self.then_condition.value
        then_sense = "E" if self.then_condition.sense == "E" else "L"

        u = f"u_ifthen[{self.id}]"

        # add variables to cplex
        variable_args = {
            "u_ifthen": get_cpx_variable_args(obj=0.0, name=u, vtype="B", ub=1.0, lb=0.0)
        }
        vars.add(**reduce(combine, variable_args.values()))

        # M*u - a[j] >= -if_val + eps
        # if (a[j] ≥ if_val + eps) then u = 1
        eps = 1e-5
        M = if_val_max - if_val + eps
        cons.add(
            names=[f"ifthen_{self.id}_if_holds"],
            lin_expr=[SparsePair(ind=[u, f"a[{if_idx}]"], val=[M, -1.0])],
            senses="G",
            rhs=[-if_val + eps],
        )

        # M*u + a[j] >= if_val - M
        # todo: ??
        cons.add(
            names=[f"ifthen_{self.id}_if_2"],
            lin_expr=[SparsePair(ind=[u, f"a[{if_idx}]"], val=[-M, 1.0])],
            senses="G",
            rhs=[if_val - M],
        )

        if if_val_max != 0:
            # u * then_val = a[j]
            cons.add(
                names=[f"ifthen_{self.id}_then"],
                lin_expr=[SparsePair(ind=[u, f"a[{then_idx}]"], val=[then_val, -1.0])],
                senses=then_sense,
                rhs=[0.0],
            )

        # update indices
        indices.append_variables(variable_args)
        indices.params.update({"M_if_then": M, "v_if": [if_val], "v_then": [then_val]})

        return cpx, indices

    def add_to_scip(self, scip, indices, x):
        """Add if-then constraints to a SCIP model using a big-M formulation."""
        from pyscipopt import Model
        from ..mip.backends.scip_utils import combine
        assert isinstance(scip, Model)

        # Compute indices/values
        if_val_max = self.adapt(x)
        if_idx = self.parent.get_feature_indices([self.if_condition.name])[0]
        if_val = self.if_condition.value
        then_idx = self.parent.get_feature_indices([self.then_condition.name])[0]
        then_val = self.then_condition.value

        # Names & existing variables
        u_name = [f"u_ifthen[{self.id}]"]
        a_if = indices.get_var(scip, f"a[{if_idx}]")
        a_then = indices.get_var(scip, f"a[{then_idx}]")

        # Add binary indicator u
        u_vars = {"names": u_name, "types": ["B"], "lb": [0.0], "ub": [1.0], "obj": [0.0]}
        variable_args = {"u": u_vars}
        reduced_args = reduce(combine, variable_args.values())
        reduced_args = {"u": reduced_args}
        indices.append_variables(scip, reduced_args)

        # Big-M constraints
        eps = 1e-5
        M = if_val_max - if_val + eps
        u = scip.addVar(name=u_name[0], obj=0.0, vtype="B", lb=0.0, ub=1.0)

        # ifthen_{id}_if_holds:  M*u - a_if >= -if_val + eps
        scip.addCons(M * u - a_if >= -if_val + eps, name=f"ifthen_{self.id}_if_holds")

        # ifthen_{id}_if_2:     -M*u + a_if >=  if_val - M
        scip.addCons(-M * u + a_if >= if_val - M, name=f"ifthen_{self.id}_if_2")

        # If then-part is active, enforce a_then = then_val * u
        if if_val_max != 0:
            scip.addCons(then_val * u - a_then == 0.0, name=f"ifthen_{self.id}_then")

        indices.params.update(
            {
                "M_if_then": M,
                "v_if": [if_val],
                "v_then": [then_val],
            }
        )

        return scip, indices
