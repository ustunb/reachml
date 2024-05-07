import numpy as np
from cplex import Cplex, SparsePair
from functools import reduce
from .abstract import ActionabilityConstraint
from ..cplex_utils import combine, get_cpx_variable_args

class MutabilitySwitch(ActionabilityConstraint):
    """
    "if x[j] is on, then x[k1]...x[km]" cannot change - a[k] = 0"
    "if x[j] is off, then x[k1]...x[km]" can change - a[k] in lower/upper bounds"

    Example:
    If "Balance_eq_0" = 1 -> [Balance_geq_20, Balance_geq_50, Balance_geq_90] are off
    If "Balance_eq_0" = 0 -> [Balance_geq_20, Balance_geq_50, Balance_geq_90] can change
    If "Balance_eq_0" = 0 and Force Change -> [Balance_geq_20, Balance_geq_50, Balance_geq_90] must change
    """
    def __init__(self, switch, targets, on_value = 1, force_change_when_off = True, parent = None):
        """
        :param self:
        :param action_set:
        :param names:
        :return:
        """
        assert isinstance(switch, str)
        if isinstance(targets, str):
            targets = [targets]
        assert switch not in targets
        assert np.isin(on_value, (0, 1))
        self._switch = str(switch)
        self._targets = targets
        self._on_value = bool(on_value)
        self._force_change_when_on = bool(force_change_when_off)
        self._parameters = ('switch', 'targets', 'on_value', 'force_change_when_off')
        super().__init__(names = [switch] + targets, parent = parent)

    @property
    def switch(self):
        return self._switch

    @property
    def targets(self):
        return self._targets

    @property
    def on_value(self):
        return self._on_value

    @property
    def force_change_when_off(self):
        return self._force_change_when_on

    def __str__(self):
        target_names = ', '.join(f"`{n}`" for n in self._targets)
        s = f"If {self.switch}={self.on_value} then {target_names} cannot change."
        if self.force_change_when_off:
            s += f"\nIf {self.switch}={not self.on_value} then {target_names} must change."
        return s

    def check_compatibility(self, action_set):
        """
        Checks that constraint is compatible with a given ActionSet
        This function will be called whenever we attach this constraint to an
        ActionSet by calling `ActionSet.constraints.add`
        :param action_set: Action Set
        :return: True if action_set contains all features listed in the constraint
                 and obey other requirements of the constraint
        """
        assert self.switch in action_set.names
        assert action_set[self.switch].actionable, f"switch feature `{self.switch}` must be actionable"
        assert action_set[self.switch].variable_type == bool, f"switch feature `{self.switch}` must be boolean"
        for n in self.targets:
            assert n in action_set.names, f"action set does not contain target feature {n}"
            assert action_set[n].actionable, f"target feature {n} must be actionable"
        return True

    def check_feasibility(self, x):
        """
        checks that point
        :param x: array-like, either a 1D feature vector with d values or
                  a 2D feature matrix with n rows and d columns
        :return: boolean indicating that point was feasible
                 if input is array then check feasibility will return an array of booleans
        """
        x = np.array(x)
        # if matrix then apply this function again for each point in the matrix
        if x.ndim == 2 and x.shape[0] > 1:
            return np.apply_over_axes(self.check_feasibility, a = x, axis = 0)
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
        """
        adapts the constraint to a feature vector x
        :param x: feature vector for
        :return: constraint parameters for point x
        """
        assert self.check_feasibility(x), f'{self.__class__} is infeasible at x = {str(x)}'
        x_switch = x[self.indices[0]]
        a_pos_max = np.abs(self.parent.get_bounds(x, bound_type = 'ub')).astype(float)
        a_neg_max = np.abs(self.parent.get_bounds(x, bound_type = 'lb')).astype(float)
        print(f'A_pos_max: {a_pos_max}')
        print(f'A_neg_max: {a_neg_max}')
        return x_switch, a_pos_max, a_neg_max

    def add_to_cpx(self, cpx, indices, x):
        """
        :param cpx: Cplex object
        :param indices:
        :param x:
        :return:
        """
        assert isinstance(cpx, Cplex)
        vars = cpx.variables
        cons = cpx.linear_constraints
        x_switch, A_pos_max, A_neg_max = self.adapt(x)
        switch_idx = self.parent.get_feature_indices(self.switch)
        target_idx = self.parent.get_feature_indices(self.targets)
        a_switch = f"a[{switch_idx}]"

        # add switching variable to CPLEX
        w = f'w[{self.id}]'
        variable_args = {'w': get_cpx_variable_args(obj = 0.0, name = w, lb = 0.0,  ub = 1.0, vtype = "B")}
        vars.add(**reduce(combine, variable_args.values()))

        # add constraint to set switching variable
        # x'[j] = self.on_value => w[j] = 1
        if self.on_value == 1:
            # w[j] = x'[j] =
            # w[j] = x[j] + a[j]
            # -> w[j] - a[j] = x[j]
            cons.add(names = [f"set_{w}"],
                     lin_expr = [SparsePair(ind = [w, a_switch], val = [1.0, -1.0])],
                     senses = "E",
                     rhs = [x_switch])
        else:
            # w[j] = 1-x'[j]
            # w[j] = 1-(x[j] + a[j])
            # -> w[j] + a[j] = 1 - x[j]
            cons.add(names = [f"set_{w}"],
                     lin_expr = [SparsePair(ind = [w,  a_switch], val = [1.0, 1.0])],
                     senses = "E",
                     rhs = [1.0 - x_switch])

        # For each, we need to add
        a_targets = [f"a[{k}]" for k in target_idx]
        a_pos_targets = [f"a[{k}]_pos" for k in target_idx]
        a_neg_targets = [f"a[{k}]_neg" for k in target_idx]
        for k in target_idx:
            a_k = f"a[{k}]"
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
            cons.add(names = [f"switch_{self.id}_for_target_{k}_pos"],
                     lin_expr = [SparsePair(ind = [a_pos_k, w], val = [1.0, A_pos])],
                     senses = "L",
                     rhs = [A_pos])
            cons.add(names = [f"switch_{self.id}_for_target_{k}_neg"],
                     lin_expr = [SparsePair(ind = [a_neg_k, w], val = [1.0, A_neg])],
                     senses = "L",
                     rhs = [A_neg])

        if self.force_change_when_off:
            n_targets = len(self.targets)
            min_step_size = np.min([a.step_size for a in self.parent if a.name in self.targets])
            min_step_size = 0.99 * min_step_size
            print(f'forcing constraint - min_step_size: {min_step_size}')
            cons.add(names = [f"switch_{self.id}_force_change_when_off"],
                     lin_expr = [SparsePair(
                             ind = a_pos_targets + a_neg_targets + [w],
                             val = np.ones(2 * n_targets).tolist() + [min_step_size]
                             )],
                     senses = "G",
                     rhs = [min_step_size])

        indices.append_variables(variable_args)
        return cpx, indices
