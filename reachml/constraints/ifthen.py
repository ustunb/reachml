import numpy as np
from cplex import SparsePair, Cplex
from functools import reduce
from .abstract import ActionabilityConstraint
from ..cplex_utils import combine, get_cpx_variable_args

class Condition(object):
    """
    :param constraint_level: Only a constraint type of action is currently supported.
    types of constraint levels are 'feature' or 'action'. ex: if is_employed = 1
    then is_ira = 1 is a 'feature' level constraint for is_employed. If
    is_employed_geq_1_yr = 1 then age increases by 1 year is an 'action' level constraint since
    if a person is not employed and become is_employed_geq_1_yr = 1 then they must increase their
    age
    :param sense: "E", "G"
    :param value: if a 'feature' level constraint then value must be between the lb and ub of the
    feature. If a 'action' level constraint then value must be between lb + value <= ub
    """

    def __init__(self, name, sense, value):
        self._name = name
        assert sense in ("E", "G")
        self._sense = sense
        self._value = float(value)

    @property
    def sense(self):
        return self._sense

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        return self._value

    def __eq__(self, other):
        out = (self.name == other.name) and (self.sense == other.sense) and (self.value) == (other.value)
        return out

    def __str__(self):
        sense = "=" if self.sense == "E" else ">"
        s = f"{self.name} {sense} {self.value}"
        return s

class IfThenConstraint(ActionabilityConstraint):

    def __init__(self, if_condition, then_condition, parent = None):
        """
        :param parent: ActionSet
        :param if_condition: names of features
        :param then_condition:
        """
        self._if_condition = if_condition
        self._then_condition = then_condition
        self._parameters = ('if_condition', 'then_condition')
        super().__init__(names = [if_condition.name, then_condition.name], parent = parent)

    @property
    def if_condition(self):
        return self._if_condition

    @property
    def then_condition(self):
        return self._then_condition

    def __str__(self):
        s = f"If {self.if_condition}, then {self.then_condition}"
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
        # check that values are within upper and lower bound
        values = [self.if_condition.value, self.then_condition.value]
        assert np.greater_equal(values, action_set[self.names].lb).all()
        assert np.less_equal(values, action_set[self.names].ub).all()
        return True

    def check_feasibility(self, x):
        return True

    def adapt(self, x):
        a_ub = self.parent.get_bounds(x, bound_type = 'ub')
        if_idx = self.parent.get_feature_indices([self.if_condition.name])[0]
        if_val_max = a_ub[if_idx]
        return if_val_max

    def add_to_cpx(self, cpx, indices, x):
        assert isinstance(cpx, Cplex)
        vars = cpx.variables
        cons = cpx.linear_constraints
        if_val_max = self.adapt(x)
        if_idx = self.parent.get_feature_indices([self.if_condition.name])[0]
        if_val = self.if_condition.value
        then_idx = self.parent.get_feature_indices([self.then_condition.name])[0]
        then_val = self.then_condition.value

        u = f'u_ifthen[{self.id}]'

        # add variables to cplex
        variable_args = {'u_ifthen': get_cpx_variable_args(obj = 0.0, name = u, vtype = "B", ub = 1.0, lb = 0.0)}
        vars.add(**reduce(combine, variable_args.values()))

        # M*u - a[j] >= -if_val + eps
        # if (a[j] ≥ if_val + eps) then u = 1
        eps = 1e-5
        M = if_val_max - if_val + eps
        cons.add(names = [f'ifthen_{self.id}_if_holds'],
                 lin_expr = [SparsePair(ind = [u,  f'a[{if_idx}]'], val = [M, -1.0])],
                 senses = "G",
                 rhs = [-if_val + eps])

        # M*u + a[j] >= if_val - M
        # todo: ??
        cons.add(names = [f'ifthen_{self.id}_if_2'],
                 lin_expr = [SparsePair(ind = [u,  f'a[{if_idx}]'], val = [-M, 1.0])],
                 senses = "G",
                 rhs = [if_val - M])

        if if_val_max != 0:
            # u * then_val = a[j]
            cons.add(names = [f'ifthen_{self.id}_then'],
                     lin_expr = [SparsePair(ind = [u,  f'a[{then_idx}]'], val = [then_val, -1.0])],
                     senses = "E",
                     rhs = [0.0])

        #update indices
        indices.append_variables(variable_args)
        indices.params.update({
            'M_if_then': M,
            'v_if': [if_val],
            'v_then': [then_val]
            })

        return cpx, indices
