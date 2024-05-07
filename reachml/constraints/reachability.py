import numpy as np
from cplex import SparsePair, Cplex
from functools import reduce
from .abstract import ActionabilityConstraint
from ..cplex_utils import combine, get_cpx_variable_args

class ReachabilityConstraint(ActionabilityConstraint):
    """
    Generalized Reachability Constraints
    - restrict actions over a subset of features to a set of values x[1], ... x[m]
    - reachability matrix indicates if we can reach x[k] from x[j]
    todo: Example
    """
    def __init__(self, names, values, parent = None, reachability = None):
        """
        :param parent: ActionSet
        :param names: names of features
        :param values: array or list-of-lists representing `m` distinct values for feature vectors with names
                       each value must be have exactly len(names)
                       must have at least 2 rows
        :param reachability: binary reachability matrix with `m` columns and `m` rows
                             R[j,k] = 1 if we can change value[j] from value[k]
        """
        # sport
        # sort_idx = np.argsort(names)
        # names = [names[i] for i in sort_idx]
        # values = values[:, sort_idx]
        n, d = values.shape
        assert n >= 1, 'values should have at least 2 rows'
        assert len(names) == d, f'values should have len(names) = {len(names)} dimensions'
        assert len(np.unique(values, axis = 0)) == len(values), 'values should be unique'
        assert np.isfinite(values).all(), 'values should be finite'
        if reachability is None:
            reachability = np.ones((n, n)) #assume all points are reachable
        else:
            reachability = np.array(reachability)
            assert n == reachability.shape[0]
            assert self.check_reachability_matrix(reachability), 'invalid reachability matrix'
        # todo: sort by name
        # names, values, reachability = self.sort_parameters(names, values, reachability)
        self._values = values
        self._reachability = reachability
        self._parameters = ('values', 'reachability')
        super().__init__(names, parent)

    @property
    def values(self):
        return self._values

    @property
    def reachability(self):
        return self._reachability

    def __str__(self):
        name_list = ', '.join(f"`{n}`" for n in self.names)
        s = f"The values of [{name_list}] must belong to one of {len(self.values)} values"
        if not np.all(self._reachability):
            s = f"{s} with custom reachability conditions."
        return s

    @staticmethod
    def sort_parameters(names, values, reachability):
        """
        sorts names, values and reachability by alphabetical order
        :param names:
        :param values:
        :param reachability:
        :return:
        """
        sort_idx = np.argsort(names)
        names = [names[i] for i in sort_idx]
        new_values = values[:, sort_idx]
        n = values.shape[0]
        new_index = {i: np.flatnonzero(np.all(new_values == v, axis = 1))[0] for i, v in enumerate(values)}
        new_reachability = np.zeros_like(reachability)
        for i in range(n):
            for j in range(n):
                new_reachability[new_index[i], new_index[j]] = reachability[i, j]
        assert np.sum(new_reachability[:]) == np.sum(new_reachability[:])
        return names, new_values, new_reachability

    @staticmethod
    def check_reachability_matrix(reachability):
        out = reachability.ndim == 2 and \
              reachability.shape[0] == reachability.shape[1] and \
              np.all(np.diagonal(reachability) == 1) and \
              np.isin(reachability, (0, 1)).all()
        return out

    def check_compatibility(self, action_set):
        """
        Checks that constraint is compatible with a given ActionSet
        This function will be called whenever we attach this constraint to an
        ActionSet by calling `ActionSet.constraints.add`
        :param action_set: Action Set
        :return: True if action_set contains all features listed in the constraint
                 and obey other requirements of the constraint
        """
        ub = np.max(self.values, axis = 0)
        lb = np.min(self.values, axis = 0)
        assert np.less_equal(ub, action_set[self.names].ub).all()
        assert np.greater_equal(lb, action_set[self.names].lb).all()
        return True

    def check_feasibility(self, x):
        x = np.array(x)
        if x.ndim == 2 and x.shape[0] > 1: # if matrix then apply this function again for each point in the matrix
            return np.apply_over_axes(self.check_feasibility, a = x, axis = 0)
        v = x[self.indices]
        out = np.all(self._values == v, axis = 1).any() #checks finite-ness
        return out

    def adapt(self, x):
        """
        adapts constraint parameters for point x
        :return:
        """
        x = np.array(x).flatten().astype(float)
        assert self.check_feasibility(x), f'{self.__class__} is infeasible at x = {str(x)}'
        v = x[self.indices]
        idx = np.flatnonzero(np.all(self._values == v, axis = 1))[0]
        reachable_points = self._reachability[idx]
        a_null = np.zeros_like(v) # null_action
        action_values = [p - v if reachable else a_null for p, reachable in zip(self._values, reachable_points)]
        return reachable_points, action_values

    def add_to_cpx(self, cpx, indices, x):
        """
        adds constraint to cplex
        :param cpx:
        :param indices:
        :param x:
        :return: Cplex MIP with `n_points` new variables and `len(names)` new constraints where:

        new variables:
        - r[id][k] = 1 if we move from x[feature_indices] to values[k, :] where values is specified by constraint id

        new constraints:
        - sum_k r[id][k] = 1
        - a[j] = sum_k r[id][k] * e[k][j]
        """
        assert isinstance(cpx, Cplex)
        reachable_points, action_values = self.adapt(x)
        n_points = len(reachable_points)
        point_indices = np.arange(n_points) #indices of points from 1...n_points
        reachability_indicators = [f'r[{self.id, k}]' for k in point_indices]

        # add variables/constraints
        vars = cpx.variables
        cons = cpx.linear_constraints

        # r[id, k] = 1 if we move from x[feature_indices] to values[k, :] where values is specified by constraint id
        variable_args = {
            'v': get_cpx_variable_args(name = reachability_indicators,
                                       obj = 0.0,
                                       lb = 0.0,
                                       ub = reachable_points,
                                       vtype = "B")
            }
        vars.add(**reduce(combine, variable_args.values()))

        # assign to exactly one reachable point:
        # sum r[const_id, k] = 1
        cons.add(names = [f'reachability_{self.id}_limit'],
                 lin_expr = [SparsePair(ind = reachability_indicators, val = [1.0] * n_points)],
                 senses = "E",
                 rhs = [1.0])


        # assign to point k, then set a[j] = action_value[k][j]
        # a[j] := sum(r[const_id, k] * action_value[k][j])
        for i, j in enumerate(self.indices):
            ind = [f'a[{j}]'] + reachability_indicators
            val = [-1.0] + [float(ak[i] * ek) for ak, ek in zip(action_values, reachable_points)]
            cons.add(names = [f'reachability_{self.id}_assignment_a[{j}]'],
                     lin_expr = [SparsePair(ind = ind, val = val)],
                     senses = "E",
                     rhs = [0.0])

        # todo: declare SOS if we can specify unique weights for each reachable point
        # reachability_weights = np.array([np.linalg.norm(x, 2) for x in action_values]).tolist()
        # cpx.SOS.add(type = "1", name = f"reachability_{self.id}_sos", SOS = SparsePair(ind = reachability_indicators, val = reachability_weights))

        # append indices
        indices.append_variables(variable_args)
        indices.params.update({'Ak': [action_values], 'Ek': [reachable_points]})
        return cpx, indices




