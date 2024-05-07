import numpy as np
from cplex import Cplex, SparsePair
from .abstract import ActionabilityConstraint
from functools import reduce
from ..cplex_utils import combine, get_cpx_variable_args


class DirectionalLinkage(ActionabilityConstraint):
    """
    Constraint to link action in a source feature to changes in target feature

    Given a set of features `names`:
    - names[0] is the "source feature"
    - names[1:] are the "target features"

    This constraint ensures that any action in a "source feature" will induce
    a S[k]-unit change in each target feature k.
    """
    def __init__(self, names, parent = None, scales = None, keep_bounds = False):
        """
        :param names: names of features that should change together
        :param parent: ActionSet (optional)
        :param scales: list or array representing the scale of features;
                       all entries must be non-zero
                       given one unit change in feature j = scale[k]/scale[j] unit change in feature k
                       set to 1.0 by default so that a unit change in one feature leads to a one unit change in other features

        :param keep_bounds: True/False to enforce existing lower/upper bounds
        on actions on target features after accounting for additional changes
        due to the actions on the source feature.

        Set `keep_bounds = True`, to ensure that actions for target feature k
        obey the normal upper and lower bounds within an ActionSet so that:

            LB ≤ a[k] ≤ UB

        where:

            LB = ActionSet[k].get_action_bound(x, bound_type = 'lb')
            UB = ActionSet[k].get_action_bound(x, bound_type = 'lb')

        Set `keep_bounds = False` to allow the actions for target feature k to
        exceed these bounds as a result of collateral effects from the source.
        In this case, the bounds will be set as:

            LB ≤ a[k] ≤ UB
            LB = LB[k] + min(LB[j]*scale[k], UB[j]*scale[k])
            UB = UB[k] + max(LB[j]*scale[k], UB[j]*scale[k])

        where:

            LB[j] = ActionSet[j].get_action_bound(x, bound_type = 'lb')
            UB[j] = ActionSet[j].get_action_bound(x, bound_type = 'ub')
            LB[k] = ActionSet[k].get_action_bound(x, bound_type = 'lb')
            UB[k] = ActionSet[k].get_action_bound(x, bound_type = 'ub')
        :return:
        """
        assert len(names) >= 2
        scales = np.ones(len(names)) if scales is None else np.array(scales).flatten()
        assert len(scales) == len(names)
        assert np.count_nonzero(scales) == len(scales)
        self._parameters = ('source', 'targets', 'scales', 'keep_bounds')
        super().__init__(names = names, parent = parent)
        self._source = self.names[0]
        self._targets = self.names[1:]
        self._scales = np.array(scales[1:]) / float(scales[0])
        self._keep_bounds = keep_bounds

    @property
    def source(self):
        return self._source

    @property
    def targets(self):
        return self._targets

    @property
    def scales(self):
        return self._scales

    @property
    def keep_bounds(self):
        return self._keep_bounds

    def check_compatibility(self, action_set):
        """
        Checks that constraint is compatible with a given ActionSet
        This function will be called whenever we attach this constraint to an
        ActionSet by calling `ActionSet.constraints.add`
        :param action_set: Action Set
        :return: True if action_set contains all features listed in the constraint
                 and obey other requirements of the constraint
        """
        # check for circular dependencies
        L = action_set.constraints.linkage_matrix
        source_index = action_set.get_feature_indices(self.source)
        target_indices = action_set.get_feature_indices(self.targets)
        for k, target in zip(target_indices, self.targets):
            assert L[k, source_index] == 0, f"Circular Dependency: " \
                                            f"Cannot link actions from {self.source}->{target}." \
                                            f"action_set already contains link from {target}->{self.source}"

        # check that source is actionable
        assert action_set[self.source].actionable

        # check that scales are compatible
        target_actions = [a for a in action_set if a.name in self.targets]
        step_compatability = [np.mod(scale, a.step_size)==0 if a.discrete else True for a, scale in zip(target_actions, self.scales)]
        assert all(step_compatability)
        return True

    def __str__(self):
        s = f"Actions on {self._source} will induce to actions on {self._targets}." \
            f"Each unit change in {self._source} leads to:" + \
            ", ".join([f"{s:1.2f}-unit change in {n}" for n,s in zip(self._targets, self._scales)])
        return s

    def check_feasibility(self, x):
        """
        checks that a feature vector is realizable under these constraints
        :param x: array-like, either a 1D feature vector with d values or
                  a 2D feature matrix with n rows and d columns
        :return: boolean indicating that point was feasible
                 if input is array then check feasibility will return an array of booleans
        """
        x = np.array(x)
        # if matrix then apply this function again for each point in the matrix
        if x.ndim == 2 and x.shape[0] > 1:
            return np.apply_over_axes(self.check_feasibility, a = x, axis = 0)
        v = x[self.indices]
        out = self.check_feature_vector(v)
        return out

    def adapt(self, x):
        """
        adapts the constraint to a feature vector x
        :param x: feature vector for
        :return: constraint parameters for point x
        """
        assert self.check_feasibility(x), f'{self.__class__} is infeasible at x = {str(x)}'
        j = self.indices[0] #j == source index
        aj_max = self.parent[j].get_action_bound(x[j], bound_type = 'ub')
        aj_min = self.parent[j].get_action_bound(x[j], bound_type = 'lb')
        b_ub = np.maximum(self.scales * aj_max, self.scales * aj_min)
        b_lb = np.minimum(self.scales * aj_max, self.scales * aj_min)
        return b_ub, b_lb

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
        b_ub, b_lb = self.adapt(x)

        # get indices of source and targets
        j = self.indices[0] #j == source index
        target_indices = self.indices[1:]

        # define variables to capture linkage effects from source
        # b[j, k]
        b = [f"b[{j},{k}]" for k in target_indices]
        variable_args = {
            'b': get_cpx_variable_args(obj = 0.0,
                                       name = b,
                                       lb = b_lb,
                                       ub = b_ub,
                                       vtype = "C")
            }
        vars.add(**reduce(combine, variable_args.values()))
        indices.append_variables(variable_args)

        # add constraint to set
        # b[j,k] = scale[k]*a[j]
        for bjk, sk in zip(b, self.scales):
            cons.add(names = [f'set_{bjk}'],
                     lin_expr = [SparsePair(ind = [bjk, f"a[{j}]"], val = [1.0, -sk])],
                     senses = 'E',
                     rhs = [0.0])

        # add linkage effect to aggregate change variables for targets
        # c[k] = a[k] + b[j][k]
        # c[k] - a[k] -b[j][k] = 0
        c = [f"c[{k}]" for k in target_indices]
        for ck, bjk in zip(c, b):
            cons.set_linear_components(f"set_{ck}", [[bjk], [-1.0]])

        # update bounds on aggregate change variables for targets
        if not self.keep_bounds:

            # update upper bound on c[k] for target
            c_ub = vars.get_upper_bounds(c) + b_ub
            vars.set_upper_bounds([(ck, uk) for ck, uk in zip(c, c_ub)])

            # update lower bound on c[k] for target
            c_lb = vars.get_lower_bounds(c) + b_lb
            vars.set_lower_bounds([(ck, lk) for ck, lk in zip(c, c_lb)])

            # update indices
            indices.ub['c'] = vars.get_upper_bounds(indices.names['c'])
            indices.lb['c'] = vars.get_lower_bounds(indices.names['c'])

        return cpx, indices