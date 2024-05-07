import numpy as np
from cplex import Cplex, SparsePair
from .abstract import ActionabilityConstraint
from ..utils import parse_attribute_name

class OneHotEncoding(ActionabilityConstraint):
    """
    Constraint to ensure that actions preserve one-hot encoding of a categorical
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
    VALID_LIMIT_TYPES = ('equal', 'max')
    def __init__(self, names, parent = None, limit = 1, limit_type = 'equal'):
        """
        :param self:
        :param action_set:
        :param names:
        :param limit: integer value representing number of
        :param limit_type: either `equal` or `max` (at most limit)
        :return:
        """
        assert isinstance(limit, int)
        assert 0 <= limit <= len(names), f"limit must be between 0 to {len(names)}"
        assert limit_type in OneHotEncoding.VALID_LIMIT_TYPES
        self._limit = limit
        self._limit_type = limit_type
        self._parameters = ('limit', 'limit_type')
        super().__init__(names = names, parent = parent)

    def check_compatibility(self, action_set):
        """
        Checks that constraint is compatible with a given ActionSet
        This function will be called whenever we attach this constraint to an
        ActionSet by calling `ActionSet.constraints.add`
        :param action_set: Action Set
        :return: True if action_set contains all features listed in the constraint
                 and obey other requirements of the constraint
        """
        assert all(vtype == bool for vtype in action_set[self.names].variable_type), 'features must be bool'
        return True

    @property
    def limit(self):
        return self._limit

    @property
    def limit_type(self):
        return self._limit_type

    def __str__(self):
        name_list = ', '.join(f"`{n}`" for n in self.names)
        attribute_name = parse_attribute_name(self.names, default_name = "categorical attribute")
        s = f"Actions on [{name_list}] must preserve one-hot encoding of {attribute_name}."
        if self.limit_type == 'equal':
            s = f"{s}. Exactly {self.limit} of [{name_list}] must be TRUE"
        elif self.limit_type == 'max':
            s = f"{s}. At most {self.limit} of [{name_list}] must be TRUE"
        return s

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
        v = x[self.indices]
        out = self.check_feature_vector(v) and np.isin(v, (0, 1)).all()
        if out:
            if self.limit_type == 'max':
                out = np.less_equal(np.sum(v), self.limit)
            elif self.limit_type == 'equal':
                out = np.equal(np.sum(v), self.limit)
        return out

    def adapt(self, x):
        """
        adapts the constraint to a feature vector x
        :param x: feature vector for
        :return: constraint parameters for point x
        """
        assert self.check_feasibility(x), f'{self.__class__} is infeasible at x = {str(x)}'
        return x[self.indices]

    def add_to_cpx(self, cpx, indices, x):
        """
        :param cpx: Cplex object
        :param indices:
        :param x:
        :return:
        """
        assert isinstance(cpx, Cplex)
        cons = cpx.linear_constraints
        x_values = self.adapt(x)
        a = [f'a[{idx}]' for idx in self.indices]
        # todo: pull constraint_id from object
        cons.add(names = [f'onehot_{self.id}'],
                 # todo, name this using constraint id
                 lin_expr = [SparsePair(ind = a, val = [1.0] * len(x_values))],
                 senses = "E" if self._limit_type == "equal" else "L",
                 rhs = [float(self._limit - np.sum(x_values))])

        return cpx, indices

