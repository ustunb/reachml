import numpy as np
from itertools import product
from .reachability import ReachabilityConstraint
from ..utils import parse_attribute_name

class OrdinalEncoding(ReachabilityConstraint):
    """
    Constraint to ensure that actions preserve one-hot encoding of an ordinal
    attribute. This constraint should be specified over a subset of boolean
    features that are produced by a one-hot encoding of an ordinal attribute Z.

    Given an ordinal attribute Z with `m` levels: `z[0], z[1], .. z[m-1]`,
    the boolean features - i.e., dummies - have the form:

      x[0] := 1[Z = 0]
      x[1] := 1[Z = 1]
      ...
      x[m-1] := 1[Z = z[m-1]]

    Here z[0] ≤ z[1] ≤ ... z[m-1] denote the levels of Z in increasing order,
    and x[k] := 1[Z = k] is a dummy variable set to 1 if and only if Z == k.

    todo: Example:
    """
    def __init__(self, names, parent = None, exhaustive = True, step_direction = 0, drop_invalid_values = True):
        """
        :param names: name of features in ordinal encoding of a feature
        :param parent: ActionSet
        :param exhaustive: set to True if one of the dummies is always on, i.e.
                           we have a dummy variable for all possible values;
                           set to False allows one of the dummies to be off
        :param step_direction: 0 if the underlying value can increase/decrease
                               1 if the underlying value can only increase
                               -1 if the underlying value can only increase
        :param drop_invalid_values: set to False to keep feature vectors that
                                    violate the encoding
        """

        assert len(names) >= 2, 'constraint only applies to 2 or more features'
        assert isinstance(exhaustive, bool)
        assert step_direction in (0, 1, -1)
        self._limit = 1
        self._exhaustive = bool(exhaustive)
        self._step_direction = step_direction

        # create values
        values = np.array(list(product([1, 0], repeat = len(names))))
        if drop_invalid_values:
            keep_idx = [self.check_encoding(v) for v in values]
            values = values[keep_idx, :]
            values = values[np.lexsort(values.T, axis = 0), :]

        n = values.shape[0]
        reachability = np.eye(n)
        for i, p in enumerate(values):
            for j, q in enumerate(values):
                if i != j:
                    out = self.check_encoding(p) and self.check_encoding(q)
                    if step_direction > 0:
                        out = out and (i < j)
                    elif step_direction < 0:
                        out = out and (j < i)
                    reachability[i, j] = out

        super().__init__(names = names, values = values, reachability = reachability, parent = parent)
        self._parameters = self._parameters + ('exhaustive', 'step_direction')

    def check_encoding(self, x):
        if self.exhaustive:
            out = np.sum(x) == self.limit
        else:
            out = np.less_equal(np.sum(x), self.limit)
        return out

    def check_feasibility(self, x):
        out = self.check_encoding(x) and super().check_feasibility(x)
        return out

    @property
    def limit(self):
        return self._limit

    @property
    def exhaustive(self):
        return self._exhaustive

    @property
    def step_direction(self):
        return self._step_direction

    def __str__(self):
        name_list = ', '.join(f"`{n}`" for n in self.names)
        attribute_name = parse_attribute_name(self.names, default_name = "ordinal attribute")
        s = f"Actions on [{name_list}] must preserve one-hot encoding of {attribute_name}."
        f"{'Exactly' if self.exhaustive else 'At most'} {self.limit} of [{name_list}] can be TRUE."
        if self.step_direction > 0:
            s = f"{s}, which can only increase." \
                f"Actions can only turn on higher-level dummies that are off" \
                f", where {self.names[0]} is the lowest-level dummy " \
                f"and {self.names[-1]} is the highest-level-dummy."
        elif self.step_direction < 0:
            s = f"{s}, which can only decrease." \
                f"Actions can only turn off higher-level dummies that are on" \
                f", where {self.names[0]} is the lowest-level dummy " \
                f"and {self.names[-1]} is the highest-level-dummy."
        return s