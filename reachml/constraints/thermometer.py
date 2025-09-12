"""Thermometer encoding constraints for monotonic dummy features."""

from itertools import product

import numpy as np

from ..utils import implies, parse_attribute_name
from .reachability import ReachabilityConstraint


class ThermometerEncoding(ReachabilityConstraint):
    r"""Constraint to maintain actions over features in a thermometer encoding.

    Given a numeric feature Z \in R, a thermometer encoding creates a set of
    m nested dummies of the form:

                      x[0] = 1[Z ≥ z[0]]
                      x[1] = 1[Z ≥ z[1]]
                      ...
                      x[m-1] = 1[Z ≥ z[m-1]]
    Here:
    - z[0] ≤ z[1] ≤ ... z[m-1] are a set of increasing threshold values on Z
    - x[k] \in {0,1} is a binary variable
    - the encoding requires that x[k] -> x[k'] for k' > k
    todo: Example
    """

    def __init__(self, names, parent=None, step_direction=0, drop_invalid_values=True):
        """:param names: names of features in thermometer encoding of a feature
        :param parent: ActionSet
        :param step_direction: 0 if the underlying value can increase/decrease
                               1 if the underlying value can only increase
                               -1 if the underlying value can only increase

        :param drop_invalid_values: set to False to keep feature vectors that
                                    violate the encoding
        """
        assert len(names) >= 2, "constraint only applies to 2 or more features"
        values = np.array(list(product([0, 1], repeat=len(names))))
        if drop_invalid_values:
            keep_idx = [self.check_encoding(v) for v in values]
            values = values[keep_idx, :]

        assert step_direction in (0, 1, -1)
        self._step_direction = step_direction
        n = values.shape[0]
        reachability = np.eye(n)
        for i, p in enumerate(values):
            for j, q in enumerate(values):
                if i != j:
                    out = self.check_encoding(p) and self.check_encoding(q)
                    if step_direction > 0:
                        out = out and implies(p, q)
                    elif step_direction < 0:
                        out = out and implies(q, p)
                    reachability[i, j] = out

        super().__init__(names=names, values=values, reachability=reachability, parent=parent)
        self._parameters = self._parameters + ("step_direction",)

    @property
    def step_direction(self):
        return self._step_direction

    @staticmethod
    def check_encoding(x):
        return np.array_equal(x, np.cumprod(x))

    def __str__(self):
        name_list = ", ".join(f"`{n}`" for n in self.names)
        attribute_name = parse_attribute_name(self.names, default_name="continuous_attribute")
        s = f"Actions on [{name_list}] must preserve thermometer encoding of {attribute_name}."
        if self.step_direction > 0:
            s = (
                f"{s}, which can only increase."
                f"Actions can only turn on higher-level dummies that are off"
                f", where {self.names[0]} is the lowest-level dummy "
                f"and {self.names[-1]} is the highest-level-dummy."
            )
        elif self.step_direction < 0:
            s = (
                f"{s}, which can only decrease."
                f"Actions can only turn off higher-level dummies that are on"
                f", where {self.names[0]} is the lowest-level dummy "
                f"and {self.names[-1]} is the highest-level-dummy."
            )
        return s
