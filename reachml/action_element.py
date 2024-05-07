from abc import ABC
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass
class ActionElement:
    name: str = field(init=True)
    lb: float = field(repr=True)
    ub: float = field(repr=True)
    actionable: Optional[bool] = field(init=False, default=True, repr=True)
    step_direction: Union[int, float] = field(init=False, default=0, repr=True)
    step_ub: float = field(init=False, default=float("inf"), repr=True)
    step_lb: float = field(init=False, default=-float("inf"), repr=True)
    variable_type: type = field(init=False, default=float, repr=True)
    discrete: type = field(init=False, default=False, repr=True)

    @staticmethod
    def from_values(name, values):
        assert len(values) >= 1, "values should be non-empty"
        assert np.isfinite(values).all(), "values should be finite"
        if np.isin(values, (0, 1)).all():  # binaries
            out = BooleanActionElement(name=name)
        elif np.equal(np.mod(values, 1), 0).all():  # integer-valued
            out = IntegerActionElement(name=name, lb=np.min(values), ub=np.max(values))
        else:
            out = FloatActionElement(name=name, lb=np.min(values), ub=np.max(values))
        return out

    def __post_init__(self):
        def setter(self, prop, val):
            if prop in ("lb", "ub", "step_lb", "step_ub"):
                assert self.__check_rep__()
            super().__setattr__(prop, val)

        self.__set_attr__ = setter

    def __check_rep__(self):
        assert self.lb <= self.ub, "lb must be <= ub"
        assert self.step_direction in (-1, 0, 1)
        if self.discrete:
            assert np.greater_equal(self.step_size, 1.0)
            if np.isfinite(self.step_lb):
                assert self.step_size <= -self.step_lb
            if np.isfinite(self.step_ub):
                assert self.step_size <= self.step_ub
        return True

    def __repr__(self):
        raise NotImplementedError()

    def get_action_bound(self, x, bound_type):
        assert bound_type in ("lb", "ub") and np.isfinite(x)
        out = 0.0
        if self.actionable:
            if bound_type == "ub" and self.step_direction >= 0:
                out = self.ub - x
                if np.isfinite(self.step_ub):
                    out = np.minimum(out, self.step_ub)
            elif bound_type == "lb" and self.step_direction <= 0:
                out = self.lb - x
                if np.isfinite(self.step_lb):
                    out = np.maximum(out, self.step_lb)
        return out


@dataclass
class FloatActionElement(ActionElement, ABC):
    variable_type: type = float
    step_size: float = field(init=False, default=1e-4, repr=False)


@dataclass
class IntegerActionElement(ActionElement, ABC):
    variable_type: type = int
    discrete: bool = True
    step_size: int = field(init=False, default=1, repr=False)

    @property
    def grid(self):
        return np.arange(self.lb, self.ub + self.step_size, self.step_size)

    def reachable_grid(self, x, return_actions=False):
        if self.actionable:
            vals = self.grid
            assert np.isin(x, vals)
            if self.step_direction == 0:
                keep = np.ones_like(vals, dtype=bool)
            elif self.step_direction > 0:
                keep = np.greater_equal(vals, x)
            else:  # self.step_direction < 0:
                keep = np.less_equal(vals, x)
            if np.isfinite(self.step_ub):
                keep &= np.less_equal(vals, x + self.step_ub)
            if np.isfinite(self.step_lb):
                keep &= np.greater_equal(vals, x + self.step_lb)
            vals = np.extract(keep, vals)
        else:
            vals = np.array([x])

        if return_actions:
            return vals - x

        return vals


@dataclass
class BooleanActionElement(IntegerActionElement, ABC):
    lb: bool = field(default=False, init=False)
    ub: bool = field(default=True, init=False)
    variable_type: type = bool
    step_size: 1 = field(init=False, repr=False)

    @property
    def grid(self):
        return np.array([0, 1])
