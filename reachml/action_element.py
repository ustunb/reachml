"""Action element primitives used by `ActionSet`.

This module defines the core element types that describe how individual features
can change: continuous (float), integer, and boolean. They encapsulate bounds,
step directions, and helper routines to compute feasible moves.
"""

from abc import ABC
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np


@dataclass
class ActionElement:
    """Base action element.

    Represents the editability and bounds of a single feature and provides
    helpers to compute feasible move magnitudes.
    """

    name: str = field(init=True)
    lb: float = field(repr=True, default=-float("inf"))
    ub: float = field(repr=True, default=float("inf"))
    actionable: Optional[bool] = field(init=False, default=True, repr=True)
    step_direction: Union[int, float] = field(init=False, default=0, repr=True)
    step_ub: float = field(init=False, default=float("inf"), repr=True)
    step_lb: float = field(init=False, default=-float("inf"), repr=True)
    variable_type: type = field(init=False, default=float, repr=True)
    discrete: type = field(init=False, default=False, repr=True)

    @staticmethod
    def from_values(name, values):
        """Infer an `ActionElement` type from observed values.

        Args:
            name: Feature name.
            values: Observed values for the feature.

        Returns:
            A concrete `ActionElement` subtype consistent with the value domain
            (boolean, integer, or float) with bounds set from the data.
        """
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
        """Validate on attribute updates during initialization."""

        def setter(self, prop, val):
            if prop in ("lb", "ub", "step_lb", "step_ub"):
                assert self.__check_rep__()
            super().__setattr__(prop, val)

        self.__set_attr__ = setter

    def __check_rep__(self):
        """Check internal representation invariants.

        Returns:
            True if the representation is valid.
        """
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
        """String representation of the action element."""
        raise NotImplementedError()

    def get_action_bound(self, x, bound_type):
        """Compute the feasible move bound at `x`.

        Args:
            x: Current feature value (finite scalar).
            bound_type: Either "lb" or "ub" indicating lower or upper move bound.

        Returns:
            The maximum allowed change (signed) from `x` in the requested
            direction, accounting for global bounds, step direction, and step
            limits.
        """
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
    """Action element for continuous (float) features."""

    variable_type: type = float
    is_discrete: bool = False
    step_size: float = field(init=False, default=1e-4, repr=False)

    def feasible_bound(self, x, return_actions=False):
        """Compute feasible bounds around `x`.

        Args:
            x: Current value.
            return_actions: If True, return deltas relative to `x`.

        Returns:
            Tuple of (lb, ub) values if `return_actions` is False; otherwise
            tuple of action deltas (a_lb, a_ub) where `lb = x + a_lb` and
            `ub = x + a_ub`.
        """
        a_lb = self.get_action_bound(x, "lb")
        a_ub = self.get_action_bound(x, "ub")

        if return_actions:
            return a_lb, a_ub

        lb = x + a_lb
        ub = x + a_ub

        return lb, ub


@dataclass
class IntegerActionElement(ActionElement, ABC):
    """Action element for integer-valued features."""

    variable_type: type = int
    discrete: bool = True
    is_discrete: bool = True
    step_size: int = field(init=False, default=1, repr=False)

    @property
    def grid(self):
        """Integer grid within [lb, ub] inclusive."""
        return np.arange(self.lb, self.ub + self.step_size, self.step_size)

    def reachable_grid(self, x, relax=False, return_actions=False):
        """Return reachable integer values from `x`.

        Args:
            x: Current value; must be in the integer grid.
            relax: If True, treat as actionable even if marked otherwise.
            return_actions: If True, return action deltas instead of values.

        Returns:
            Numpy array of reachable values, or action deltas if
            `return_actions` is True.
        """
        if self.actionable or relax:
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
    """Action element for boolean features (0/1)."""

    lb: bool = field(default=False, init=False)
    ub: bool = field(default=True, init=False)
    variable_type: type = bool
    step_size: 1 = field(init=False, repr=False)
    discrete: bool = True

    @property
    def grid(self):
        """Boolean grid {0, 1}."""
        return np.array([0, 1])
