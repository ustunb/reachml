"""Enumerators for reachable sets over actionable features.

Provides grid-based and MIP-based enumeration of feasible actions/points.
"""

from itertools import product

import numpy as np

from .action_set import ActionSet
from .mip import EnumeratorMIP
from .mip.backends.cplex_utils import set_mip_node_limit, set_mip_time_limit
from .utils import DEFAULT_SOLVER


class ReachableSetEnumerator:
    """Enumerate reachable sets over discrete feature spaces via decomposition."""

    SETTINGS = {
        "eps_min": 0.99,
    }

    def __init__(self, action_set, x, print_flag=False, solver=DEFAULT_SOLVER, **kwargs):
        """Initialize enumerator for a full action set and point `x`.

        Args:
            action_set: Full `ActionSet`.
            x: Current feature vector.
            print_flag: If True, enable verbose solver output.
            solver: Solver backend identifier.
            **kwargs: Override default settings.
        """
        assert isinstance(action_set, ActionSet)
        self._action_set = action_set

        # attach initial point
        assert isinstance(x, (list, np.ndarray))
        assert np.isfinite(x).all()
        x = np.array(x, dtype=np.float64).flatten()
        assert len(x) == len(self._action_set)
        self._x = x
        self.solver = solver

        # parse settings
        self.settings = dict(ReachableSetEnumerator.SETTINGS) | kwargs
        self.print_flag = print_flag

        # setup enumerators
        self._enumerators = {}
        for i, part in enumerate(self.partition):
            if len(part) == 1 and action_set[part[0]].discrete:
                self._enumerators[i] = ReachableGrid(
                    action_set=self.action_set[part],
                    x=self.x[part],
                    **self.settings,
                )
            else:
                self._enumerators[i] = ReachableSetEnumerationMIP(
                    action_set=self.action_set[part],
                    x=self.x[part],
                    print_flag=self.print_flag,
                    solver=self.solver,
                    **self.settings,
                )

    @property
    def action_set(self):
        """Underlying `ActionSet`."""
        return self._action_set

    @property
    def partition(self):
        """Actionable partitions of the action set."""
        return self._action_set.actionable_partition

    @property
    def x(self):
        """Current feature vector as 1D array."""
        return self._x

    @property
    def complete(self):
        """Whether all partitions have been fully enumerated."""
        return all(e.complete for e in self._enumerators.values())

    @property
    def reachable_points(self):
        """All reachable points as `feasible_actions + x`."""
        return np.add(self.feasible_actions, self._x)

    @property
    def feasible_actions(self):
        """All feasible action vectors across actionable partitions."""
        actions_per_part = [
            self.convert_to_full_action(e.feasible_actions, part)
            for e, part in zip(self._enumerators.values(), self.partition, strict=False)
        ]
        if len(actions_per_part) == 0:
            assert np.logical_not(self.action_set.actionable).all()
            actions = np.zeros(shape=(1, len(self._x)))
        else:
            combos = list(product(*actions_per_part))  # todo: this is probably blowing up
            actions = np.sum(combos, axis=1)
            null_action_idx = np.flatnonzero(np.invert(np.any(actions, axis=1)))
            if len(null_action_idx) > 1:
                actions = np.delete(actions, null_action_idx[1:], axis=0)
        return actions

    def enumerate(self, max_points=float("inf"), time_limit=None, node_limit=None, **kwargs):
        """Enumerate reachable points by iterating per-part enumerators.

        Args:
            max_points: Max solutions to request from each enumerator.
            time_limit: Solver time limit in seconds.
            node_limit: Solver node limit (int).
        """
        for e in self._enumerators.values():
            e.enumerate(max_points=max_points, time_limit=time_limit, node_limit=node_limit)

    def convert_to_full_action(self, actions, part):
        full_action = np.zeros((len(actions), len(self.x)))
        full_action[:, part] = actions

        return full_action

    def __repr__(self):
        return f"ReachableSetEnumerator<x = {str(self.x)}>"


class ReachableGrid:
    def __init__(self, action_set, x, **kwargs):
        """Grid enumerator for a single discrete actionable feature.

        Args:
            action_set: `ActionSet` with length 1.
            x: Feature value.
        """
        assert len(action_set) == 1 and action_set.actionable[0]
        self.feasible_actions = action_set[0].reachable_grid(x, return_actions=True).reshape(-1, 1)
        self.complete = True
        self.action_set = action_set

    def enumerate(self, **kwargs):
        """No-op for grid enumerator (already complete)."""
        pass


class ReachableSetEnumerationMIP:
    SETTINGS = {
        "eps_min": 0.5,
        # todo: set MIP parameters here
    }

    def __init__(self, action_set, x, print_flag=False, solver=DEFAULT_SOLVER, **kwargs):
        """MIP-based enumerator for a multi-feature actionable partition.

        Args:
            action_set: Partition `ActionSet`.
            x: Current feature subvector.
            print_flag: If True, enable verbose solver output.
            solver: Solver backend identifier.
            **kwargs: Override default settings.
        """
        assert isinstance(action_set, ActionSet)
        assert any(action_set.actionable)
        # assert part in action_set.actionable_partition
        self._action_set = action_set
        # self.part = part

        # attach initial point
        assert isinstance(x, (list, np.ndarray))
        x = np.array(x, dtype=np.float64).flatten()
        assert len(x) == len(self._action_set)
        self._x = x

        # set actionable indices
        self.actionable_indices = list(range(len(self._x)))

        # complete returns True if and only if we have enumerated all points in this set
        self._complete = False

        # parse remaining settings
        self.settings = dict(ReachableSetEnumerationMIP.SETTINGS) | kwargs
        self.print_flag = print_flag

        # build base MIP
        self.mip_obj = EnumeratorMIP(action_set, x, print_flag=print_flag, solver=solver)
        self.mip, self.indices = self.mip_obj.mip, self.mip_obj.indices

        # initialize reachable points with null vector
        self._feasible_actions = [[0.0] * len(x)]
        self.mip_obj.remove_actions(actions=[self._feasible_actions[-1]])

    @property
    def action_set(self):
        """Partition `ActionSet`."""
        return self._action_set

    @property
    def x(self):
        """Current partition feature vector."""
        return self._x

    @property
    def feasible_actions(self):
        """Feasible actions accumulated so far as an array."""
        return np.array(self._feasible_actions)

    @property
    def complete(self):
        """Whether enumeration has exhausted all feasible actions."""
        return self._complete

    def enumerate(self, max_points=None, time_limit=None, node_limit=None):
        """Repeatedly solve the MIP to discover new feasible actions.

        Args:
            max_points: Optional cap on number of actions to add.
            time_limit: Solver time limit in seconds.
            node_limit: Solver node limit.
        """
        # pull size limit
        max_points = float("inf") if max_points is None else max_points

        # update time limit and node limit
        if time_limit is not None:
            self.mip = set_mip_time_limit(self.mip, time_limit)
        if node_limit is not None:
            self.mip = set_mip_node_limit(self.mip, node_limit)

        k = 0
        while k < max_points:
            self.mip_obj.solve_model()
            if not self.mip_obj.solution_exists:
                self._complete = True
                break

            self.mip_obj.check_solution()  # whatever unpacks `current_solution` into `self._feasible_actions`
            a = self.mip_obj.current_solution
            self._feasible_actions.append(a)
            self.mip_obj.remove_actions(actions=[a])
            k = k + 1

    def __check_rep__(self):
        """Check internal consistency of enumeration results and indices."""
        assert self.indices.check_cpx(self.mip)
        AU = np.unique(self._feasible_actions, axis=0)
        assert len(self._feasible_actions) == AU.shape[0], "solutions are not unique"
        assert np.all(AU == 0, axis=1).any(), "feasible actions do not contain null action"
