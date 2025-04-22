from itertools import product

import numpy as np

from .action_set import ActionSet
from .cplex_utils import (
    set_mip_node_limit,
    set_mip_time_limit,
)
from .mip import EnumeratorMIP


class ReachableSetEnumerator:
    """
    Class to enumerate reachable sets over discrete feature spaces via decomposition
    """

    SETTINGS = {
        "eps_min": 0.99,
    }

    def __init__(self, action_set, x, print_flag=False, **kwargs):
        """
        :param action_set: full action set
        :param x:
        :param print_flag:
        :param kwargs:
        """
        assert isinstance(action_set, ActionSet)
        self._action_set = action_set

        # attach initial point
        assert isinstance(x, (list, np.ndarray))
        assert np.isfinite(x).all()
        x = np.array(x, dtype=np.float64).flatten()
        assert len(x) == len(self._action_set)
        self._x = x

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
                    **self.settings,
                )

    @property
    def action_set(self):
        return self._action_set

    @property
    def partition(self):
        return self._action_set.actionable_partition

    @property
    def x(self):
        return self._x

    @property
    def complete(self):
        return all(e.complete for e in self._enumerators.values())

    @property
    def reachable_points(self):
        return np.add(self.feasible_actions, self._x)

    @property
    def feasible_actions(self):
        actions_per_part = [
            self.convert_to_full_action(e.feasible_actions, part)
            for e, part in zip(self._enumerators.values(), self.partition)
        ]
        if len(actions_per_part) == 0:
            assert np.logical_not(self.action_set.actionable).all()
            actions = np.zeros(shape=(1, len(self._x)))
        else:
            combos = list(
                product(*actions_per_part)
            )  # todo: this is probably blowing up
            actions = np.sum(combos, axis=1)
            null_action_idx = np.flatnonzero(np.invert(np.any(actions, axis=1)))
            if len(null_action_idx) > 1:
                actions = np.delete(actions, null_action_idx[1:], axis=0)
        return actions

    def enumerate(
        self, max_points=float("inf"), time_limit=None, node_limit=None, **kwargs
    ):
        """
        Repeatedly solves MIP to enumerate reachable points in discrete feature space
        :param max_points: number of points to enumerate
        :param time_limit: the time limit on the solver (seconds).
        :param node_limit: the node limit on the solver (integer).
        """
        for e in self._enumerators.values():
            e.enumerate(
                max_points=max_points, time_limit=time_limit, node_limit=node_limit
            )

    def convert_to_full_action(self, actions, part):
        full_action = np.zeros((len(actions), len(self.x)))
        full_action[:, part] = actions

        return full_action

    def __repr__(self):
        return f"ReachableSetEnumerator<x = {str(self.x)}>"


class ReachableGrid:
    def __init__(self, action_set, x, **kwargs):
        """
        :param action_set: ActionSet with length 1
        :param x: feature value
        """
        assert len(action_set) == 1 and action_set.actionable[0]
        self.feasible_actions = (
            action_set[0].reachable_grid(x, return_actions=True).reshape(-1, 1)
        )
        self.complete = True

    def enumerate(self, **kwargs):
        pass


class ReachableSetEnumerationMIP:
    SETTINGS = {
        "eps_min": 0.5,
        # todo: set MIP parameters here
    }

    def __init__(self, action_set, x, print_flag=False, **kwargs):
        """
        :param action_set:
        :param x:
        :param print_flag:
        :param kwargs:
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
        self.mip_obj = EnumeratorMIP(action_set, x, print_flag=print_flag)
        self.mip, self.indices = self.mip_obj.mip, self.mip_obj.indices

        # initialize reachable points with null vector
        self._feasible_actions = [[0.0] * len(x)]
        self.mip_obj.remove_actions(actions=[self._feasible_actions[-1]])

    @property
    def action_set(self):
        return self._action_set

    @property
    def x(self):
        return self._x

    @property
    def feasible_actions(self):
        return np.array(self._feasible_actions)

    @property
    def complete(self):
        return self._complete

    def enumerate(self, max_points=None, time_limit=None, node_limit=None):
        """
        Repeatedly solves MIP to enumerate reachable points in discrete feature space
        :param time_limit: the time limit on the solver (seconds).
        :param node_limit: the node limit on the solver (integer).
        :return: solutions: list of solution dicts.
        each solution dict contains an optimal action, its cost, and other information pulled from recourse MIP
        :return:
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
            self.mip.solve()
            if not self.mip_obj.solution_exists:
                self._complete = True
                break
            self.mip_obj.check_solution()
            a = self.mip_obj.current_solution
            self._feasible_actions.append(a)
            self.mip_obj.remove_actions(actions=[a])
            k = k + 1

    def __check_rep__(self):
        assert self.indices.check_cpx(self.mip)
        AU = np.unique(self._feasible_actions, axis=0)
        assert len(self._feasible_actions) == AU.shape[0], "solutions are not unique"
        assert np.all(AU == 0, axis=1).any(), (
            "feasible actions do not contain null action"
        )
