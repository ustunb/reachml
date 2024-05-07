import numpy as np
from cplex import Cplex, SparsePair
from functools import reduce
from itertools import chain, product
from .action_set import ActionSet
from .reachable_set import ReachableSet
from .cplex_utils import (
    combine,
    get_cpx_variable_types,
    get_cpx_variable_args,
    has_solution,
    CplexGroupedVariableIndices,
    set_mip_time_limit,
    set_mip_node_limit,
    get_mip_stats,
)


class ReachableSetEnumerator:
    """
    Class to enumerate reachable sets over discrete feature spaces via decomposition
    """

    SETTINGS = {
        "eps_min": 0.99,
    }

    def __init__(self, action_set, x, print_flag=False, **kwargs):
        """
        :param action_set:
        :param x:
        :param print_flag:
        :param kwargs:
        """
        assert isinstance(action_set, ActionSet)
        self._action_set = action_set

        # attach initial point
        assert isinstance(x, (list, np.ndarray))
        assert np.isfinite(x).all()
        x = np.array(x, dtype=np.float_).flatten()
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
                    action_set=self.action_set,
                    x=self.x[part],
                    part=part,
                    **self.settings,
                )
            else:
                self._enumerators[i] = ReachableSetEnumerationMIP(
                    action_set=self.action_set,
                    x=self.x,
                    part=part,
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
    def reachable_set(self):
        """returns reachable_set from object"""
        return ReachableSet(
            action_set=self.action_set,
            x=self.x,
            complete=self.complete,
            values=self.feasible_actions,
            initialize_from_actions=True,
        )

    @property
    def feasible_actions(self):
        actions_per_part = [e.feasible_actions for e in self._enumerators.values()]
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

    def enumerate(self, max_points=float("inf"), time_limit=None, node_limit=None):
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
        return self.reachable_set

    def __repr__(self):
        return f"ReachableSetEnumerator<x = {str(self.x)}>"


class ReachableGrid:
    def __init__(self, action_set, x, part, **kwargs):
        """
        :param action_set: ActionSet
        :param x: value of feature j
        :param part: list of length 1 containing index of feature in partition
        """
        assert len(part) == len(x) == 1 and part in action_set.actionable_partition
        j = part[0]
        self.part = j
        actions = action_set[j].reachable_grid(x, return_actions=True)
        self.feasible_actions = np.zeros(shape=(len(actions), len(action_set)))
        self.feasible_actions[:, j] = actions
        self.complete = True

    def enumerate(self, **kwargs):
        pass


class ReachableSetEnumerationMIP:
    SETTINGS = {
        "eps_min": 0.5,
        # todo: set MIP parameters here
    }

    def __init__(self, action_set, x, part, print_flag=False, **kwargs):
        """
        :param action_set:
        :param x:
        :param part:
        :param print_flag:
        :param kwargs:
        """
        assert isinstance(action_set, ActionSet)
        assert part in action_set.actionable_partition
        self._action_set = action_set
        self.part = part

        # attach initial point
        assert isinstance(x, (list, np.ndarray))
        x = np.array(x, dtype=np.float_).flatten()
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
        cpx, indices = self.build_mip()

        # add non-separable constraints
        for con in action_set.constraints:
            mip, indices = con.add_to_cpx(cpx=cpx, indices=indices, x=self.x)

        # set MIP parameters
        cpx = self.set_solver_parameters(cpx, print_flag=self.print_flag)
        self.mip, self.indices = cpx, indices

        # initialize reachable points with null vector
        self._feasible_actions = [[0.0] * len(x)]
        self.remove_actions(actions=[self._feasible_actions[-1]])

    @property
    def action_set(self):
        return self._action_set

    @property
    def x(self):
        return self._x

    @property
    def feasible_actions(self):
        return self._feasible_actions

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
            s = self.solution_info
            if not s["has_solution"]:
                self._complete = True
                break
            self.check_solution()
            a = self.action
            self._feasible_actions.append(a)
            self.remove_actions(actions=[a])
            k = k + 1

    ### MIP Functions ###
    def build_mip(self):
        """
        build CPLEX mip object of actions
        :return: `cpx` Cplex MIP Object
                 `indices` CplexGroupVariableIndices
        ----
        Variables
        ----------------------------------------------------------------------------------------------------------------
        name                  length              type        description
        ----------------------------------------------------------------------------------------------------------------
        a[j]                  d x 1               real        action on variable j
        a_pos[j]              d x 1               real        absolute value of a[j]
        a_neg[j]              d x 1               real        absolute value of a[j]
        """

        # Setup cplex object
        cpx = Cplex()
        cpx.set_problem_type(cpx.problem_type.MILP)
        cpx.objective.set_sense(cpx.objective.sense.minimize)
        vars = cpx.variables
        cons = cpx.linear_constraints

        # variable parameters
        a_lb = self.action_set.get_bounds(self.x, bound_type="lb", part=self.part)
        a_ub = self.action_set.get_bounds(self.x, bound_type="ub", part=self.part)
        a_pos_max = np.abs(a_ub)
        a_neg_max = np.abs(a_lb)
        a_types = get_cpx_variable_types(self.action_set, self.actionable_indices)

        # add variables to CPLEX
        variable_args = {
            "a": get_cpx_variable_args(
                obj=0,
                name=[f"a[{j}]" for j in self.actionable_indices],
                lb=a_lb,
                ub=a_ub,
                vtype=a_types,
            ),
            "a_pos": get_cpx_variable_args(
                obj=1.0,
                name=[f"a[{j}]_pos" for j in self.actionable_indices],
                lb=0.0,
                ub=a_pos_max,
                vtype=a_types,
            ),
            "a_neg": get_cpx_variable_args(
                obj=1.0,
                name=[f"a[{j}]_neg" for j in self.actionable_indices],
                lb=0.0,
                ub=a_neg_max,
                vtype=a_types,
            ),
            "a_sign": get_cpx_variable_args(
                obj=0.0,
                name=[f"a[{j}]_sign" for j in self.actionable_indices],
                lb=0.0,
                ub=1.0,
                vtype="B",
            ),
            #
            "c": get_cpx_variable_args(
                obj=0,
                name=[f"c[{j}]" for j in self.actionable_indices],
                lb=a_lb,
                ub=a_ub,
                vtype=a_types,
            ),
        }
        vars.add(**reduce(combine, variable_args.values()))

        # store information about variables for manipulation / debugging
        indices = CplexGroupedVariableIndices()
        indices.append_variables(variable_args)
        names = indices.names

        for j, a_j, a_pos_j, a_neg_j, a_sign_j, c_j in zip(
            self.actionable_indices,
            names["a"],
            names["a_pos"],
            names["a_neg"],
            names["a_sign"],
            names["c"],
        ):
            # a_pos_j - a_j ≥ 0
            cons.add(
                names=[f"abs_val_pos_{a_j}"],
                lin_expr=[SparsePair(ind=[a_pos_j, a_j], val=[1.0, -1.0])],
                senses="G",
                rhs=[0.0],
            )

            # a_neg_j + a_j ≥ 0
            cons.add(
                names=[f"abs_val_neg_{a_j}"],
                lin_expr=[SparsePair(ind=[a_neg_j, a_j], val=[1.0, 1.0])],
                senses="G",
                rhs=[0.0],
            )

            cons.add(
                names=[f"set_{a_j}_sign_pos"],
                lin_expr=[
                    SparsePair(ind=[a_pos_j, a_sign_j], val=[1.0, -a_pos_max[j]])
                ],
                senses="L",
                rhs=[0.0],
            )

            cons.add(
                names=[f"set_{a_j}_sign_neg"],
                lin_expr=[SparsePair(ind=[a_neg_j, a_sign_j], val=[1.0, a_neg_max[j]])],
                senses="L",
                rhs=[a_neg_max[j]],
            )

            cons.add(
                names=[f"set_{a_j}"],
                lin_expr=[
                    SparsePair(ind=[a_j, a_pos_j, a_neg_j], val=[1.0, -1.0, 1.0])
                ],
                senses="E",
                rhs=[0.0],
            )

            cons.add(
                names=[f"set_{a_j}"],
                lin_expr=[
                    SparsePair(ind=[a_j, a_pos_j, a_neg_j], val=[1.0, -1.0, 1.0])
                ],
                senses="E",
                rhs=[0.0],
            )

            cons.add(
                names=[f"set_{c_j}"],
                lin_expr=[SparsePair(ind=[c_j, a_j], val=[1.0, -1.0])],
                senses="E",
                rhs=[0.0],
            )

        return cpx, indices

    def remove_actions(self, actions):
        """
        adds variables and constraints to remove actions from the feasible region of the MIP
        :param actions: list of actions to remove
        :return:
        """
        # todo: this should only run for a single set of actions rather than multiple actions
        assert isinstance(actions, list)
        vars = self.mip.variables
        cons = self.mip.linear_constraints
        indices = self.indices

        # Basic MIP Parameters
        # a_ub = indices.ub["a"]
        # a_lb = indices.lb["a"]
        # a_types = indices.types["a"]
        a_ub = indices.ub["c"]
        a_lb = indices.lb["c"]
        a_types = indices.types["c"]
        d = len(self.actionable_indices)

        # distance-based parameters
        n_points = len(actions)
        A_nogood = np.vstack(actions)
        D_pos = a_ub - A_nogood
        D_neg = A_nogood - a_lb

        # get number of existing parameters
        start_index = indices.counts.get("nogood", 0)
        point_indices = range(start_index, start_index + n_points)

        # add variables to cpx
        variable_args = {
            #
            "delta_pos": get_cpx_variable_args(
                obj=0.0,
                name=[
                    f"delta[{j, k}]_pos"
                    for k in point_indices
                    for j in self.actionable_indices
                ],
                lb=0.0,
                ub=list(chain.from_iterable(D_pos.tolist())),
                vtype=a_types * n_points,
            ),
            #
            "delta_neg": get_cpx_variable_args(
                obj=0.0,
                name=[
                    f"delta[{j, k}]_neg"
                    for k in point_indices
                    for j in self.actionable_indices
                ],
                lb=0.0,
                ub=list(chain.from_iterable(D_neg.tolist())),
                vtype=a_types * n_points,
            ),
            #
            "delta_sign": get_cpx_variable_args(
                obj=0.0,
                name=[
                    f"delta[{j, k}]_sign"
                    for k in point_indices
                    for j in self.actionable_indices
                ],
                lb=0.0,
                ub=1.0,
                vtype="B",
            ),
        }
        vars.add(**reduce(combine, variable_args.values()))

        # add constraints for each point
        for k, ak, D_pos_k, D_neg_k in zip(point_indices, actions, D_pos, D_neg):
            delta_pos_k = [f"delta[{j, k}]_pos" for j in self.actionable_indices]
            delta_neg_k = [f"delta[{j, k}]_neg" for j in self.actionable_indices]
            delta_sign_k = [f"delta[{j, k}]_sign" for j in self.actionable_indices]

            # sum(delta[j,k]_pos + delta[j,k]_neg) ≥ eps_min
            cons.add(
                names=[f"sum_abs_dist_val"],
                lin_expr=[
                    SparsePair(
                        ind=delta_pos_k + delta_neg_k, val=np.ones(2 * d).tolist()
                    )
                ],
                senses="G",
                rhs=[self.settings["eps_min"]],
            )

            for j, delta_pos_jk, delta_neg_jk, delta_sign_jk in zip(
                self.actionable_indices, delta_pos_k, delta_neg_k, delta_sign_k
            ):
                c_j = f"c[{j}]"

                # if delta[j,k]_pos > 0 => delta[j,k]_sign = 1
                cons.add(
                    names=[f"set_{delta_pos_jk}"],
                    lin_expr=[
                        SparsePair(
                            ind=[delta_pos_jk, delta_sign_jk], val=[1.0, -D_pos_k[j]]
                        )
                    ],
                    senses="L",
                    rhs=[0.0],
                )

                # if delta[j,k]_neg > 0 => delta[j,k]_sign = 0
                cons.add(
                    names=[f"set_{delta_neg_jk}"],
                    lin_expr=[
                        SparsePair(
                            ind=[delta_neg_jk, delta_sign_jk], val=[1.0, D_neg_k[j]]
                        )
                    ],
                    senses="L",
                    rhs=[D_neg_k[j]],
                )

                # c[j] - delta[j,k]_pos + delta[j,k]_neg = a[j,k]
                cons.add(
                    names=[f"set_dist_{j, k}"],
                    lin_expr=[
                        SparsePair(
                            ind=[c_j, delta_pos_jk, delta_neg_jk], val=[1.0, -1.0, 1.0]
                        )
                    ],
                    senses="E",
                    rhs=[ak[j]],
                )

                # a[j] - delta[j,k]_pos + delta[j,k]_neg = a[j,k]
                # a_j = f"a[{j}]"
                # cons.add(names=[f'set_dist_{j, k}'],
                #          lin_expr=[SparsePair(ind=[a_j, delta_pos_jk, delta_neg_jk], val=[1.0, -1.0, 1.0])],
                #          senses="E",
                #          rhs=[ak[j]])

        # update indices
        indices.append_variables(variable_args)
        indices.counts["nogood"] = start_index + n_points
        return True

    def check_solution(self):
        """checks the solution of a MIP is it exists"""

        # check indices
        self.__check_rep__()
        indices = self.indices
        names = indices.names

        # check solution
        assert has_solution(self.mip)
        sol = self.mip.solution

        # check that optimal values of variables are within bounds
        for name, lb, ub in zip(
            names.values(), indices.lb.values(), indices.ub.values()
        ):
            value = sol.get_values(name)
            assert np.greater_equal(value, lb).all()
            assert np.greater_equal(ub, value).all()

        # get actions
        a = np.array(sol.get_values(names["a"]))
        a_pos = np.array(sol.get_values(names["a_pos"]))
        a_neg = np.array(sol.get_values(names["a_neg"]))

        # check that a_pos and a_neg are capturing absolute value
        # try:
        #     assert np.isclose(a, a_pos - a_neg).all()
        #     assert np.isclose(np.abs(a), a_pos + a_neg).all()
        # except AssertionError as e:
        #     print(f"AssertionError", e)
        #     print(f'a: {str(a)}')
        #     print(f'a_pos: {str(a_pos)}')
        #     print(f'a_neg: {str(a_neg)}')
        assert np.isclose(a, a_pos - a_neg).all()
        assert np.isclose(np.abs(a), a_pos + a_neg).all()
        assert np.all(a_neg[np.greater(a_pos, 0)] == 0)
        assert np.all(a_pos[np.greater(a_neg, 0)] == 0)

        # check that total change is inline
        c = np.array(sol.get_values(names["c"]))
        # assert np.array_equal(a, c)

        num_sols = len(self.feasible_actions)
        d = len(self.actionable_indices)
        if num_sols > 0:
            s = np.array(sol.get_values(names["delta_sign"]))
            delta_pos = np.array(sol.get_values(names["delta_pos"]))
            delta_neg = np.array(sol.get_values(names["delta_neg"]))

            # check distance constraints
            s = np.reshape(s, (num_sols, d))
            delta_pos = np.reshape(delta_pos, (num_sols, d))
            delta_neg = np.reshape(delta_neg, (num_sols, d))
            delta_pos_off = np.isclose(delta_pos, 0.0)
            delta_neg_off = np.isclose(delta_neg, 0.0)
            delta_pos_on = np.logical_not(delta_pos_off)
            delta_neg_on = np.logical_not(delta_neg_off)
            # check that we are at least eps_min from each previous solution
            assert np.greater_equal(
                np.sum(delta_pos + delta_neg, axis=1), self.settings["eps_min"]
            ).all()
            # check delta_pos[j] and delta_neg[j] are not both positive for each j
            assert (
                np.logical_and(delta_pos_on, delta_neg_on).any() == False
            ), "delta_pos or delta_neg can be positive, not both"
            # check that s[j] = 1 if delta_pos[j] = 1, and s[j] = 0 if delta_neg[j] = 0
            assert np.isclose(
                s[delta_pos_on], 1.0
            ).all(), "delta_pos > 0 should imply delta_sign = 1"
            assert np.isclose(
                s[delta_neg_on], 0.0
            ).all(), "delta_neg > 0 should imply delta_sign = 0"

        # todo: add checks for constraints
        return True

    @property
    def action(self):
        # return self.mip.solution.get_values(self.indices.names['a'])
        return self.mip.solution.get_values(self.indices.names["c"])

    @property
    def solution_info(self):
        """returns information associated with the current best solution for the mip"""
        return get_mip_stats(self.mip)

    @staticmethod
    def set_solver_parameters(cpx, print_flag):
        p = cpx.parameters
        p.emphasis.numerical.set(1)
        p.mip.tolerances.integrality.set(1e-7)
        p.mip.tolerances.mipgap.set(0.0)
        p.mip.tolerances.absmipgap.set(0.0)
        p.mip.display.set(print_flag)
        p.simplex.display.set(print_flag)
        p.paramdisplay.set(print_flag)
        if not print_flag:
            cpx.set_results_stream(None)
            cpx.set_log_stream(None)
            cpx.set_error_stream(None)
            cpx.set_warning_stream(None)

        return cpx

    def __check_rep__(self):
        assert self.indices.check_cpx(self.mip)
        AU = np.unique(self._feasible_actions, axis=0)
        assert len(self._feasible_actions) == AU.shape[0], "solutions are not unique"
        assert np.all(
            AU == 0, axis=1
        ).any(), "feasible actions do not contain null action"
