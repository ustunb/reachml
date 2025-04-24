from functools import reduce
from itertools import chain

import numpy as np
from cplex import Cplex, SparsePair

from .action_set import ActionSet
from .cplex_utils import (
    CplexGroupedVariableIndices,
    combine,
    get_cpx_variable_args,
    get_cpx_variable_types,
    get_mip_stats,
    has_solution,
)


class BaseMIP:
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
        self._action_set = action_set

        # attach initial point
        assert isinstance(x, (list, np.ndarray))
        x = np.array(x, dtype=np.float64).flatten()
        assert len(x) == len(self._action_set)
        self._x = x

        # set actionable indices
        self.actionable_indices = list(range(len(self._x)))

        # parse remaining settings
        self.settings = dict(BaseMIP.SETTINGS) | kwargs
        self.print_flag = print_flag

        # build base MIP
        cpx, indices = self.build_mip()

        # add non-separable constraints
        for con in action_set.constraints:
            mip, indices = con.add_to_cpx(cpx=cpx, indices=indices, x=self.x)

        # set MIP parameters
        cpx = self.set_solver_parameters(cpx, print_flag=self.print_flag)
        self.mip, self.indices = cpx, indices

    @property
    def action_set(self):
        return self._action_set

    @property
    def x(self):
        return self._x

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
        a_lb = self.action_set.get_bounds(self.x, bound_type="lb")
        a_ub = self.action_set.get_bounds(self.x, bound_type="ub")
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


class EnumeratorMIP(BaseMIP):
    def __init__(self, action_set, x, print_flag=False, **kwargs):
        """
        :param action_set:
        :param x:
        :param print_flag:
        :param kwargs:
        """
        super().__init__(action_set, x, print_flag=print_flag, **kwargs)
        self.n_sols = 0

    @property
    def current_solution(self):
        """returns the current best solution for the mip"""
        return self.mip.solution.get_values(self.indices.names["c"])

    @property
    def solution_info(self):
        """returns information associated with the current best solution for the mip"""
        return get_mip_stats(self.mip)

    @property
    def solution_exists(self):
        """returns true if the MIP has a solution"""
        return self.solution_info["has_solution"]

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

        # Basic MIP Parameters
        # a_ub = indices.ub["a"]
        # a_lb = indices.lb["a"]
        # a_types = indices.types["a"]
        a_ub = self.indices.ub["c"]
        a_lb = self.indices.lb["c"]
        a_types = self.indices.types["c"]
        d = len(self.actionable_indices)

        # distance-based parameters
        n_points = len(actions)
        A_nogood = np.vstack(actions)
        D_pos = a_ub - A_nogood
        D_neg = A_nogood - a_lb

        # get number of existing parameters
        start_index = self.indices.counts.get("nogood", 0)
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
                names=["sum_abs_dist_val"],
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
        self.indices.append_variables(variable_args)
        self.indices.counts["nogood"] = start_index + n_points

        # update number of solutions
        self.n_sols += n_points

        return True

    def check_solution(self):
        """checks the solution of a MIP is it exists"""

        # check indices
        assert self.indices.check_cpx(self.mip)
        names = self.indices.names

        # check solution
        assert has_solution(self.mip)
        sol = self.mip.solution

        # check that optimal values of variables are within bounds
        for name, lb, ub in zip(
            names.values(), self.indices.lb.values(), self.indices.ub.values()
        ):
            # value = sol.get_values(name)
            value = np.round(sol.get_values(name), 10)
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

        d = len(self.actionable_indices)
        if self.n_sols > 0:
            s = np.array(sol.get_values(names["delta_sign"]))
            delta_pos = np.array(sol.get_values(names["delta_pos"]))
            delta_neg = np.array(sol.get_values(names["delta_neg"]))

            # check distance constraints
            s = np.reshape(s, (self.n_sols, d))
            delta_pos = np.reshape(delta_pos, (self.n_sols, d))
            delta_neg = np.reshape(delta_neg, (self.n_sols, d))
            delta_pos_off = np.isclose(delta_pos, 0.0)
            delta_neg_off = np.isclose(delta_neg, 0.0)
            delta_pos_on = np.logical_not(delta_pos_off)
            delta_neg_on = np.logical_not(delta_neg_off)
            # check that we are at least eps_min from each previous solution
            assert np.greater_equal(
                np.sum(delta_pos + delta_neg, axis=1), self.settings["eps_min"]
            ).all()
            # check delta_pos[j] and delta_neg[j] are not both positive for each j
            assert np.logical_and(delta_pos_on, delta_neg_on).any() == False, (
                "delta_pos or delta_neg can be positive, not both"
            )
            # check that s[j] = 1 if delta_pos[j] = 1, and s[j] = 0 if delta_neg[j] = 0
            assert np.isclose(s[delta_pos_on], 1.0).all(), (
                "delta_pos > 0 should imply delta_sign = 1"
            )
            assert np.isclose(s[delta_neg_on], 0.0).all(), (
                "delta_neg > 0 should imply delta_sign = 0"
            )

        # todo: add checks for constraints
        return True
