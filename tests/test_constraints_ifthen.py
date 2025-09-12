"""
Test Strategy
todo
"""

import pytest
import pandas as pd
from reachml import *
from reachml.constraints.ifthen import *
from reachml.utils import SUPPORTED_SOLVERS


def test_initialization():
    pass
    # params = {'limit': limit_value, 'limit_type': limit_type,}
    # cons = OneHotEncoding(names = dataset_actionset['names'], **params)
    # print(str(cons))
    # assert set(cons.parameters) == set(params.keys())
    # for name, value in params.items():
    #     assert cons.__getattribute__(name) == value


def test_equals():
    pass
    # params = {'limit': limit_value, 'limit_type': limit_type}
    # cons = OneHotEncoding(names = dataset_actionset['names'], **params)
    #
    # same_names = list(dataset_actionset['names'])
    # same_names.reverse()
    # same_cons = OneHotEncoding(names = same_names, **params)
    # assert cons == same_cons
    #
    # diff_names = dataset_actionset['names']
    # if len(diff_names) >= 2:
    #     diff_names.pop(0)
    #     diff_cons = OneHotEncoding(names = diff_names, **params)
    #     assert not (cons != diff_cons)

@pytest.mark.parametrize("solver", SUPPORTED_SOLVERS)
def test_add_ifthen_values_at_bounds(solver):
    X = pd.DataFrame(columns=["x0", "x1"], data=[[1.0, 0.0], [0.0, 1.0]])
    A = ActionSet(X)
    A.step_direction = 1
    A.constraints.add(
        constraint=IfThenConstraint(
            if_condition=Condition("x0", "G", 1),
            then_condition=Condition("x1", "E", 1),
        )
    )
    x = X.iloc[0].values  # x = [0.0, 1.0]
    enumerator = ReachableSetEnumerator(x=x, action_set=A, solver=solver)
    enumerator.enumerate()

    # a_values = mip.solution.get_values(indices.names['a'])
    # self.assertEqual(a_values[0], 1.0)
    # self.assertEqual(a_values[1], 0.0)

@pytest.mark.parametrize("solver", SUPPORTED_SOLVERS)
def test_add_ifthen_unseen_then_value(solver):
    X = pd.DataFrame(columns=["x0", "x1"], data=[[0.0, 0.0], [1.0, 20.0], [0.0, 11.0]])
    A = ActionSet(X)
    A.constraints.add(
        constraint=IfThenConstraint(
            if_condition=Condition("x0", "G", 1.0),
            then_condition=Condition("x1", "E", 2.0),
        )
    )

    x = X.iloc[0].values  # [0, 10]
    enumerator = ReachableSetEnumerator(x=x, action_set=A, solver=solver)
    enumerator.enumerate()
    # self.assertEqual(a_values[0], 1.0)
    # self.assertEqual(a_values[1], 2.0)

@pytest.mark.parametrize("solver", SUPPORTED_SOLVERS)
def test_add_ifthen_unseen_ifvalue(solver):
    X = pd.DataFrame(columns=["x0", "x1"], data=[[0.0, 0.0], [0.0, 11.0]])
    A = ActionSet(X)
    A["x0"].lb = 0
    A["x0"].ub = 1
    A["x0"].variable_type = bool
    A.constraints.add(
        constraint=IfThenConstraint(
            if_condition=Condition("x0", "G", 1.0),
            then_condition=Condition("x1", "E", 1.0),
        )
    )

    x = X.iloc[1].values  # x = [0, 11]
    enumerator = ReachableSetEnumerator(x=x, action_set=A, solver=solver)
    enumerator.enumerate()
    # self.assertEqual(a_values[0], 1.0)
    # self.assertEqual(a_values[1], 1.0)

@pytest.mark.parametrize("solver", SUPPORTED_SOLVERS)
def test_add_ifthen_overlapping(solver):
    X = pd.DataFrame(
        columns=["x0", "x1", "x2", "x3"],
        data=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 11.0]],
    )

    A = ActionSet(X)
    A.constraints.add(
        constraint=IfThenConstraint(
            if_condition=Condition("x1", "G", 1),
            then_condition=Condition("x3", "E", 1),
        )
    )

    A.constraints.add(
        constraint=IfThenConstraint(
            if_condition=Condition("x2", "G", 1), then_condition=Condition("x3", "E", 1)
        )
    )

    # x = [1.0, 0.0, 0.0, 10.0]
    x = X.iloc[0].values
    enumerator = ReachableSetEnumerator(x=x, action_set=A, solver=solver)
    enumerator.enumerate()

    # a_values = mip.solution.get_values(indices.names['a'])
    # self.assertEqual(a_values[1], 1.0)
    # self.assertEqual(a_values[3], 1.0)

@pytest.mark.parametrize("solver", SUPPORTED_SOLVERS)
def test_returns_solution_for_if_then_constraint_for_multiple_constraints_increase_second_constraint(solver):
    X = pd.DataFrame(
        columns=["x0", "x1", "x2", "x3"],
        data=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 11.0]],
    )

    A = ActionSet(X)
    A.constraints.add(
        constraint=IfThenConstraint(
            if_condition=Condition("x1", "G", 1),
            then_condition=Condition("x3", "E", 1),
        )
    )

    A.constraints.add(
        constraint=IfThenConstraint(
            if_condition=Condition("x2", "G", 1), then_condition=Condition("x3", "E", 1)
        )
    )

    # x = [1.0, 0.0, 0.0, 10.0]
    x = X.iloc[0].values
    enumerator = ReachableSetEnumerator(x=x, action_set=A, solver=solver)
    enumerator.enumerate()

    # force a_0 to increase by 1
    # mip, indices = executor.build_feasibility_mip(A, x)
    # mip.linear_constraints.add(
    #         names=[f'increase_a[2]'],
    #         lin_expr=[cplex.SparsePair(ind=['a[2]'], val=[1.0])],
    #         senses="E",
    #         rhs=[1.0]
    #         )
    # check_cpx_formulation(mip, indices)
    # mip.solve()
    # executor.check_solution(mip, indices, A, x)
    # a_values = mip.solution.get_values(indices.names['a'])
    # self.assertEqual(a_values[2], 1.0)
    # self.assertEqual(a_values[3], 1.0)

if __name__ == "__main__":
    pytest.main()
