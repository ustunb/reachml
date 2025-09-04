"""
Test Strategy
target_mutability: True, False
target_size: [1, 2]
force: [True, False]
change_violates_target_bound: [True, False]
change_violates_target_sign: [True, False]
"""
import pytest
import pandas as pd
import numpy as np
from reachml import *
from reachml.reachable_set import EnumeratedReachableSet
from reachml.constraints.switch import MutabilitySwitch
from reachml.utils import SUPPORTED_SOLVERS

sortrows = lambda v: v[np.lexsort(v.T, axis=0), :]


@pytest.fixture(params=[0, 1])
def on_value(request):
    return request.param


@pytest.fixture(params=[True, False])
def force(request):
    return request.param


def get_test_case(on_value, force):
    X = pd.DataFrame(
        columns=["x0", "x1", "x2"], data=[[0, 0, -1], [0, 0, 0], [1, 1, 1]]
    )

    all_values = np.array(
        [
            [0, 0, -1],
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, -1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, -1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, -1],
            [1, 0, 0],
            [1, 1, 1],
        ]
    )

    expected_sets = {}
    if on_value == 0 and force == True:
        expected_sets = {
            (0, 0, -1): [],  # infeasible because x[0] = 0 -> x[1] ≠ 0 due to force
            (0, 0, 0): [
                [0, 0, 0],  # current point
                [1, 1, 0],  # change x[1]
                [1, 0, 1],
                [1, 0, -1],  # change x[2]
                [1, 1, -1],
                [1, 1, 1],  # change x[1] and x[2]
            ],
            (1, 1, 1): [
                [1, 1, 1],  # current point
                [1, 0, 1],  # change x[1]
                [1, 1, 0],
                [1, 1, -1],  # change x[2]
                [1, 0, 0],
                [1, 0, -1],  # change x[1] and x[2]
                [0, 1, 1],  # turn on - no change,
            ],
        }
    elif on_value == 0 and force == False:
        expected_sets = {
            (0, 0, -1): [],  # infeasible because x[0] = 0 -> x[2] = 0
            #
            (0, 0, 0): [
                [0, 0, 0],  # current point
                [1, 1, 0],  # change x[1]
                [1, 0, -1],
                [1, 0, 0],
                [1, 0, 1],  # change x[2]
                [1, 1, -1],
                [1, 1, 1],  # change x[1] and x[2]
            ],
            #
            (1, 1, 1): [
                [1, 1, 1],  # current point
                [0, 1, 1],  # turning x[0] on -> no changes
                [1, 0, 1],  # change x[1]
                [1, 1, 0],
                [1, 1, -1],  # change x[2]
                [1, 0, 0],
                [1, 0, -1],  # change x[1] and x[2]
            ],
        }
    elif on_value == 1 and force == True:
        expected_sets = {
            (0, 0, -1): [],  # infeasible because x[0] = 0 -> x[1] ≠ 0
            (0, 0, 0): [],  # infeasible because x[0] = 0 -> x[1] ≠ 0, x[2] ≠ 0
            (1, 1, 1): [],  # infeasible because x[0] = 1 -> x[1] = 0, x[2] = 0
        }
    elif on_value == 1 and force == False:
        expected_sets = {
            (0, 0, -1): [
                [0, 0, -1],  # current point
                [1, 0, -1],  # turning x[0] = 1 -> no changes
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, -1],
                [0, 1, 0],
                [0, 1, 1],  # keeping x[0] = 0 allows changes
            ],
            (0, 0, 0): [
                [0, 0, 0],  # current point
                [1, 0, 0],  # turning x[0] = 1 -> no changes
                [0, 0, -1],
                [0, 0, 1],
                [0, 1, -1],
                [0, 1, 0],
                [0, 1, 1],  # keeping x[0] = 0 allows changes
            ],
            (1, 1, 1): [],  # infeasible because x[0] = 1 -> x[1] = 0, x[2] = 0
        }

    out = {
        "X": X,
        "A": ActionSet(X),
        "all_values": sortrows(all_values),
        "expected_sets": {
            tuple(k): np.array(v) if len(v) > 0 else np.array(v)
            for k, v in expected_sets.items()
        },
    }

    return out


def test_initialization(on_value, force):
    test_case = get_test_case(on_value, force)
    X = test_case["X"]
    A = test_case["A"]
    names = X.columns.tolist()
    params = {
        "switch": names[0],
        "targets": names[1:],
        "on_value": on_value,
        "force_change_when_off": force,
    }
    cons = MutabilitySwitch(**params)
    print(str(cons))
    assert set(cons.parameters) == set(params.keys())
    for name, value in params.items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(cons.__getattribute__(name), value)

    # add
    const_id = A.constraints.add(cons)
    assert cons.id == const_id

    # adding it again raises an error
    with pytest.raises(AssertionError):
        A.constraints.add(cons)

    dropped = A.constraints.drop(const_id)
    assert dropped

@pytest.mark.parametrize("solver", SUPPORTED_SOLVERS)
def test_enumeration_with_switch_constraints(on_value, force, solver):
    test_case = get_test_case(on_value, force)
    X = test_case["X"]
    A = test_case["A"]
    print(A)
    names = X.columns.tolist()
    cons = MutabilitySwitch(
        switch=names[0],
        targets=names[1:],
        on_value=on_value,
        force_change_when_off=force,
    )
    A.constraints.add(constraint=cons)
    print(cons)
    for idx, x in enumerate(X.values):
        if cons.check_feasibility(x):
            print(f"enumeration for x={x}\n")
            expected_set = test_case["expected_sets"].get(tuple(x))
            reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver=solver)
            reachable_set.generate()
            assert reachable_set.complete
            # print(f'expected_set.X\n{expected_set}\n')
            # print(f'reachable_set.X\n{reachable_set.X}\n')
            # print(f'expected actions \n{np.subtract(expected_set, x)}\n')
            # print(f'enumerated actions\n{reachable_set.actions}\n')
            assert np.array_equal(sortrows(reachable_set.X), sortrows(expected_set))
        else:
            with pytest.raises(AssertionError):
                ReachableSetEnumerator(x=x, action_set=A, solver=solver)


def test_enumeration_with_switch_constraints_scip_and_cplex(on_value, force):
    try:
        assert "scip" in SUPPORTED_SOLVERS and "cplex" in SUPPORTED_SOLVERS
    except AssertionError:
        print("SCIP and CPLEX are not both supported solvers.")
        pytest.skip()

    test_case = get_test_case(on_value, force)
    X = test_case["X"]
    A = test_case["A"]
    print(A)
    names = X.columns.tolist()
    cons = MutabilitySwitch(
        switch=names[0],
        targets=names[1:],
        on_value=on_value,
        force_change_when_off=force,
    )
    A.constraints.add(constraint=cons)
    print(cons)
    for idx, x in enumerate(X.values):
        if cons.check_feasibility(x):
            print(f"enumeration for x={x}\n")
            expected_set = test_case["expected_sets"].get(tuple(x))
            scip_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver="scip")
            scip_reachable_set.generate()
            assert scip_reachable_set.complete
            assert np.array_equal(sortrows(scip_reachable_set.X), sortrows(expected_set))
            cplex_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver="cplex")
            cplex_reachable_set.generate()
            assert cplex_reachable_set.complete
            assert np.array_equal(sortrows(cplex_reachable_set.X), sortrows(expected_set))
            assert np.array_equal(
                sortrows(scip_reachable_set.X), sortrows(cplex_reachable_set.X)
            )
        else:
            with pytest.raises(AssertionError):
                ReachableSetEnumerator(x=x, action_set=A, solver="scip")
            with pytest.raises(AssertionError):
                ReachableSetEnumerator(x=x, action_set=A, solver="cplex")
    


if __name__ == "__main__":
    pytest.main()
