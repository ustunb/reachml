"""
Test Strategy
todo
"""
import pytest
import pandas as pd
import numpy as np
from reachml import *
from reachml.reachable_set import EnumeratedReachableSet
from reachml.constraints.ordinal import OrdinalEncoding
from reachml.utils import SUPPORTED_SOLVERS

sortrows = lambda v: v[np.lexsort(v.T, axis=0), :]


@pytest.fixture(params=[True, False])
def exhaustive(request):
    return request.param


@pytest.fixture(params=[0, 1, -1])
def step_direction(request):
    return request.param


def get_test_case(exhaustive, step_direction):
    X = pd.DataFrame(
        columns=["x0", "x1", "x2"], data=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )

    valid_values = np.sort(X.values, axis=0)

    all_values = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
            [1, 0, 1],
            [0, 1, 1],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )

    if step_direction == 0:
        expected_sets = {
            (0, 0, 0): [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            (1, 0, 0): [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            (0, 1, 0): [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            (0, 0, 1): [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        }
    elif step_direction > 0:
        expected_sets = {
            (0, 0, 0): [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            (1, 0, 0): [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            (0, 1, 0): [[0, 1, 0], [0, 0, 1]],
            (0, 0, 1): [[0, 0, 1]],
        }
    elif step_direction < 0:
        expected_sets = {
            (0, 0, 0): [[0, 0, 0]],
            (1, 0, 0): [[0, 0, 0], [1, 0, 0]],
            (0, 1, 0): [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            (0, 0, 1): [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        }

    expected_sets = {k: np.array(v) for k, v in expected_sets.items()}
    if exhaustive:
        exhaustive_sets = {}
        for k, v in expected_sets.items():
            if np.sum(k) == 1:
                to_keep = np.sum(v, axis=1) == 1
                exhaustive_sets[k] = v[to_keep, :]
        expected_sets = exhaustive_sets

    expected_sets = {k: sortrows(v) for k, v in expected_sets.items()}
    out = {
        "X": X,
        "all_values": all_values,
        "valid_values": valid_values,
        "expected_sets": expected_sets,
    }

    return out


def test_initialization(exhaustive, step_direction):
    test_case = get_test_case(exhaustive, step_direction)
    X = test_case["X"]
    names = X.columns.tolist()
    params = {"exhaustive": exhaustive, "step_direction": step_direction}
    cons = OrdinalEncoding(names=names, **params)
    print(str(cons))
    assert set(params.keys()).issubset(set(cons.parameters))
    for name, value in params.items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(cons.__getattribute__(name), value)

    # add
    A = ActionSet(X)
    const_id = A.constraints.add(cons)
    assert cons.id == const_id

    # adding it again raises an error
    with pytest.raises(AssertionError):
        A.constraints.add(cons)

    # drop
    dropped = A.constraints.drop(const_id)
    assert dropped

@pytest.mark.parametrize("solver", SUPPORTED_SOLVERS)
def test_enumeration_with_ordinal_constraints(exhaustive, step_direction, solver):
    test_case = get_test_case(exhaustive, step_direction)
    X = test_case["X"]
    valid_values = test_case["valid_values"]
    all_values = test_case["all_values"]
    names = X.columns.tolist()

    # create constraint
    constraint = OrdinalEncoding(
        names=names, exhaustive=exhaustive, step_direction=step_direction
    )
    A = ActionSet(X)
    A.constraints.add(constraint=constraint)

    for idx, x in enumerate(all_values):
        expected_set = test_case["expected_sets"].get(tuple(x))
        if constraint.check_encoding(x):
            reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver=solver)
            reachable_set.generate()
            assert reachable_set.complete
            assert np.array_equal(sortrows(reachable_set.X), expected_set)
        else:
            with pytest.raises(AssertionError):
                EnumeratedReachableSet(x=x, action_set=A, solver=solver).generate()


def test_enumeration_with_ordinal_constraints_scip_and_cplex(exhaustive, step_direction):
    try:
        assert "scip" in SUPPORTED_SOLVERS and "cplex" in SUPPORTED_SOLVERS
    except AssertionError:
        print("SCIP and CPLEX are not both supported solvers.")
        pytest.skip()


    test_case = get_test_case(exhaustive, step_direction)
    X = test_case["X"]
    valid_values = test_case["valid_values"]
    all_values = test_case["all_values"]
    names = X.columns.tolist()

    # create constraint
    constraint = OrdinalEncoding(
        names=names, exhaustive=exhaustive, step_direction=step_direction
    )
    A = ActionSet(X)
    A.constraints.add(constraint=constraint)

    for idx, x in enumerate(all_values):
        expected_set = test_case["expected_sets"].get(tuple(x))
        if constraint.check_encoding(x):
            reachable_set_scip = EnumeratedReachableSet(
                x=x, action_set=A, solver="scip"
            )
            reachable_set_scip.generate()
            assert reachable_set_scip.complete
            assert np.array_equal(sortrows(reachable_set_scip.X), expected_set)

            reachable_set_cplex = EnumeratedReachableSet(
                x=x, action_set=A, solver="cplex"
            )
            reachable_set_cplex.generate()
            assert reachable_set_cplex.complete
            assert np.array_equal(sortrows(reachable_set_cplex.X), expected_set)
            assert reachable_set_scip == reachable_set_cplex
        else:
            with pytest.raises(AssertionError):
                EnumeratedReachableSet(x=x, action_set=A, solver="scip").generate()
            with pytest.raises(AssertionError):
                EnumeratedReachableSet(x=x, action_set=A, solver="cplex").generate()
