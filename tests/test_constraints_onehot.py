"""
Test Strategy
todo
"""
import pytest
import pandas as pd
import numpy as np
from itertools import product
from reachml.paths import tests_dir
from reachml import *
from reachml.reachable_set import EnumeratedReachableSet
from reachml.constraints.onehot import OneHotEncoding
from reachml.utils import SUPPORTED_SOLVERS


@pytest.fixture(params=["credit_onehot", "credit_onehot_all_immutable"])
def test_case(request):
    if "credit_onehot" in request.param:
        X = pd.read_csv(tests_dir / "credit.csv")
        names = ["Age_lt_25", "Age_in_25_to_40", "Age_in_40_to_59", "Age_geq_60"]

    A = ActionSet(X)
    if "all_immutable" in request.param:
        A[names].actionable = False

    out = {"X": X, "A": A, "names": names}

    return out


@pytest.fixture(params=["max", "equal"])
def limit_type(request):
    return request.param


@pytest.fixture(params=[0, 1, 2])
def limit_value(request):
    return request.param


def test_initialization(test_case, limit_value, limit_type):
    params = {
        "limit": limit_value,
        "limit_type": limit_type,
    }
    cons = OneHotEncoding(names=test_case["names"], **params)
    print(str(cons))
    assert set(cons.parameters) == set(params.keys())
    for name, value in params.items():
        assert cons.__getattribute__(name) == value

    # add
    A = test_case["A"]
    const_id = A.constraints.add(cons)
    assert cons.id == const_id

    # adding it again raises an error
    with pytest.raises(AssertionError):
        A.constraints.add(cons)

    dropped = A.constraints.drop(const_id)
    assert dropped


def test_equals(test_case, limit_value, limit_type):
    params = {"limit": limit_value, "limit_type": limit_type}
    cons = OneHotEncoding(names=test_case["names"], **params)

    same_names = list(test_case["names"])
    same_names.reverse()
    same_cons = OneHotEncoding(names=same_names, **params)
    assert cons == same_cons

    diff_names = test_case["names"]
    if len(diff_names) >= 2:
        diff_names.pop(0)
        diff_cons = OneHotEncoding(names=diff_names, **params)
        assert not (cons != diff_cons)

@pytest.mark.parametrize("solver", SUPPORTED_SOLVERS)
def test_enumeration_with_onehot_constraints(limit_type, limit_value, solver):
    X = pd.DataFrame(
        columns=["x0", "x1", "x2"],
        data=[
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
    )

    A = ActionSet(X)
    cons = OneHotEncoding(
        names=X.columns.tolist(), limit=limit_value, limit_type=limit_type
    )
    A.constraints.add(constraint=cons)
    for idx, x in enumerate(X.values):
        if cons.check_feasibility(x):
            reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver=solver)
            reachable_set.generate()
            assert reachable_set.complete, "reachable set is not complete"
            assert all([cons.check_feasibility(x) for x in reachable_set.X]), "violates constraint feasibility"
        else:
            with pytest.raises(AssertionError):
                EnumeratedReachableSet(x=x, action_set=A, solver=solver).generate()
            with pytest.raises(AssertionError):
                cons.adapt(x)


@pytest.mark.parametrize("solver", SUPPORTED_SOLVERS)
def test_enumeration_with_onehot_constraints_overlapping(solver):
    """
    assume that x1, x2 = (0, 1) is infeasible
    this is similar to x2 = 1 -> x1 = 1 so
    :return:
    """
    X = pd.DataFrame(
        columns=["x0", "x1", "x2", "x3"],
        data=[  #
            [1, 0, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 1],
            #
            [0, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
        ],
    )

    A = ActionSet(X)
    con_A = OneHotEncoding(names=["x0", "x1"], limit=1, limit_type="equal")
    con_B = OneHotEncoding(names=["x1", "x2", "x3"], limit=2, limit_type="max")
    A.constraints.add(constraint=con_A)
    A.constraints.add(constraint=con_B)
    SA = A.get_feature_indices(["x0", "x1"])
    SB = A.get_feature_indices(["x1", "x2", "x3"])
    feature_indices = list(set(SA + SB))
    assert feature_indices in A.partition
    for idx, x in enumerate(X.values):
        reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver=solver)
        reachable_set.generate()
        assert np.equal(np.sum(reachable_set.X[:, SA], axis=1), 1.0).all(), "Doesn't pass EQUAL check"
        assert np.less_equal(
            np.sum(reachable_set.X[:, SB], axis=1), 2.0
        ).all(), "Doesn't pass LEQ check"


@pytest.mark.parametrize("limit_value,limit_type,solver", list(product([1], ["equal", "max"], SUPPORTED_SOLVERS)))
def test_enumeration_with_onehot_constraints_immutable(limit_type, limit_value, solver):
    immutable_idx = [0]
    X = pd.DataFrame(columns=["x0", "x1", "x2"], data=[[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    A = ActionSet(X)
    constraint = OneHotEncoding(
        names=["x0", "x1", "x2"], limit=limit_value, limit_type=limit_type
    )
    A.constraints.add(constraint=constraint)
    A[constraint.names].actionable = True
    A[immutable_idx].actionable = False
    is_feasible = lambda z: constraint.check_feasibility(z) and np.all(
        z[immutable_idx] == x[immutable_idx]
    )
    for idx, x in enumerate(X.values):
        if is_feasible(x):
            reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver=solver)
            reachable_set.generate()
            assert reachable_set.complete
            assert all([is_feasible(x) for x in reachable_set.X])
        else:
            with pytest.raises(AssertionError):
                EnumeratedReachableSet(x=x, action_set=A, solver=solver)


@pytest.mark.parametrize("solver", SUPPORTED_SOLVERS)
def test_enumeration_with_onehot_constraints_monotonic(solver):
    X = pd.DataFrame(
        columns=["x0", "x1", "x2"], data=[[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]
    )

    A = ActionSet(X)
    A.constraints.add(
        OneHotEncoding(names=X.columns.tolist(), limit=1, limit_type="max")
    )
    increasing = [0]
    decreasing = [1]
    A[increasing].step_direction = 1
    A[decreasing].step_direction = -1
    expected_reachable_sets = {
        (0, 0, 0): [[0, 0, 0], [1, 0, 0], [0, 0, 1]],
        (0, 0, 1): [[0, 0, 0], [1, 0, 0], [0, 0, 1]],
        (0, 1, 0): [[0, 1, 0], [0, 0, 0], [1, 0, 0], [0, 0, 1]],
        (1, 0, 0): [[1, 0, 0]],
    }

    for idx, x in enumerate(X.values):
        reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver=solver)
        reachable_set.generate()
        assert reachable_set.complete
        expected_set = np.array(expected_reachable_sets.get(tuple(x)))
        assert expected_set in reachable_set
        assert len(reachable_set) == len(reachable_set)
        if x[increasing] == 1:
            assert len(reachable_set) == 1
        if x[decreasing] == 0:
            assert np.all(reachable_set.X[:, decreasing] == 0)


def test_enumeration_with_onehot_constraints_dropping_scip_and_cplex(test_case, limit_value, limit_type):
    try:
        assert "scip" in SUPPORTED_SOLVERS and "cplex" in SUPPORTED_SOLVERS
    except AssertionError:
        print("SCIP and CPLEX are not both supported solvers.")
        pytest.skip()

    X = pd.DataFrame(
        columns=["x0", "x1", "x2"],
        data=[
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
    )

    A = ActionSet(X)
    cons = OneHotEncoding(
        names=X.columns.tolist(), limit=limit_value, limit_type=limit_type
    )
    A.constraints.add(constraint=cons)
    for idx, x in enumerate(X.values):
        if cons.check_feasibility(x):
            scip_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver="scip")
            scip_reachable_set.generate()
            assert scip_reachable_set.complete, "reachable set is not complete"
            assert all([cons.check_feasibility(x) for x in scip_reachable_set.X]), "violates constraint feasibility"
            cplex_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver="cplex")
            cplex_reachable_set.generate()
            assert cplex_reachable_set.complete, "reachable set is not complete"
            assert all([cons.check_feasibility(x) for x in cplex_reachable_set.X]), "violates constraint feasibility"
            assert len(scip_reachable_set) == len(cplex_reachable_set), "reachable sets differ between solvers"
            assert all([x in cplex_reachable_set for x in scip_reachable_set.X]), "reachable sets differ between solvers"

        else:
            with pytest.raises(AssertionError):
                EnumeratedReachableSet(x=x, action_set=A, solver="scip").generate()
                EnumeratedReachableSet(x=x, action_set=A, solver="cplex").generate()
            with pytest.raises(AssertionError):
                cons.adapt(x)

#### to be implemented ####
# def test_check_compatability(dataset_actionset, limit_value, limit_type):
#     # todo: throws error if variables are not binary
#     pytest.skip()
#     assert True
#
# def test_adapt(dataset_actionset, limit_value, limit_type):
#     # todo: check that you call adapt
#     pytest.skip()
#     assert True
#
# def test_adapt_point_does_not_exist(dataset_actionset, limit_value, limit_type):
#     # todo: adapting to x that violates onehotconstraint should throw error
#     pytest.skip()
#     assert True
#
# def test_add_to_cpx(dataset_actionset, limit_value, limit_type):
#     # todo: add to cplex, then check that we have added the right constraint
#     # todo: check for extra constraint
#     pytest.skip()
#     assert True

if __name__ == "__main__":
    pytest.main()
