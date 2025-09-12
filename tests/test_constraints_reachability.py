"""
Test Strategy
todo
"""
import pytest
import pandas as pd
import numpy as np
from reachml import *
from reachml.reachable_set import EnumeratedReachableSet
from reachml.paths import tests_dir
from reachml.constraints.reachability import ReachabilityConstraint
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


@pytest.fixture(params=["all"])
def values(request, test_case):
    X = test_case["X"]
    names = test_case["names"]
    values = np.unique(X[names], axis=0)
    return values


@pytest.fixture(params=["vacuous", "fixed"])
def reachability(request, values):
    n = values.shape[0]
    if request.param == "vacuous":
        reachability = np.ones(shape=(n, n))
    elif request.param == "fixed":
        reachability = np.eye(n)
    return reachability


def test_initialization(test_case, values, reachability):
    X = test_case["X"]
    A = test_case["A"]
    names = test_case["names"]
    params = {"names": names, "values": values, "reachability": reachability}
    cons = ReachabilityConstraint(**params)
    print(str(cons))
    assert set(cons.parameters) == set(params.keys())
    for name, value in params.items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(cons.__getattribute__(name), value)

    # add
    const_id = A.constraints.add(cons)
    assert cons.id == const_id

    # drop
    dropped = A.constraints.drop(const_id)
    assert dropped

@pytest.mark.parametrize("solver", SUPPORTED_SOLVERS)
def test_contains(test_case, solver):
    A = test_case["A"]
    x = np.array([0.0, 0.0, 0.0, 0.0])
    reachable_set = EnumeratedReachableSet(A, x, complete=True, values=np.vstack([x, np.eye(4)]), solver=solver)

    y = np.array([2, 2, 2, 2])
    assert y not in reachable_set

    x1 = np.array([0, 0, 0, 0])
    assert x1 in reachable_set

    # x1_list = [0, 0, 0, 0]
    # assert x1_list in reachable_set

    x2 = np.array([1, 0, 0, 0])
    assert np.array([x1, x2]) in reachable_set


def test_equals(test_case, values, reachability):
    X = test_case["X"]
    A = test_case["A"]
    params = {"values": values, "reachability": reachability}
    cons = ReachabilityConstraint(names=test_case["names"], **params)
    same_names = list(test_case["names"])
    same_names.reverse()
    same_cons = ReachabilityConstraint(names=same_names, **params)
    assert cons != same_cons


#### enumeration
@pytest.mark.parametrize("solver", SUPPORTED_SOLVERS)
def test_vacuous_reachability_constraints(dataset_actionset_2d, solver):
    X = dataset_actionset_2d["X"]
    A = dataset_actionset_2d["A"]
    expected_reachable_set = dataset_actionset_2d["R"]
    print(f"X: {X}")
    print(f"A: {A}")

    for x in X.values:
        reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver=solver)
        R = np.array(expected_reachable_set.get(tuple(x)))
        n_expected = R.shape[0]
        reachable_set.generate(max_points=n_expected)
        assert reachable_set.complete
        assert len(reachable_set) == n_expected
        assert R in reachable_set

        # calling generate again should not change anything
        reachable_set.generate()
        assert len(reachable_set) == n_expected and R in reachable_set

        # adding a reachability constraint should not change anything
        const_id = A.constraints.add(
            ReachabilityConstraint(names=X.columns.tolist(), values=X.values)
        )
        constrained_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver=solver)
        constrained_reachable_set.generate()
        assert reachable_set == constrained_reachable_set
        A.constraints.drop(const_id)

@pytest.mark.parametrize("solver", SUPPORTED_SOLVERS)
def test_vacuous_reachability_constraints_with_overlap(solver):
    X = pd.DataFrame(
        columns=["x0", "x1", "x2"],
        data=[
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
    )
    values = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]
    )

    for idx, x in enumerate(X.values):
        # adding a reachability constraint should not change anything
        A = ActionSet(X)
        A.constraints.add(
            ReachabilityConstraint(
                names=["x0", "x1"], values=np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            )
        )
        A.constraints.add(
            ReachabilityConstraint(
                names=["x1", "x2"],
                values=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
                reachability=np.array(
                    [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
                ),
            )
        )
        feature_indices = A.get_feature_indices(["x0", "x1", "x2"])
        assert feature_indices in A.partition

        constrained_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver=solver)
        constrained_reachable_set.generate()
        assert values in constrained_reachable_set
        assert constrained_reachable_set.complete
        assert len(constrained_reachable_set) == values.shape[0]

        # dropping the constraint should lead to the right solution
        A.constraints.clear()
        reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver=solver)
        reachable_set.generate()
        assert values in reachable_set
        assert reachable_set.complete
        assert reachable_set == constrained_reachable_set

@pytest.mark.parametrize("solver", SUPPORTED_SOLVERS)
def test_reachability_constraints_for_fixed_point(dataset_actionset_2d, solver):
    """
    check to make sure that we can remove all other points from current point using reachability matrix
    :return:
    """

    X = dataset_actionset_2d["X"]
    A = dataset_actionset_2d["A"]
    expected_reachable_set = dataset_actionset_2d["R"]
    print(f"X: {X}")
    print(f"A: {A}")

    n_values = X.values.shape[0]
    for idx, x in enumerate(X.values):
        # adding a reachability constraint should not change anything
        reachability_matrix = np.ones(shape=(n_values, n_values))
        reachability_matrix[idx, :] = np.zeros(n_values)
        reachability_matrix[idx, idx] = 1.0
        const_id = A.constraints.add(
            constraint=ReachabilityConstraint(
                names=X.columns.tolist(),
                values=X.values,
                reachability=reachability_matrix,
            )
        )
        constrained_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver=solver)
        constrained_reachable_set.generate()
        assert constrained_reachable_set.complete
        assert len(constrained_reachable_set) == 1
        assert np.array_equal(constrained_reachable_set.X[0, :], x)

        # dropping the constraint should lead to the right solution
        A.constraints.drop(const_id)
        R = np.array(expected_reachable_set.get(tuple(x)))
        n_expected = R.shape[0]
        reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver=solver)
        reachable_set.generate(max_points=n_expected)
        assert reachable_set.complete
        assert len(reachable_set) == n_expected
        assert R in reachable_set

@pytest.mark.parametrize("solver", SUPPORTED_SOLVERS)
def test_enumeration_with_overlapping(solver):
    """
    assume that x1, x2 = (0, 1) is infeasible
    this is similar to x2 = 1 -> x1 = 1 so
    :return:
    """

    X = pd.DataFrame(
        columns=["x0", "x1", "x2"],
        data=[
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
            # [0, 0, 1],
            [0, 1, 1],
            # [1, 0, 1],
            [1, 1, 1],
        ],
    )

    values = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
            # [0, 0, 1],
            [0, 1, 1],
            # [1, 0, 1],
            [1, 1, 1],
        ]
    )

    for idx, x in enumerate(X.values):
        # adding a reachability constraint should not change anything
        A = ActionSet(X)
        A.constraints.add(
            constraint=ReachabilityConstraint(
                names=["x0", "x1"], values=np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            )
        )
        A.constraints.add(
            constraint=ReachabilityConstraint(
                names=["x1", "x2"],
                values=np.array([[0, 0], [1, 0], [1, 1]]),
            )
        )

        feature_indices = A.get_feature_indices(["x0", "x1", "x2"])
        assert feature_indices in A.partition

        constrained_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver=solver)
        constrained_reachable_set.generate()
        assert [1, 0, 1] not in constrained_reachable_set
        assert [0, 0, 1] not in constrained_reachable_set
        assert values in constrained_reachable_set
        assert constrained_reachable_set.complete
        assert len(constrained_reachable_set) == values.shape[0]

        # dropping the constraint should lead to the right solution
        A.constraints.clear()
        reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver=solver)
        reachable_set.generate()
        assert reachable_set.complete
        assert values in reachable_set
        print(reachable_set.X)
        print(constrained_reachable_set.X)
        print(reachable_set.X in constrained_reachable_set)
        assert constrained_reachable_set.X in reachable_set
        assert reachable_set.X not in constrained_reachable_set

@pytest.mark.parametrize("solver", SUPPORTED_SOLVERS)
def test_enumeration_for_reachability_on_onehot_encoding(solver):
    """
    check that we can use reachability constraints to force
    at most 1 of (x1, x2, x3) to be on
    """

    X = pd.DataFrame(
        columns=["x0", "x1", "x2"], data=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )

    values = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]
    )

    for idx, x in enumerate(X.values):
        # adding a reachability constraint should not change anything
        A = ActionSet(X)
        A.constraints.add(
            constraint=ReachabilityConstraint(
                names=["x0", "x1", "x2"],
                values=values,
                reachability=np.array(
                    [
                        [1, 1, 1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1],
                    ]
                ),
            )
        )

        constrained_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver=solver)
        constrained_reachable_set.generate()
        assert constrained_reachable_set.complete
        limits = np.sum(constrained_reachable_set.X, axis=1)
        assert np.all(limits <= 1)

        # dropping the constraint should lead to the right solution
        A.constraints.clear()
        reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver=solver)
        reachable_set.generate()
        assert reachable_set.complete
        assert values in reachable_set
        assert len(reachable_set) == values.shape[0]

@pytest.mark.parametrize("solver", SUPPORTED_SOLVERS)
def test_enumeration_for_reachability_on_ordinal_encoding_with_1step(solver):
    """
    check that we can use reachability constraints to force
    at most 1 of (x1, x2, x3) to be on
    """

    X = pd.DataFrame(
        columns=["x0", "x1", "x2"], data=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )

    values = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    expected_reachable_sets = {
        (0, 0, 0): [[0, 0, 0], [1, 0, 0]],
        (1, 0, 0): [[1, 0, 0], [0, 1, 0]],
        (0, 1, 0): [[0, 1, 0], [0, 0, 1]],
        (0, 0, 1): [[0, 0, 1]],
    }

    # adding a reachability constraint should not change anything
    A = ActionSet(X)
    A.constraints.add(
        constraint=ReachabilityConstraint(
            names=["x0", "x1", "x2"],
            values=values,
            reachability=np.array(
                [
                    [1, 1, 0, 0],
                    [0, 1, 1, 0],
                    [0, 0, 1, 1],
                    [0, 0, 0, 1],
                ]
            ),
        )
    )

    for idx, x in enumerate(X.values):
        reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver=solver)
        reachable_set.generate()
        expected_set = np.array(expected_reachable_sets.get(tuple(x)))
        limits = np.sum(reachable_set.X, axis=1)
        assert np.all(limits <= 1)
        assert reachable_set.complete
        assert expected_set in reachable_set
        assert len(reachable_set) == len(expected_set)

@pytest.mark.parametrize("solver", SUPPORTED_SOLVERS)
def test_enumeration_for_reachability_on_thermometer_encoding(solver):
    """
    check that we can use reachability constraints to enforce therometer encoding
        x1 = 1[x ≥ v1]
        x2 = 1[x ≥ v2]
        x3 = 1[x ≥ v3]
    where v[1] ≤ v[2] ≤ v[3]
    """

    X = pd.DataFrame(
        columns=["x0", "x1", "x2"], data=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]]
    )

    values = np.array(
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

    for idx, x in enumerate(X.values):
        # adding a reachability constraint should not change anything
        A = ActionSet(X)
        A.constraints.add(
            constraint=ReachabilityConstraint(
                names=["x0", "x1", "x2"],
                values=values,
                reachability=np.array(
                    [
                        [1, 1, 1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1],
                    ]
                ),
            )
        )

        constrained_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver=solver)
        constrained_reachable_set.generate()
        assert constrained_reachable_set.complete
        assert X.values in constrained_reachable_set
        assert len(constrained_reachable_set) == X.values.shape[0]

        # dropping the constraint should lead to the right solution
        A.constraints.clear()
        reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver=solver)
        reachable_set.generate()
        assert reachable_set.complete
        assert values in reachable_set


@pytest.mark.parametrize("solver", SUPPORTED_SOLVERS)
def test_enumeration_for_reachability_on_thermometer_encoding_monotonic(solver):
    """
    check that we can use reachability constraints to enforce therometer encoding
        x1 = 1[x ≥ v1]
        x2 = 1[x ≥ v2]
        x3 = 1[x ≥ v3]
    where v[1] ≤ v[2] ≤ v[3]
    """

    X = pd.DataFrame(columns=["x0", "x1", "x2"], data=[[0, 0, 0], [1, 0, 0], [1, 1, 1]])

    values = np.array(
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

    for idx, x in enumerate(X.values):
        # adding a reachability constraint should not change anything
        A = ActionSet(X)
        A.step_direction = 1
        A.constraints.add(
            constraint=ReachabilityConstraint(
                names=["x0", "x1", "x2"],
                values=values,
                reachability=np.array(
                    [
                        [1, 1, 1, 1, 0, 0, 0, 0],
                        [0, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1],
                    ]
                ),
            )
        )

        constrained_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver=solver)
        constrained_reachable_set.generate()
        assert constrained_reachable_set.complete
        assert X.values[idx:,] in constrained_reachable_set

        # dropping the constraint lead to infeasible actions
        A.constraints.clear()
        reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver=solver)
        reachable_set.generate()
        assert reachable_set.complete
        assert len(reachable_set) == 2 ** (len(A) - np.sum(x == 1))

        # infeasible actions from [0, 0, 0]
        if np.array_equal(x, [0, 0, 0]):
            assert [0, 0, 1] in reachable_set
            assert [0, 0, 1] not in constrained_reachable_set
            assert [0, 1, 0] in reachable_set
            assert [0, 1, 0] not in constrained_reachable_set
            assert [1, 0, 1] in reachable_set
            assert [1, 0, 1] not in constrained_reachable_set

        # infeasible actions from [1, 0, 0]
        if np.array_equal(x, [1, 0, 0]):
            assert [1, 0, 1] in reachable_set
            assert [1, 0, 1] not in constrained_reachable_set


@pytest.mark.parametrize("solver", SUPPORTED_SOLVERS)
def test_enumeration_in_1d(solver):
    pytest.skip()
    # we do not currently allow for 1D reachability constraints
    # not implemenging in SCIP and CPLEX comparison, since we are skipping

    X = pd.DataFrame(
        columns=["has_phd", "percent_monthly_savings"],
        data=[[0, 0.0], [1, 0.5], [0, 0.7], [1, 1.0]],
    )

    A = ActionSet(X)
    A["has_phd"].step_direction = 1
    A.constraints.add(
        constraint=ReachabilityConstraint(
            names=["percent_monthly_savings"],
            values=[[0.0], [0.5], [0.7], [1.0]],
            reachability=np.array(
                [[1, 1, 1, 1], [0, 1, 0, 1], [1, 1, 1, 1], [0, 0, 0, 1]]
            ),
        )
    )

    #
    x = X.iloc[0].values
    reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver=solver)
    reachable_set.generate()
    assert reachable_set.complete
    R = reachable_set.X
    assert np.isin(X.values[:, 0], [0, 1]).all()
    assert np.isin(X.values[:, 1], [0.0, 0.5, 0.7, 1.0]).all()

def test_contains_scip_and_cplex(test_case):
    try:
        assert "scip" in SUPPORTED_SOLVERS and "cplex" in SUPPORTED_SOLVERS
    except AssertionError:
        print("SCIP and CPLEX are not both supported solvers.")
        pytest.skip()

    A = test_case["A"]
    x = np.array([0.0, 0.0, 0.0, 0.0])
    scip_reachable_set = EnumeratedReachableSet(A, x, complete=True, values=np.vstack([x, np.eye(4)]), solver="scip")
    cplex_reachable_set = EnumeratedReachableSet(A, x, complete=True, values=np.vstack([x, np.eye(4)]), solver="cplex")

    y = np.array([2, 2, 2, 2])
    assert y not in scip_reachable_set
    assert y not in cplex_reachable_set

    x1 = np.array([0, 0, 0, 0])
    assert x1 in scip_reachable_set
    assert x1 in cplex_reachable_set

    x2 = np.array([1, 0, 0, 0])
    assert np.array([x1, x2]) in scip_reachable_set
    assert np.array([x1, x2]) in cplex_reachable_set

    assert scip_reachable_set == cplex_reachable_set

def test_vacuous_reachability_constraints(dataset_actionset_2d):
    try:
        assert "scip" in SUPPORTED_SOLVERS and "cplex" in SUPPORTED_SOLVERS
    except AssertionError:
        print("SCIP and CPLEX are not both supported solvers.")
        pytest.skip()    
    
    X = dataset_actionset_2d["X"]
    A = dataset_actionset_2d["A"]
    expected_reachable_set = dataset_actionset_2d["R"]
    print(f"X: {X}")
    print(f"A: {A}")

    for x in X.values:
        x = np.array(x, dtype=int).tolist()
        scip_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver="scip")
        cplex_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver="cplex")
        R = np.array(expected_reachable_set.get(tuple(x)))
        n_expected = R.shape[0]
        scip_reachable_set.generate(max_points=n_expected)
        cplex_reachable_set.generate(max_points=n_expected)
        assert scip_reachable_set.complete
        assert cplex_reachable_set.complete
        assert len(scip_reachable_set) == n_expected
        assert len(cplex_reachable_set) == n_expected
        assert R in scip_reachable_set
        assert R in cplex_reachable_set
        assert scip_reachable_set == cplex_reachable_set

        # calling generate again should not change anything
        scip_reachable_set.generate()
        cplex_reachable_set.generate()
        assert len(scip_reachable_set) == n_expected and R in scip_reachable_set
        assert len(cplex_reachable_set) == n_expected and R in cplex_reachable_set
        assert scip_reachable_set == cplex_reachable_set

        # adding a reachability constraint should not change anything
        const_id = A.constraints.add(
            ReachabilityConstraint(names=X.columns.tolist(), values=X.values)
        )
        scip_constrained_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver="scip")
        cplex_constrained_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver="cplex")
        scip_constrained_reachable_set.generate()
        cplex_constrained_reachable_set.generate()
        assert scip_reachable_set == scip_constrained_reachable_set
        assert cplex_reachable_set == cplex_constrained_reachable_set
        assert scip_constrained_reachable_set == cplex_constrained_reachable_set
        A.constraints.drop(const_id)

def test_vacuous_reachability_constraints_with_overlap():
    try:
        assert "scip" in SUPPORTED_SOLVERS and "cplex" in SUPPORTED_SOLVERS
    except AssertionError:
        print("SCIP and CPLEX are not both supported solvers.")
        pytest.skip()
    
    X = pd.DataFrame(
        columns=["x0", "x1", "x2"],
        data=[
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
    )
    values = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]
    )

    for idx, x in enumerate(X.values):
        x = np.array(x, dtype=int).tolist()
        # adding a reachability constraint should not change anything
        A = ActionSet(X)
        A.constraints.add(
            ReachabilityConstraint(
                names=["x0", "x1"], values=np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            )
        )
        A.constraints.add(
            ReachabilityConstraint(
                names=["x1", "x2"],
                values=np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
                reachability=np.array(
                    [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
                ),
            )
        )
        feature_indices = A.get_feature_indices(["x0", "x1", "x2"])
        assert feature_indices in A.partition

        scip_constrained_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver="scip")
        cplex_constrained_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver="cplex")
        scip_constrained_reachable_set.generate()
        cplex_constrained_reachable_set.generate()
        assert values in scip_constrained_reachable_set
        assert values in cplex_constrained_reachable_set
        assert scip_constrained_reachable_set.complete
        assert cplex_constrained_reachable_set.complete
        assert len(scip_constrained_reachable_set) == values.shape[0]
        assert len(cplex_constrained_reachable_set) == values.shape[0]
        assert scip_constrained_reachable_set == cplex_constrained_reachable_set

        # dropping the constraint should lead to the right solution
        A.constraints.clear()
        scip_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver="scip")
        cplex_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver="cplex")
        scip_reachable_set.generate()
        cplex_reachable_set.generate()
        assert values in scip_reachable_set
        assert values in cplex_reachable_set
        assert scip_reachable_set.complete
        assert cplex_reachable_set.complete
        assert scip_constrained_reachable_set == scip_reachable_set
        assert cplex_constrained_reachable_set == cplex_reachable_set
        assert scip_reachable_set == cplex_reachable_set

def test_reachability_constraints_for_fixed_point_scip_and_cplex(dataset_actionset_2d):
    try:
        assert "scip" in SUPPORTED_SOLVERS and "cplex" in SUPPORTED_SOLVERS
    except AssertionError:
        print("SCIP and CPLEX are not both supported solvers.")
        pytest.skip()

    X = dataset_actionset_2d["X"]
    A = dataset_actionset_2d["A"]
    expected_reachable_set = dataset_actionset_2d["R"]
    print(f"X: {X}")
    print(f"A: {A}")

    n_values = X.values.shape[0]
    for idx, x in enumerate(X.values):
        x = np.array(x, dtype=int).tolist()
        # adding a reachability constraint should not change anything
        reachability_matrix = np.ones(shape=(n_values, n_values))
        reachability_matrix[idx, :] = np.zeros(n_values)
        reachability_matrix[idx, idx] = 1.0
        const_id = A.constraints.add(
            constraint=ReachabilityConstraint(
                names=X.columns.tolist(),
                values=X.values,
                reachability=reachability_matrix,
            )
        )
        scip_constrained_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver="scip")
        cplex_constrained_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver="cplex")
        scip_constrained_reachable_set.generate()
        cplex_constrained_reachable_set.generate()
        assert scip_constrained_reachable_set.complete
        assert cplex_constrained_reachable_set.complete
        assert len(scip_constrained_reachable_set) == 1
        assert len(cplex_constrained_reachable_set) == 1
        assert np.array_equal(scip_constrained_reachable_set.X[0, :], x)
        assert np.array_equal(cplex_constrained_reachable_set.X[0, :], x)
        assert scip_constrained_reachable_set == cplex_constrained_reachable_set

        # dropping the constraint should lead to the right solution
        A.constraints.drop(const_id)
        R = np.array(expected_reachable_set.get(tuple(x)))
        n_expected = R.shape[0]
        scip_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver="scip")
        cplex_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver="cplex")
        scip_reachable_set.generate(max_points=n_expected)
        cplex_reachable_set.generate(max_points=n_expected)
        assert scip_reachable_set.complete
        assert cplex_reachable_set.complete
        assert len(scip_reachable_set) == n_expected
        assert len(cplex_reachable_set) == n_expected
        assert R in scip_reachable_set
        assert R in cplex_reachable_set
        assert scip_reachable_set == cplex_reachable_set

def test_enumeration_with_overlap_scip_and_cplex():
    try:
        assert "scip" in SUPPORTED_SOLVERS and "cplex" in SUPPORTED_SOLVERS
    except AssertionError:
        print("SCIP and CPLEX are not both supported solvers.")
        pytest.skip()

    """
    assume that x1, x2 = (0, 1) is infeasible
    this is similar to x2 = 1 -> x1 = 1 so
    :return:
    """

    X = pd.DataFrame(
        columns=["x0", "x1", "x2"],
        data=[
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
            # [0, 0, 1],
            [0, 1, 1],
            # [1, 0, 1],
            [1, 1, 1],
        ],
    )

    values = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
            # [0, 0, 1],
            [0, 1, 1],
            # [1, 0, 1],
            [1, 1, 1],
        ]
    )

    for idx, x in enumerate(X.values):
        x = np.array(x, dtype=int).tolist()
        # adding a reachability constraint should not change anything
        A = ActionSet(X)
        A.constraints.add(
            constraint=ReachabilityConstraint(
                names=["x0", "x1"], values=np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            )
        )
        A.constraints.add(
            constraint=ReachabilityConstraint(
                names=["x1", "x2"],
                values=np.array([[0, 0], [1, 0], [1, 1]]),
            )
        )

        feature_indices = A.get_feature_indices(["x0", "x1", "x2"])
        assert feature_indices in A.partition

        scip_constrained_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver="scip")
        cplex_constrained_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver="cplex")
        scip_constrained_reachable_set.generate()
        cplex_constrained_reachable_set.generate()
        assert [1, 0, 1] not in scip_constrained_reachable_set
        assert [1, 0, 1] not in cplex_constrained_reachable_set
        assert [0, 0, 1] not in scip_constrained_reachable_set
        assert [0, 0, 1] not in cplex_constrained_reachable_set
        assert values in scip_constrained_reachable_set
        assert values in cplex_constrained_reachable_set
        assert scip_constrained_reachable_set.complete
        assert cplex_constrained_reachable_set.complete
        assert len(scip_constrained_reachable_set) == values.shape[0]
        assert len(cplex_constrained_reachable_set) == values.shape[0]
        assert scip_constrained_reachable_set == cplex_constrained_reachable_set

def test_enumeration_for_reachability_on_onehot_encoding_scip_and_cplex():
    try:
        assert "scip" in SUPPORTED_SOLVERS and "cplex" in SUPPORTED_SOLVERS
    except AssertionError:
        print("SCIP and CPLEX are not both supported solvers.")
        pytest.skip()

    """
    check that we can use reachability constraints to force
    at most 1 of (x1, x2, x3) to be on
    """

    X = pd.DataFrame(
        columns=["x0", "x1", "x2"], data=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )

    values = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]
    )

    for idx, x in enumerate(X.values):
        x = np.array(x, dtype=int).tolist()
        # adding a reachability constraint should not change anything
        A = ActionSet(X)
        A.constraints.add(
            constraint=ReachabilityConstraint(
                names=["x0", "x1", "x2"],
                values=values,
                reachability=np.array(
                    [
                        [1, 1, 1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1],
                    ]
                ),
            )
        )
        scip_constrained_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver="scip")
        cplex_constrained_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver="cplex")
        scip_constrained_reachable_set.generate()
        cplex_constrained_reachable_set.generate()
        assert scip_constrained_reachable_set.complete
        assert cplex_constrained_reachable_set.complete
        limits = np.sum(scip_constrained_reachable_set.X, axis=1)
        assert np.all(limits <= 1)
        limits = np.sum(cplex_constrained_reachable_set.X, axis=1)
        assert np.all(limits <= 1)
        assert scip_constrained_reachable_set == cplex_constrained_reachable_set
        # dropping the constraint should lead to the right solution
        A.constraints.clear()
        scip_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver="scip")
        cplex_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver="cplex")
        scip_reachable_set.generate()
        cplex_reachable_set.generate()
        assert scip_reachable_set.complete
        assert cplex_reachable_set.complete
        assert values in scip_reachable_set
        assert values in cplex_reachable_set
        assert len(scip_reachable_set) == values.shape[0]
        assert len(cplex_reachable_set) == values.shape[0]
        assert scip_reachable_set == cplex_reachable_set

def test_enumeration_for_reachability_on_ordinal_encoding_with_1step_scip_and_cplex():
    try:
        assert "scip" in SUPPORTED_SOLVERS and "cplex" in SUPPORTED_SOLVERS
    except AssertionError:
        print("SCIP and CPLEX are not both supported solvers.")
        pytest.skip()

    """
    check that we can use reachability constraints to force
    at most 1 of (x1, x2, x3) to be on
    """

    X = pd.DataFrame(
        columns=["x0", "x1", "x2"], data=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )

    values = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    expected_reachable_sets = {
        (0, 0, 0): [[0, 0, 0], [1, 0, 0]],
        (1, 0, 0): [[1, 0, 0], [0, 1, 0]],
        (0, 1, 0): [[0, 1, 0], [0, 0, 1]],
        (0, 0, 1): [[0, 0, 1]],
    }

    # adding a reachability constraint should not change anything
    A = ActionSet(X)
    A.constraints.add(
        constraint=ReachabilityConstraint(
            names=["x0", "x1", "x2"],
            values=values,
            reachability=np.array(
                [
                    [1, 1, 0, 0],
                    [0, 1, 1, 0],
                    [0, 0, 1, 1],
                    [0, 0, 0, 1],
                ]
            ),
        )
    )

    for idx, x in enumerate(X.values):
        scip_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver="scip")
        cplex_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver="cplex")
        scip_reachable_set.generate()
        cplex_reachable_set.generate()
        expected_set = np.array(expected_reachable_sets.get(tuple(x)))
        limits = np.sum(scip_reachable_set.X, axis=1)
        assert np.all(limits <= 1)
        limits = np.sum(cplex_reachable_set.X, axis=1)
        assert np.all(limits <= 1)
        assert scip_reachable_set.complete
        assert cplex_reachable_set.complete
        assert expected_set in scip_reachable_set
        assert expected_set in cplex_reachable_set
        assert len(scip_reachable_set) == len(expected_set)
        assert len(cplex_reachable_set) == len(expected_set)
        assert scip_reachable_set == cplex_reachable_set

def test_enumeration_for_reachability_on_thermometer_encoding_scip_and_cplex():
    try:
        assert "scip" in SUPPORTED_SOLVERS and "cplex" in SUPPORTED_SOLVERS
    except AssertionError:
        print("SCIP and CPLEX are not both supported solvers.")
        pytest.skip()

    """
    check that we can use reachability constraints to enforce therometer encoding
        x1 = 1[x ≥ v1]
        x2 = 1[x ≥ v2]
        x3 = 1[x ≥ v3]
    where v[1] ≤ v[2] ≤ v[3]
    """

    X = pd.DataFrame(
        columns=["x0", "x1", "x2"], data=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]]
    )

    values = np.array(
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

    for idx, x in enumerate(X.values):
        x = np.array(x, dtype=int).tolist()
        # adding a reachability constraint should not change anything
        A = ActionSet(X)
        A.constraints.add(
            constraint=ReachabilityConstraint(
                names=["x0", "x1", "x2"],
                values=values,
                reachability=np.array(
                    [
                        [1, 1, 1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1],
                    ]
                ),
            )
        )
        scip_constrained_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver="scip")
        cplex_constrained_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver="cplex")
        scip_constrained_reachable_set.generate()
        cplex_constrained_reachable_set.generate()
        assert scip_constrained_reachable_set.complete
        assert cplex_constrained_reachable_set.complete
        assert X.values in scip_constrained_reachable_set
        assert X.values in cplex_constrained_reachable_set
        assert len(scip_constrained_reachable_set) == X.values.shape[0]
        assert len(cplex_constrained_reachable_set) == X.values.shape[0]
        assert scip_constrained_reachable_set == cplex_constrained_reachable_set

        # dropping the constraint should lead to the right solution
        A.constraints.clear()
        scip_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver="scip")
        cplex_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver="cplex")
        scip_reachable_set.generate()
        cplex_reachable_set.generate()
        assert scip_reachable_set.complete
        assert cplex_reachable_set.complete
        assert values in scip_reachable_set
        assert values in cplex_reachable_set
        assert scip_reachable_set == cplex_reachable_set

def test_enumeration_for_reachability_on_thermometer_encoding_monotonic_scip_and_cplex():
    try:
        assert "scip" in SUPPORTED_SOLVERS and "cplex" in SUPPORTED_SOLVERS
    except AssertionError:
        print("SCIP and CPLEX are not both supported solvers.")
        pytest.skip()

    """
    check that we can use reachability constraints to enforce therometer encoding
        x1 = 1[x ≥ v1]
        x2 = 1[x ≥ v2]
        x3 = 1[x ≥ v3]
    where v[1] ≤ v[2] ≤ v[3]
    """

    X = pd.DataFrame(columns=["x0", "x1", "x2"], data=[[0, 0, 0], [1, 0, 0], [1, 1, 1]])

    values = np.array(
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

    for idx, x in enumerate(X.values):
        # adding a reachability constraint should not change anything
        A = ActionSet(X)
        A.step_direction = 1
        A.constraints.add(
            constraint=ReachabilityConstraint(
                names=["x0", "x1", "x2"],
                values=values,
                reachability=np.array(
                    [
                        [1, 1, 1, 1, 0, 0, 0, 0],
                        [0, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1],
                    ]
                ),
            )
        )
        scip_constrained_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver="scip")
        cplex_constrained_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver="cplex")
        scip_constrained_reachable_set.generate()
        cplex_constrained_reachable_set.generate()
        assert scip_constrained_reachable_set.complete
        assert cplex_constrained_reachable_set.complete
        assert X.values[idx:,] in scip_constrained_reachable_set
        assert X.values[idx:,] in cplex_constrained_reachable_set

        # dropping the constraint lead to infeasible actions
        A.constraints.clear()
        scip_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver="scip")
        cplex_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver="cplex")
        scip_reachable_set.generate()
        cplex_reachable_set.generate()
        assert scip_reachable_set.complete
        assert cplex_reachable_set.complete
        assert len(scip_reachable_set) == 2 ** (len(A) - np.sum(x == 1))
        assert len(cplex_reachable_set) == 2 ** (len(A) - np.sum(x == 1))

        if np.array_equal(x, [0, 0, 0]):
            assert [0, 0, 1] in scip_reachable_set
            assert [0, 0, 1] not in scip_constrained_reachable_set
            assert [0, 1, 0] in scip_reachable_set
            assert [0, 1, 0] not in scip_constrained_reachable_set
            assert [1, 0, 1] in scip_reachable_set
            assert [1, 0, 1] not in scip_constrained_reachable_set

            assert [0, 0, 1] in cplex_reachable_set
            assert [0, 0, 1] not in cplex_constrained_reachable_set
            assert [0, 1, 0] in cplex_reachable_set
            assert [0, 1, 0] not in cplex_constrained_reachable_set
            assert [1, 0, 1] in cplex_reachable_set
            assert [1, 0, 1] not in cplex_constrained_reachable_set

        if np.array_equal(x, [1, 0, 0]):
            assert [1, 0, 1] in scip_reachable_set
            assert [1, 0, 1] not in scip_constrained_reachable_set

            assert [1, 0, 1] in cplex_reachable_set
            assert [1, 0, 1] not in cplex_constrained_reachable_set

if __name__ == "__main__":
    pytest.main()
