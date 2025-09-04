import pytest
import numpy as np
from reachml.reachable_set import EnumeratedReachableSet
from reachml.constraints import *
from reachml.utils import SUPPORTED_SOLVERS


@pytest.fixture(params=[True, False])
def vacuous_reachability_constraint(request):
    return request

@pytest.mark.parametrize("solver", SUPPORTED_SOLVERS)
def test_enumeration(discrete_test_case, vacuous_reachability_constraint, solver):
    X = discrete_test_case["X"]
    A = discrete_test_case["A"]
    print(f"X: {X}")
    print(f"A: {A}")

    expected_reachable_set = discrete_test_case["R"]
    for x in X.values:
        x = np.array(x, dtype=int).tolist()
        reachable_set = EnumeratedReachableSet(action_set=A, x=x, solver=solver)
        R = expected_reachable_set.get(tuple(x))
        n_expected = len(R)
        for k in range(0, n_expected):
            reachable_set.generate(max_points=1)
            assert reachable_set.complete is False or len(reachable_set) == n_expected

        # reachable set should be complete
        assert reachable_set.complete
        assert len(reachable_set) == n_expected
        for xr in R:
            assert np.all(reachable_set.X == xr, axis=1).any()

        # calling enumerate should not change anything
        reachable_set.generate(max_points=1)
        assert len(reachable_set) == n_expected
        for xr in R:
            assert np.all(reachable_set.X == xr, axis=1).any()

        if vacuous_reachability_constraint:
            # adding a reachability constraint should not change anything
            const_id = A.constraints.add(
                ReachabilityConstraint(names=X.columns.tolist(), values=reachable_set.X)
            )
            constrained_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver=solver)
            constrained_reachable_set.generate()
            print(f"constrained set: {constrained_reachable_set.X}")
            print(f"reachable set: {reachable_set.X}")
            assert reachable_set == constrained_reachable_set
            A.constraints.drop(const_id)


def test_enumeration_scip_and_cplex(discrete_test_case, vacuous_reachability_constraint):
    try:
        assert "scip" in SUPPORTED_SOLVERS and "cplex" in SUPPORTED_SOLVERS
    except AssertionError:
        print("SCIP and CPLEX are not both supported solvers.")
        return
    X = discrete_test_case["X"]
    A = discrete_test_case["A"]
    print(f"X: {X}")
    print(f"A: {A}")

    # generate and validate SCIP solution
    expected_reachable_set = discrete_test_case["R"]
    for x in X.values:
        x = np.array(x, dtype=int).tolist()
        scip_reachable_set = EnumeratedReachableSet(action_set=A, x=x, solver="scip")
        R = expected_reachable_set.get(tuple(x))
        n_expected = len(R)
        for k in range(0, n_expected):
            scip_reachable_set.generate(max_points=1)
            assert scip_reachable_set.complete is False or len(scip_reachable_set) == n_expected

        # reachable set should be complete
        assert scip_reachable_set.complete
        assert len(scip_reachable_set) == n_expected
        assert len(scip_reachable_set.X) == n_expected
        for xr in R:
            assert np.all(scip_reachable_set.X == xr, axis=1).any()

        # calling enumerate should not change anything
        scip_reachable_set.generate(max_points=1)
        assert len(scip_reachable_set) == n_expected
        for xr in R:
            assert np.all(scip_reachable_set.X == xr, axis=1).any()

    # generate and validate CPLEX solution
    for x in X.values:
        x = np.array(x, dtype=int).tolist()
        cplex_reachable_set = EnumeratedReachableSet(action_set=A, x=x, solver="cplex")
        R = expected_reachable_set.get(tuple(x))
        n_expected = len(R)
        for k in range(0, n_expected):
            cplex_reachable_set.generate(max_points=1)
            assert cplex_reachable_set.complete is False or len(cplex_reachable_set) == n_expected

        # reachable set should be complete
        assert cplex_reachable_set.complete
        assert len(cplex_reachable_set) == n_expected
        for xr in R:
            assert np.all(cplex_reachable_set.X == xr, axis=1).any()
        
    for xr in scip_reachable_set.X:
        assert np.all(cplex_reachable_set.X == xr, axis=1).any()

    # calling enumerate should not change anything
    scip_reachable_set.generate(max_points=1)
    assert len(scip_reachable_set) == n_expected
    for xr in R:
        assert np.all(scip_reachable_set.X == xr, axis=1).any()

    if vacuous_reachability_constraint:
        # adding a reachability constraint should not change anything
        const_id = A.constraints.add(
            ReachabilityConstraint(names=X.columns.tolist(), values=scip_reachable_set.X)
        )
        scip_constrained_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver="scip")
        scip_constrained_reachable_set.generate()
        try:
            assert scip_reachable_set == scip_constrained_reachable_set
        except AssertionError:
            for xr in scip_reachable_set.X:
                assert np.all(scip_constrained_reachable_set.X == xr, axis=1).any()
        A.constraints.drop(const_id)

        cplex_constrained_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver="cplex")
        cplex_constrained_reachable_set.generate()
        assert cplex_reachable_set == cplex_constrained_reachable_set
        A.constraints.drop(const_id)

        assert scip_constrained_reachable_set == cplex_constrained_reachable_set


if __name__ == "__main__":
    pytest.main()
