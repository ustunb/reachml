import pytest
import numpy as np
from reachml.reachable_set import EnumeratedReachableSet
from reachml.constraints import *


@pytest.fixture(params=[True, False])
def vacuous_reachability_constraint(request):
    return request


def test_enumeration(discrete_test_case, vacuous_reachability_constraint):
    X = discrete_test_case["X"]
    A = discrete_test_case["A"]
    print(f"X: {X}")
    print(f"A: {A}")

    expected_reachable_set = discrete_test_case["R"]
    for x in X.values:
        x = np.array(x, dtype=int).tolist()
        reachable_set = EnumeratedReachableSet(action_set=A, x=x)
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
            constrained_reachable_set = EnumeratedReachableSet(x=x, action_set=A)
            constrained_reachable_set.generate()
            assert reachable_set == constrained_reachable_set
            A.constraints.drop(const_id)


if __name__ == "__main__":
    pytest.main()
