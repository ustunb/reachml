import pytest
import pandas as pd
import numpy as np
from reachml.paths import tests_dir
from reachml.action_set import ActionSet
from reachml.constraints import *


@pytest.fixture(params=["credit"])
def test_case(request):
    X = pd.read_csv(tests_dir / "credit.csv").drop(columns=["NoDefaultNextMonth"])
    A = ActionSet(X)
    A["Married"].actionable = False
    A[
        ["Age_lt_25", "Age_in_25_to_40", "Age_in_40_to_59", "Age_geq_60"]
    ].actionable = False
    A["EducationLevel"].step_direction = 1
    A["EducationLevel"].lb = 0
    A["EducationLevel"].ub = 3
    A["TotalMonthsOverdue"].lb = 0
    A["TotalMonthsOverdue"].ub = 100
    out = {"X": X, "A": A}
    return out


def test_action_bounds(test_case):
    X = test_case["X"]
    A = test_case["A"]
    x = X.iloc[9].values  # = [0, 1, 0, 1, 0, 0, 1, 420, 400, 1, 4, 0, 0, 0, 0, 0, 0]
    expected_lb = np.array([0, -1, 0, 0, 0, 0, 0, -420, -400, -1, -4, 0, 0, 0, 0, 0, 0])
    expected_ub = np.array(
        [0, 0, 0, 0, 0, 0, 2, 50390, 51030, 5, 2, 6, 29450, 26670, 3, 100, 1]
    )
    assert np.array_equal(expected_lb, A.get_bounds(x, bound_type="lb"))
    assert np.array_equal(expected_ub, A.get_bounds(x, bound_type="ub"))


def test_action_bounds_negative():
    X = pd.DataFrame(columns=["x1", "x2"], data=[[-3, -7], [-1, -5]])
    A = ActionSet(X)
    expected_ub = np.max(X.values, axis=0)
    expected_lb = np.min(X.values, axis=0)
    for x in X.values:
        assert np.array_equal(expected_ub - x, A.get_bounds(x, bound_type="ub"))
        assert np.array_equal(expected_lb - x, A.get_bounds(x, bound_type="lb"))


def test_action_bounds_immutable(test_case):
    X = test_case["X"].iloc[:, 0:2]
    A = ActionSet(X)
    A["Married"].actionable = False
    # immutable attributes should have the same ub= lb
    x = X.iloc[0].values
    ub_mutable = np.array([1, 1])
    lb_mutable = np.array([0, 0])
    expected_ub = ub_mutable
    expected_lb = lb_mutable
    immutable_idx = np.flatnonzero(np.logical_not(A.actionable))
    expected_ub[immutable_idx] = 0.0
    expected_lb[immutable_idx] = 0.0
    assert np.array_equal(expected_ub, A.get_bounds(x, bound_type="ub"))
    assert np.array_equal(expected_lb, A.get_bounds(x, bound_type="lb"))


def test_action_bounds_monotonic(test_case):
    X = test_case["X"].iloc[:, 0:7]
    A = ActionSet(X)

    A = ActionSet(X)
    A["Married"].actionable = False
    A[
        ["Age_lt_25", "Age_in_25_to_40", "Age_in_40_to_59", "Age_geq_60"]
    ].actionable = False
    A["EducationLevel"].ub = 3
    A["EducationLevel"].lb = 0

    x = X.iloc[0].values  # x = [1, 0, 1, 0, 0, 0, 2]
    # no constraint
    expected_lb = [0, 0, 0, 0, 0, 0, A["EducationLevel"].lb]
    expected_ub = [0, 1, 0, 0, 0, 0, A["EducationLevel"].ub]
    education_idx = A.get_feature_indices("EducationLevel")
    expected_ub[education_idx] = A["EducationLevel"].ub - x[education_idx]
    expected_lb[education_idx] = A["EducationLevel"].lb - x[education_idx]
    assert np.array_equal(expected_lb, A.get_bounds(x, bound_type="lb"))
    assert np.array_equal(expected_ub, A.get_bounds(x, bound_type="ub"))

    # set education level to increasing
    A["EducationLevel"].step_direction = 1  ## force conditional immutability.
    expected_ub[education_idx] = A["EducationLevel"].ub - x[education_idx]
    expected_lb[education_idx] = 0.0
    assert np.array_equal(expected_lb, A.get_bounds(x, bound_type="lb"))
    assert np.array_equal(expected_ub, A.get_bounds(x, bound_type="ub"))

    # set education level to decreasing
    A["EducationLevel"].step_direction = -1
    expected_ub[education_idx] = 0.0
    expected_lb[education_idx] = A["EducationLevel"].lb - x[education_idx]
    assert np.array_equal(expected_lb, A.get_bounds(x, bound_type="lb"))
    assert np.array_equal(expected_ub, A.get_bounds(x, bound_type="ub"))


def test_action_bounds_partition_masking(test_case):
    X = test_case["X"]
    A = ActionSet(X)
    A["Married"].actionable = False
    A["EducationLevel"].step_direction = 1  ## force conditional immutability.
    A["EducationLevel"].ub = 3
    A["EducationLevel"].lb = 0
    A["TotalMonthsOverdue"].lb = 0
    A["TotalMonthsOverdue"].ub = 100

    const_id = A.constraints.add(
        OneHotEncoding(
            names=["Age_lt_25", "Age_in_25_to_40", "Age_in_40_to_59", "Age_geq_60"]
        )
    )

    const_id = A.constraints.add(
        IfThenConstraint(
            if_condition=Condition("MaxBillAmountOverLast6Months", "G", 100),
            then_condition=Condition("MaxPaymentAmountOverLast6Months", "E", 100),
        )
    )

    x = X.iloc[0].values
    part = A.actionable_partition[1]
    part_A = A[part]
    n_actionable_in_part = sum(part_A.actionable)

    # expected
    expected_lb = np.zeros(len(x), dtype=float)
    expected_lb[2] = -1.0
    expected_ub = np.zeros(len(x), dtype=float)
    expected_ub[[3, 4, 5]] = [1.0, 1.0, 1.0]

    # actual
    lb = part_A.get_bounds(x[part], bound_type="lb")
    ub = part_A.get_bounds(x[part], bound_type="ub")
    assert np.count_nonzero(lb) <= n_actionable_in_part
    assert np.count_nonzero(ub) <= n_actionable_in_part
    assert np.array_equal(lb, expected_lb[part])
    assert np.array_equal(ub, expected_ub[part])


if __name__ == "__main__":
    pytest.main()
