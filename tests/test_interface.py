import pytest
import numpy as np
import pandas as pd
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
    ].actionable = True
    A["EducationLevel"].step_direction = 1
    A["EducationLevel"].lb = 0
    A["EducationLevel"].ub = 3
    A["TotalMonthsOverdue"].lb = 0
    A["TotalMonthsOverdue"].ub = 100
    out = {"X": X, "A": A}
    return out


@pytest.fixture(params=["onehot", "reachability", "mixed"])
def constraint_info(request, test_case):
    constraint_info = []
    if request.param in ("onehot", "mixed"):
        constraint_info.append(
            {
                "class": OneHotEncoding,
                "names": [
                    "Age_lt_25",
                    "Age_in_25_to_40",
                    "Age_in_40_to_59",
                    "Age_geq_60",
                ],
                "parameters": {"limit": 1, "limit_type": "equal"},
            }
        )

    if request.param in ("reachability", "mixed"):
        names = ["Age_lt_25", "Age_in_25_to_40", "Age_in_40_to_59", "Age_geq_60"]
        values = np.unique(test_case["X"][names], axis=0)
        constraint_info.append(
            {
                "class": ReachabilityConstraint,
                "names": [
                    "Age_lt_25",
                    "Age_in_25_to_40",
                    "Age_in_40_to_59",
                    "Age_geq_60",
                ],
                "parameters": {"values": values, "reachability": np.eye(len(values))},
            }
        )

    return constraint_info


def test_add_drop_constraint(test_case, constraint_info):
    A = test_case["A"]

    constraint_ids = []

    for info in constraint_info:
        constraintClass = info["class"]
        cons = constraintClass(names=info["names"], **info["parameters"])

        # fields should raise error if accessed before attachment
        post_add_fields = ["indices", "id"]
        for name in post_add_fields:
            assert cons.parent is None
            with pytest.raises(ValueError):
                cons.__getattribute__(name)

        # adding should fix this
        const_id = A.constraints.add(cons)
        constraint_ids.append(const_id)
        assert const_id == len(A.constraints._map) - 1
        assert A is cons.parent
        assert const_id == cons.id
        assert set(cons.indices) == set(A.get_feature_indices(info["names"]))

        # once you add a constraint, you cannot add it again
        with pytest.raises(AssertionError):
            A.constraints.add(cons)

        # cannot add a constraint with the same parameters either
        same_cons = constraintClass(names=info["names"], **info["parameters"])
        assert cons == same_cons
        assert not cons is same_cons

        with pytest.raises(AssertionError):
            A.constraints.add(same_cons)

    for const_id in constraint_ids:
        cons = A.constraints._map.get(const_id)

        # drop constraint
        dropped = A.constraints.drop(const_id)
        assert dropped

        # cannot drop dropped constraint
        dropped = A.constraints.drop(const_id)
        assert not dropped

        assert cons.parent is None
        for name in post_add_fields:
            with pytest.raises(ValueError):
                cons.__getattribute__(name)


def test_partition_separable(test_case):
    X = test_case["X"]
    A = test_case["A"]
    partition = A.partition
    actionable_partition = A.actionable_partition
    for part in actionable_partition:
        assert len(part) == 1
        assert part in partition

    for part in partition:
        assert len(part) == 1
        if part not in actionable_partition:
            assert A[part[0]].actionable == False

    assert A.separable


def test_partition_with_constraints(test_case):
    X = test_case["X"]
    A = test_case["A"]

    onehot_names = ["Age_lt_25", "Age_in_25_to_40", "Age_in_40_to_59", "Age_geq_60"]
    const_id = A.constraints.add(
        OneHotEncoding(names=onehot_names, limit_type="equal", limit=1)
    )
    A[onehot_names].actionable = True

    ifthen_names = ["MaxBillAmountOverLast6Months", "MaxPaymentAmountOverLast6Months"]
    const_id = A.constraints.add(
        IfThenConstraint(
            Condition(name="MaxBillAmountOverLast6Months", sense="E", value=100),
            Condition(name="MaxPaymentAmountOverLast6Months", sense="E", value=100),
        )
    )
    A[ifthen_names].actionable = True

    onehot_indices = A.get_feature_indices(onehot_names)
    ifthen_indices = A.get_feature_indices(ifthen_names)

    assert any([set(onehot_indices) == set(part) for part in A.actionable_partition])
    assert any([set(ifthen_indices) == set(part) for part in A.actionable_partition])

    for part in A.actionable_partition:
        if not (
            (set(part) == set(onehot_indices)) or (set(part) == set(ifthen_indices))
        ):
            assert len(part) == 1

    assert not A.separable


def test_partition_with_constraints_on_immutable(test_case):
    X = test_case["X"]
    A = test_case["A"]

    onehot_names = ["Age_lt_25", "Age_in_25_to_40", "Age_in_40_to_59", "Age_geq_60"]
    const_id = A.constraints.add(
        OneHotEncoding(names=onehot_names, limit_type="equal", limit=1)
    )
    A[onehot_names].actionable = False
    onehot_indices = A.get_feature_indices(onehot_names)
    assert onehot_indices in A.partition
    assert onehot_indices not in A.actionable_partition
    # assert not A.separable


def test_partition_with_overlapping_constraints(test_case):
    X = test_case["X"]
    A = test_case["A"]

    onehot_names = ["Age_lt_25", "Age_in_25_to_40"]
    A[onehot_names].actionable = True
    const_id = A.constraints.add(
        OneHotEncoding(names=onehot_names, limit_type="max", limit=1)
    )
    onehot_indices = A.get_feature_indices(onehot_names)
    assert any([set(onehot_indices) == set(part) for part in A.actionable_partition])

    other_onehot_names = ["Age_in_25_to_40", "Age_in_40_to_59", "Age_geq_60"]
    A[other_onehot_names].actionable = True
    const_id = A.constraints.add(
        OneHotEncoding(names=other_onehot_names, limit_type="max", limit=2)
    )
    other_onehot_indices = A.get_feature_indices(other_onehot_names)
    all_onehot_indices = A.get_feature_indices(
        list(set(onehot_names + other_onehot_names))
    )
    assert any([set(onehot_indices) != set(part) for part in A.actionable_partition])
    assert any(
        [set(other_onehot_indices) != set(part) for part in A.actionable_partition]
    )
    assert any(
        [set(all_onehot_indices) == set(part) for part in A.actionable_partition]
    )
    for part in A.actionable_partition:
        if not (set(part) == set(all_onehot_indices)):
            assert len(part) == 1


if __name__ == "__main__":
    pytest.main()
