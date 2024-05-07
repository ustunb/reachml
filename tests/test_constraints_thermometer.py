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
from reachml.constraints.thermometer import ThermometerEncoding

sortrows = lambda v: v[np.lexsort(v.T, axis=0), :]


@pytest.fixture(params=[0, 1, -1])
def step_direction(request):
    return request.param


@pytest.fixture(params=[True, False])
def drop_invalid(request):
    return request.param


def get_test_case(step_direction):
    X = pd.DataFrame(
        columns=["x0", "x1", "x2"], data=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]]
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
            (0, 0, 0): [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
            (1, 0, 0): [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
            (1, 1, 0): [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
            (1, 1, 1): [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
        }
    elif step_direction > 0:
        expected_sets = {
            (0, 0, 0): [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
            (1, 0, 0): [[1, 0, 0], [1, 1, 0], [1, 1, 1]],
            (1, 1, 0): [[1, 1, 0], [1, 1, 1]],
            (1, 1, 1): [[1, 1, 1]],
        }
    elif step_direction < 0:
        expected_sets = {
            (0, 0, 0): [[0, 0, 0]],
            (1, 0, 0): [[0, 0, 0], [1, 0, 0]],
            (1, 1, 0): [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
            (1, 1, 1): [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
        }

    for v in all_values:
        if tuple(v) not in expected_sets:
            expected_sets[tuple(v)] = [list(v)]

    expected_sets = {k: sortrows(np.array(v)) for k, v in expected_sets.items()}

    out = {
        "X": X,
        "all_values": sortrows(all_values),
        "valid_values": sortrows(valid_values),
        "expected_sets": expected_sets,
    }

    return out


def test_enumeration_with_thermometer_constraints(step_direction, drop_invalid):
    test_case = get_test_case(step_direction)
    X = test_case["X"]
    valid_values = test_case["valid_values"]
    all_values = test_case["all_values"]
    names = X.columns.tolist()

    # create thermometer constraint
    thermometer_constraint = ThermometerEncoding(
        names=names, step_direction=step_direction, drop_invalid_values=drop_invalid
    )
    if drop_invalid:
        assert np.array_equal(
            sortrows(thermometer_constraint.values), sortrows(valid_values)
        )
    else:
        assert np.array_equal(
            sortrows(thermometer_constraint.values), sortrows(all_values)
        )

    A = ActionSet(X)
    A.constraints.add(constraint=thermometer_constraint)

    for idx, x in enumerate(all_values):
        expected_set = test_case["expected_sets"].get(tuple(x))
        value_is_valid = np.all(valid_values == x, axis=1).any()
        # print(f'x: {x}')
        # print(f'valid_values: {valid_values}')

        if (value_is_valid == False) and (drop_invalid == True):
            with pytest.raises(AssertionError):
                ReachableSetEnumerator(x=x, action_set=A)

        if (value_is_valid == True) or (drop_invalid == False):
            enumerator = ReachableSetEnumerator(x=x, action_set=A)
            reachable_set = enumerator.enumerate()
            assert reachable_set.complete
            assert np.array_equal(np.sort(reachable_set.X, axis=0), expected_set)
            # print(f'reachable_set.X: {reachable_set.X}')
            # print(f'expected_set.X: {expected_set}')
            # print(thermometer_constraint.reachability)


if __name__ == "__main__":
    pytest.main()
