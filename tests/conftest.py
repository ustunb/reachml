"""
This file contains fixtures to create datasets and action sets for testing enumeration
"""

import pytest
import numpy as np
import pandas as pd
from reachml.action_set import ActionSet

BOOLEAN_TEST_CASES_1D = [
    "boolean_1d",
    "boolean_1d_immutable",
    "boolean_1d_increasing",
    "boolean_1d_decreasing",
]

UNSIGNED_INTEGER_TEST_CASES_1D = [
    "uint_1d",
    "uint_1d_immutable",
    "uint_1d_increasing",
    "uint_1d_decreasing",
]

INTEGER_TEST_CASES_1D = [
    "int_1d",
    "int_1d_immutable",
    "int_1d_increasing",
    "int_1d_decreasing",
]

SEPARABLE_BOOLEAN_TEST_CASES_2D = [
    "boolean_2d",
    "boolean_2d_immutable",
    "boolean_2d_monotonic",
]

TEST_CASES_STEP_CONSTRAINTS = [
    "step_mixed_3d_increasing",
    "step_mixed_3d_decreasing",
    "step_mixed_3d_increasing_ub_set_automatically",
]

SEPARABLE_TEST_CASES = (
    BOOLEAN_TEST_CASES_1D
    + UNSIGNED_INTEGER_TEST_CASES_1D
    + INTEGER_TEST_CASES_1D
    + SEPARABLE_BOOLEAN_TEST_CASES_2D
    + TEST_CASES_STEP_CONSTRAINTS
)

SEPARABLE_TEST_CASES_2D = SEPARABLE_BOOLEAN_TEST_CASES_2D


@pytest.fixture(params=SEPARABLE_TEST_CASES)
def discrete_test_case(request):
    name = request.param
    if "1d" in name:
        out = generate_1d_test_case(name)
    elif "2d" in name:
        out = generate_2d_test_case(name)
    elif "step" in name:
        out = generate_dataset_actionset_with_step_constraints(name)
    return out


@pytest.fixture(params=SEPARABLE_TEST_CASES_2D)
def dataset_actionset_2d(request):
    dataset_actionset_name = request.param
    out = generate_2d_test_case(dataset_actionset_name)
    return out


def generate_1d_test_case(dataset_actionset_name):
    assert (
        dataset_actionset_name
        in BOOLEAN_TEST_CASES_1D
        + INTEGER_TEST_CASES_1D
        + UNSIGNED_INTEGER_TEST_CASES_1D
    )

    if "boolean" in dataset_actionset_name:
        values = [0, 1]
        X = pd.DataFrame(columns=["has_phd"], data=values)
    elif "uint" in dataset_actionset_name:
        values = np.arange(0, 5).tolist()
        X = pd.DataFrame(columns=["n_accounts"], data=values)
    elif "int" in dataset_actionset_name:
        values = np.arange(-2, 2 + 1).tolist()
        X = pd.DataFrame(columns=["delta_accounts"], data=values)

    A = ActionSet(X)
    if "increasing" in dataset_actionset_name:
        A.step_direction = 1
        R = {tuple([v]): [w for w in values if w >= v] for v in values}
    elif "decreasing" in dataset_actionset_name:
        A.step_direction = -1
        R = {tuple([v]): [w for w in values if w <= v] for v in values}
    elif "immutable" in dataset_actionset_name:
        A.actionable = False
        R = {tuple([v]): [v] for v in values}
    else:
        R = {tuple([v]): [w for w in values] for v in values}

    out = {
        "X": X,
        "A": A,
        "R": R,
    }

    return out


def generate_2d_test_case(dataset_actionset_name):
    X = pd.DataFrame(
        columns=["has_phd", "has_children"], data=[[0, 0], [0, 1], [1, 0], [1, 1]]
    )
    A = ActionSet(X)

    if dataset_actionset_name == "boolean_2d":
        R = {
            (0, 0): X.values.tolist(),
            (1, 0): X.values.tolist(),
            (0, 1): X.values.tolist(),
            (1, 1): X.values.tolist(),
        }

    elif dataset_actionset_name == "boolean_2d_immutable":
        A.actionable = False
        R = {
            (0, 0): [[0, 0]],
            (1, 0): [[1, 0]],
            (0, 1): [[0, 1]],
            (1, 1): [[1, 1]],
        }

    elif dataset_actionset_name == "boolean_2d_monotonic":
        A["has_phd"].step_direction = 1
        A["has_children"].step_direction = 1
        R = {
            (0, 0): [[0, 0], [1, 0], [0, 1], [1, 1]],
            (1, 0): [[1, 0], [1, 1]],
            (0, 1): [[0, 1], [1, 1]],
            (1, 1): [[1, 1]],
        }

    out = {
        "X": X,
        "A": A,
        "R": R,
    }

    return out


def generate_dataset_actionset_with_step_constraints(dataset_actionset_name):
    X = pd.DataFrame(
        columns=["x1", "x2", "x3"], data=[[0, 0, 0], [0, 1, 5], [1, 0, 10]]
    )
    A = ActionSet(X)

    if dataset_actionset_name == "step_mixed_3d_increasing":
        A.step_direction = 1
        A["x3"].step_ub = 1
        A["x3"].ub = 20

        R = {
            (0, 0, 0): [
                [0, 0, 0],
                [0, 0, 1],
                [1, 0, 0],
                [1, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 1, 0],
                [1, 1, 0],
            ],
            #
            (0, 1, 5): [[0, 1, 5], [0, 1, 6], [1, 1, 5], [1, 1, 6]],
            #
            (1, 0, 10): [[1, 0, 10], [1, 0, 11], [1, 1, 10], [1, 1, 11]],
        }

    if dataset_actionset_name == "step_mixed_3d_decreasing":
        A.step_direction = 1
        A["x3"].step_direction = -1
        A["x3"].step_lb = -2
        A["x3"].lb = -1

        R = {
            (0, 0, 0): [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, -1],
                [1, 0, -1],
                [0, 1, -1],
                [1, 1, -1],
            ],
            #
            (0, 1, 5): [
                [0, 1, 5],
                [1, 1, 5],
                [0, 1, 4],
                [1, 1, 4],
                [0, 1, 3],
                [1, 1, 3],
            ],
            #
            (1, 0, 10): [
                [1, 0, 10],
                [1, 1, 10],
                [1, 0, 9],
                [1, 1, 9],
                [1, 0, 8],
                [1, 1, 8],
            ],
        }

    if dataset_actionset_name == "step_mixed_3d_increasing_ub_set_automatically":
        A.step_direction = 1
        A["x3"].step_ub = 1
        R = {
            (0, 0, 0): [
                [0, 0, 0],
                [0, 0, 1],
                [1, 0, 0],
                [1, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 1, 0],
                [1, 1, 0],
            ],
            #
            (0, 1, 5): [[0, 1, 5], [0, 1, 6], [1, 1, 5], [1, 1, 6]],
            #
            (1, 0, 10): [[1, 0, 10], [1, 1, 10]],
        }

    out = {"X": X, "A": A, "R": R}
    return out
