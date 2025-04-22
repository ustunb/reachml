import pytest
import pandas as pd
import numpy as np

from reachml.action_set import ActionSet
from reachml.constraints import ThermometerEncoding
from reachml.scoring import ResponsivenessScorer

TEST_PARAMS = [
    "separable_discrete",
    "separable_discrete_immutable",
    "separable_discrete_monotonic",
    "joint_thermometer_discrete",
    "joint_thermometer_discrete_immutable",
    "joint_thermometer_discrete_monotonic",
    # "separable_continuous",
    # "separable_continuous_monotonic",
    # "joint_continuous",
    # "joint_continuous_monotonic",
]

@pytest.fixture(params=TEST_PARAMS)
def discrete_test_case(request):
    if "separable" in request.param:
        out = generate_discrete_separable_test_case(request.param)
    elif "joint" in request.param:
        if "thermometer" in request.param:
            out = generate_joint_thermometer_test_case(request.param)

    return out

def generate_discrete_separable_test_case(name):
    X1_MAX = 5
    X = pd.DataFrame(
        columns=["x1", "x2"], 
        data=[[0, 0], [2, 1]]
        )
    A = ActionSet(X)
    A["x1"].ub = X1_MAX

    element_clf = lambda x: 1 if x[0] > 2 else -1
    clf = lambda x: np.array([element_clf(xi) for xi in x])

    if "immutable" in name:
        A.actionable = False
        expected_set = {
            (0, 0): {
                "Rj": [[], []],
                "score": [0, 0]
            },
            (2, 1): {
                "Rj": [[], []],
                "score": [0, 0]
            }
        }
    elif "monotonic" in name:
        A.step_direction = 1
        expected_set = {
            (0, 0): {
                "Rj": [[[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]], [[0, 1]]], 
                "score": [0.6, 0]
            },
            (2, 1): {
                "Rj": [[[3, 1], [4, 1], [5, 1]], []],
                "score": [1, 0]
            }
        }
    else:
        expected_set = {
            (0, 0): {
                "Rj": [[[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]], [[0, 1]]], 
                "score": [0.6, 0]
            },
            (2, 1): {
                "Rj": [[[0, 1], [1, 1], [3, 1], [4, 1], [5, 1]], [[2, 0]]],
                "score": [0.6, 0]
            }
        }

    out = {
        "X": X,
        "A": A,
        "clf": clf,
        "expected_set": expected_set,
    }

    return out

def generate_joint_thermometer_test_case(name):
    step_direction = -1 if "monotonic" in name else 0

    X = pd.DataFrame(
        columns=["CreditUtil_geq_25", "CreditUtil_geq_50", "CreditUtil_geq_75", "CreditUtil_geq_100"],
        data=[[0, 0, 0, 0], [1, 1, 1, 1]]
    )
    A = ActionSet(X)

    # add thermometer encoding
    therm_const = ThermometerEncoding(
        names=X.columns.tolist(), step_direction=step_direction
    )
    A.constraints.add(constraint=therm_const)

    elem_clf = lambda x: 1 if x[2] < 1 else -1
    clf = lambda x: np.array([elem_clf(xi) for xi in x])
    
    if "immutable" in name:
        A.actionable = False
        expected_set = {
            (0, 0, 0, 0): {
                "Rj": [[], [], [], []],
                "score": [0, 0, 0, 0]
            },
            (1, 1, 1, 1): {
                "Rj": [[], [], [], []],
                "score": [0, 0, 0, 0]
            }
        }

    elif "monotonic" in name:
        expected_set = {
            (0, 0, 0, 0): {
                "Rj": [[], [], [], []],
                "score": [0, 0, 0, 0]
            },
            (1, 1, 1, 1): {
                "Rj": [[[0, 0, 0, 0,]], 
                       [[0, 0, 0, 0], [1, 0, 0, 0]], 
                       [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0]], 
                       [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]], 
                "score": [1, 1, 1, 0.75]
            }
        }
    else:
        expected_set = {
            (0, 0, 0, 0): {
                "Rj": [[[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]],
                       [[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]],
                       [[1, 1, 1, 0], [1, 1, 1, 1]],
                       [[1, 1, 1, 1]]],
                "score": [0.5, 1/3, 0, 0]
            },
            (1, 1, 1, 1): {
                "Rj": [[[0, 0, 0, 0,]], 
                       [[0, 0, 0, 0], [1, 0, 0, 0]], 
                       [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0]], 
                       [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]], 
                "score": [1, 1, 1, 0.75]
            }
        }

    out = {
        "X": X,
        "A": A,
        "clf": clf,
        "expected_set": expected_set,
    }
    
    return out

def test_enumerating_scorer(discrete_test_case):
    X = discrete_test_case["X"]
    A = discrete_test_case["A"]
    clf = discrete_test_case["clf"]
    expected_set = discrete_test_case["expected_set"]

    scorer = ResponsivenessScorer(
        action_set=A,
        method="enumerate"
    )

    for i, x in enumerate(X.values):
        score = scorer.score(x, clf)
        # check score
        assert np.isclose(score, expected_set[tuple(x)]["score"]).all()

        # check interventions
        for j in range(len(x)):
            inter = scorer.inter[scorer._get_inter_key(x)].get(j, np.array([]))

            if inter.size == 0:
                assert expected_set[tuple(x)]["Rj"][j] == []
            else:
                assert np.array_equal(
                        np.sort(x + inter, axis=0), 
                        np.array(expected_set[tuple(x)]["Rj"][j], dtype=np.float64)
                    )

def test_discrete_sampling(discrete_test_case):
    X = discrete_test_case["X"]
    A = discrete_test_case["A"]
    clf = discrete_test_case["clf"]
    expected_set = discrete_test_case["expected_set"]

    scorer = ResponsivenessScorer(
        action_set=A,
        method="sample",
    )

    for i, x in enumerate(X.values):
        score = scorer.score(x, clf, n=1)
        # check interventions
        for j in range(len(x)):
            inter = scorer.inter[scorer._get_inter_key(x)].get(j, np.array([]))

            if inter.size == 0:
                assert expected_set[tuple(x)]["Rj"][j] == []
            else:
                assert np.array(expected_set[tuple(x)]["Rj"][j], dtype=np.float64) in x + inter