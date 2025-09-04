import numpy as np
import pytest
import sys
import os
import pandas as pd

sys.path.append(os.getcwd())

from reachml.action_element import *
from reachml.action_set import ActionSet
from reachml.sampling import *
from reachml.constraints import OneHotEncoding
from reachml.utils import SUPPORTED_SOLVERS

testing_params = [("Actionable", "Actionable"), ("Actionable", "Inactionable"), ("Inactionable", "Actionable"),
                  ("Inactionable", "Monotonic"), ("Actionable", "Monotonic"), ("Monotonic", "Monotonoic"), ("Monotonic", "Actionable"), ("Monotonic", "Inactionable")]

@pytest.fixture(params=testing_params)
def setup_bool_data(request):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    A = ActionSet(X, names=['x1', 'x2'])

    # Unpack the parameter tuple
    x1_status, x2_status = request.param
    A["x1"].actionable = False if x1_status == "Inactionable" else True
    A["x2"].actionable = False if x2_status == "Inactionable" else True
    A["x1"].step_direction = 1 if x1_status == "Monotonic" else 0
    A["x2"].step_direction = 1 if x2_status == "Monotonic" else 0
    return A, x1_status, x2_status


@pytest.mark.parametrize("solver", SUPPORTED_SOLVERS)
def test_bool_data(setup_bool_data, solver):
    A, x1_status, x2_status = setup_bool_data

    x1_actionable = False if x1_status == "Inactionable" else True
    x2_actionable = False if x2_status == "Inactionable" else True
    x1_monotonic = (1 if x1_status == "Monotonic" else 0)
    x2_monotonic = (1 if x2_status == "Monotonic" else 0)

    assert A["x1"].actionable == x1_actionable
    assert A["x2"].actionable == x2_actionable
    assert A["x1"].step_direction == x1_monotonic
    assert A["x2"].step_direction == x2_monotonic

    x = np.array([0,0])
    S = ReachableSetSampler(A, x, solver=solver)
    samples = S.sample(10)
    for sample in samples:
        if x1_actionable == False:
            assert x[0] == 0
        if x2_actionable == False:
            assert x[1] == 0
    
    x = np.array([1, 0])
    S = ReachableSetSampler(A, x, solver=solver)
    samples = S.sample(10)
    for sample in samples:
        if x1_actionable == False:
            assert x[0] == 1
        if x2_actionable == False:
            assert x[1] == 0
        if x1_monotonic == True:
            assert x[0] == 1
    
    x = np.array([1, 1])
    S = ReachableSetSampler(A, x, solver=solver)
    samples = S.sample(10)
    for sample in samples:
        if x1_actionable == False:
            assert x[0] == 1
        if x2_actionable == False:
            assert x[1] == 1
        if x1_monotonic == True:
            assert x[0] == 1
        if x2_monotonic == True:
            assert x[1] == 1
    
    x = np.array([0, 1])
    S = ReachableSetSampler(A, x, solver=solver)
    samples = S.sample(10)
    for sample in samples:
        if x1_actionable == False:
            assert x[0] == 0
        if x2_actionable == False:
            assert x[1] == 1
        if x2_monotonic == True:
            assert x[1] == 1

@pytest.fixture(params=testing_params)
def setup_continuous_data(request):
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    A = ActionSet(X, names=['x1', 'x2'])

    # Unpack the parameter tuple
    x1_status, x2_status = request.param
    A["x1"].actionable = False if x1_status == "Inactionable" else True
    A["x2"].actionable = False if x2_status == "Inactionable" else True
    A["x1"].step_direction = 1 if x1_status == "Monotonic" else 0
    A["x2"].step_direction = 1 if x2_status == "Monotonic" else 0
    return A, x1_status, x2_status

@pytest.mark.parametrize("solver", SUPPORTED_SOLVERS)
def test_continuous_data(setup_continuous_data, solver):
    A, x1_status, x2_status = setup_continuous_data

    x1_actionable = False if x1_status == "Inactionable" else True
    x2_actionable = False if x2_status == "Inactionable" else True
    x1_monotonic = (1 if x1_status == "Monotonic" else 0)
    x2_monotonic = (1 if x2_status == "Monotonic" else 0)

    assert A["x1"].actionable == x1_actionable
    assert A["x2"].actionable == x2_actionable
    assert A["x1"].step_direction == x1_monotonic
    assert A["x2"].step_direction == x2_monotonic

    x = np.array([0.0,0.0])
    S = ReachableSetSampler(A, x, solver=solver)
    samples = S.sample(10)
    for sample in samples:
        if x1_actionable == False:
            assert x[0] == 0
        if x2_actionable == False:
            assert x[1] == 0
        if x1_monotonic == False:
            assert x[1] >= 0
        if x2_monotonic == False:
            assert x[0] >= 0
    
    x = np.array([1.0, 0.0])
    S = ReachableSetSampler(A, x, solver=solver)
    samples = S.sample(10)
    for sample in samples:
        if x1_actionable == False:
            assert x[0] == 1
        if x2_actionable == False:
            assert x[1] == 0
        if x1_monotonic == True:
            assert x[0] == 1
        if x2_monotonic == True:
            assert x[1] >= 0
    
    x = np.array([1, 1])
    S = ReachableSetSampler(A, x, solver=solver)
    samples = S.sample(10)
    for sample in samples:
        if x1_actionable == False:
            assert x[0] == 1
        if x2_actionable == False:
            assert x[1] == 1
        if x1_monotonic == True:
            assert x[0] == 1
        if x2_monotonic == True:
            assert x[1] == 1
    
    x = np.array([0, 1])
    S = ReachableSetSampler(A, x, solver=solver)
    samples = S.sample(10)
    for sample in samples:
        if x1_actionable == False:
            assert x[0] == 0
        if x2_actionable == False:
            assert x[1] == 1
        if x2_monotonic == True:
            assert x[1] == 1
        if x1_monotonic == True:
            assert x[0] >= 0.0

@pytest.mark.parametrize("solver", SUPPORTED_SOLVERS)
def test_categorical_data(solver):
    data = {"color":["red", "blue", "green"], "true":["yes", "no", "maybe"]}
    cat_df = pd.DataFrame(data=data)

    cat_df = pd.get_dummies(cat_df, columns=["color", "true"])
    X = cat_df.to_numpy().astype(int)

    A = ActionSet(X, names=["color_red", "color_blue", "color_green", "true_yes", "true_no", "true_maybe"])

    A.constraints.add(OneHotEncoding(names=["color_red", "color_blue", "color_green"]))
    A.constraints.add(OneHotEncoding(names = [ "true_yes", "true_no", "true_maybe"]))
    x = np.array([0, 1, 0, 1, 0, 0])
    S = ReachableSetSampler(A, x)
    samples = S.sample(25)

    for sample in samples:
        if sample[0] == [1]:
            assert sample[1] == 0
            assert sample[2] == 0
        if sample[1] == 1:
            assert sample[0] == 0
            assert sample[2] == 0
        if sample[2] == 1:
            assert sample[0] == 0
            assert sample[1] == 0
        if sample[3] == 1:
            assert sample[4] == 0
            assert sample[5] == 0
        if sample[4] == 1:
            assert sample[3] == 0
            assert sample[5] == 0
        if sample[5] == 1:
            assert sample[4] == 0
            assert sample[3] == 0
    
    A["color_blue"].actionable = False
    A["true_yes"].actionable = False
    x = np.array([0, 1, 0, 1, 0, 0])
    S = ReachableSetSampler(A, x, solver=solver)
    samples = S.sample(25)

    for sample in samples:
        assert sample[0] == 0
        assert sample[1] == 1
        assert sample[2] == 0
        assert sample[3] == 1
        assert sample[4] == 0
        assert sample[5] == 0
    

@pytest.mark.parametrize("solver", SUPPORTED_SOLVERS)
def test_bool_and_continuous(solver):
    X = np.array([[0, 0.0], [0, 0.5], [1, 0.0], [1, 0.5], [0, 1.0], [1, 1.0]])
    A = ActionSet(X, names=['x1', 'x2'])
    A["x1"].actionable = False
    x = np.array([0,0.5])
    S = ReachableSetSampler(A, x, solver=solver)
    samples = S.sample(10)
    for sample in samples:
        assert sample[0] == 0

    A["x1"].actionable = True
    A["x2"].actionable = False
    S = ReachableSetSampler(A, x, solver=solver)
    samples = S.sample(10)
    for sample in samples:
        assert sample[1] == 0.5
        assert (sample[0] == 1 or sample[0] == 0)

    A["x2"].actionable = True
    A["x2"].step_direction = 1
    S = ReachableSetSampler(A, x, solver=solver)
    samples = S.sample(10)
    for sample in samples:
        assert sample[1] >= 0.5

@pytest.mark.parametrize("solver", SUPPORTED_SOLVERS)
def test_continuous_and_cat(solver):
    data = {"color":["red", "blue", "green"]}
    cat_df = pd.DataFrame(data=data)

    cat_df = pd.get_dummies(cat_df, columns=["color"])
    X = cat_df.to_numpy().astype(int)
    test = np.array([[0.33], [0.66], [1]])
    result = np.hstack((test, X))
    A = ActionSet(result, names=["x1", "color_red", "color_blue", "color_green"])
    A.constraints.add(OneHotEncoding(names=["color_red", "color_blue", "color_green"]))
    x = np.array([0.5, 0, 1, 0])
    S = ReachableSetSampler(A, x, solver=solver)
    samples = S.sample(30)
    for sample in samples:
        if sample[1] == 1:
            assert sample[2] == 0
            assert sample[3] == 0
        if sample[2] == 1:
            assert sample[1] == 0
            assert sample[3] == 0
        if sample[3] == 1:
            assert sample[1] == 0
            assert sample[2] == 0
        assert (sample[0] <= 1.0 and sample[0] >= 0.33)
    
    A["color_red"].actionable = False
    A["x1"].step_direction = 1
    x = np.array([0.5, 0, 1, 0])
    S = ReachableSetSampler(A, x, solver=solver)
    samples = S.sample(30)
    for sample in samples:
        assert sample[1] == 0
        assert ((sample[2] == 0 and sample[3] == 1) or (sample[2] == 1 and sample[3] == 0))
        assert (sample[0] <= 1.0 and sample[0] >= 0.5)


@pytest.mark.parametrize("solver", SUPPORTED_SOLVERS)
def test_bool_and_cat(solver):
    data = {"color":["red", "blue", "green"]}
    cat_df = pd.DataFrame(data=data)
    cat_df = pd.get_dummies(cat_df, columns=["color"])
    X = cat_df.to_numpy().astype(int)
    test = np.array([[0], [1], [1]])
    result = np.hstack((test, X))
    A = ActionSet(result, names=["x1", "color_red", "color_blue", "color_green"])
    A.constraints.add(OneHotEncoding(names=["color_red", "color_blue", "color_green"]))
    A["x1"].actionable = False
    x = np.array([0, 0, 1, 0])
    S = ReachableSetSampler(A, x, solver=solver)
    samples = S.sample(25)
    for sample in samples:
        assert sample[0] == 0
        if sample[1] == 1:
            assert sample[2] == 0
            assert sample[3] == 0
        if sample[2] == 1:
            assert sample[1] == 0
            assert sample[3] == 0
        if sample[3] == 1:
            assert sample[1] == 0
            assert sample[2] == 0
    
    A["x1"].actionable = True
    x = np.array([1, 0, 1, 0])
    A["x1"].step_direction = 1
    A["color_green"].actionable = False
    S = ReachableSetSampler(A, x, solver=solver)
    samples = S.sample(25)
    for sample in samples:
        assert sample[0] == 1
        if sample[1] == 1:
            assert (sample[2] == 0 and sample[3] == 0)
        elif sample[2] == 1:
            assert (sample[1] == 0 and sample[3] == 0)
        else:
            assert sample[3] == 0

@pytest.mark.parametrize("solver", SUPPORTED_SOLVERS)
def test_sampling_accuracy(solver):
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    A = ActionSet(X, names=['x1', 'x2'])

    x = np.array([0.5, 0.5])
    S = ReachableSetSampler(A, x, solver=solver)
    samples = S.sample(1000)
    upperx = 0.75
    uppery = 1.0
    lowerx = 0.5
    lowery = 0.5

    area = (upperx-lowerx)*(uppery-lowery)
    count = 0
    for sample in samples:
        if (((lowerx <= sample[0]) and (sample[0] <= upperx)) and ((lowery <= sample[1]) and (sample[1] <= uppery))):
            count+= 1
    
    error = 0.05
    assert (((count/1000) >= area - error) and ((count/1000) <= area + error))

    A["x1"].step_direction = 1
    S = ReachableSetSampler(A, x, solver=solver)
    samples = S.sample(2000)
    count = 0
    for sample in samples:
        if (((lowerx <= sample[0]) and (sample[0] <= upperx)) and ((lowery <= sample[1]) and (sample[1] <= uppery))):
            count+= 1
    assert (((count/2000) >= 2*area - error) and ((count/2000) <= 2*area + error))

    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    A = ActionSet(X, names=['x1', 'x2'])

    n_samples = 50000
    x = np.array([0.0, 0.0])
    S = ReachableSetSampler(A, x, solver=solver)
    samples = S.sample(n_samples)
    x_1 = 0.5
    x_2 = 0.0

    count = 0
    for sample in samples:
        if ((sample[0] >= x_1) and sample[1] >= x_2):
            count+=1
    assert ((count/n_samples) >= 0.485 and (count/n_samples) <= 0.515)

    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    A = ActionSet(X, names=['x1', 'x2'])
    x = np.array([1.0, 0.0])
    S = ReachableSetSampler(A, x, solver=solver)
    samples = S.sample(2500)
    upperx = 1
    uppery = 0.75
    lowerx = 0.5
    lowery = 0.5

    area = (upperx-lowerx)*(uppery-lowery)
    count = 0
    for sample in samples:
        if (((lowerx <= sample[0]) and (sample[0] <= upperx)) and ((lowery <= sample[1]) and (sample[1] <= uppery))):
            count+= 1
    error = 0.05
    assert (((count/2500) >= area - error) and ((count/2500) <= area + error))

    data = {"color":["red", "blue", "green"], "true":["yes", "no", "maybe"]}
    cat_df = pd.DataFrame(data=data)

    cat_df = pd.get_dummies(cat_df, columns=["color", "true"])
    X = cat_df.to_numpy().astype(int)

    A = ActionSet(X, names=["color_red", "color_blue", "color_green", "true_yes", "true_no", "true_maybe"])

    A.constraints.add(OneHotEncoding(names=["color_red", "color_blue", "color_green"]))
    A.constraints.add(OneHotEncoding(names = [ "true_yes", "true_no", "true_maybe"]))
    x = np.array([0, 1, 0, 1, 0, 0])
    S = ReachableSetSampler(A, x, solver=solver)
    samples = S.sample(3000)
    counts = {}
    for sample in samples:
        for i in range(len(sample)):
            if sample[i] == 1:
                if i not in counts.keys():
                    counts[i] = 1
                else:
                    counts[i] += 1
    
    for i in counts.keys():
        assert (counts[i] >= 800 and counts[i] <= 1150)

@pytest.mark.parametrize("solver", SUPPORTED_SOLVERS)
def test_1d_discrete(solver):
    X = np.array([[0, 1], [1, 1]])
    A = ActionSet(X, names=['x1'])

    x = np.array([0])
    S = ReachableSetSampler(A, x, solver=solver)
    samples = S.sample(100)
    for sample in samples:
        assert (x[0] == 0 or x[0] == 1)
    
    A['x1'].actionable = True
    A['x1'].step_direction = 1
    S = ReachableSetSampler(A, x, solver=solver)
    samples = S.sample(5000)
    for sample in samples:
        assert x[0] == 0 or x[0] == 1
    
    x = np.array([1])
    A['x1'].actionable = True
    A['x1'].step_direction = 1
    S = ReachableSetSampler(A, x, solver=solver)
    samples = S.sample(5000)
    for sample in samples:
        assert x[0] == 1

@pytest.mark.parametrize("solver", SUPPORTED_SOLVERS)
def test_1d_continuous(solver):
    X = np.array([[0.5], [200]])
    A = ActionSet(X, names=['x1'])

    x = np.array([3])
    S = ReachableSetSampler(A, x, solver=solver)
    samples = S.sample(100)
    for sample in samples:
        assert (x >= 0.5 and x <= 200)

    A["x1"].step_direction = 1
    x = np.array([75])
    samples = S.sample(100)
    for sample in samples:
        assert (x >= 75)
    
    A["x1"].step_direction = -1
    x = np.array([20])
    samples = S.sample(100)
    for sample in samples:
        assert (x <= 20)

if __name__ == "__main__":
    pytest.main()