# reachml

[![python](https://img.shields.io/badge/Python-3.10-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2308.12820-b31b1b.svg)](https://arxiv.org/abs/2308.12820)
[![CI](https://github.com/ustunb/reachml/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ustunb/reachml/actions/workflows/ci.yml)

`reach-ml` is a library for recourse verification. 

## Background

*Recourse* is the ability of a decision subject to change the prediction of a machine learning model through actions on its features. *Recourse verification* aims to tell if a decision subject is assigned a prediction that is fixed.  

## Installation

The library relies on [IBM CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio). As
of time of writing, the free version of CPLEX has two issues: (1) it does not work well with **Mac
M1 architecture**, and (2) it has a limit on the number of constraints compared to the
commercial/academic version. For most useful use cases, you want to install the package followed by
an installation of the full version of CPLEX:
```
pip install git+https://github.com/ustunb/reachml#egg=reachml
python path/to/cplex/setup.py install
```
where the path to the installer could be, e.g., `/opt/ibm/ILOG/CPLEX_Studio221/python/`.

You can install the version of the library packaged with free CPLEX right away:
```
pip install "git+https://github.com/ustunb/reachml#egg=reachml[cplex]"
```

## Quickstart

The following example shows how to specify actions over the features in a classification task using an `ActionSet`. Given the `ActionSet` and a set of feature vectors $X$, we can then construct a `ReachableSet` for each point.

```python
import pandas as pd
from reachml import ActionSet, ReachableSet, ReachableDatabase
from reachml.constraints import OneHotEncoding, DirectionalLinkage

# Simple toy dataset with 3 points and 
X = pd.DataFrame(
    {
        "age": [32, 19, 52],
        "marital_status": [1, 0, 0],
        "years_since_last_default": [5, 0, 21],
        "job_type_a": [0, 1, 1], # categorical feature with one-hot encoding
        "job_type_b": [1, 0, 0],
        "job_type_c": [0, 0, 0],
    }
)

# Create an action set
action_set = ActionSet(X)

# Specify constraints on individual features
action_set["age"].actionable = False # cannot change age
action_set["marital_status"].actionable = False # should not change marital status
action_set["years_since_last_default"].step_direction = 1 # can only increase
action_set["years_since_last_default"].step_ub = 1 # should have recourse within 1 year

# Capture a one hot-encoding for job-type
action_set.constraints.add(
    constraint=OneHotEncoding(names=["job_type_a", "job_type_b", "job_type_c"])
)

# Capture deterministic causal changes - if `years_since_last_default` increases, `age` must increase
# This constraint will ensure that  `age` will change even though it is not actionable
action_set.constraints.add(
    constraint=DirectionalLinkage(
        names=["years_since_last_default", "age"], scales=[1, 1]
    )
)

print(action_set)
# should return the following output
##+---+--------------------------+--------+------------+----+----+----------------+---------+---------+
##|   | name                     |  type  | actionable | lb | ub | step_direction | step_ub | step_lb |
##+---+--------------------------+--------+------------+----+----+----------------+---------+---------+
##| 0 | age                      | <int>  |   False    | 19 | 52 |              0 |         |         |
##| 1 | marital_status           | <bool> |   False    | 0  | 1  |              0 |         |         |
##| 2 | years_since_last_default | <int>  |    True    | 0  | 21 |              1 |       1 |         |
##| 3 | job_type_a               | <bool> |    True    | 0  | 1  |              0 |         |         |
##| 4 | job_type_b               | <bool> |    True    | 0  | 1  |              0 |         |         |
##| 5 | job_type_c               | <bool> |    True    | 0  | 1  |              0 |         |         |
##+---+--------------------------+--------+------------+----+----+----------------+---------+---------+

# `ActionSet` infers absolute lower and upper bounds from the dataset so you will have to correct these manually
action_set["years_since_last_default"].ub = 100

# validate the action set
assert action_set.validate(data)

# Create the database of reachable sets for all points in a given dataset,
# and save it to ./reachable_db.h5 file
db = ReachableSetDatabase(action_set, path="reachable_db.h5")
db.generate(data, overwrite=True)

# Get the reachable set of a point
x = data.iloc[0]
reachable_set = db[x]
print(reachable_set)` should return the following output:
##    age  marital_status  years_since_last_default  job_type_a  job_type_b  job_type_c
## 0  32.0             1.0                       5.0         0.0         1.0         0.0
## 1  32.0             1.0                       5.0         0.0         0.0         1.0
## 2  32.0             1.0                       5.0         1.0         0.0         0.0
## 3  33.0             1.0                       6.0         0.0         0.0         1.0
## 4  33.0             1.0                       6.0         0.0         1.0         0.0
## 5  33.0             1.0                       6.0         1.0         0.0         0.0

# Check if the point is assigned a fixed prediction
np.any(clf.predict(reachable_set.X))
```
Given a reachable set and a classifier `clf`, you can check if a point has recourse as `np.any(clf.predict(reachable_set.X))`

For more examples, check out [this
script](https://github.com/ustunb/reachml/blob/main/research/iclr2024/scripts/setup_dataset_actionset_fico.py) which sets up the action set for the FICO dataset.

### Resources and Citation

For more about recourse verification, check out our paper ICLR 2024:

[Prediction without Preclusion](https://openreview.net/forum?id=SCQfYpdoGE)

The code to accompany the paper is available under `[research/iclr2024](https://github.com/ustunb/reachml/tree/main/research/iclr2024/`


If you use this library in your research, we would appreciate a citation:
```
@inproceedings{
kothari2024prediction,
title={Prediction without Preclusion: Recourse Verification with Reachable Sets},
author={Avni Kothari and Bogdan Kulynych and Tsui-Wei Weng and Berk Ustun},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=SCQfYpdoGE}
}
```

