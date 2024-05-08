# reachml

[![python](https://img.shields.io/badge/Python-3.10-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2308.12820-b31b1b.svg)](https://arxiv.org/abs/2308.12820)
[![CI](https://github.com/ustunb/reachml/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ustunb/reachml/actions/workflows/ci.yml)


Library for recourse verification and the accompanying code for the paper "[Prediction without Preclusion: Recourse Verification with Reachable Sets](https://arxiv.org/abs/2308.12820)" (ICLR 2024 Spotlight).

## Getting Started

### Installing the Library
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

If you want to use simple applications and are not using the Mac M1 architecture, you can install
the version of the library packaged with free CPLEX right away:
```
pip install git+https://github.com/ustunb/reachml#egg=reachml[cplex]
```

## Quickstart

### Defining the Actionability Constraints

The first step to recourse verification through reachable sets is defining the actionability
constraints. This is done by configuring an `ActionSet` object.
```python
import pandas as pd

from reachml import ActionSet
from reachml.constraints import OneHotEncoding, DirectionalLinkage

# An example dataset in credit scoring.
data = pd.DataFrame(
    {
        # Simple features.
        "age": [32, 19, 52],
        "marital_status": [1, 0, 0],
        "years_since_last_default": [5, 0, 21],
        # A one-hot encoded job type feature.
        "job_type_a": [0, 1, 1],
        "job_type_b": [1, 0, 0],
        "job_type_c": [0, 0, 0],
    }
)

# Let's encode some inherent actionability constraints in this data.
action_set = ActionSet(data)

# We don't consider actions that increase age.
action_set["age"].actionable = False

# We do not consider actions that change the marital status.
action_set["marital_status"].actionable = False

# We assume individuals can change job types, and so we nave to preserve one-hot encoding.
action_set.constraints.add(
    constraint=OneHotEncoding(names=["job_type_a", "job_type_b", "job_type_c"])
)

# We only consider actions that increase the years since the last default if it happened
action_set["years_since_last_default"].step_direction = +1
# ...and we only consider actions that make the individual wait for up to one year.
action_set["years_since_last_default"].step_ub = 1

# If years_since_last_default increases, age also has to increase.
action_set.constraints.add(
    constraint=DirectionalLinkage(
        names=["years_since_last_default", "age"], scales=[1, 1]
    )
)

# Validate that the dataset matches the constraints.
assert action_set.validate(data)
print(action_set)
```

You should see the the following output:
```
+---+--------------------------+--------+------------+----+----+----------------+---------+---------+
|   | name                     |  type  | actionable | lb | ub | step_direction | step_ub | step_lb |
+---+--------------------------+--------+------------+----+----+----------------+---------+---------+
| 0 | age                      | <int>  |   False    | 19 | 52 |              0 |         |         |
| 1 | marital_status           | <bool> |   False    | 0  | 1  |              0 |         |         |
| 2 | years_since_last_default | <int>  |    True    | 0  | 21 |              1 |       1 |         |
| 3 | job_type_a               | <bool> |    True    | 0  | 1  |              0 |         |         |
| 4 | job_type_b               | <bool> |    True    | 0  | 1  |              0 |         |         |
| 5 | job_type_c               | <bool> |    True    | 0  | 1  |              0 |         |         |
+---+--------------------------+--------+------------+----+----+----------------+---------+---------+
```
Note that by default the absolute lower and upper bounds of numeric features are inferred from the
input dataset, and if the actual upper bound of an actionable feature should be different, you
have to override it:
```
action_set["years_since_last_default"].ub = 100
```

### Generating Reachable Sets
For a given initial feature vector, a reachable set is the set of all other feature vectors that
are achievable through actions applied to the initial feature vectore. To generate a reachable set, use a
`ReachableSetDatabase`:

```python
from reachml import ReachableSetDatabase
# Generate the database of reachable sets for all points in a given dataset,
# and save it to ./reachable_db.h5 file
db = ReachableSetDatabase(action_set, path="reachable_db.h5")
db.generate(data, overwrite=True)
```
Note that by default the generation will not overwrite existing reachable sets even if the action
set has changed.

You can now retrieve the generated reachable sets from the database by calling `db[x]` with
a given initial feature vector `x`:
```
# Get the reachable set of the first example.
reachable_set = db[data.iloc[0]]
print(pd.DataFrame(reachable_set.X, columns=data.columns))
```

You should see the following output:
```
    age  marital_status  years_since_last_default  job_type_a  job_type_b  job_type_c
0  32.0             1.0                       5.0         0.0         1.0         0.0
1  32.0             1.0                       5.0         0.0         0.0         1.0
2  32.0             1.0                       5.0         1.0         0.0         0.0
3  33.0             1.0                       6.0         0.0         0.0         1.0
4  33.0             1.0                       6.0         0.0         1.0         0.0
5  33.0             1.0                       6.0         1.0         0.0         0.0
```

The 0-th entry is the original feature vector `x`. As you can see, the reachable set contains
feature vectors obtained via changing the job type, and increasing `years_since_last_default`.
Although we have set age to be immutable, because `years_since_last_default` is mutable and is
linked to `age` through the `DirectionalLinkage` constraint, it wil change.

### Using Reachable Sets
Having the reachable set as a matrix of points, i.e., you can analyze it or use it to query
a model to audit for its sensitivity to specified actions, e.g., checking that there is
any positive prediction for points in the reachable set `np.any(clf.predict(reachable_set.X))`.

### More Examples
Check out [this
script](https://github.com/ustunb/reachml/blob/main/iclr2024/scripts/setup_dataset_actionset_fico.py)
which sets up the action set for the Fico dataset for a realistic example.
