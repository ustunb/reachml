import pandas as pd

from reachml import ActionSet
from reachml import ReachableSetDatabase
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

# Generate the database of reachable sets for all points in a given dataset,
# and save it to ./reachable_db.h5 file
db = ReachableSetDatabase(action_set, path="reachable_db.h5")
db.generate(data, overwrite=True)

# Get the reachable set of the first example.
reachable_set = db[data.iloc[0]]
print(pd.DataFrame(reachable_set.X, columns=data.columns))

# The reachable set can be used for querying the model or other analyses.
