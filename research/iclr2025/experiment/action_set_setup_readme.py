# fmt: off
import os
import sys

sys.path.append(os.getcwd())

from reachml import ActionSet
from reachml import downstream
from reachml.constraints import *
import pandas as pd
from src.ext.data import BinaryClassificationDataset
from src.paths import *
from src.ext import fileutils

contribution_limit = 6500

def setup_readme_action_set() -> ActionSet:
    df = pd.read_csv("readme_example_data.csv")
    data = BinaryClassificationDataset.from_df(df, outcome_id=0)
    strata = df["loanApproved"].to_numpy()
    data.generate_cvindices(
        strata=strata,
        total_folds_for_cv=[1, 3, 5],
        replicates=3,
        seed=0,
    )

    # build action set from features
    A = ActionSet(data.X_df)

    # AGE (not actionable, as you wanted)
    A["age"].actionable = False
    A["age"].lb = 16
    A["age"].ub = 120
    A["age"].step_direction = 1

    # EMPLOYMENT
    A["employed"].lb = 0
    A["employed"].ub = 1

    # INCOME
    A["monthly_income"].lb = 0
    A["monthly_income"].ub = 10_000_000  # example bounds

    # RETIREMENT SAVINGS
    A["retirement_savings"].lb = 0
    A["retirement_savings"].ub = 100_000_000  # example bounds

    # PORTFOLIO
    A["portfolio_allocation"].lb = 0.0
    A["portfolio_allocation"].ub = 1.0

    # -------------------
    # IF–THEN CONSTRAINTS
    # -------------------

    # if not employed -> freeze retirement_savings
    A.constraints.add(
        IfThenConstraint(
            if_condition=Condition("employed", "E", 0),
            then_condition=Condition("retirement_savings", "E", 0),
        )
    )

    # if age >= 73 -> freeze retirement_savings
    A.constraints.add(
        IfThenConstraint(
            if_condition=Condition("age", "G", 73),
            then_condition=Condition("retirement_savings", "E", 0),
        )
    )

    # --- directional linkage part ---
    # your DirectionalLinkage makes retirement_savings the source and age the target
    # that creates a teeny scale for age (= 1/6500) and the original code asserts
    # "target scale must be a multiple of its step". We can make age non-discrete
    # so the compatibility check passes, without making age actionable.
    A["age"].discrete = False

    A.constraints.add(
        constraint=DirectionalLinkage(
            names=["retirement_savings", "age"],
            # source = retirement_savings (scale = 6500)
            # target = age (scale = 1)
            scales=[contribution_limit, 1],
        )
    )

    # save dataset
    fileutils.save(
        data,
        path=get_data_file("readme_example", action_set_name="readme_example_action_set"),
        overwrite=True,
        check_save=False,
    )

    # save action set (use A, not 'action_set')
    fileutils.save(
        A,
        path=get_action_set_file("readme_example", action_set_name="readme_example_action_set"),
        overwrite=True,
        check_save=True,
    )

    return A

if __name__ == "__main__":
    action_set = setup_readme_action_set()
