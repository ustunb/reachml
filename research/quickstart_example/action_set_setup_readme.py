from reachml import ActionSet
from reachml import downstream
from reachml.constraints import *
import pandas as pd
from neurips2025.src.data import BinaryClassificationDataset

contribution_limit = 6500

def setup_readme_action_set() -> ActionSet:
    df = pd.read_csv("readme_example_data.csv")
    data = BinaryClassificationDataset.from_df(df)


    A = ActionSet(data)
    A["loanApproved"].actionable = False
    A["age"].actionable = False
    A["age"].lb = 16
    A["age"].ub = 120
    A["employed"].lb = 0
    A["employed"].ub = 1
    A["monthly_income"].lb = 0
    A["monthly_income"].ub = 10000000 # Example bounds
    A["retirement_savings"].lb = 0
    A["retirement_savings"].ub = 120 * contribution_limit  # Example bounds
    A["portfolio_allocation"].lb = 0.0
    A["portfolio_allocation"].ub = 1.0

    # Set up constraints
    # Example: Age can only increase
    A["age"].step_direction = 1
	
    A.constraints.add(
		constraint = IfThenConstraint(
            A["employed"] == 0,
            A["retirement_savings"].step_direction <= 0
        )
    )

    A.constraints.add(
        constraint = IfThenConstraint(
            A["age"] >= 73,
            A["retirement_savings"].step_direction <= 0
        )
    )

    A.constraints.add(
        constraint=DirectionalLinkage(
            names=["age", "retirement_savings"],
            # source = age (scale = 1)
            # target = retirement_savings (scale = contribution_limit)
            scales=[1, contribution_limit],
        )
    )

    A.constraints.add(
        constraint = 
    )

    return A

if __name__ == "__main__":
    action_set = setup_readme_action_set()
    downstream.utils.fileutils.save(action_set, "readme_example_action_set.actionset")

