import os
import sys
import itertools

import pandas as pd
from src.paths import *
from src import fileutils
from src.data import BinaryClassificationDataset

audit_output_file = results_dir / "audit_stats.csv"
model_output_file = results_dir / "model_stats.csv"
data_output_file = results_dir / "data_stats.csv"

# Merge audit stats.
all_stats_files = [f for f in results_dir.glob("*.stats")]
all_stats_csvs = [pd.read_csv(f) for f in all_stats_files]
pd.concat(all_stats_csvs).to_csv(audit_output_file, index=False)
print("audit stats saved to:", audit_output_file)


# Merge model stats.
datasets = ["fico", "german", "givemecredit"]
action_sets = ["simple_1D", "complex_1D", "complex_nD"]
model_types = ["logreg", "xgb", "rf"]

model_stats = []
for data_name, action_set_name, model_type in itertools.product(
    datasets, action_sets, model_types
):
    try:
        model_file = get_model_file(
            data_name=data_name, action_set_name=action_set_name, model_type=model_type
        )
        model_data = fileutils.load(model_file)

        model_stats.append(
            dict(
                data_name=data_name,
                action_set_name=action_set_name,
                model_type=model_type,
                train_auc=model_data["train"]["auc"],
                test_auc=model_data["test"]["auc"],
                train_error=model_data["test"]["error"],
                test_error=model_data["test"]["error"],
            )
        )

    except Exception as e:
        print("issue with: ", data_name, action_set_name, model_type)
        print(e)

pd.DataFrame(model_stats).to_csv(model_output_file, index=False)
print("model stats saved to:", model_output_file)


# Merge dataset stats.
data_stats = []
for data_name, action_set_name in itertools.product(datasets, action_sets):
    dataset = fileutils.load(
        get_data_file(data_name=data_name, action_set_name=action_set_name)
    )
    data_stats.append(
        dict(
            data_name=data_name,
            action_set_name=action_set_name,
            n=dataset.n,
            d=dataset.d,
        )
    )

pd.DataFrame(data_stats).to_csv(data_output_file, index=False)
print("data stats saved to:", data_output_file)
