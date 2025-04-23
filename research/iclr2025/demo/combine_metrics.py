import os
import sys

import pandas as pd
import rich
from rich.panel import Panel

sys.path.append(os.getcwd())
import subprocess

from src.ext import fileutils
from src.paths import *

os.chdir(repo_dir)

datasets = ["givemecredit_cts"]
models = ["logreg", "xgb"]
explainers = ["LIME", "SHAP", "LIME_actionAware", "SHAP_actionAware"]

ACTION_SET = "complex_nD"
overwrite = False

cols = [
    "data_name",
    "action_set_name",
    "model_type",
    "explainer_type",
    "metric",
    "value",
    "n",
]

df = pd.DataFrame(columns=cols)


def add_columns(df, **kwargs):
    for k, v in kwargs.items():
        df[k] = v


def model_metric(dataset, action_set, model):
    model = fileutils.load(
        get_model_file(data_name=dataset, action_set_name=action_set, model_type=model)
    )
    model_metrics = {
        "train_error": model["train"]["error"],
        "train_auc": model["train"]["auc"],
        "test_error": model["test"]["error"],
        "test_auc": model["test"]["auc"],
    }

    return model_metrics


df_lst = [df]

for dataset in datasets:
    for model in models:
        mod_met = model_metric(dataset, ACTION_SET, model)
        dm_metric_df = (
            pd.DataFrame.from_dict(mod_met, orient="index")
            .reset_index()
            .rename(columns={"index": "metric", 0: "value"})
        )
        add_columns(
            dm_metric_df,
            data_name=dataset,
            action_set_name=ACTION_SET,
            model_type=model,
        )

        df_lst.append(dm_metric_df)

        for exp in explainers:
            dset_model_exp_metric_f = (
                results_dir / f"{dataset}_{ACTION_SET}_{model}_{exp}_demo_metrics.csv"
            )

            if not dset_model_exp_metric_f.exists() or overwrite:
                rich.print(
                    Panel(
                        f"Generating dataset + model + explainer metric for {dataset} + {model} + {exp}..."
                    )
                )
                subprocess.run(
                    [
                        "python3",
                        "demos/calc_metrics.py",
                        f"--data_name={dataset}",
                        f"--action_set_name={ACTION_SET}",
                        f"--model_type={model}",
                        f"--explainer_type={exp}",
                    ]
                )

            dme_metric_df = pd.read_csv(dset_model_exp_metric_f)
            add_columns(
                dme_metric_df,
                data_name=dataset,
                action_set_name=ACTION_SET,
                model_type=model,
                explainer_type=exp,
            )

            df_lst.append(dme_metric_df)

out_df = pd.concat(df_lst)
out_df.to_csv(results_dir / "demo_all_metrics.csv", index=False)
print(f"Saved to {results_dir / 'demo_all_metrics.csv'}")
