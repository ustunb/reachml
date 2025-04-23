import argparse
import os
import sys

import pandas as pd
import rich
from rich.panel import Panel

sys.path.append(os.getcwd())
import subprocess

from src.ext import fileutils
from src.paths import *

os.chdir(paper_dir)

datasets = ["german", "givemecredit", "fico"]
models = ["logreg", "rf", "xgb"]
explainers = ["resp", "LIME", "SHAP", "LIME_actionAware", "SHAP_actionAware"]

ACTION_SET = "complex_nD"

cols = [
    "data_name",
    "action_set_name",
    "model_type",
    "explainer_type",
    "metric",
    "value",
]


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", type=str, default=datasets)
    parser.add_argument("--models", nargs="+", type=str, default=models)
    parser.add_argument(
        "--explainers",
        nargs="+",
        type=str,
        default=explainers,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    df_lst = []

    for dataset in args.datasets:
        for model in models:
            dset_model_metric_f = (
                results_dir / f"{dataset}_{ACTION_SET}_{model}.metrics"
            )

            if not dset_model_metric_f.exists() or args.overwrite:
                rich.print(
                    Panel(
                        f"Generating dataset + model metric for {dataset} + {model}..."
                    )
                )
                subprocess.run(
                    [
                        "python3",
                        "experiment/dataset_model_metric.py",
                        f"--data_name={dataset}",
                        f"--action_set_name={ACTION_SET}",
                        f"--model_type={model}",
                    ]
                )

            dset_model_metric = fileutils.load(dset_model_metric_f)
            dset_model_metric.update(model_metric(dataset, ACTION_SET, model))
            dm_metric_df = (
                pd.DataFrame.from_dict(dset_model_metric, orient="index")
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

            for exp in args.explainers:
                dset_model_exp_metric_f = (
                    results_dir / f"{dataset}_{ACTION_SET}_{model}_{exp}.metrics"
                )

                if not dset_model_exp_metric_f.exists() or args.overwrite:
                    rich.print(
                        Panel(
                            f"Generating dataset + model + explainer metric for {dataset} + {model} + {exp}..."
                        )
                    )
                    subprocess.run(
                        [
                            "python3",
                            "experiment/dataset_model_explainer_metric.py",
                            f"--data_name={dataset}",
                            f"--action_set_name={ACTION_SET}",
                            f"--model_type={model}",
                            f"--explainer_type={exp}",
                        ]
                    )

                dset_model_exp_metric = fileutils.load(dset_model_exp_metric_f)
                dme_metric_df = (
                    pd.DataFrame.from_dict(dset_model_exp_metric, orient="index")
                    .reset_index()
                    .rename(columns={"index": "metric", 0: "value"})
                )
                add_columns(
                    dme_metric_df,
                    data_name=dataset,
                    action_set_name=ACTION_SET,
                    model_type=model,
                    explainer_type=exp,
                )

                df_lst.append(dme_metric_df)

    out_df = pd.concat(df_lst)
    out_df = out_df[cols]
    out_df.to_csv(results_dir / "all_metrics.csv", index=False)


if __name__ == "__main__":
    main()
