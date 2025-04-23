import argparse
import os
import sys

import numpy as np
import pandas as pd
import psutil

sys.path.append(os.getcwd())

from src.ext import fileutils
from src.paths import *

from reachml.database import ReachableSetDatabase

DB_ACTION_SET_NAME = "complex_nD"

settings = {
    "data_name": "german",
    "action_set_name": "complex_nD",
    "model_type": "xgb",
    "method_name": "reach",
    "explainer_type": "rij",
    "overwrite": False,
    "hard_cap": None,
}

ppid = os.getppid()
process_type = psutil.Process(ppid).name()  # e.g. pycharm, bash
if process_type not in ("Code Helper (Plugin)"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--action_set_name", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--explainer_type", type=str, required=True)
    parser.add_argument("--hard_cap", type=int, default=settings["hard_cap"])
    parser.add_argument(
        "--overwrite", default=settings["overwrite"], action="store_true"
    )
    args, _ = parser.parse_known_args()
    settings.update(vars(args))

# load files
data = fileutils.load(get_data_file(**settings))
action_set = fileutils.load(get_action_set_file(**settings))
db = ReachableSetDatabase(
    action_set=action_set,
    path=get_reachable_db_file(
        data_name=settings["data_name"], action_set_name=DB_ACTION_SET_NAME
    ),
)
audit_results = fileutils.load(get_audit_results_file(**settings))
audit_df = pd.DataFrame.from_dict(audit_results, orient="index")
infeasible_pts = set(data.inv[audit_df[~audit_df["recourse_exists"]].index])

# load model
model_results = fileutils.load(get_model_file(**settings))
clf = model_results["model"]
scaler = model_results["scaler"]
reformat = lambda x: x.reshape(1, -1) if x.ndim == 1 else x
if scaler is None:
    rescale = lambda x: reformat(x)
else:
    rescale = lambda x: scaler.transform(reformat(x))


def compute_plot_data():
    neg_pred_idx = audit_df[~audit_df["orig_prediction"]].index
    neg_inv_idx = np.unique(data.inv[neg_pred_idx])

    rij_df = fileutils.load(get_resp_df_file(**settings))
    rij_df = rij_df[rij_df["u_index"].isin(neg_inv_idx)]

    if settings["explainer_type"] == "rij":
        rij_df["exp_abs"] = rij_df["rij"]
        melted = (
            pd.DataFrame(index=neg_inv_idx, columns=np.arange(data.d))
            .melt(ignore_index=False)
            .reset_index()
            .rename(columns={"variable": "feature"})
            .drop(columns="value")
        )
    else:
        exp_obj = fileutils.load(get_explainer_file(**settings))
        df = exp_obj.get_explanations()["values"]
        df.rename(
            {feat: idx for idx, feat in enumerate(exp_obj.data.names.X)},
            axis=1,
            inplace=True,
        )
        df = df.iloc[neg_inv_idx]

        melted = df.melt(
            var_name="feature", value_name="exp", ignore_index=False
        ).reset_index()
        melted["exp_abs"] = melted["exp"].abs()

    mrged = melted.merge(
        rij_df,
        left_on=["index", "feature"],
        right_on=["u_index", "feature"],
        how="left",
    )
    mrged.fillna(0, inplace=True)
    mrged["exp_rank"] = mrged.groupby("index")[["exp_abs"]].rank(
        ascending=False, method="first"
    )
    mrged = mrged[mrged["exp_abs"] > 0]

    mrged["count"] = mrged["index"].map(lambda x: data.cnt[x])

    # rank and rij_zero counts (for each rank) for each feature
    data_df = (
        mrged.groupby(["feature", "exp_rank"])
        .agg(
            # counting all instances
            count_rank=("count", "sum"),
            # counting non-responsive instances
            count_rij_zero=("count", lambda x: x[mrged["rij"] == 0].sum()),
        )
        .reset_index()
    )

    data_df["feature"] = data_df["feature"].map(lambda x: data.names.X[x])

    return data_df


def main():
    fileutils.save(compute_plot_data(), get_plot_data_file(**settings), overwrite=True)


if __name__ == "__main__":
    main()
