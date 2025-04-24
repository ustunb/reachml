import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.getcwd())
import argparse

import psutil
from src.ext import fileutils
from src.paths import *

from reachml import ReachableSetDatabase

settings = {
    "data_name": "givemecredit",
    "action_set_name": "complex_nD",
    "model_type": "logreg",
    "explainer_type": "LIME",
    "method_name": "reach",
    "overwrite": False,
    "masker": None,
}

ppid = os.getppid()
process_type = psutil.Process(ppid).name()
if process_type not in ("Code Helper (Plugin)"):  # add your favorite IDE process here
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--explainer_type", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument(
        "--overwrite", default=settings["overwrite"], action="store_true"
    )
    # parser.add_argument("--seed", type=int, default=settings["random_seed"])
    args, _ = parser.parse_known_args()
    settings.update(vars(args))

# load dataset
data = fileutils.load(get_data_file(**settings))

action_set = fileutils.load(get_action_set_file(**settings))
db = ReachableSetDatabase(action_set=action_set, path=get_reachable_db_file(**settings))

audit_results = fileutils.load(get_audit_results_file(**settings))
audit_df = pd.DataFrame.from_dict(audit_results, orient="index")

K = 4
DESC_STATS = ["min", "50%", "max", "mean"]


def main():
    metric = {}

    neg_pred_idx = audit_df[~audit_df["orig_prediction"]].index

    n_denied = data.cnt[neg_pred_idx.values.astype(int)].sum()

    resp_df = fileutils.load(get_resp_df_file(**settings))

    if settings["explainer_type"] == "resp":
        resp_df["exp_abs"] = resp_df["resp"]
        melted = (
            pd.DataFrame(index=neg_pred_idx, columns=np.arange(data.d))
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
        df = df.iloc[neg_pred_idx]

        melted = df.melt(
            var_name="feature", value_name="exp", ignore_index=False
        ).reset_index()
        melted["exp_abs"] = melted["exp"].abs()

    mrged = melted.merge(
        resp_df,
        left_on=["index", "feature"],
        right_on=["u_index", "feature"],
        how="left",
    )
    mrged.fillna(0, inplace=True)
    mrged["exp_rank"] = mrged.groupby("index")[["exp_abs"]].rank(
        ascending=False, method="first"
    )

    top_k = mrged[(mrged["exp_rank"] <= K)]
    top_k = top_k[top_k["exp_abs"] > 0]

    # 1. % Presented with reasons
    n_with_reasons = data.cnt[top_k["index"].unique()].sum()
    r_with_reasons = n_with_reasons / n_denied

    metric["r_with_reasons"] = r_with_reasons

    # 2. % with all invalid
    all_invalid = top_k.groupby("index").filter(lambda x: x["resp"].sum() == 0)
    n_all_invalid = data.cnt[all_invalid["index"].unique()].sum()
    r_all_invalid = n_all_invalid / n_with_reasons

    metric["r_all_invalid"] = r_all_invalid

    # 3. % with at least 1 invalid
    at_least_1_invalid = top_k.groupby("index").filter(
        lambda x: (x["resp"] == 0).sum() > 0
    )
    n_at_least_1_invalid = data.cnt[at_least_1_invalid["index"].unique()].sum()
    r_at_least_1_invalid = n_at_least_1_invalid / n_with_reasons

    metric["r_at_least_1_invalid"] = r_at_least_1_invalid

    # 4. % with at least 1 responsive
    at_least_1_responsive = top_k.groupby("index").filter(
        lambda x: (x["resp"] > 0).sum() > 0
    )
    n_at_least_1_responsive = data.cnt[at_least_1_responsive["index"].unique()].sum()
    r_at_least_1_responsive = n_at_least_1_responsive / n_with_reasons

    metric["r_at_least_1_responsive"] = r_at_least_1_responsive

    # 5. % with all responsive
    all_responsive = top_k.groupby("index").filter(
        lambda x: (x["resp"] > 0).sum() == x["exp_rank"].max()
    )
    n_all_responsive = data.cnt[all_responsive["index"].unique()].sum()
    r_all_responsive = n_all_responsive / n_with_reasons

    metric["r_all_responsive"] = r_all_responsive

    # 6. Number of reasons
    avg_n_reasons = top_k.groupby("index").size().mean()

    metric["avg_n_reasons"] = avg_n_reasons

    fileutils.save(metric, get_metrics_file(**settings), overwrite=True)

    return metric


if __name__ == "__main__":
    main()
