import os
import sys

import numpy as np

sys.path.append(os.getcwd())
import argparse

import psutil
from src.ext import fileutils
from src.paths import *

from reachml import ReachableSetDatabase

settings = {
    "data_name": "german",
    "action_set_name": "complex_nD",
    "model_type": "logreg",
    "method_name": "reach",
    "overwrite": False,
    "masker": None,
}

ppid = os.getppid()
process_type = psutil.Process(ppid).name()
if process_type not in ("Code Helper (Plugin)"):  # add your favorite IDE process here
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--action_set_name", type=str, required=True)
    parser.add_argument(
        "--overwrite", default=settings["overwrite"], action="store_true"
    )
    args, _ = parser.parse_known_args()
    settings.update(vars(args))

# load dataset
data = fileutils.load(get_data_file(**settings))

action_set = fileutils.load(get_action_set_file(**settings))
db = ReachableSetDatabase(action_set=action_set, path=get_reachable_db_file(**settings))

resp_df = fileutils.load(get_resp_df_file(**settings))

act_feats = np.where(action_set.actionable)[0]
ass_feats = list(
    [set(action_set.constraints.get_associated_features(f)) for f in act_feats]
)

# percentiles
Q = [10, 50, 90]


def main():
    metric = {}

    # 1-D, n-D actions (metric: percentile, mean)
    df = resp_df.groupby("u_index")[["marginal_action_count"]].sum()
    df["total_action_count"] = [db[x].actions.shape[0] for x in data.U]
    df["nD_action_count"] = df["total_action_count"] - df["marginal_action_count"]
    n_actions = df[["marginal_action_count", "nD_action_count"]].values

    perct = np.percentile(np.repeat(n_actions, data.cnt, axis=0), Q, axis=0)

    for i, q in enumerate(Q):
        metric[f"n_1D_actions_percentile_{q}"] = perct[i, 0]
        metric[f"n_nD_actions_percentile_{q}"] = perct[i, 1]

    fileutils.save(metric, get_metrics_file(**settings), overwrite=True)


if __name__ == "__main__":
    main()
