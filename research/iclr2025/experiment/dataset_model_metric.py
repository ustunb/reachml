import os
import sys

import pandas as pd

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

# percentiles
Q = [10, 50, 90]


def main():
    metric = {}

    denied = audit_df[~audit_df["orig_prediction"]]

    # 1. num people denied (metric: ratio/percentage)
    n_denied = data.cnt[denied.index].sum()
    r_denied = n_denied / data.n

    metric["r_denied"] = r_denied

    # 2. people with fixed predictions (among denied) (metric: ratio/percentage)
    fixed = denied[denied["flip_action_idx"].apply(len) == 0]
    n_fixed = data.cnt[fixed.index].sum()
    r_fixed = n_fixed / n_denied

    metric["r_denied_fixed"] = r_fixed

    # 3. people with n-D recourse but not 1-D (metric: ratio/percentage)
    resp_df = fileutils.load(get_resp_df_file(**settings))
    cnt_1D = resp_df.groupby("u_index")[["resp"]].sum().reset_index()
    not_fixed_df = cnt_1D[
        cnt_1D["u_index"].isin(denied[denied["recourse_exists"]].index)
    ]
    nD_only = not_fixed_df[not_fixed_df["resp"] == 0]

    n_nD_only = data.cnt[nD_only.index].sum()
    r_nD_only = n_nD_only / n_denied

    metric["r_nD_recourse_only"] = r_nD_only

    # # 4. number of 1D/nD actions that lead to recourse per person (metric: percentiles)
    # den_cnts = data.cnt[denied.index]

    # denied_1D = cnt_1D[cnt_1D['u_index'].isin(denied.index)]
    # action_cnt_df = denied.merge(denied_1D, left_index=True, right_on='u_index')
    # action_cnt_df['total_resp'] = action_cnt_df['flip_action_idx'].apply(len)
    # action_cnt_df['nD_resp'] = action_cnt_df['total_resp'] - action_cnt_df['resp']

    # perct = np.percentile(
    #     np.repeat(
    #         action_cnt_df[['resp', 'nD_resp']].values,
    #         den_cnts,
    #         axis=0
    #         ),
    #     Q,
    #     axis=0
    # )

    # for i, q in enumerate(Q):
    #     metric[f'n_1D_actions_recourse_percentile_{q}'] = perct[i,0]
    #     metric[f'n_nD_actions_recourse_percentile_{q}'] = perct[i,1]

    fileutils.save(metric, get_metrics_file(**settings), overwrite=True)


if __name__ == "__main__":
    main()
