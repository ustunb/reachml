import os
import sys
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from psutil import Process

sys.path.append(os.getcwd())

from src.ext import fileutils
from src.paths import *

DB_ACTION_SET_NAME = "complex_nD"
TARGET = 1

settings = {
    "data_name": "givemecredit_cts",
    "action_set_name": "complex_nD",
    "model_type": "xgb",
    "explainer_type": "SHAP",
    "overwrite": False,
}

# parse arguments when script is run from Terminal / not iPython console
if Process(pid=os.getppid()).name() not in ("pycharm"):
    p = ArgumentParser()
    p.add_argument("--data_name", type=str, required=True)
    p.add_argument("--action_set_name", type=str, required=True)
    p.add_argument("--model_type", type=str, required=True)
    p.add_argument("--explainer_type", type=str, required=True)
    p.add_argument("--overwrite", default=settings["overwrite"], action="store_true")
    args, _ = p.parse_known_args()
    settings.update(vars(args))

# load
data = fileutils.load(get_data_file(**settings))
action_set = fileutils.load(get_action_set_file(**settings))
exp_obj = fileutils.load(get_explainer_file(**settings))
clf = fileutils.load(get_model_file(**settings))["model"]
resp_df = fileutils.load(get_resp_df_file(**settings))
resp_sc = fileutils.load(get_scorer_file(**settings), decompress=True)

neg_pred_idx = np.where(clf.predict(data.U) != TARGET)[0]

df = exp_obj.get_explanations()["values"]
df.rename(
    {feat: idx for idx, feat in enumerate(exp_obj.data.names.X)}, axis=1, inplace=True
)
df = df.iloc[neg_pred_idx]

melted = df.melt(var_name="feature", value_name="exp", ignore_index=False).reset_index()
melted["exp_abs"] = melted["exp"].abs()

mrged = melted.merge(
    resp_df, left_on=["index", "feature"], right_on=["u_index", "feature"], how="left"
)
mrged.fillna(0, inplace=True)
mrged["exp_rank"] = mrged.groupby("index")[["exp_abs"]].rank(
    ascending=False, method="first"
)

top_k = mrged[(mrged["exp_rank"] <= 4)]
top_k = top_k[top_k["exp_abs"] > 0]
n_pts = top_k["index"].nunique()

# number of responsive reasons in top k
n_resp_reasons = (
    top_k.groupby("index")["resp"]
    .apply(lambda x: np.count_nonzero(x))
    .value_counts()
    .to_dict()
)

# number of responsive + monotonic
resp_reasons = top_k[top_k["resp"] > 0].reset_index(drop=True)


def check_monotonic(idx, j):
    x = data.U[neg_pred_idx[idx]]
    j_inter = resp_sc.inter[resp_sc._get_inter_key(x)][j].toarray()
    pos = j_inter[:, j] > 0
    neg = j_inter[:, j] < 0
    pred = clf.predict(x + j_inter)

    targ = pred == TARGET
    offt = pred != TARGET

    if targ.all() or offt.all():
        return True, "constant"

    pos_diff = np.diff(pred[pos])
    neg_diff = np.diff(pred[neg])

    pos_mono = (pos_diff >= 0).all()
    neg_mono = (neg_diff <= 0).all()
    pos_offt = (offt[pos]).all()
    neg_offt = (offt[neg]).all()

    if pos_mono and neg_offt:
        out_bool = True
        out_type = "positive"
    elif neg_mono and pos_offt:
        out_bool = True
        out_type = "negative"
    else:
        out_bool = False
        out_type = "n/a"

    return out_bool, out_type


mono_out = []
for _, j in resp_reasons[["index", "feature"]].iterrows():
    mono_out.append([*check_monotonic(j["index"], j["feature"])])

mono_df = pd.DataFrame(mono_out, columns=["mono", "mono_type"])
all_df = pd.concat([resp_reasons, mono_df], axis=1)

# NOTE: number with 0 is going to be this plus from n_resp_reasons
n_resp_mono_reasons_ser = all_df.groupby("index")["mono"].sum().value_counts()
n_resp_mono_reasons = n_resp_mono_reasons_ser.to_dict()
n_resp_mono_reasons[0] = int(n_pts - n_resp_mono_reasons_ser.loc[1:].sum())

# number of responsive + monotonic + direction is intuitive
intuit_map = {
    2: "negative",
    3: "positive",
    4: "negative",
    5: "negative",
    6: "negative",
}

all_df["intuitive"] = all_df["feature"].map(intuit_map) == all_df["mono_type"]
n_resp_mono_intuit_reasons_ser = (
    all_df.groupby("index")["intuitive"].sum().value_counts()
)
n_resp_mono_intuit_reasons = n_resp_mono_intuit_reasons_ser.to_dict()
n_resp_mono_intuit_reasons[0] = int(
    n_pts - n_resp_mono_intuit_reasons_ser.loc[1:].sum()
)

metric_df_lst = []

for n, v in n_resp_reasons.items():
    metric_df_lst.append(
        {
            "metric": f"n_resp_reasons_{n}",
            "value": v,
        }
    )

for n, v in n_resp_mono_reasons.items():
    metric_df_lst.append(
        {
            "metric": f"n_resp_mono_reasons_{n}",
            "value": v,
        }
    )

for n, v in n_resp_mono_intuit_reasons.items():
    metric_df_lst.append(
        {
            "metric": f"n_resp_mono_intuit_reasons_{n}",
            "value": v,
        }
    )

metric_df = pd.DataFrame(metric_df_lst)
metric_file = (
    results_dir
    / f"{settings['data_name']}_{settings['action_set_name']}_{settings['model_type']}_{settings['explainer_type']}_demo_metrics.csv"
)
metric_df.to_csv(metric_file, index=False)
print(f"Metrics saved to {metric_file}")
