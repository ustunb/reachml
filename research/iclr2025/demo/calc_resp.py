import os
import sys
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from psutil import Process

sys.path.append(os.getcwd())

from src.ext import fileutils
from src.paths import *

from reachml.scoring import ResponsivenessScorer

DB_ACTION_SET_NAME = "complex_nD"
TARGET = 1

settings = {
    "data_name": "givemecredit_cts",
    "action_set_name": "complex_nD",
    "model_type": "xgb",
    "overwrite": False,
}

# parse arguments when script is run from Terminal / not iPython console
if Process(pid=os.getppid()).name() not in ("pycharm"):
    p = ArgumentParser()
    p.add_argument("--data_name", type=str, required=True)
    p.add_argument("--action_set_name", type=str, required=True)
    p.add_argument("--model_type", type=str, required=True)
    p.add_argument("--overwrite", default=settings["overwrite"], action="store_true")
    args, _ = p.parse_known_args()
    settings.update(vars(args))

# load
data = fileutils.load(get_data_file(**settings))
action_set = fileutils.load(get_action_set_file(**settings))
model = fileutils.load(get_model_file(**settings))
clf = model["model"]

act_feats = np.array(list(action_set.actionable_features))
act_feats.sort()

# note: among unique data points
# denied individuals (too costly to calculate for all)
neg_pred_idx = np.where(clf.predict(data.U) != TARGET)[0]

# only calculate resp for these points (takes too long for other points)
if settings["overwrite"] or not os.path.exists(get_scorer_file(**settings)):
    resp_sc = ResponsivenessScorer(action_set)
else:
    resp_sc = fileutils.load(get_scorer_file(**settings), decompress=True)

resp_out = resp_sc(data.U[neg_pred_idx], clf)
resp_act_feat = resp_out[:, act_feats]
row_df = pd.DataFrame(resp_act_feat, columns=act_feats)
resp_df = row_df.melt(ignore_index=False).reset_index()
resp_df.columns = ["u_index", "feature", "resp"]
resp_df.sort_values(by=["u_index", "feature"], inplace=True)

fileutils.save(resp_df, path=get_resp_df_file(**settings), overwrite=True)
fileutils.save(
    resp_sc,
    get_scorer_file(**settings),
    overwrite=True,
    check_save=False,
    compress=True,
)
