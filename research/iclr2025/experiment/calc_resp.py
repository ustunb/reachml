import os
import sys

import psutil

sys.path.append(os.getcwd())

import argparse

import numpy as np
import pandas as pd

from reachml import ReachableSetDatabase
from reachml.scoring import ResponsivenessScorer

DB_ACTION_SET_NAME = "complex_nD"

settings = {
    "data_name": "german",
    "action_set_name": "complex_nD",
    "overwrite": False,
}

all_models = ["logreg", "xgb", "rf"]

ppid = os.getppid()
process_type = psutil.Process(ppid).name()
if process_type not in ("pycharm"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--action_set_name", type=str, required=True)
    parser.add_argument("--model_type", default=argparse.SUPPRESS, type=str)
    parser.add_argument(
        "--overwrite", default=settings["overwrite"], action="store_true"
    )
    args, _ = parser.parse_known_args()
    settings.update(vars(args))

from src.ext import fileutils
from src.paths import *

# load action set and processed data
data = fileutils.load(get_data_file(**settings))
action_set = fileutils.load(get_action_set_file(**settings))

# load database
db = ReachableSetDatabase(
    action_set=action_set,
    path=get_reachable_db_file(
        data_name=settings["data_name"], action_set_name=DB_ACTION_SET_NAME
    ),
)

resp_file_path = get_scorer_file(**settings)

if settings["overwrite"] or not os.path.exists(resp_file_path):
    resp_sc = ResponsivenessScorer(action_set, reach_db=db)
else:
    resp_sc = fileutils.load(resp_file_path)

act_feats = np.array(list(action_set.actionable_features))
act_feats.sort()


def create_resp_df(model_type, save=True):
    temp_settings = settings.copy()
    temp_settings["model_type"] = model_type
    model_results = fileutils.load(get_model_file(**temp_settings))
    clf = model_results["model"]

    resp_out = resp_sc(data.U, clf, save=False)
    resp_act_feat = resp_out[:, act_feats]
    row_df = pd.DataFrame(resp_act_feat, columns=act_feats)
    resp_df = row_df.melt(ignore_index=False).reset_index()
    resp_df.columns = ["u_index", "feature", "resp"]
    resp_df.sort_values(by=["u_index", "feature"], inplace=True)

    if save:
        fileutils.save(
            resp_df,
            path=get_resp_df_file(**temp_settings),
            overwrite=True,
            check_save=False,
        )

    return resp_df


if __name__ == "__main__":
    models = [settings["model_type"]] if settings.get("model_type") else all_models
    for model_type in models:
        create_resp_df(model_type)

    # save resp_sc
    if settings["overwrite"] or not os.path.exists(resp_file_path):
        fileutils.save(
            resp_sc, path=get_scorer_file(**settings), overwrite=True, check_save=False
        )
