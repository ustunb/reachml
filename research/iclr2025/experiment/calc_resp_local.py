import os
import sys

import psutil

sys.path.append(os.getcwd())

import argparse

import pandas as pd

from reachml.scoring import ResponsivenessScorer

DB_ACTION_SET_NAME = "complex_nD"

settings = {
    "data_name": "givemecredit_cts",
    "action_set_name": "complex_nD",
    "overwrite": False,
    "method": "auto",
}

all_models = ["logreg", "xgb", "rf"]

ppid = os.getppid()
process_type = psutil.Process(ppid).name()
if process_type not in ("pycharm"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--action_set_name", type=str, required=True)
    parser.add_argument("--model_type", type=str, default=argparse.SUPPRESS)
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

if settings["method"] == "auto":
    settings["method"] = "enumerate" if action_set.can_enumerate else "sample"

# create scorer
if os.path.exists(get_scorer_file(**settings)) and not settings["overwrite"]:
    resp_sc = fileutils.load(get_scorer_file(**settings))
else:
    if settings["method"] == "enumerate":
        from reachml import ReachableSetDatabase

        settings["reach_db"] = ReachableSetDatabase(
            action_set=action_set,
            path=get_reachable_db_file(
                data_name=settings["data_name"], action_set_name=DB_ACTION_SET_NAME
            ),
        )
    resp_sc = ResponsivenessScorer(action_set, **settings)


def create_resp_df(model_type, X=None, save=True):
    temp_settings = settings.copy()
    temp_settings["model_type"] = model_type
    clf = fileutils.load(get_model_file(**temp_settings))["model"]

    if X is None:
        X = data.U

    raw_scores = resp_sc(X, clf)
    row_df = pd.DataFrame(raw_scores[:, resp_sc.act_feats])
    resp_df = row_df.melt(
        var_name="feature", value_name="resp", ignore_index=False
    ).reset_index()

    if save:
        fileutils.save(
            resp_df,
            path=get_resp_df_file(**temp_settings),
            overwrite=True,
            check_save=False,
        )

    return resp_df


if __name__ == "__main__":
    models = [settings["model_type"]] if "model_type" in settings else all_models
    for m in models:
        create_resp_df(m)

    if settings["overwrite"] or not os.path.exists(get_scorer_file(**settings)):
        fileutils.save(resp_sc, path=get_scorer_file(**settings), overwrite=True)
