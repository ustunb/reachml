import os
import sys

import psutil

sys.path.append(os.getcwd())

import argparse

import numpy as np
from tqdm.auto import tqdm

from reachml import ReachableSetDatabase

DB_ACTION_SET_NAME = "complex_nD"

settings = {
    "data_name": "givemecredit",
    "action_set_name": "complex_nD",
    "model_type": "xgb",
}

ppid = os.getppid()
process_type = psutil.Process(ppid).name()
if process_type not in ("pycharm"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--action_set_name", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
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

# load processed model
model_results = fileutils.load(get_model_file(**settings))
clf = model_results["model"]

# run audit
null_action = np.zeros(data.d)
nan_action = np.repeat(np.nan, data.d)
results = {}
predictions = clf.predict(data.U)

for idx, (x, y, fx) in tqdm(
    list(enumerate(zip(data.U, data.y[data.u_idx], predictions)))
):
    # pull reachable set
    R = db[x]
    flipped_idx = np.flatnonzero(clf.predict(R.actions + x) != fx)
    feasible = len(flipped_idx) > 0
    recourse_exists = bool(R.complete and feasible)

    results[idx] = {
        "y_true": y > 0,
        "orig_prediction": fx > 0,
        "flip_action_idx": flipped_idx if feasible else [],
        "actionable": R.actions.shape[0] > 0,
        "abstains": (R.complete == False) and not feasible,
        "recourse_exists": recourse_exists,
    }

# save results
fileutils.save(
    results,
    path=get_audit_results_file(**settings),
    overwrite=True,
    check_save=False,
)
