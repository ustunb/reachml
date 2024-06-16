import sys
import os
import time
import psutil
import argparse
import contextlib
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import recourse as rs
from tqdm.auto import tqdm

from src.paths import *
from src import fileutils


def undo_coefficient_scaling(clf=None, coefficients=None, intercept=0.0, scaler=None):
    """
    given coefficients and data for scaled data, returns coefficients and intercept for unnormalized data

    w = w_scaled / sigma
    b = b_scaled - (w_scaled / sigma).dot(mu) = b_scaled - w.dot(mu)

    :param sklearn linear classifier
    :param coefficients: vector of coefficients
    :param intercept: scalar for the intercept function
    :param scaler: sklearn.Scaler or

    :return: coefficients and intercept for unnormalized data

    """
    if coefficients is None:
        assert clf is not None
        assert intercept == 0.0
        assert hasattr(clf, "coef_")
        coefficients = clf.coef_
        intercept = clf.intercept_ if hasattr(clf, "intercept_") else 0.0

    if scaler is None:
        w = np.array(coefficients)
        b = float(intercept)
    else:
        isinstance(scaler, StandardScaler)
        x_shift = np.array(scaler.mean_)
        x_scale = np.sqrt(scaler.var_)
        w = coefficients / x_scale
        w = np.array(w).flatten()
        w[np.isnan(w)] = 0

        b = intercept - np.dot(w, x_shift)
        b = float(b)

    return w, b
    # coefficients_unnormalized = scaler.inverse_transform(coefficients.reshape(1, -1))


# args data name, hard cap, action set name
settings = {
    "data_name": "fico",
    "action_set_name": "complex_nD",
    "model_type": "logreg",
    "method_name": "ar",
    "total_cfs": 1,
    "random_seed": 2338,
}

# parse the settings when the script is run from the command line
ppid = os.getppid()  # get parent process id
process_type = psutil.Process(ppid).name()  # e.g. pycharm, bash
if process_type not in ("pycharm"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--action_set_name", type=str, required=True)
    # parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--total_cfs", type=int, default=settings["total_cfs"])
    parser.add_argument("--random_seed", type=int, default=settings["random_seed"])
    args, _ = parser.parse_known_args()
    settings.update(vars(args))

# load dataset and actionset
data = fileutils.load(get_data_file(**settings))
action_set = fileutils.load(get_action_set_file(**settings))

# load model
model_results = fileutils.load(get_model_file(**settings))
clf = model_results["model"]
scaler = model_results["scaler"]
if scaler is None:
    predictions = clf.predict(data.X)
else:
    predictions = clf.predict(scaler.transform(data.X))
w, b = undo_coefficient_scaling(clf, scaler=scaler)

# setup action set for AR
LB = data.X.min(axis=0)
UB = data.X.max(axis=0)
bounds = {n: (lb, ub, "absolute") for n, lb, ub in zip(data.names.X, LB, UB)}
A = rs.ActionSet(X=data.X, names=data.names.X, custom_bounds=bounds)
A.step_type = "absolute"
for a in action_set:
    A[a.name].step_size = a.step_size
    A[a.name].actionable = a.actionable
    A[a.name].step_direction = a.step_direction
A.set_alignment(coefficients=w, intercept=b)

# audit
results = []
null_action = np.zeros(data.d)
nan_action = np.repeat(np.nan, data.d)
for idx, (x, fx) in enumerate(tqdm(list(zip(data.X, predictions)))):
    if fx > 0:
        results.append(
            {
                "id_for_x": idx,
                "ctf_id": 0,
                "a": null_action,
                "found_a": False,
                "abstains": False,
                "certifies_infeasibility": False,
            }
        )
    else:
        fs = rs.Flipset(x=x, action_set=A, coefficients=w, intercept=b)
        try:
            start_time = time.process_time()
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull):
                    fs.populate(
                        enumeration_type="distinct_subsets",
                        total_items=settings["total_cfs"],
                        cost_type="total",
                    )
            exec_time = time.process_time() - start_time
            for k in range(settings["total_cfs"]):
                if k < len(fs.actions):
                    results.append(
                        {
                            "id_for_x": idx,
                            "ctf_id": k,
                            "a": fs.actions[k],
                            "found_a": True,
                            "abstains": False,
                            "certifies_infeasibility": False,
                        }
                    )
                else:  # at this point AR proves infeasibility
                    results.append(
                        {
                            "id_for_x": idx,
                            "ctf_id": k,
                            "a": nan_action,
                            "found_a": False,
                            "abstains": False,
                            "certifies_infeasibility": True,
                        }
                    )

        except Exception as e:
            if "cost_var_names" in str(e):
                results.append(
                    {
                        "id_for_x": idx,
                        "ctf_id": 0,
                        "found_a": False,
                        "a": nan_action,
                        "abstains": True,
                        "certifies_infeasibility": False,
                    }
                )
            else:
                raise e

# convert results to dataframe, prepend settings, and save
df = pd.DataFrame(data=results)
stat_names = df.columns.tolist()
fileutils.save(
    results,
    path=get_benchmark_results_file(**settings),
    overwrite=True,
    check_save=False,
)
