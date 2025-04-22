# generic imports
import sys
import os
from psutil import Process
from argparse import ArgumentParser

# script imports
sys.path.append(os.getcwd()) # add repo dir to path to import source code
from src.paths import *

from src.ext import fileutils

import pandas as pd
import itertools

# script settings / default command line arguments
settings = {
    "data_name": "german",
    "action_set_name": "complex_nD",
    "model_type": "logreg",
    "method_name": "ar",
}

# parse arguments when script is run from Terminal / not iPython console
if Process(pid=os.getppid()).name() not in ("pycharm"):
    p = ArgumentParser()
    p.add_argument("--data_name", type=str, required=True)
    p.add_argument("--action_set_name", type=str, required=True)
    p.add_argument("--model_type", type=str, required=True)
    p.add_argument("--method_name", type=str, required=True)
    args, _ = p.parse_known_args()
    settings.update(vars(args))

audit_results = fileutils.load(get_audit_results_file(**settings))
microdata = pd.DataFrame.from_dict(audit_results, orient="index")
pts_with_recourse_pct = microdata.query(
    "orig_prediction == False"
).recourse_exists.mean()
print(f"{100-pts_with_recourse_pct*100:.2f}% points without recourse")

# fmt: off
def get_metrics(chunk):
    stats = dict(
        n=len(chunk),
        finds_action_cnt=(chunk.found_a).sum(),
        finds_no_action_cnt=(~chunk.found_a).sum(),
        certifies_infeasibility_cnt=chunk.certifies_infeasibility.sum(),
        abstains_cnt=chunk.abstains.sum(),
        finds_good_action_cnt=(chunk.found_a & chunk.actionable).sum(),
        loophole_cnt=(chunk.found_a & ~chunk.actionable).sum(),
        blindspot_cnt=((~chunk.found_a | chunk.abstains) & chunk.recourse_exists).sum(),
    )

    return pd.DataFrame([dict(stat_name=k, stat_value=v) for k, v in stats.items()])


key_to_vals = {"pos": [True], "neg": [False], "all": [False, True]}

metric_chunks = []
for (y_true_key, y_true_vals), (pred_key, pred_vals) in itertools.product(
    key_to_vals.items(), key_to_vals.items()
):
    if y_true_key == pred_key == "all":
        continue

    data_chunk = microdata.query(
        f"y_true in {y_true_vals} and orig_prediction in {pred_vals}"
    )

    metric_chunks.append(
        get_metrics(data_chunk).assign(
            label_type=y_true_key,
            prediction_type=pred_key,
            data_name=settings["data_name"],
            actionset_name=settings["action_set_name"],
            model_type=settings["model_type"],
            method_name=settings["method_name"],
        )
    )

metrics = pd.concat(metric_chunks)

print(
    metrics.query("prediction_type == 'neg' and label_type == 'all'")
    .pivot(
        index=["data_name", "model_type", "method_name", "actionset_name"],
        columns="stat_name",
        values="stat_value",
    )
    .T
)

stats_file = get_stats_file(**settings)
metrics.to_csv(stats_file, index=False)
print(f"saved to: {stats_file}")
