import itertools
import os
import sys
import psutil
import rich

import numpy as np
import pandas as pd
import argparse
from tqdm.auto import tqdm
from reachml import ReachableSetDatabase

settings = {
    "data_name": "german",
    "action_set_name": "complex_nD",
    "model_type": "xgb",
}

ppid = os.getppid()
process_type = psutil.Process(ppid).name()
if process_type not in ("pycharm"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", nargs="+", type=str, default=["german"])
    parser.add_argument(
        "--action_set_name", nargs="+", type=str, default=["complex_nD"]
    )
    parser.add_argument(
        "--model_type",
        nargs="+",
        type=str,
        default=["logreg", "rf", "xgb"],
    )
    args = parser.parse_args()
    settings.update(vars(args))
    data_names = args.data_name

from src.paths import *
from src import fileutils

data_chunks = []

for data_name, action_set_name, model_type in itertools.product(
    args.data_name, args.action_set_name, args.model_type
):
    # load processed model
    model_results = fileutils.load(
        get_model_file(
            data_name=data_name, action_set_name=action_set_name, model_type=model_type
        )
    )

    data_chunks.append(
        dict(
            dataset=data_name,
            action_set=action_set_name,
            model=model_type,
            train_auc=model_results["train"]["auc"],
            train_error=model_results["train"]["error"],
            test_auc=model_results["test"]["auc"],
            test_error=model_results["test"]["error"],
        )
    )

df = pd.DataFrame(data_chunks)
print(df)

print(df.style.to_latex())
