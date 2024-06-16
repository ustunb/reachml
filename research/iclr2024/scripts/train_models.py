import sys
import os

import psutil
import argparse
from src import fileutils
from src.paths import *

settings = {
    "data_name": "fico",
    "action_set_name": "complex_nD",
    "model_type": "logreg",
    "fold_id": "K05N01",
    "fold_num_validation": None,
    "fold_num_test": 5,
    "random_seed": 2338,
}

ppid = os.getppid()
process_type = psutil.Process(ppid).name()
if process_type not in ("pycharm"):  # add your favorite IDE process here
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--action_set_name", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--fold_id", type=str, default=settings["fold_id"])
    parser.add_argument(
        "--fold_num_validation", type=int, default=settings["fold_num_validation"]
    )
    parser.add_argument("--fold_num_test", type=int, default=settings["fold_num_test"])
    parser.add_argument("--seed", type=int, default=settings["random_seed"])
    args, _ = parser.parse_known_args()
    settings.update(vars(args))

# load datasets
data = fileutils.load(get_data_file(**settings))
data.split(
    fold_id=settings["fold_id"],
    fold_num_validation=settings["fold_num_validation"],
    fold_num_test=settings["fold_num_test"],
)

# pick training function
if settings["model_type"] == "logreg":
    from src.training import train_logreg as train_model
elif settings["model_type"] == "rf":
    from src.training import train_rf as train_model
elif settings["model_type"] == "xgb":
    from src.training import train_xgb as train_model
elif settings["model_type"] == "dnn":
    raise NotImplementedError()  # todo: implement
    from src.training import train_dnn as train_model

rebalance = None if settings["data_name"] != "givemecredit" else "over"
results = train_model(data, seed=settings["random_seed"], rebalance=rebalance)
results = results | settings

fileutils.save(results, path=get_model_file(**settings), overwrite=True)
