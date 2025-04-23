import os
import sys

sys.path.append(os.getcwd())
import argparse

import psutil
from src.ext import fileutils, training
from src.paths import get_data_file, get_model_file

settings = {
    "data_name": "german",
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
if settings["model_type"] == "logreg":
    settings["rescale"] = True
elif settings["model_type"] == "xgb":
    settings["label_encoding"] = (0, 1)
    if "givemecredit" in settings["data_name"]:
        settings["early_stopping_rounds"] = 10
        settings["fold_num_validation"] = 4

if "givemecredit" in settings["data_name"]:
    settings["rebalance"] = "over"

data = fileutils.load(get_data_file(**settings))
data.split(
    fold_id=settings["fold_id"],
    fold_num_validation=settings["fold_num_validation"],
    fold_num_test=settings["fold_num_test"],
)

results = training.train_model(data, **settings)
results = results | settings
fileutils.save(results, path=get_model_file(**settings), overwrite=True)
