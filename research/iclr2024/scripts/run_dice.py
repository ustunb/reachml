import os
import sys
import time
import psutil
import argparse
import contextlib
import warnings

warnings.simplefilter(action="ignore", category=UserWarning)

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from copy import deepcopy
from raiutils.exceptions import UserConfigValidationException
import dice_ml
from src.paths import *
from src import fileutils

settings = {
    "data_name": "german",
    "action_set_name": "complex_nD",
    "model_type": "xgb",
    "method_name": "dice",
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
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--total_cfs", type=int, default=settings["total_cfs"])
    parser.add_argument("--random_seed", type=int, default=settings["random_seed"])
    args, _ = parser.parse_known_args()
    settings.update(vars(args))

# load dataset and actionset
data = fileutils.load(get_data_file(**settings))
action_set = fileutils.load(get_action_set_file(**settings))

# load model results
model_results = fileutils.load(get_model_file(**settings, is_raw=False))
clf, scaler = model_results["model"], model_results["scaler"]
if scaler is not None:
    rescale = lambda x: (
        scaler.transform(x.reshape(1, -1)) if x.ndim == 1 else scaler.transform(x)
    )

    # dice rescaler needs kwargs
    def dice_transformer_func(x, **kwargs):
        return scaler.transform(x.to_numpy())

else:
    rescale = lambda x: x.reshape(1, -1) if x.ndim == 1 else x
    dice_transformer_func = None

predictions = clf.predict(rescale(data.X))
prediction_handle = lambda x: clf.predict(rescale(x))


# setup dice
binary_features = [a.name for a in action_set if a.variable_type == bool]
if settings["model_type"] == "xgb":
    binary_features = []  # todo: why do we set to [] for XGB?
continuous_features = [name for name in action_set.names if name not in binary_features]
features_to_vary = [a.name for a in action_set if a.actionable]

# setup dice
df = deepcopy(data.df.astype(int))  # todo: why is this set to int?
dice_df = dice_ml.Data(
    dataframe=df, continuous_features=continuous_features, outcome_name=data.names.y
)
dice_model = dice_ml.Model(model=clf, backend="sklearn", func=dice_transformer_func)
explainer = dice_ml.Dice(dice_df, dice_model)


def get_dice_permitted_range(
    query_instance, binary_features, features_to_vary, action_set
):
    """
    :param query_instance: can be a dictionary
    :param binary_features:
    :param features_to_vary:
    :param action_set:
    :return:
    """
    x, names = query_instance.values[0], query_instance.columns
    lbs = x + action_set.get_bounds(x, bound_type="lb")
    ubs = x + action_set.get_bounds(x, bound_type="ub")
    out = {}
    for name, lb, ub in zip(names, lbs, ubs):
        if name in features_to_vary:
            if name in binary_features:
                out[name] = [str(int(lb)), str(int(ub))]
            else:
                out[name] = [int(lb), int(ub)]
    return out


record_df = df.drop(columns=[data.names.y])
unique_df = pd.DataFrame(record_df)
unique_df.drop_duplicates(subset=action_set.names, keep="first", inplace=True)
nan_action = np.repeat(np.nan, data.d)

unique_results = []
# get results for distinct values of x
for idx, row in tqdm(list(unique_df.iterrows())):
    x = row.to_numpy()
    new_info = info = {
        "id_for_x": idx,
        "ctf_id": None,
        "a": nan_action,
        "found_a": False,
        "certifies_infeasibility": False,
        "abstains": False,
    }
    # only generate counterfactuals for negative predictions #todo: only loop over these
    if predictions[idx] <= 0:
        try:
            query_instance = pd.DataFrame(data=[row]).astype(int)
            permitted_range = get_dice_permitted_range(
                query_instance,
                binary_features=binary_features,
                features_to_vary=features_to_vary,
                action_set=action_set,
            )
            #            start = time.process_time()
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stderr(devnull):
                    explanation = explainer.generate_counterfactuals(
                        query_instance,
                        total_CFs=settings["total_cfs"],
                        permitted_range=permitted_range,
                        features_to_vary=features_to_vary,
                        desired_class="opposite",
                        random_seed=settings["random_seed"],
                        verbose=False,
                    )
            #            exec_time = time.process_time() - start
            dice_cfs = explanation.cf_examples_list[0].final_cfs_df[data.names.X].values
            dice_outcomes = (
                explanation.cf_examples_list[0].final_cfs_df[data.names.y].values
            )
            for ctf_id, (yp, xp) in enumerate(zip(dice_outcomes, dice_cfs)):
                xp = np.array(xp, dtype=float)
                new_info = dict(info)
                new_info.update(
                    {
                        "ctf_id": ctf_id,
                        "found_a": True,
                        "a": np.subtract(xp, np.array(x)),
                        "certifies_infeasibility": False,
                        "abstains": False,
                    }
                )
                unique_results.append(new_info)
        except UserConfigValidationException as e:
            if "No counterfactuals found for any of the query points" in str(e):
                unique_results.append(info)
        except Exception as e:
            import ipdb

            ipdb.set_trace()
            print("code should not get here")
    unique_results.append(new_info)

# copy results for other points
duplicate_results = []
for idx, row in tqdm(list(unique_df.iterrows())):
    match_idx = (row == record_df).all(1)
    if sum(match_idx) > 1:
        row_results = [info for info in unique_results if info["id_for_x"] == idx]
        duplicate_idx = record_df[match_idx].index.tolist()[1:]
        for dup_idx in duplicate_idx:
            for info in row_results:
                dup_info = dict(info) | {"id_for_x": dup_idx}
                duplicate_results.append(dup_info)

all_results = unique_results + duplicate_results
# convert results to dataframe, prepend settings, and save
df = pd.DataFrame(data=all_results)
stat_names = df.columns.tolist()
fileutils.save(
    all_results,
    path=get_benchmark_results_file(**settings),
    overwrite=True,
    check_save=False,
)
