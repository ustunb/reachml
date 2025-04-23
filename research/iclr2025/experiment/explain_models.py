import sys
import os
import psutil
import argparse

sys.path.append(os.getcwd())
from src.paths import *

from src.ext import fileutils

from sklearn.pipeline import Pipeline

settings = {
    "data_name": "givemecredit_cts",
    "action_set_name": "complex_nD",
    "model_type": "logreg",
    "explainer_type": "SHAP",
    "overwrite": False,
}

ppid = os.getppid()
process_type = psutil.Process(ppid).name()
if process_type not in ("Code Helper (Plugin)"):  # add your favorite IDE process here
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--action_set_name", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--explainer_type", type=str, required=True)
    parser.add_argument("--overwrite", default=settings["overwrite"], action="store_true")
    # parser.add_argument("--seed", type=int, default=settings["random_seed"])
    args, _ = parser.parse_known_args()
    settings.update(vars(args))

# load dataset
data = fileutils.load(get_data_file(**settings))

# load model
model_results = fileutils.load(get_model_file(**settings))
clf = model_results["model"]

# load action set
action_set = fileutils.load(get_action_set_file(**settings))

# pick explainer
if "SHAP" in settings["explainer_type"]:
    from src.explainer import SHAP_Explainer as Explainer
    if settings["model_type"] == "logreg":  # this way we can use LinearSHAP
        clf = model_results["model"]["clf"]
        settings["scaler"] = model_results["model"]["scaler"]
elif "LIME" in settings["explainer_type"]:
    from src.explainer import LIME_Explainer as Explainer

# SHAP masking
if settings["model_type"] == "logreg" and \
    isinstance(clf, Pipeline) and \
        settings["explainer_type"] == "SHAP":
    settings["scaler"] = clf["scaler"]
    clf = clf["clf"]

if "actionAware" in settings["explainer_type"]:
    from src.explainer import actionAwareExplainer

    og_settings = settings.copy()
    og_settings['explainer_type'] = og_settings['explainer_type'].replace("_actionAware", "")
    og_exp_obj = fileutils.load(get_explainer_file(**og_settings))
    exp_obj = actionAwareExplainer(og_exp_obj, action_set)
else:
    exp_obj = Explainer(model=clf, data=data, **settings)

f = get_explainer_file(**settings)
if not f.is_file() or settings["overwrite"]:
    print(f"Generating explainer and saving {f.name}...")
    exp_obj._generate_exp(**settings)
    fileutils.save(exp_obj, path=f, overwrite=True)
else:
    print(f"{f.name} exists. Using existing one. Specify overwrite=True to overwrite.")