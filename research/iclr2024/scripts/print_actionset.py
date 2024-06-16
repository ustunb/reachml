import sys
import os
import psutil
import argparse

from src.paths import *
from src import fileutils
from pathlib import Path

settings = {
    "data_name": "german",
    "action_set_name": "complex_nD",
    "output_dir": "/Users/berk/Dropbox (Harvard University)/Apps/Overleaf/reachable-sets/tables/",
}

ppid = os.getppid()
process_type = psutil.Process(ppid).name()  # e.g. pycharm, bash
if process_type not in ("pycharm"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--action_set_name", type=str, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args, _ = parser.parse_known_args()
    settings.update(vars(args))

action_set = fileutils.load(get_action_set_file(**settings))

# setup output directory
output_dir = Path(settings["output_dir"])
actionset_table_file = output_dir / f"{settings['data_name']}_actionset_table.tex"
actionset_constraint_file = (
    output_dir / f"{settings['data_name']}_actionset_constraints.tex"
)
print(f"printing actionset tex files to: {output_dir}")
print(f"table: {actionset_table_file}")
print(f"constraints: {actionset_constraint_file}")


def sanitize_names(name, wrapper_function="textfn"):
    """
    :param name: string for feature name
    :param latex function name to apply to features:
    :return:
    """
    s = str(name)
    s = s.replace("_geq_", "$\geq$")
    s = s.replace("_leq_", "$\leq$")
    s = s.replace("_lt_", "$<$")
    s = s.replace("_gt_", "$>$")
    s = s.replace("_eq_", "$=$")
    s = s.replace("_is_", "$=$")
    if wrapper_function is not None:
        s = "\\textfn{" + s + "}"
    return s


def to_latex(action_set, reorder=False, header_wrapper_function="textheader"):
    """
    :param action_set: ActionSet object
    :return: formatted latex table summarizing the action set for publications
    """
    df = action_set.df
    tex_binary_str = "$\\{0,1\\}$"
    tex_integer_str = "$\\mathbb{Z}$"
    tex_real_str = "$\\mathbb{R}$"

    new_types = [tex_real_str] * len(df)
    new_ub = [f"{v:1.1f}" for v in df["ub"].values]
    new_lb = [f"{v:1.1f}" for v in df["lb"].values]

    for i, t in enumerate(df["variable_type"]):
        ub, lb = df["ub"][i], df["lb"][i]
        if t in (int, bool):
            new_ub[i] = f"{int(ub)}"
            new_lb[i] = f"{int(lb)}"
            new_types[i] = tex_binary_str if t is bool else tex_integer_str

    df["variable_type"] = new_types
    df["ub"] = new_ub
    df["lb"] = new_lb

    # reorder rows
    if reorder:
        actionable_idx = df.query("actionable == True").index.tolist()
        immutable_idx = df.query("actionable == False").index.tolist()
        new_index = immutable_idx + actionable_idx
        df = df.loc[new_index]

    # monotonicity
    df["monotonicity"] = df["actionable"]
    up_idx = df["actionable"] & df["step_direction"] > 0
    dn_idx = df["actionable"] & df["step_direction"] < 0
    df.loc[up_idx, "monotonicity"] = "$+$"
    df.loc[dn_idx, "monotonicity"] = "$-$"
    df.loc[~df["actionable"], "monotonicity"] = ""
    df = df.drop(["step_direction"], axis=1)

    # actionability
    df["actionable"] = df["actionable"].map({False: "No", True: "Yes"})  # todo change

    # select columns
    df = df[["name", "variable_type", "lb", "ub", "actionable", "monotonicity"]]

    # parse names
    df["name"] = df["name"].apply(lambda row: sanitize_names(row))

    column_renamer = {
        "name": "Name",
        "variable_type": "Type",
        "actionable": "Actionability",
        "lb": "LB",
        "ub": "UB",
        "monotonicity": "Sign",
    }
    if header_wrapper_function is not None:
        column_renamer = {
            k: "\\" + header_wrapper_function + "{" + v + "}"
            for k, v in column_renamer.items()
        }
    df = df.rename(columns=column_renamer)
    table = df.to_latex(index=False, escape=False)
    table = table.replace("_", "\_")
    return table


constraint_list = ["\\begin{constraints}"]
for con in action_set.constraints:
    constraint_name = con.__class__.__name__
    text = f"{str(con)}"
    for name in con.names:
        text = text.replace(name, sanitize_names(name))
    text = text.replace("`", "")
    text = text.replace("_", "")
    constraint_list.append("\\item " + constraint_name + ": " + text)
constraint_list.append("\\end{constraints}")
constraint_list = "\n".join(constraint_list)

with open(actionset_table_file, "w") as f:
    print(to_latex(action_set), file=f)

with open(actionset_constraint_file, "w") as f:
    print(constraint_list, file=f)
