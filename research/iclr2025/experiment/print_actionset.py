import sys
import os
from psutil import Process
from argparse import ArgumentParser

# script / source code imports
sys.path.append(os.getcwd()) # add repo dir to path to import source code
from src.paths import *
from src.ext import fileutils

# script settings / default command line arguments
settings = {
    "data_name": "givemecredit",
    "action_set_name": "complex_nD",
    "output_dir": "/Users/harrycheon/Dropbox/Apps/Overleaf/reasons-without-recourse/tables/",
}

# parse settings from command line when script is not run from iPython console
if Process(pid=os.getppid()).name() not in ("pycharm"):
    p = ArgumentParser()
    p.add_argument("--data_name", type=str, required=True)
    p.add_argument("--action_set_name", type=str, default=settings["action_set_name"])
    p.add_argument("--output_dir", type=str, default=settings["output_dir"])
    args, _ = p.parse_known_args()
    settings.update(vars(args))

# setup output directory and file names
output_dir = results_dir if len(settings["output_dir"]) == 0 else Path(settings["output_dir"])
table_file = output_dir / f"{settings['data_name']}_actionset_table.tex"
constraint_file = output_dir / f"{settings['data_name']}_actionset_constraints.tex"


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
    df["monotonicity"] = ""
    up_idx = df["actionable"] & df["step_direction"] > 0
    dn_idx = df["actionable"] & df["step_direction"] < 0
    df.loc[up_idx, "monotonicity"] = "$+$"
    df.loc[dn_idx, "monotonicity"] = "$-$"
    df = df.drop(["step_direction"], axis=1)

    # actionability
    df["actionable"] = df["actionable"].map({False: "No", True: "Yes"})  # todo change

    # select columns
    df = df[["name", "variable_type", "lb", "ub", "actionable", "monotonicity"]]

    # format names
    df["name"] = df["name"].apply(lambda row: sanitize_names(row))

    # format columns
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
    styler = df.style
    (styler.hide(level=0, axis=0))
    table = styler.to_latex(hrules=True)
    table = table.replace("_", "\_")
    return table


# load action set
action_set = fileutils.load(get_action_set_file(**settings))

# print constraint list
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

with open(table_file, "w") as f:
    print(to_latex(action_set), file=f)
    print(f"Saved Table to:{f}")

with open(constraint_file, "w") as f:
    print(constraint_list, file=f)
    print(f"Saved Constraint List to:{f}")
