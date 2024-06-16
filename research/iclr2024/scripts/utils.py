import numpy as np
import pandas as pd
import prettytable
import rich
from prettytable.colortable import ColorTable
from src.data import BinaryClassificationDataset
from src.training import train_logreg_vanilla
from operator import itemgetter
from collections import Counter


def check_processing_loss(
    data,
    data_raw,
    model_type="logreg",
    fold_id="K05N01",
    fold_num_test=1,
    rebalance="over",
    seed=2338,
    **kwargs,
):
    """
    checks loss in data processing between two binary classification datasets
    :param data:
    :param data_raw:
    :param model_type:
    :param fold_id:
    :param fold_num_test:
    :param rebalance:
    :param seed:
    :param kwargs:
    :return:
    """
    assert isinstance(data, BinaryClassificationDataset)
    assert isinstance(data_raw, BinaryClassificationDataset)
    assert np.array_equal(data.cvindices[fold_id], data_raw.cvindices[fold_id])
    data.split(fold_id=fold_id, fold_num_validation=None, fold_num_test=fold_num_test)
    data_raw.split(
        fold_id=fold_id,
        fold_num_validation=kwargs.get("fold_num_validation)"),
        fold_num_test=fold_num_test,
    )
    if model_type == "logreg":
        from src.training import train_logreg_vanilla as train_model
    elif model_type == "xgb":
        from src.training import train_xgb as train_model
    elif model_type == "rf":
        from src.training import train_rf as train_model
    elif model_type == "dnn":
        raise NotImplementedError()
    out = {
        "model": train_model(data, rebalance=rebalance, seed=seed),
        "model_raw": train_model(data_raw, rebalance=rebalance, seed=seed),
    }
    return out


def check_responsiveness(data, database):

    # todo: plop into database
    model_info = train_logreg_vanilla(data)
    clf = model_info["model"]
    if model_info["scaler"] is None:
        rescale = lambda x: x
    else:
        reformat = lambda x: x.reshape(1, -1) if x.ndim == 1 else x
        rescale = lambda x: model_info["scaler"].transform(reformat(x))

    results = {}
    predictions = clf.predict(rescale(data.X))
    for idx, (x, y, fx) in enumerate(zip(data.X, data.y, predictions)):
        R = database[x]
        flipped_idx = np.flatnonzero(clf.predict(rescale(R.X)) > 0)
        n_feasible_actions = len(flipped_idx)
        feasible = n_feasible_actions > 0
        if feasible:
            xp = R.X[flipped_idx[0], :]
            fxp = clf.predict(rescale(xp)).squeeze()
        else:
            fxp = fx.squeeze()
            # a = np.repeat(np.nan, data.d)
        results[idx] = {
            "y": y > 0,
            "yhat": fx > 0,
            "yhat_post": fxp > 0,  # = f(x + a)
            # "a":
            "recourse": feasible,
            "n_reachable": len(R),
            "n_feasible": n_feasible_actions,
            "complete": R.complete,
            "abstain": (R.complete == False) and not feasible,
        }

    df = pd.DataFrame.from_dict(results, orient="index")
    return df


#### TABLES
COLORS = {
    "bold": "\033[1;38;107m",
    "red": "\033[0;31;108m",
    "blue": "\033[0;34;108m",
    "grey": "\033[0;37;108m",
    "immutable": "\033[1;32;107m",
}


def highlight(strings, flags=None, invert=False, code=None):
    assert isinstance(strings, list)
    strings = [str(s) for s in strings]

    if flags is None:
        flags = [True] * len(strings)
    else:
        assert isinstance(flags, list) and len(flags) == len(strings)

    if invert:
        flags = [not (f) for f in flags]

    if code is None:
        code = "\033[1;38;107m"  # RED

    out = [code + s + "\033[0m" if f else s for f, s in zip(flags, strings)]
    return out


def tabulate_actions(action_set):
    # todo: update table to show partitions
    # todo: add also print constraints
    """
    prints a table with information about each element in the action set
    :param action_set: ActionSet object
    :return:
    """
    # fmt:off
    TYPES = {bool: "bool", int: "int", float: "float"}
    FMT = {bool: "1.0f", int: "1.0f", float: "1.2f"}

    # create table
    t = ColorTable()
    #t.border = False
    t.hrules = prettytable.HEADER
    t.vrules = prettytable.NONE

    indices = list(range(len(action_set)))
    indices = highlight(indices, code = COLORS["grey"])
    t.add_column("", indices, align="r")

    names = highlight(action_set.name, action_set.actionable, invert = True, code = COLORS["red"])
    t.add_column("name", names, align="r")

    vtypes = [TYPES[v] for v in action_set.variable_type]
    t.add_column("type", vtypes, align="r")

    actionable = highlight(action_set.actionable, action_set.actionable, invert = True, code = COLORS["red"])
    t.add_column("actionable", actionable,  align="r")

    # UB
    t.add_column("lb", [f"{a.lb:{FMT[a.variable_type]}}" for a in action_set], align="r")
    t.add_column("ub", [f"{a.ub:{FMT[a.variable_type]}}" for a in action_set], align="r")

    # LB
    directions = [s if s != 0 else "" for s in action_set.step_direction]
    t.add_column("step_dir", directions, align="c")
    t.add_column("step_ub", [v if np.isfinite(v) else "" for v in action_set.step_ub], align="r")
    t.add_column("step_lb", [v if np.isfinite(v) else "" for v in action_set.step_lb], align="r")
    return str(t)


#### DATA PROCESSING
def tally(values):
    c = Counter(values)
    return sorted(c.items(), key=itemgetter(0))


# to remove
def or_conditions_met(features):
    transformed = np.any(features == 1, axis=1)
    return np.where(transformed == True, 1, 0)


filter_cond = lambda cond: np.where(cond, 1, 0)


#### Posthoc Analysis
def tally_predictions(i, database, data, predictor, target=1):
    point_df = pd.DataFrame(data.X_df.iloc[i, :])
    point_df.columns = [f"x_{i}"]
    x = point_df.T.values
    R = database[x]
    S = np.equal(predictor(R.X), target)
    point_df["total"] = database[x].scores(weigh_changes=False)
    point_df["flip"] = database[x].scores(point_mask=S, weigh_changes=False)
    point_df["same"] = database[x].scores(point_mask=~S, weigh_changes=False)
    return point_df


def visualize_diff(x, x_prime):
    df = pd.DataFrame(
        index=x.index,
        columns=["x", "x'"],
        data=np.vstack([x.values.squeeze(), x_prime.squeeze()]).T,
    )
    max_index_length = max([len(s) for s in df.index])
    max_value_length = df[["x", "x'"]].astype(str).applymap(len).max().max().astype(int)

    # Add a column 'Difference' to highlight differing rows
    df["Difference"] = np.where(df["x"] != df["x'"], "DIFFERENT", "")
    for index, row in df.iterrows():
        padded_index = f"{index: <{max_index_length}}"
        padded_x = f"{row['x']: >{max_value_length + 1}}"
        x_prime_key = "x'"
        padded_x_prime = f"{row[x_prime_key]: >{max_value_length + 1}}"
        if row["Difference"] == "DIFFERENT":
            rich.print(padded_index, padded_x, "[red]{}[/red]".format(padded_x_prime))
        else:
            rich.print(padded_index, padded_x, padded_x_prime)


###
