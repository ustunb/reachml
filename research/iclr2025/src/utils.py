"""
TODO: describe what this file contains
"""

from collections import Counter
from itertools import chain
from operator import itemgetter
from os.path import commonprefix

import numpy as np
import pandas as pd
import prettytable
import rich
from prettytable.colortable import ColorTable
from sklearn.preprocessing import StandardScaler
from src.ext.training import train_model


def has_feature_vector_discrete(X, x):
    return np.all(X == x, axis=1).any()


def has_feature_vector_float(X, x, atol):
    return np.isclose(X, x, atol=atol).all(axis=1).any()


def expand_values(value, m):
    """
    expands value m times
    :param value:
    :param m:
    :return:
    """
    assert isinstance(m, int) and m >= 1

    if not isinstance(value, (np.ndarray, list, str, bool, int, float)):
        raise ValueError(f"unsupported variable type {type(value)}")

    if isinstance(value, np.ndarray):
        if len(value) == m:
            arr = value
        elif value.size == 1:
            arr = np.repeat(value, m)
        else:
            raise ValueError(f"length mismatch; need either 1 or {m} values")
    elif isinstance(value, list):
        if len(value) == m:
            arr = value
        elif len(value) == 1:
            arr = [value] * m
        else:
            raise ValueError(f"length mismatch; need either 1 or {m} values")
    elif isinstance(value, str):
        arr = [str(value)] * m
    elif isinstance(value, bool):
        arr = [bool(value)] * m
    elif isinstance(value, int):
        arr = [int(value)] * m
    elif isinstance(value, float):
        arr = [float(value)] * m

    return arr


def check_feature_matrix(X, d=1):
    """
    :param X: feature matrix
    :param d:
    :return:
    """
    assert X.ndim == 2, "`X` must be a matrix"
    assert X.shape[0] >= 1, "`X` must have at least 1 row"
    assert X.shape[1] >= d, f"`X` must contain at least {d} column"
    assert np.issubdtype(X.dtype, np.number), "X must be numeric"
    assert np.isfinite(X).all(), "X must be finite"
    return True


def check_variable_names(names):
    """
    checks variable names
    :param names: list of names for each feature in a dataset.
    :return:
    """
    assert isinstance(names, list), "`names` must be a list"
    assert all([isinstance(n, str) for n in names]), "`names` must be a list of strings"
    assert len(names) >= 1, "`names` must contain at least 1 element"
    assert all([len(n) > 0 for n in names]), (
        "elements of `names` must have at least 1 character"
    )
    assert len(names) == len(set(names)), "names must be distinct"
    return True


def check_partition(action_set, partition):
    """
    :param action_set:
    :param partition:
    :return:
    """
    assert isinstance(partition, list)
    assert [isinstance(p, list) for p in partition]

    # check coverage
    all_indices = range(len(action_set))
    flattened = list(chain.from_iterable(partition))
    assert set(flattened) == set(all_indices), "partition should include each index"
    assert len(flattened) == len(set(flattened)), "parts are not mutually exclusive"

    # check minimality
    is_minimal = True
    for part in partition:
        for j in part:
            if not set(part) == set(action_set.constraints.get_associated_features(j)):
                is_minimal = False
                break
    assert is_minimal
    return True


def implies(a, b):
    return np.all(b[a == 1] == 1)


def parse_attribute_name(dummy_names, default_name=""):
    """
    parse attribute name from
    :param dummy_names: list of names of a dummy variable
    :param default_name: default name to return if nothing is parsed
    :return: string containing the attribute name or default name if no common prefix
    """
    out = commonprefix(dummy_names)
    if len(out) == 0:
        out = default_name
    return out


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
    assert np.array_equal(data.cvindices[fold_id], data_raw.cvindices[fold_id])
    data.split(fold_id=fold_id, fold_num_validation=None, fold_num_test=fold_num_test)
    data_raw.split(
        fold_id=fold_id,
        fold_num_validation=kwargs.get("fold_num_validation)"),
        fold_num_test=fold_num_test,
    )

    if model_type == "logreg":
        kwargs["rescale"] = True
    elif model_type == "xgb":
        kwargs["label_encoding"] = (0, 1)

    out = {
        "model": train_model(
            data, model_type=model_type, rebalance=rebalance, seed=seed, **kwargs
        ),
        "model_raw": train_model(
            data_raw, model_type=model_type, rebalance=rebalance, seed=seed, **kwargs
        ),
    }
    return out


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
    # coefficients_unnormalized = scaler.inverse_transform(coefficients.reshape(1, -1))
    return w, b
