"""Utilities for validation, display, and helpers.

This module provides helper functions used across ReachML, including validation
of feature matrices and names, pretty-printed tables, small analysis helpers,
and solver detection.
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


def has_feature_vector_discrete(X, x):
    """Return True if the exact vector `x` appears in `X` (discrete match)."""
    return np.all(X == x, axis=1).any()


def has_feature_vector_float(X, x, atol):
    """Return True if a row in `X` is approximately equal to `x` within `atol`."""
    return np.isclose(X, x, atol=atol).all(axis=1).any()


def expand_values(value, m):
    """Broadcast a scalar or list-like to length `m`.

    Args:
        value: Scalar or 1D list/array-like to expand.
        m: Target length (int, >= 1).

    Returns:
        A list or array with length `m`.
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
    """Validate that `X` is a proper feature matrix.

    Args:
        X: Numpy array of shape (n, d').
        d: Minimum number of columns required.

    Returns:
        True if the matrix passes shape and finiteness checks.
    """
    assert X.ndim == 2, "`X` must be a matrix"
    assert X.shape[0] >= 1, "`X` must have at least 1 row"
    assert X.shape[1] >= d, f"`X` must contain at least {d} column"
    assert np.issubdtype(X.dtype, np.number), "X must be numeric"
    assert np.isfinite(X).all(), "X must be finite"
    return True


def check_variable_names(names):
    """Validate variable names.

    Args:
        names: List of feature names.

    Returns:
        True if names are non-empty, strings, unique, and non-null.
    """
    assert isinstance(names, list), "`names` must be a list"
    assert all([isinstance(n, str) for n in names]), "`names` must be a list of strings"
    assert len(names) >= 1, "`names` must contain at least 1 element"
    assert all([len(n) > 0 for n in names]), "elements of `names` must have at least 1 character"
    assert len(names) == len(set(names)), "names must be distinct"
    return True


def check_partition(action_set, partition):
    """Validate a partition of features.

    Args:
        action_set: The `ActionSet` instance.
        partition: List of lists with feature indices.

    Returns:
        True if the partition covers all indices, has no overlap, and each
        part is minimal under the constraint associations.
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
    """Return True if all indices where `a == 1` also satisfy `b == 1`."""
    return np.all(b[a == 1] == 1)


def parse_attribute_name(dummy_names, default_name=""):
    """Parse a common attribute name from dummy variable names.

    Args:
        dummy_names: List of dummy/one-hot variable names.
        default_name: Fallback name if no common prefix exists.

    Returns:
        The parsed attribute name or `default_name`.
    """
    out = commonprefix(dummy_names)
    if len(out) == 0:
        out = default_name
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
    """Highlight a list of strings conditionally using ANSI codes.

    Args:
        strings: List of strings or values to render.
        flags: Optional booleans selecting which entries to highlight.
        invert: If True, invert the flags.
        code: ANSI code prefix to apply when highlighting.

    Returns:
        A list of string values with ANSI styling applied where selected.
    """
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

    out = [code + s + "\033[0m" if f else s for f, s in zip(flags, strings, strict=False)]
    return out


def tabulate_actions(action_set):
    # todo: update table to show partitions
    # todo: add also print constraints
    """Build a table with per-feature action parameters.

    Args:
        action_set: The `ActionSet` to display.

    Returns:
        A string representation of the table.
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

    names = highlight(
        action_set.name, action_set.actionable, invert=True, code=COLORS["red"]
    )
    t.add_column("name", names, align="r")

    vtypes = [TYPES[v] for v in action_set.variable_type]
    t.add_column("type", vtypes, align="r")

    actionable = highlight(
        action_set.actionable,
        action_set.actionable,
        invert=True,
        code=COLORS["red"],
    )
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
    """Count unique values and return sorted pairs.

    Args:
        values: Iterable of hashable values.

    Returns:
        List of `(value, count)` tuples sorted by value.
    """
    c = Counter(values)
    return sorted(c.items(), key=itemgetter(0))


# to remove
def or_conditions_met(features):
    """Return 1 if any feature equals 1 across the row, else 0.

    Args:
        features: 2D array-like of shape (n, d) with binary indicators.

    Returns:
        Numpy array of shape (n,) with 0/1 values.
    """
    transformed = np.any(features == 1, axis=1)
    return np.where(transformed, 1, 0)


def filter_cond(cond):
    """Convert a boolean condition to 0/1 integers."""
    return np.where(cond, 1, 0)


#### Posthoc Analysis
def tally_predictions(i, database, data, predictor, target=1):
    """Summarize scores and flips for an instance over a reachable set.

    Args:
        i: Row index in `data`.
        database: Reachable set database.
        data: Dataset with `X_df`.
        predictor: Callable or model with `__call__` over feature arrays.
        target: Target label to count as a flip.

    Returns:
        DataFrame with total, flip, and same scores for the instance.
    """
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
    """Pretty-print row-wise differences between two aligned Series/DataFrames.

    Args:
        x: Pandas Series (original).
        x_prime: Pandas Series (modified).
    """
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
    """Convert scaled model coefficients back to the original feature scale.

    Given a linear model trained on standardized features, compute the
    equivalent coefficients and intercept for the unscaled data.

    Args:
        clf: Sklearn linear classifier with `coef_`/`intercept_` (optional if
            `coefficients` and `intercept` are provided).
        coefficients: 1D coefficients array.
        intercept: Intercept term.
        scaler: `sklearn.preprocessing.StandardScaler` instance.

    Returns:
        Tuple `(w, b)` of unscaled coefficients and intercept.
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


### MIP Settings
def _check_solver_cpx():
    """Return True if IBM CPLEX is installed and importable."""
    chk = False
    try:
        import importlib.util

        chk = importlib.util.find_spec("cplex") is not None
    except ImportError:
        pass

    return chk


def _check_solver_scip():
    """Return True if PySCIPOpt (SCIP) is installed and importable."""
    chk = False
    try:
        import importlib.util

        chk = importlib.util.find_spec("pyscipopt") is not None
    except ImportError:
        pass
    return chk


_SOLVER_TYPE_CPX = "cplex"
_SOLVER_TYPE_SCIP = "scip"


# Set Default Solver
def set_default_solver():
    """Detect and return the default available solver as a string."""
    if _check_solver_cpx():
        return _SOLVER_TYPE_CPX

    if _check_solver_scip():
        return _SOLVER_TYPE_SCIP

    raise ModuleNotFoundError("could not find installed MIP solver")


DEFAULT_SOLVER = set_default_solver()

# Build List of Supported Solvers
SUPPORTED_SOLVERS = []

if _check_solver_cpx():
    SUPPORTED_SOLVERS.append(_SOLVER_TYPE_CPX)

if _check_solver_scip():
    SUPPORTED_SOLVERS.append(_SOLVER_TYPE_SCIP)

SUPPORTED_SOLVERS = tuple(SUPPORTED_SOLVERS)
