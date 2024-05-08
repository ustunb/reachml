import numpy as np
import pandas as pd
from itertools import chain
from os.path import commonprefix


def has_feature_vector_discrete(X, x):
    return np.all(X == x, axis=1).any()


def has_feature_vector_float(X, x, atol):
    return np.isclose(X, x, atol=atol).all(axis=1).any()


def ensure_matrix(X, names=None):
    if not isinstance(X, (pd.DataFrame, np.ndarray)):
        raise TypeError("`X` must be pandas.DataFrame or numpy.ndarray")
    if isinstance(X, pd.DataFrame):
        if names is not None:
            raise ValueError("Should not supply names when X is a dataframe.")
        names = X.columns.tolist()
        X = X.values

    return X, names


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
    assert all(
        [len(n) > 0 for n in names]
    ), "elements of `names` must have at least 1 character"
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


implies = lambda a, b: np.all(b[a == 1] == 1)


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
