import warnings
from copy import deepcopy
from itertools import chain

import numpy as np
import pandas as pd
from prettytable import PrettyTable

from .action_element import ActionElement
from .constraints.abstract import ActionabilityConstraint
from .constraints.directional_linkage import DirectionalLinkage
from .utils import check_feature_matrix, check_variable_names, expand_values


class ActionSet:
    """
    Class to represent and manipulate feasible actions for the features in a dataset
    """

    def __init__(
        self,
        X,
        names=None,
        indices=None,
        elements=None,
        constraints=None,
        parent=None,
        **kwargs,
    ):
        """
        :param X: pandas.DataFrame or numpy matrix representing a feature matrix (features are columns, samples are rows)
                  X must contain at least 1 column and at least 1 row
        :param names: list of strings containing variable names.
                      names is only required if X is a numpy matrix
        """
        # validate X/Names if creating from scratch
        if elements is None:
            assert isinstance(X, (pd.DataFrame, np.ndarray)), (
                "`X` must be pandas.DataFrame or numpy.ndarray"
            )
            if isinstance(X, pd.DataFrame):
                names = X.columns.tolist()
                X = X.values
            assert check_variable_names(names)
            assert check_feature_matrix(X, d=len(names))

        # key properties
        self._names = names if names is not None else [str(n) for n in names]
        self._indices = (
            indices
            if indices is not None
            else {n: j for j, n in enumerate(self._names)}
        )
        self._elements = (
            elements
            if elements is not None
            else {
                n: ActionElement.from_values(name=n, values=X[:, j])
                for j, n in enumerate(self._names)
            }
        )
        self._constraints = _ConstraintInterface(parent=self)

        # build constraints
        if constraints is not None:
            for con in constraints:
                if parent is not None:  # this is a slice of an existing action set
                    con = deepcopy(con)  # monitor memory usage
                self._constraints.add(con)

        self._parent = parent
        assert self._check_rep()

    # harry: what is this for?
    def _check_rep(self):
        """
        checks representation
        :return: True if representation is valid
        """
        # check that names and indices are consistent
        assert set(self._names) == set(self._indices.keys())
        assert set(self._indices.values()) == set(range(len(self)))
        # check that elements are consistent
        assert set(self._names) == set(self._elements.keys())
        # check that constraints are consistent
        assert self._constraints.__check_rep__()
        return True

    @property
    def names(self):
        return self._names

    @property
    def parent(self):
        return self._parent

    @property
    def discrete(self):
        """:return: True if action set is discrete i.e., all actionable features are discrete"""
        return all([e.variable_type in (int, bool) for e in self if e.actionable])

    @property
    def can_enumerate(self):
        """:return: True if action set can be enumerated"""
        return any(self.actionable) and all(
            [e.variable_type in (int, bool) for e in self if e.actionable]
        )

    @property
    def actionable_features(self):
        """:return: list of actionable feature indices"""
        return {self._indices[e.name] for e in self if e.actionable}

    def get_feature_indices(self, names):
        """
        returns list of indices for feature names
        :param names: string or list of strings for feature names
        :return: index or list of indices
        """
        if isinstance(names, list):
            return [self._indices.get(n) for n in names]
        assert names in self._indices
        return self._indices.get(names)

    @property
    def constraints(self):
        return self._constraints

    def validate(self, X, warn=True, return_df=False):
        """
        check if feature vectors obey the bounds and constraints in an action set
        this function should be used as a minimal test for validity
        :param X: feature matrix
        :param warn: if True will issue a warning
        :param return_df: if True, will return a dataframe highlighting which points are infeasible
        :return: True/False if X obeys all bounds and constraints in this action set (default)
                 if return_df = True, then it will return a DataFrame showing which points in X are violated
        """
        assert check_feature_matrix(X, d=len(self))
        # todo: add fast return
        # fast_return = warn == False and return_df == False

        mutable_features = self.get_feature_indices(
            [a.name for a in self if a.actionable]
        )
        UM, u_to_x, counts = np.unique(
            X[:, mutable_features], axis=0, return_counts=True, return_inverse=True
        )
        U = np.zeros(shape=(UM.shape[0], len(self)))
        U[:, mutable_features] = UM

        # check feasibility of upper/lower bounds
        ub_mutable = self[mutable_features].ub
        lb_mutable = self[mutable_features].lb
        ub_chk = np.array([np.less_equal(x, ub_mutable).all() for x in UM])
        lb_chk = np.array([np.greater_equal(x, lb_mutable).all() for x in UM])
        valid_lb = np.all(lb_chk)
        valid_ub = np.all(ub_chk)

        # todo: handle for immutable attributes within constraints
        # check feasibility of each constraint
        # con_chk = {con.id: np.apply_along_axis(con.check_feasibility, arr = U, axis = 0) for con in self.constraints}
        con_chk = {
            con.id: np.array([con.check_feasibility(x) for x in U])
            for con in self.constraints
        }
        violated_constraints = [k for k, v in con_chk.items() if not np.all(v)]
        valid_constraints = len(violated_constraints) == 0
        out = valid_lb and valid_ub and valid_constraints

        if warn:
            if not valid_lb:
                warnings.warn("X contains points that exceed lower bounds")

            if not valid_ub:
                warnings.warn("X contains points that exceed upper bounds")

            if not valid_constraints:
                warnings.warn(
                    f"X contains points that violate constraints: {violated_constraints}",
                    stacklevel=2,
                )

        if return_df:
            out = (
                pd.DataFrame({"ub": ub_chk, "lb": lb_chk} | con_chk)
                .iloc[u_to_x]
                .reset_index(drop=True)
            )

        return out

    @property
    def partition(self):
        """
        :return: most granular partition of features in ActionSet
                 list of lists, where each inner is a set of feature indices
        """
        partition = []
        remaining_indices = list(range(len(self)))
        while len(remaining_indices) > 0:
            j = remaining_indices.pop(0)
            part = set(self.constraints.get_associated_features(j))
            overlap = False
            for part_id, other_part in enumerate(partition):
                if not other_part.isdisjoint(part):
                    partition[part_id] = other_part.union(part)
                    overlap = True
                    break
            if not overlap:
                partition.append(part)
            remaining_indices = [
                j for j in remaining_indices if j not in chain.from_iterable(partition)
            ]
        partition = [sorted(list(part)) for part in partition]
        return partition

    @property
    def actionable_partition(self):
        """
        :return: most granular partition of features in ActionSet
                 each set includes at least one actionable feature
                 list of lists, where each list if a set of feature indices
        """
        return [part for part in self.partition if any(self[part].actionable)]

    @property
    def separable(self):
        """:return: True if action set is separable in features that are actionable and non-actionable"""
        return all(len(part) == 1 for part in self.partition)

    @property
    def df(self):
        """
        :return: data frame containing key action set parameters
        """
        df = pd.DataFrame(
            {
                "name": self.name,
                "variable_type": self.variable_type,
                "lb": self.lb,
                "ub": self.ub,
                "actionable": self.actionable,
                "step_direction": self.step_direction,
            }
        )
        return df

    def get_bounds(self, x, bound_type):
        """
        :param x: point
        :param bound_type: 'lb' or 'ub'
        :param part: list of feature indices for partitioning
        :return:
        """
        assert bound_type in ("lb", "ub"), f"invalid bound_type: {bound_type}"
        out = [
            aj.get_action_bound(xj, bound_type=bound_type) for aj, xj in zip(self, x)
        ]
        return out

    #### built-ins ####
    def __len__(self):
        return len(self._names)

    def __iter__(self):
        return (self._elements[n] for n in self._names)

    def __str__(self):
        return tabulate_actions(self)

    def __repr__(self):
        return tabulate_actions(self)

    def __eq__(self, other):
        out = (
            isinstance(other, ActionSet)
            and self._names == other._names
            and self.constraints == other.constraints
            and all([a == b for a, b in zip(self, other)])
        )
        return out

    #### getter/setter methods ####
    def __setitem__(self, name, e):
        assert isinstance(e, ActionElement), "ActionSet can only contain ActionElements"
        assert name in self._names, f"no variable with name {name} in ActionSet"
        self._elements.update({name: e})

    def __getitem__(self, index):
        match index:
            case str():
                out = self._elements[index]
            case int() | np.int_():
                out = self._elements[self._names[index]]
            case list() | slice() | np.ndarray():
                # transform array or slice to list
                if isinstance(index, np.ndarray):
                    index = index.tolist()
                elif isinstance(index, slice):
                    index = list(range(len(self)))[index]

                # discover components
                if isinstance(index[0], int):
                    names = [self._names[j] for j in index]
                    idx_lst = index
                elif isinstance(index[0], bool):
                    names = [self._names[j] for j, v in enumerate(index) if v]
                    idx_lst = [j for j, v in enumerate(index) if v]
                elif isinstance(index[0], str):
                    names = index
                    idx_lst = [self._indices[n] for n in names]

                # constraints
                out = ActionSet(
                    X=[],
                    names=names,
                    indices={n: j for j, n in enumerate(names)},
                    elements={n: self._elements[n] for n in names},
                    constraints=self.constraints.get_associated_constraints(idx_lst),
                    parent=self,
                )
            case _:
                raise IndexError(
                    "index must be str, int, slice, or a list of names or indices"
                )

        return out

    def __getattribute__(self, name):
        if name[0] == "_" or (name not in ActionElement.__annotations__):
            return object.__getattribute__(self, name)
        else:
            return [getattr(self._elements[n], name) for n, j in self._indices.items()]

    def __setattr__(self, name, value):
        """
        sets attribuets with broadcasting
        :param name:
        :param value:
        :return:
        """
        # broadcast values
        if hasattr(self, "_elements") and hasattr(ActionElement, name):
            attr_values = expand_values(value, len(self))
            for n, j in self._indices.items():
                self._elements[n].__setattr__(name, attr_values[j])
        else:
            object.__setattr__(self, name, value)


class _ConstraintInterface:
    """
    Class to represent and manipulate actionability constraints that involve 2+ features
    """

    def __init__(self, parent=None):
        self._parent = parent
        self._map = {}
        self._df = pd.DataFrame(columns=["const_id", "feature_name", "feature_idx"])
        self._next_id = 0

    def __check_rep__(self):
        """checks representation"""
        all_ids = list(self._map.keys())
        assert np.greater_equal(all_ids, 0).all(), "ids should be positive integers"
        assert set(all_ids) == set(self._df.const_id), "map ids should match df ids"
        for i, cons in self._map.items():
            assert len(self._df.const_id == i) >= 1, (
                "expecting at least 1 feature per constraint"
            )
            # todo: check that self._df only contains 1 feature_idx per constraint_id pair
        if len(all_ids) > 0:
            assert self._next_id > max(all_ids), (
                "next_id should exceed current largest constraint id"
            )
        return True

    @property
    def parent(self):
        return self._parent

    @property
    def df(self):
        """const_id, name, index triplets"""
        return self._df

    @property
    def linkage_matrix(self):
        """
        matrix of linkages between the features in the action set
        L[j,k] = change in feature k that result from action on feature j
        """
        get_index = self.parent.get_feature_indices
        L = np.eye(len(self.parent))
        linkage_constraints = filter(
            lambda x: isinstance(x, DirectionalLinkage), self._map.values()
        )
        for cons in linkage_constraints:
            j = get_index(cons.source)
            for target, scale in zip(cons.targets, cons.scales):
                k = get_index(target)
                L[j, k] = scale
        # todo: account for standard linkages
        return L

    def add(self, constraint):
        """
        adds a constraint to the set of constraints
        :param constraint:
        :return:
        """
        assert isinstance(constraint, ActionabilityConstraint)
        assert not self.__contains__(constraint)
        constraint.parent = self.parent
        const_id = self._next_id
        self._map.update({const_id: constraint})
        self._next_id += 1
        # add to feature_df
        df_new = pd.DataFrame(
            data={
                "const_id": const_id,
                "feature_name": constraint.names,
                "feature_idx": self.parent.get_feature_indices(constraint.names),
            }
        )
        self._df = pd.concat([self._df, df_new]).reset_index(drop=True)
        assert self.__check_rep__()
        return const_id

    def drop(self, const_id):
        """
        drops a constraint from the set of constraints
        :param const_id: id for dropped constraint
        :return: True if dropped
        """
        dropped = False
        if const_id in self._map:
            cons = self._map.pop(const_id)
            self._df = self._df[self._df.const_id != const_id]
            cons.parent = None
            assert self.__check_rep__()
            dropped = True
        return dropped

    def clear(self):
        """
        drops all constraints from the set of constraints
        :param const_id: id for dropped constraint
        :return: True if dropped
        """
        to_drop = list(self._map.keys())
        dropped = True
        for const_id in to_drop:
            dropped = dropped and self.drop(const_id)
        if dropped:
            self._next_id = 0
        return dropped

    def get_associated_features(self, i, return_constraint_ids=False):
        """
        returns a list of features linked with feature i via constraints
        :param i: feature index
        :return: list of feature indices
        """
        df = self._df
        constraint_matches = {}
        feature_matches = df.feature_idx.isin([i])
        if any(feature_matches):
            constraint_matches = set(df[feature_matches].const_id)
            pull_idx = df.const_id.isin(constraint_matches)
            out = list(set(df[pull_idx].feature_idx))
            out.sort()
        else:
            out = [i]

        if return_constraint_ids:
            out = (out, constraint_matches)

        return out

    def get_associated_constraints(self, features, return_ids=False):
        """
        returns constraints associated with a set of features
        :param features: list of feature indices
        :param return_ids: if True, will return a list of constraint ids
        :return: list of constraints or constraint ids
        """
        const_bool = (
            self.df.groupby("const_id")[["feature_idx"]]
            .agg(set)
            .apply(lambda x: x["feature_idx"].issubset(set(features)), axis=1)
        )  # boolean series

        const_ids = const_bool[const_bool].index

        if return_ids:
            out = const_ids
        else:
            out = [self._map[i] for i in const_ids]

        return out

    def find(self, constraint):
        """
        returns const_id of a constraint
        :param constraint: ActionabilityConstraint
        :return: index of constraint; or -1 if none
        """
        for k, v in self._map.items():
            if v is constraint:
                return k
        return -1

    #### built-ins ####
    def __contains__(self, constraint):
        """
        :param constraint:
        :return:
        """
        for v in self._map.values():
            if v == constraint:
                return True
        return False

    def __iter__(self):
        """iterate over constraint objects"""
        return self._map.values().__iter__()

    def __eq__(self, other):
        """returns True if other ConstraintInterface has the same map, df, id"""
        out = (
            isinstance(other, _ConstraintInterface)
            and self._map == other._map
            and all(self._df == other.df)
            and self._next_id == other._next_id
        )
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
    TYPES = {bool: "<bool>", int: "<int>", float: "<float>"}
    FMT = {bool: "1.0f", int: "1.0f", float: "1.2f"}
    t = PrettyTable()
    vtypes = [TYPES[v] for v in action_set.variable_type]
    # t.add_column("", list(range(len(action_set))), align="r")
    t.add_column("", list(action_set._indices.values()), align="r")
    t.add_column("name", action_set.name, align="l")
    t.add_column("type", vtypes, align="c")
    t.add_column("actionable", action_set.actionable, align="c")
    t.add_column("lb", [f"{a.lb:{FMT[a.variable_type]}}" for a in action_set], align="c")
    t.add_column("ub", [f"{a.ub:{FMT[a.variable_type]}}" for a in action_set], align="c")
    t.add_column("step_direction", action_set.step_direction, align="r")
    t.add_column("step_ub", [v if np.isfinite(v) else "" for v in action_set.step_ub], align="r")
    t.add_column("step_lb", [v if np.isfinite(v) else "" for v in action_set.step_lb], align="r")
    return str(t)
