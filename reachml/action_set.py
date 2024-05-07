import numpy as np
import pandas as pd
from itertools import chain
import warnings
from prettytable import PrettyTable
from .action_element import ActionElement
from .constraints.abstract import ActionabilityConstraint
from .constraints.directional_linkage import DirectionalLinkage
from .utils import check_variable_names, expand_values, check_feature_matrix


class ActionSet(object):
    """
    Class to represent and manipulate feasible actions for the features in a dataset
    """

    def __init__(self, X, names=None, **kwargs):
        """
        :param X: pandas.DataFrame or numpy matrix representing a feature matrix (features are columns, samples are rows)
                  X must contain at least 1 column and at least 1 row
        :param names: list of strings containing variable names.
                      names is only required if X is a numpy matrix
        """
        assert isinstance(
            X, (pd.DataFrame, np.ndarray)
        ), "`X` must be pandas.DataFrame or numpy.ndarray"
        if isinstance(X, pd.DataFrame):
            names = X.columns.tolist()
            X = X.values

        # validate X/Names
        assert check_variable_names(names)
        assert check_feature_matrix(X, d=len(names))

        # key properties
        self._names = [str(n) for n in names]
        self._indices = {n: j for j, n in enumerate(self._names)}
        self._elements = {
            n: ActionElement.from_values(name=n, values=X[:, j])
            for j, n in enumerate(self._names)
        }
        self._constraints = _ConstraintInterface(parent=self)
        assert self._check_rep()

    def _check_rep(self):
        """check if representation invariants are True"""
        # elements = self._elements.values()
        # assert all([isinstance(e, ActionElement) for e in elements])
        return True

    @property
    def names(self):
        return self._names

    @property
    def discrete(self):
        """:return: True if action set is discrete"""
        return all([e.variable_type in (int, bool) for e in self if e.actionable])

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
        validate the bounds and constraints in the action set on a set of feature vectors
        :param X: feature matrix
        :param warn: if True will issue a warning
        :param return_df: if True, will return a dataframe highlighting which points are infeasible
        :return: True/False if X meets all the bounds and constraints in this action set (default)
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

        # todo: handling for immutable attribuets within constraints
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
                warnings.warn(f"X contains points that exceed lower bounds")

            if not valid_ub:
                warnings.warn(f"X contains points that exceed upper bounds")

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

    def get_bounds(self, x, bound_type, part=None):
        """
        :param x: point
        :param bound_type: 'lb' or 'ub'
        :param part: list of feature indices for partitioning
        :return:
        """
        assert bound_type in ("lb", "ub"), f"invalid bound_type: {bound_type}"
        if part is None:
            out = [
                aj.get_action_bound(xj, bound_type=bound_type)
                for aj, xj in zip(self, x)
            ]
        else:
            out = [
                aj.get_action_bound(xj, bound_type=bound_type) if j in part else 0.0
                for j, (aj, xj) in enumerate(zip(self, x))
            ]
        return out

    #### built-ins ####
    def __len__(self):
        return len(self._names)

    def __iter__(self):
        return (self._elements[n] for n in self._names)

    def __eq__(self, other):
        out = (
            isinstance(other, ActionSet)
            and self._names == other._names
            and self.constraints == other.constraints
            and all([a == b for a, b in zip(self, other)])
        )
        return out

    #### getter/setter methods ####
    def __getitem__(self, index):
        if isinstance(index, str):
            return self._elements[index]
        elif isinstance(index, (int, np.int_)):
            return self._elements[self._names[index]]
        elif isinstance(index, list):
            if isinstance(index[0], int):
                names = [self._names[j] for j in index]
            elif isinstance(index[0], bool):
                names = [self._names[j] for j, v in enumerate(index) if v]
            elif isinstance(index[0], str):
                names = index
            return _ActionSlice([self._elements[n] for n in names])
        elif isinstance(index, np.ndarray):
            names = np.array(self._names)[index].tolist()
            return _ActionSlice([self._elements[n] for n in names])
        elif isinstance(index, slice):
            return _ActionSlice([self._elements[n] for n in self._names[index]])
        else:
            raise IndexError("index must be str, int, a list of strings/int or a slice")

    def __setitem__(self, name, e):
        assert isinstance(e, ActionElement), "ActionSet can only contain ActionElements"
        assert name in self._names, f"no variable with name {name} in ActionSet"
        self._elements.update({name: e})

    def __getattribute__(self, attr_name):
        if attr_name[0] == "_" or (attr_name not in ActionElement.__annotations__):
            return object.__getattribute__(self, attr_name)
        else:
            return [
                getattr(self._elements[n], attr_name) for n, j in self._indices.items()
            ]

    def __setattr__(self, name, value):
        if hasattr(self, "_elements") and hasattr(ActionElement, name):
            attr_values = expand_values(value, len(self))
            for n, j in self._indices.items():
                self._elements[n].__setattr__(name, attr_values[j])
        else:
            object.__setattr__(self, name, value)

    #### printing
    def __str__(self):
        return tabulate_actions(self)

    def __repr__(self):
        return tabulate_actions(self)

    def to_latex(self):
        """
        :param action_set: ActionSet object
        :return: formatted latex table summarizing the action set for publications
        """
        df = self.df
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
                new_types[i] = tex_binary_str if t in bool else tex_integer_str

        df["variable_type"] = new_types
        df["ub"] = new_ub
        df["lb"] = new_lb

        df["mutability"] = df["actionable"].map(
            {False: "no", True: "yes"}
        )  # todo change
        up_idx = df["actionable"] & df["step_direction"] > 0
        dn_idx = df["actionable"] & df["step_direction"] < 0
        df.loc[up_idx, "mutability"] = "only increases"
        df.loc[dn_idx, "mutability"] = "only decreases"

        df = df.drop(["actionable", "step_direction"], axis=1)
        df = df[["name", "variable_type", "lb", "ub", "mutability"]]
        df = df.rename(
            columns={
                "name": "Name",
                "variable_type": "Type",
                "actionability": "Actionability",
                "lb": "LB",
                "ub": "UB",
            }
        )

        table = df.to_latex(index=False, escape=False)
        table = table.replace("_", "\_")
        return table


class _ConstraintInterface(object):
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
            assert (
                len(self._df.const_id == i) >= 1
            ), "expecting at least 1 feature per constraint"
            # todo: check that self._df only contains 1 feature_idx per constraint_id pair
        if len(all_ids) > 0:
            assert self._next_id > max(
                all_ids
            ), "next_id should exceed current largest constraint id"
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
        self._df = pd.concat([self._df, df_new])
        constraint.parent = self.parent
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
        :param i:
        :return:
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


class _ActionSlice(object):
    """
    Class to set ActionElement properties by slicing.
    This class allows us to support commands like:
        a = ActionSet(...)
        a[1:2].ub = 2
    """

    def __init__(self, action_elements):
        self._indices = {e.name: j for j, e in enumerate(action_elements)}
        self._elements = {e.name: e for e in action_elements}

    def __getattr__(self, name):
        if name in ("_indices", "_elements"):
            object.__getattr__(self, name)
        else:
            return [getattr(self._elements[n], name) for n, j in self._indices.items()]

    def __setattr__(self, name, value):
        if name in ("_indices", "_elements"):
            object.__setattr__(self, name, value)
        else:
            assert hasattr(ActionElement, name)
            attr_values = expand_values(value, len(self._indices))
            for n, j in self._indices.items():
                setattr(self._elements[n], name, attr_values[j])

    def __len__(self):
        return len(self._indices)

    def __str__(self):
        return tabulate_actions(self)

    def __repr__(self):
        return str(self)




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
    t.add_column("", list(range(len(action_set))), align="r")
    t.add_column("name", action_set.name, align="l")
    t.add_column("type", vtypes, align="c")
    t.add_column("actionable", action_set.actionable, align="c")
    t.add_column("lb", [f"{a.lb:{FMT[a.variable_type]}}" for a in action_set], align="c")
    t.add_column("ub", [f"{a.ub:{FMT[a.variable_type]}}" for a in action_set], align="c")
    t.add_column("step_direction", action_set.step_direction, align="r")
    t.add_column("step_ub", [v if np.isfinite(v) else "" for v in action_set.step_ub], align="r")
    t.add_column("step_lb", [v if np.isfinite(v) else "" for v in action_set.step_lb], align="r")
    return str(t)
