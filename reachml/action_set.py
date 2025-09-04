"""Action set and constraint management.

This module defines `ActionSet`, a container of per-feature action elements and
their cross-feature constraints, along with helpers for validation and display.
"""

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
    """Represent and manipulate feasible actions over features."""

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
        """Initialize an action set from data or elements.

        Args:
            X: Feature matrix (`pandas.DataFrame` or `numpy.ndarray`), rows are
                samples and columns are features. Must have at least one column
                and one row when inferring elements.
            names: Optional list of feature names (required if `X` is an
                `ndarray`).
            indices: Optional mapping from name to column index.
            elements: Optional mapping from name to `ActionElement` instances.
            constraints: Optional iterable of actionability constraints.
            parent: Optional parent `ActionSet` when creating a slice.
            **kwargs: Reserved for future options.
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
            indices if indices is not None else {n: j for j, n in enumerate(self._names)}
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
        """Check internal representation invariants.

        Returns:
            True if representation is valid.
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
        """List of feature names in order."""
        return self._names

    @property
    def parent(self):
        """Parent `ActionSet` when this is a slice; otherwise None."""
        return self._parent

    @property
    def discrete(self):
        """Whether all actionable features are discrete (int or bool)."""
        return all([e.variable_type in (int, bool) for e in self if e.actionable])

    @property
    def can_enumerate(self):
        """Whether this action set can be enumerated (all actionable discrete)."""
        return any(self.actionable) and all(
            [e.variable_type in (int, bool) for e in self if e.actionable]
        )

    @property
    def actionable_features(self):
        """Set of actionable feature indices."""
        return {self._indices[e.name] for e in self if e.actionable}

    def get_feature_indices(self, names):
        """Return indices for feature name(s).

        Args:
            names: A feature name or list of names.

        Returns:
            An index (int) or list of indices.
        """
        if isinstance(names, list):
            return [self._indices.get(n) for n in names]
        assert names in self._indices
        return self._indices.get(names)

    @property
    def constraints(self):
        """Constraint interface accessor."""
        return self._constraints

    def validate(self, X, warn=True, return_df=False):
        """Validate feature vectors against bounds and constraints.

        Args:
            X: Feature matrix with `len(self)` columns.
            warn: If True, emit warnings for violations.
            return_df: If True, return a DataFrame highlighting violations per
                unique mutable pattern, aligned back to `X`.

        Returns:
            Boolean if `return_df` is False; otherwise a `pandas.DataFrame`
            with violation indicators per row.
        """
        assert check_feature_matrix(X, d=len(self))
        # todo: add fast return
        # fast_return = warn == False and return_df == False

        mutable_features = self.get_feature_indices([a.name for a in self if a.actionable])
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
        # Example vectorized feasibility check (not used):
        # con_chk = {con.id: np.apply_along_axis(con.check_feasibility, arr=U, axis=0)
        #            for con in self.constraints}
        con_chk = {
            con.id: np.array([con.check_feasibility(x) for x in U]) for con in self.constraints
        }
        violated_constraints = [k for k, v in con_chk.items() if not np.all(v)]
        valid_constraints = len(violated_constraints) == 0
        out = valid_lb and valid_ub and valid_constraints

        if warn:
            if not valid_lb:
                warnings.warn("X contains points that exceed lower bounds", stacklevel=2)

            if not valid_ub:
                warnings.warn("X contains points that exceed upper bounds", stacklevel=2)

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
        """Most granular partition of linked features.

        Returns:
            List of lists where each inner list is a set of feature indices
            linked by constraints.
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
        """Partition subsets that include at least one actionable feature."""
        return [part for part in self.partition if any(self[part].actionable)]

    @property
    def separable(self):
        """Whether all partitions are singletons (fully separable)."""
        return all(len(part) == 1 for part in self.partition)

    @property
    def df(self):
        """DataFrame with key action set parameters per feature."""
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

    @property
    def summary(self):
        """Returns a dictionary with summary statistics of the action set.

        Dictionary keys:
            - "total_features": Total number of features.
            - "num_immutable_features": Count of immutable (not actionable).
            - "num_mutable_features": Count of mutable features.
            - "num_actionable_features": Count of actionable features.
            - "num_partitions": Total count of partitions.
            - "num_separable_features": Partitions of size 1.
            - "num_nonseparable_features": `total_features - num_separable_features`.
            - "min_partition_size": Min size (if any multi-feature partitions).
            - "median_partition_size": Median size (multi-feature partitions).
            - "max_partition_size": Max size (multi-feature partitions).
        """
        d = len(self)
        num_mutable = mutable_indices = set()
        for part in self.partition:
            if any(self._elements[self._names[i]].actionable for i in part):
                for i in part:
                    if not self._elements[self._names[i]].actionable:
                        mutable_indices.add(i)
        num_mutable = len(mutable_indices)
        num_actionable = sum(1 for e in self if e.actionable)
        num_immutable = d - (num_mutable + num_actionable)
        num_mutable += num_actionable
        partitions = self.partition
        num_partitions = len(partitions)
        num_separable = sum(1 for part in partitions if len(part) == 1)

        multi_partition_sizes = [len(part) for part in partitions if len(part) > 1]
        if multi_partition_sizes:
            min_size = min(multi_partition_sizes)
            median_size = np.median(multi_partition_sizes)
            max_size = max(multi_partition_sizes)
        else:
            min_size = median_size = max_size = None

        stats = {
            "total_features": d,
            "num_immutable_features": num_immutable,
            "num_mutable_features": num_mutable,
            "num_actionable_features": num_actionable,
            "num_partitions": num_partitions,
            "num_separable_features": num_separable,
            "num_nonseparable_features": d - num_separable,
            "min_partition_size": min_size,
            "median_partition_size": median_size,
            "max_partition_size": max_size,
        }
        return stats

    def get_bounds(self, x, bound_type):
        """Return per-feature move bounds at `x`.

        Args:
            x: Point (feature vector).
            bound_type: Either "lb" or "ub".

        Returns:
            List of feasible move magnitudes per feature.
        """
        assert bound_type in ("lb", "ub"), f"invalid bound_type: {bound_type}"
        out = [
            aj.get_action_bound(xj, bound_type=bound_type) for aj, xj in zip(self, x, strict=False)
        ]
        return out

    #### built-ins ####
    def __len__(self):
        """Number of features in the action set."""
        return len(self._names)

    def __iter__(self):
        """Iterate over `ActionElement`s in name order."""
        return (self._elements[n] for n in self._names)

    def __str__(self):
        """Pretty table rendering of the action set."""
        return tabulate_actions(self)

    def __repr__(self):
        """Debug-friendly table rendering of the action set."""
        return tabulate_actions(self)

    def __eq__(self, other):
        """Structural equality: names, constraints, and elements."""
        out = (
            isinstance(other, ActionSet)
            and self._names == other._names
            and self.constraints == other.constraints
            and all([a == b for a, b in zip(self, other, strict=False)])
        )
        return out

    #### getter/setter methods ####
    def __setitem__(self, name, e):
        """Replace the `ActionElement` for a named feature."""
        assert isinstance(e, ActionElement), "ActionSet can only contain ActionElements"
        assert name in self._names, f"no variable with name {name} in ActionSet"
        self._elements.update({name: e})

    def __getitem__(self, index):
        """Return an element or a sliced `ActionSet` by index, name, or mask."""
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
                raise IndexError("index must be str, int, slice, or a list of names or indices")

        return out

    def __getattribute__(self, name):
        """Support list-style attribute access for `ActionElement` fields."""
        if name[0] == "_" or (name not in ActionElement.__annotations__):
            return object.__getattribute__(self, name)
        else:
            return [getattr(self._elements[n], name) for n, j in self._indices.items()]

    def __setattr__(self, name, value):
        """Broadcast `ActionElement` attribute updates across features.

        Args:
            name: Attribute name on `ActionElement`.
            value: Scalar or list-like to broadcast across elements.
        """
        # broadcast values
        if hasattr(self, "_elements") and hasattr(ActionElement, name):
            attr_values = expand_values(value, len(self))
            for n, j in self._indices.items():
                self._elements[n].__setattr__(name, attr_values[j])
        else:
            object.__setattr__(self, name, value)


class _ConstraintInterface:
    """Represent and manipulate multi-feature actionability constraints."""

    def __init__(self, parent=None):
        self._parent = parent
        self._map = {}
        self._df = pd.DataFrame(columns=["const_id", "feature_name", "feature_idx"])
        self._next_id = 0

    def __check_rep__(self):
        """Check internal representation invariants for constraints."""
        all_ids = list(self._map.keys())
        assert np.greater_equal(all_ids, 0).all(), "ids should be positive integers"
        assert set(all_ids) == set(self._df.const_id), "map ids should match df ids"
        for i, _cons in self._map.items():
            assert len(self._df.const_id == i) >= 1, "expecting at least 1 feature per constraint"
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
        """Constraint table with `(const_id, name, index)` triplets."""
        return self._df

    @property
    def linkage_matrix(self):
        """Linkage matrix L where L[j, k] gives induced change in k by j."""
        get_index = self.parent.get_feature_indices
        L = np.eye(len(self.parent))
        linkage_constraints = filter(
            lambda x: isinstance(x, DirectionalLinkage), self._map.values()
        )
        for cons in linkage_constraints:
            j = get_index(cons.source)
            for target, scale in zip(cons.targets, cons.scales, strict=False):
                k = get_index(target)
                L[j, k] = scale
        # todo: account for standard linkages
        return L

    def add(self, constraint):
        """Add a constraint and return its id.

        Args:
            constraint: An `ActionabilityConstraint` instance.

        Returns:
            The assigned constraint id (int).
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
        """Drop a constraint by id.

        Args:
            const_id: Constraint id to remove.

        Returns:
            True if a constraint was removed.
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
        """Drop all constraints and reset ids.

        Returns:
            True if all constraints were removed.
        """
        to_drop = list(self._map.keys())
        dropped = True
        for const_id in to_drop:
            dropped = dropped and self.drop(const_id)
        if dropped:
            self._next_id = 0
        return dropped

    def get_associated_features(self, i, return_constraint_ids=False):
        """Return features linked with feature `i` via constraints.

        Args:
            i: Feature index.
            return_constraint_ids: If True, also return the matching constraint ids.

        Returns:
            List of feature indices; optionally a tuple with ids.
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
        """Return constraints associated with a set of features.

        Args:
            features: List of feature indices.
            return_ids: If True, return ids instead of constraint objects.

        Returns:
            List of constraints or their ids.
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
        """Return the id for a constraint object (or -1 if not found)."""
        for k, v in self._map.items():
            if v is constraint:
                return k
        return -1

    #### built-ins ####
    def __contains__(self, constraint):
        """Membership test based on equality with an existing constraint."""
        for v in self._map.values():
            if v == constraint:
                return True
        return False

    def __iter__(self):
        """Iterate over constraint objects."""
        return self._map.values().__iter__()

    def __eq__(self, other):
        """Returns True if other ConstraintInterface has the same map, df, id."""
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
    """Build a table with per-feature action parameters.

    Args:
        action_set: The `ActionSet` to display.

    Returns:
        A string representation of the table.
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
