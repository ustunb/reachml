from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import pandas as pd

from .action_set import ActionSet
from .enumeration import ReachableSetEnumerator

# default threshold
THRESH = 0.1


class ReachableSet(ABC):
    """
    Class to represent or manipulate a reachable set over a discrete feature space
    """

    _TOLERANCE = 1e-16
    _METADATA_KEYS = ["complete"]

    def __init__(
        self,
        action_set: ActionSet,
        x: Optional[np.ndarray] = None,
        complete: bool = False,
        values: Optional[np.ndarray] = None,
        initialize_from_actions: bool = False,
        **kwargs,
    ):
        """
        :param action_set: action set
        :param x: source point
        :param complete: indicator if complete
        :param values: array-like -- set of d-dimensional feature vectors or actions to add
        :param initialize_from_actions: set to True if values that we pass are actions
        """

        if x is None:
            if initialize_from_actions:
                raise ValueError(
                    "Cannot initialize from actions without the initial point."
                )
            if values is None or len(values) == 0:
                raise ValueError(
                    "Need to provide values if the initial point is not given."
                )
            else:
                x = values[0].flatten()

        self._action_set = action_set
        self._complete = complete
        self._x = x
        self._generator = None
        self._time = kwargs.get("time", None)

    @property
    def action_set(self):
        """return action set"""
        return self._action_set

    @property
    def discrete(self) -> bool:
        """returns True if fixed point"""
        return self._action_set.discrete

    @property
    def x(self):
        """return source point"""
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def d(self):
        """returns number of dimensions"""
        return len(self._x)

    @property
    def X(self):
        """returns reachable feature vectors"""
        return self._X

    @property
    def actions(self) -> np.ndarray:
        """returns action vectors, computed on the fly"""
        return np.subtract(self.X, self.x)

    @property
    def complete(self) -> bool:
        """returns True if reachable set contains all reachable points"""
        return self._complete

    @property
    def time(self) -> Optional[float]:
        """returns time taken to generate reachable set"""
        return self._time

    @property
    def fixed(self) -> bool:
        """returns True if fixed point"""
        return len(self) == 1 and self._complete

    @property
    def generator(self):
        """returns generator"""
        if self._generator is None:
            self._generator = self._initialize_generator()

        return self._generator

    @abstractmethod
    def generate(self, **kwargs):
        """generate reachable set, points are stored in self.X"""
        pass

    @abstractmethod
    def _initialize_generator(self):
        """initialize generator"""
        pass

    def extract(self, other):
        """extract points from another reachable set"""
        raise NotImplementedError()

    def find(self, clf, target):
        """
        :param clf: classifier with a predict function
        :param target: float/int that attains a target class, or array-like or target classes
        :return: first reachable point that attains a target prediction from the classifier
        """
        # check that clf has a predict function
        assert hasattr(clf, "predict")

        if not isinstance(target, (list, tuple)):
            target = np.float(target)

        # todo: check that target classes are in classifier.classes
        # todo: optimize for loop using e.g. numba or using the size of X
        out = (False, None)
        for x in self.X:
            if clf.predict(x) in target:
                out = (True, x)
                break
        return out

    def reset(self):
        """reset reachable set"""
        self._complete = False

    def __getitem__(self, i: int) -> np.ndarray:
        return self._X[i]

    def __check_rep__(self) -> bool:
        """returns True if class invariants hold"""
        assert self.X.shape[0] == len(self)
        assert self.has_feature_vector(self.x)
        return True

    def __contains__(self, item: Union[np.ndarray, list]):
        """returns True if reachable set contains all reachable points"""
        if isinstance(item, list):
            out = np.all(self.X == item, axis=1).any()
        elif isinstance(item, np.ndarray) and item.ndim == 1:
            out = self.has_feature_vector(item)
        elif isinstance(item, np.ndarray) and item.ndim == 2:
            out = np.all([self.has_feature_vector(x) for x in item])
        else:
            out = False
        return out

    def __is_comparable_to__(self, other):
        out = (
            isinstance(other, ReachableSet)
            and self.action_set.names == other.action_set.names
        )
        return out

    def __eq__(self, other):
        out = (
            self.__is_comparable_to__(other)
            and np.array_equal(self.x, other.x)
            and len(self) == len(other)
            and self.__contains__(other.X)
        )
        return out

    def __add__(self, other):
        """add two reachable sets together"""
        raise NotImplementedError()
        assert isinstance(other, ReachableSet)
        # todo: check that reachable sets are compatible

    def __len__(self) -> int:
        """returns number of points in the reachable set, including the original point"""
        return self._X.shape[0]

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}<n={len(self)}, complete={bool(self._complete)}>"

    def _get_metadata(self) -> pd.Series:
        metadata = pd.Series(dict(complete=self._complete))
        assert all(metadata.index == ReachableSet._METADATA_KEYS)
        return metadata


class EnumeratedReachableSet(ReachableSet):
    """
    Class to represent or manipulate a reachable set over a discrete feature space
    """

    def __init__(
        self,
        action_set: ActionSet,
        x: Optional[np.ndarray] = None,
        complete: bool = False,
        values: Optional[np.ndarray] = None,
        initialize_from_actions: bool = False,
        **kwargs,
    ):
        """
        :param action_set: action set
        :param x: source point
        """
        super().__init__(
            action_set=action_set, x=x, complete=complete, values=values, **kwargs
        )
        self._X = np.array(x).reshape((1, self.d))

        if self.discrete:
            self.has_feature_vector = lambda x: np.all(self._X == x, axis=1).any()
        else:
            self.has_feature_vector = (
                lambda x: np.isclose(self._X, x, atol=self._TOLERANCE).all(axis=1).any()
            )

        if values is not None:
            self.add(values=values, actions=initialize_from_actions, **kwargs)

    def _initialize_generator(self):
        """initialize generator"""
        return ReachableSetEnumerator(action_set=self.action_set, x=self.x)

    def generate(self, **kwargs):
        """generate reachable set using enumeration"""
        self.generator.enumerate(**kwargs)
        vals = self.generator.feasible_actions

        self.add(values=vals, actions=True)
        self._complete = True

        return len(vals)

    def add(
        self,
        values: np.ndarray,
        actions: bool = False,
        check_distinct: bool = True,
        check_exists: bool = True,
        **kwargs,
    ):
        """
        :param values: array-like -- feature vectors / to add
        :param actions: set to True if `values` are actions rather than feature vectors
        :param check_distinct: set to False if `values` are distinct
        :param check_exists: set to False if `values` do not exist in current reachable set
        :return:
        """
        if isinstance(values, list):
            values = np.vstack(values)

        assert values.ndim == 2
        assert values.shape[0] > 0
        assert values.shape[1] == self.d

        if check_distinct:
            values = np.unique(values, axis=0)

        if actions:
            values = self._x + values

        if check_exists:
            keep_idx = [not self.has_feature_vector(x) for x in values]
            values = values[keep_idx]

        self._X = np.append(self._X, values=values, axis=0)
        out = values.shape[0]
        return out

    def reset(self):
        """reset reachable set"""
        super().reset()
        self._X = np.array(self.x).reshape((1, self.d))