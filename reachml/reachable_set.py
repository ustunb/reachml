"""Reachable-set abstractions and concrete implementations."""

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import pandas as pd

from .action_set import ActionSet
from .enumeration import ReachableSetEnumerator
from .sampling import ReachableSetSampler
from .utils import DEFAULT_SOLVER

# default threshold
THRESH = 0.1


"""Reachable-set abstractions for enumeration and sampling over actions."""


class ReachableSet(ABC):
    """Represent or manipulate a reachable set for a point `x`."""

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
        """Initialize a reachable set.

        Args:
            action_set: Action set describing feasible actions.
            x: Source point.
            complete: If True, the set contains all reachable points.
            values: Optional initial feature vectors or actions.
            initialize_from_actions: If True, `values` are actions.
            **kwargs: Additional metadata such as `time`.
        """
        if x is None:
            if initialize_from_actions:
                raise ValueError("Cannot initialize from actions without the initial point.")
            if values is None or len(values) == 0:
                raise ValueError("Need to provide values if the initial point is not given.")
            else:
                x = values[0].flatten()

        self._action_set = action_set
        self._complete = complete
        self._x = x
        self._generator = None
        self._time = kwargs.get("time", None)

    @property
    def action_set(self):
        """Return action set."""
        return self._action_set

    @property
    def discrete(self) -> bool:
        """Returns True if fixed point."""
        return self._action_set.discrete

    @property
    def x(self):
        """Return source point."""
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def d(self):
        """Returns number of dimensions."""
        return len(self._x)

    @property
    def X(self):
        """Returns reachable feature vectors."""
        return self._X

    @property
    def actions(self) -> np.ndarray:
        """Returns action vectors, computed on the fly."""
        return np.subtract(self.X, self.x)

    @property
    def complete(self) -> bool:
        """Returns True if reachable set contains all reachable points."""
        return self._complete

    @property
    def time(self) -> Optional[float]:
        """Returns time taken to generate reachable set."""
        return self._time

    @property
    def fixed(self) -> bool:
        """Returns True if fixed point."""
        return len(self) == 1 and self._complete

    @property
    def generator(self):
        """Returns generator."""
        if self._generator is None:
            self._generator = self._initialize_generator()

        return self._generator

    @abstractmethod
    def generate(self, **kwargs):
        """Generate reachable set, points are stored in self.X."""
        pass

    @abstractmethod
    def _initialize_generator(self):
        """Initialize generator."""
        pass

    def extract(self, other):
        """Extract points from another reachable set."""
        raise NotImplementedError()

    def find(self, clf, target):
        """Return first reachable point that attains a target prediction.

        Args:
            clf: Classifier with `predict`.
            target: Class or list/tuple of classes to match.
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
        """Reset reachable set."""
        self._complete = False

    def __getitem__(self, i: int) -> np.ndarray:
        """Return the i-th point in the reachable set."""
        return self._X[i]

    def __check_rep__(self) -> bool:
        """Returns True if class invariants hold."""
        assert self.X.shape[0] == len(self)
        assert self.has_feature_vector(self.x)
        return True

    def __contains__(self, item: Union[np.ndarray, list]):
        """Returns True if reachable set contains all reachable points."""
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
        """Return True if `other` has the same action set names."""
        out = isinstance(other, ReachableSet) and self.action_set.names == other.action_set.names
        return out

    def __eq__(self, other):
        """Equality based on action set names, x, size, and containment."""
        out = (
            self.__is_comparable_to__(other)
            and np.array_equal(self.x, other.x)
            and len(self) == len(other)
            and self.__contains__(other.X)
        )
        return out

    def __add__(self, other):
        """Add two reachable sets together."""
        raise NotImplementedError()
        assert isinstance(other, ReachableSet)
        # todo: check that reachable sets are compatible

    def __len__(self) -> int:
        """Returns number of points in the reachable set, including the original point."""
        return self._X.shape[0]

    def __repr__(self) -> str:
        """Debug string with size and completeness of the reachable set."""
        return f"<{self.__class__.__name__}<n={len(self)}, complete={bool(self._complete)}>"

    def _get_metadata(self) -> pd.Series:
        metadata = pd.Series(dict(complete=self._complete))
        assert all(metadata.index == ReachableSet._METADATA_KEYS)
        return metadata


class EnumeratedReachableSet(ReachableSet):
    """Class to represent or manipulate a reachable set over a discrete feature space."""

    def __init__(
        self,
        action_set: ActionSet,
        x: Optional[np.ndarray] = None,
        complete: bool = False,
        values: Optional[np.ndarray] = None,
        initialize_from_actions: bool = False,
        solver: Optional[str] = DEFAULT_SOLVER,
        **kwargs,
    ):
        """Initialize an enumerated reachable set for `x` over `action_set`."""
        super().__init__(action_set=action_set, x=x, complete=complete, values=values, **kwargs)
        self._X = np.array(x).reshape((1, self.d))

        if self.discrete:
            self.has_feature_vector = lambda x: np.all(self._X == x, axis=1).any()
        else:
            self.has_feature_vector = (
                lambda x: np.isclose(self._X, x, atol=self._TOLERANCE).all(axis=1).any()
            )

        if values is not None:
            self.add(values=values, actions=initialize_from_actions, **kwargs)

        self.solver = solver

    def _initialize_generator(self):
        """Initialize generator."""
        return ReachableSetEnumerator(action_set=self.action_set, x=self.x, solver=self.solver)

    def generate(self, **kwargs):
        """Generate reachable set using enumeration."""
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
        """Add feature vectors or actions to the enumerated reachable set.

        Args:
            values: Feature vectors to add (or actions if `actions=True`).
            actions: If True, `values` are actions rather than points.
            check_distinct: If True, keep only unique rows from `values`.
            check_exists: If True, skip rows that already exist in the set.
            **kwargs: Reserved for future options (unused).
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
        """Reset reachable set."""
        super().reset()
        self._X = np.array(self.x).reshape((1, self.d))


class SampledReachableSet(ReachableSet):
    """Class to represent or manipulate a reachable set over a discrete feature space."""

    _METADATA_KEYS = ReachableSet._METADATA_KEYS + ["resp_thresh"]

    def __init__(
        self,
        action_set: ActionSet,
        x: Optional[np.ndarray] = None,
        complete: bool = False,
        values: Optional[np.ndarray] = None,
        initialize_from_actions: bool = False,
        resp_thresh=THRESH,
        **kwargs,
    ):
        """Initialize a sampled reachable set for `x` over `action_set`."""
        super().__init__(
            action_set=action_set,
            x=x,
            complete=complete,
            values=values,
            resp_thresh=resp_thresh,
            **kwargs,
        )

        self._resp_thresh = resp_thresh

        self._X = None

        if values is not None:
            self.add(values=values, actions=initialize_from_actions, **kwargs)

        self.seed = kwargs.get("seed", None)

    @property
    def resp_thresh(self):
        """Responsiveness threshold epsilon used for sampling termination."""
        return self._resp_thresh

    @ReachableSet.x.setter
    def x(self, value):
        """Update the source point for this reachable set."""
        # sometimes throws errors, check
        # super(SampledReachableSet, self.__class__).x.fset(self, value)
        self._x = value
        if self.generator is not None:
            self.generator.x = value

    def _initialize_generator(self):
        """Initialize generator."""
        return ReachableSetSampler(action_set=self.action_set, x=self.x, seed=self.seed)

    def generate(self, **kwargs):
        """Generate reachable set using sampling.

        Args:
            **kwargs: keyword arguments
                alpha (float, optional): significance level for sampling
                                         (between 0 and 1)
                n (int, optional): number of samples to draw, overrides alpha
        """
        assert self.generator is not None, "Generator not initialized"
        n = kwargs.get("n", None)
        if n is None:
            n = self.calculate_n(alpha=kwargs.get("alpha", 0.05))

        vals = self.generator.sample(n)

        self.add(values=vals, actions=False, check_distinct=False)
        self._complete = True

        return len(vals)

    # TODO: #4 update function to match paper @harrycheon
    def calculate_n(self, alpha, **kwargs):
        """Calculate number of samples to draw."""
        return np.ceil(np.log(alpha) / np.log(1 - self.resp_thresh)).astype(int)

    def reset(self, **kwargs):
        """Reset reachable set and refresh rng for sampler."""
        super().reset()
        self._X = None
        if self.generator is not None:
            self.generator.reset_rng()

    def add(self, values: np.ndarray, actions: bool = False, **kwargs):
        """Add `values` to the reachable set, optionally as actions.

        Args:
            values: Feature vectors to add (or actions if `actions=True`).
            actions: If True, `values` are action vectors instead of points.
            check_distinct: If False, skip duplicates check.
            check_exists: If False, skip containment check in current set.
            **kwargs: Reserved for future options (unused).
        """
        if isinstance(values, list):
            values = np.vstack(values)

        assert values.ndim == 2
        assert values.shape[0] > 0
        assert values.shape[1] == self.d

        if actions:
            values = self._x + values

        if self._X is None:
            self._X = np.array(values)
        else:
            self._X = np.append(self._X, values=values, axis=0)

        out = values.shape[0]
        return out

    def _get_metadata(self) -> pd.Series:
        metadata = super()._get_metadata()
        metadata["resp_thresh"] = self.resp_thresh
        return metadata
