"""Abstract base definitions for actionability constraints."""

from abc import ABC, abstractmethod

import numpy as np

from ..utils import check_variable_names


class ActionabilityConstraint(ABC):
    """Abstract class for actionability constraints.

    All constraint classes inherit from this class.
    """

    def __init__(self, names, parent=None, **kwargs):
        """Initialize a constraint over a list of feature names."""
        assert check_variable_names(names)
        assert self._parameters is not None
        self._names = names
        self._parent = None
        if parent is not None:
            self.parent = parent

    @property
    def parameters(self):
        """Tuple of constraint parameters -- changes per class."""
        return self._parameters

    @property
    def names(self):
        """List of feature names."""
        return self._names

    @property
    def parent(self):
        """Pointer to parent action set."""
        return self._parent

    @parent.setter
    def parent(self, action_set):
        """Pointer to parent action set."""
        if action_set is not None:
            missing_features = set(self.names) - set(action_set.names)
            assert len(missing_features) == 0, (
                f"Cannot attach {self.__class__} to ActionSet. Missing: {missing_features}"
            )
            assert self.check_compatibility(action_set)
        self._parent = action_set

    @property
    def id(self):
        """Unique id of this constraint within the parent action set."""
        if self.parent is None:
            raise ValueError("constraint must be attached to ActionSet to have an id")
        return self.parent.constraints.find(self)

    @property
    def indices(self):
        """List of feature indices with respect to the parent; used to determine partitions."""
        if self.parent is None:
            raise ValueError("constraint must be attached to ActionSet to have indices")
        return self._parent.get_feature_indices(self._names)

    # @abstractmethod
    # def is_encoding_constraint(self):
    #     """Return True if constraint specifies categorical/ordinal encoding."""
    #     raise NotImplementedError()
    #
    # @abstractmethod
    # def is_causal_constraint(self):
    #     """Return True if constraint specifies downstream effects."""
    #     raise NotImplementedError()

    # built-ins
    def __eq__(self, other):
        """Structural equality by names and parameter fields."""
        if not isinstance(other, type(self)):
            return False
        if not set(self.names) == set(other.names):
            return False
        out = True
        for p in self._parameters:
            a = self.__getattribute__(p)
            b = other.__getattribute__(p)
            if isinstance(a, np.ndarray):
                out = out & np.array_equal(a, b)
            else:
                out = out & (a == b)
        return out

    def __repr__(self):
        """Debug representation with parameter values."""
        name = self.__class__.__name__
        fields = ",".join([f"{p}={self.__getattribute__(p)}" for p in self._parameters])
        out = f"<{name}(names={self._names},{fields})>"
        return out

    def __str__(self):
        """String representation."""
        return self.__repr__()

    # basic checks on features
    def check_feature_vector(self, x):
        """Return True if `x` is a finite feature vector of expected length.

        Args:
            x: List or ndarray.
        """
        out = len(x) == len(self.names) and np.isfinite(x).all()
        return out

    # methods that each subclass needs to implement
    @abstractmethod
    def check_compatibility(self, action_set):
        """Returns True if constraint is compatible with the action set."""
        raise NotImplementedError()

    @abstractmethod
    def check_feasibility(self, x):
        """Returns True if current point is feasible with the constraint."""
        raise NotImplementedError()

    @abstractmethod
    def adapt(self, x):
        """Adapts constraint parameters for point x."""
        raise NotImplementedError()

    @abstractmethod
    def add_to_cpx(self, cpx, indices, x):
        """Add constraint to CPLEX model for enumeration."""
        raise NotImplementedError()

    @abstractmethod
    def add_to_scip(self, scip, indices, x):
        """Add constraint to PySCIPOpt model for enumeration."""
        raise NotImplementedError()
