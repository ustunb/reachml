import numpy as np
from abc import ABC, abstractmethod
from ..utils import check_variable_names


class ActionabilityConstraint(ABC):
    """
    Abstract Class for Actionability Constraints
    All Constraint Classes inherit from this class
    """

    def __init__(self, names, parent=None, **kwargs):
        assert check_variable_names(names)
        assert self._parameters is not None
        self._names = names
        self._parent = None
        if parent is not None:
            self.parent = parent

    @property
    def parameters(self):
        """tuple of constraint parameters -- changes per class"""
        return self._parameters

    @property
    def names(self):
        """list of feature names"""
        return self._names

    @property
    def parent(self):
        """pointer to parent action set"""
        return self._parent

    @parent.setter
    def parent(self, action_set):
        """pointer to parent action set"""
        if action_set is not None:
            missing_features = set(self.names) - set(action_set.names)
            assert (
                len(missing_features) == 0
            ), f"Cannot attach {self.__class__} to ActionSet. ActionSet is missing the following features: {missing_features}"
            assert self.check_compatibility(action_set)
        self._parent = action_set

    @property
    def id(self):
        if self.parent is None:
            raise ValueError("constraint must be attached to ActionSet to have an id")
        return self.parent.constraints.find(self)

    @property
    def indices(self):
        """list of feature indices with respect to the parent; used to determine partitions"""
        if self.parent is None:
            raise ValueError("constraint must be attached to ActionSet to have indices")
        return self._parent.get_feature_indices(self._names)

    # @abstractmethod
    # def is_encoding_constraint(self):
    #     """returns True if constraint specifies an encoding over categorical or ordinal features"""
    #     raise NotImplementedError()
    #
    # @abstractmethod
    # def is_causal_constraint(self):
    #     """returns True if constraint specifies downstream effects"""
    #     raise NotImplementedError()

    # built-ins
    def __eq__(self, other):
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
        name = self.__class__.__name__
        fields = ",".join([f"{p}={self.__getattribute__(p)}" for p in self._parameters])
        out = f"<{name}(names={self._names},{fields})>"
        return out

    def __str__(self):
        return self.__repr__()

    # basic checks on features
    def check_feature_vector(self, x):
        """
        returns true if x is a finite feature vector with d elements
        :param x: list or nd.array
        """
        out = len(x) == len(self.names) and np.isfinite(x).all()
        return out

    # methods that each subclass needs to implement
    @abstractmethod
    def check_compatibility(self, action_set):
        """
        returns True if constraint is compatible with the action set
        """
        raise NotImplementedError()

    @abstractmethod
    def check_feasibility(self, x):
        """
        returns True if current point is feasible with the constraint
        """
        raise NotImplementedError()

    @abstractmethod
    def adapt(self, x):
        """
        adapts constraint parameters for point x
        """
        raise NotImplementedError()

    @abstractmethod
    def add_to_cpx(self, cpx, indices, x):
        """
        adds constraint to ReachableSetEnumeratorMIP
        :param cpx: Cplex object
        :return: nothing
        """
        raise NotImplementedError()
