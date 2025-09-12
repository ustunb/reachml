"""Sampling utilities for reachable sets and partitions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import reduce
from typing import List

import numpy as np

from .action_set import ActionSet
from .constraints import ThermometerEncoding
from .enumeration import ReachableSetEnumerationMIP
from .mip import BaseMIP
from .utils import DEFAULT_SOLVER


class ReachableSetSampler:
    """Object to sample from the reachable set for a point x."""

    _RND_PRECISION = 4

    def __init__(self, action_set, x, solver=DEFAULT_SOLVER, **kwargs):
        """Initialize the sampler with an action set and point `x`."""
        self._action_set = action_set
        self._x = x
        self._solver = solver

        # random seed
        default_seed = abs(
            hash(x.astype(np.float32).round(ReachableSetSampler._RND_PRECISION).tobytes())
        )
        seed = kwargs.get("seed", default_seed)
        rng = np.random.default_rng(seed)
        self._master_rng = rng
        self._seed = seed

        # samplers
        samplers = {
            i: PartitionSampler.from_values(
                x[part], action_set[part], rng.spawn(1)[0], self._solver
            )
            for i, part in enumerate(self.partition)
        }
        self._partition_samplers = samplers

    @property
    def action_set(self):
        """Action set whose partitions are sampled."""
        return self._action_set

    @property
    def partition(self):
        """Actionable partitions of the action set."""
        return self._action_set.actionable_partition

    @property
    def x(self):
        """Current point being sampled around."""
        return self._x

    @property
    def seed(self):
        """Seed used for the random number generator."""
        return self._seed

    @x.setter
    def x(self, value):
        self._x = value

    @seed.setter
    def seed(self, value):
        self._seed = value
        self._master_rng = np.random.default_rng(value)

        for v in self._partition_samplers.values():
            v.rng = self._master_rng.spawn(1)[0]

    def sample(self, n):
        """Sample `n` full points by stitching per-partition samples."""
        part_samples = [s.sample(n) for s in self._partition_samplers.values()]
        part_idx = reduce(lambda x, y: x + y, self.partition)
        Xs = np.repeat(self.x.reshape(1, -1), n, axis=0)
        Xs[:, part_idx] = np.hstack(part_samples)  # update the sampled part

        # do we want to return or store the samples in the object?
        return Xs

    def reset_rng(self):
        """Reset the random number generator."""
        ReachableSetSampler.generate_seed(self.x)

    @staticmethod
    def generate_seed(x):
        """Generate a deterministic seed from a numeric vector `x`."""
        return abs(hash(x.astype(np.float32).round(ReachableSetSampler._RND_PRECISION).tobytes()))


@dataclass
class PartitionSampler(ABC):
    """Object to sample points over a subset of features."""

    x: np.ndarray = field(init=True)
    action_set: ActionSet = field(init=True, repr=False)
    # part: list[int] = field(init=True)
    rng: np.random.Generator = field(init=True, repr=False)
    solver: str = field(init=True, default=DEFAULT_SOLVER, repr=True)

    def __post_init__(self):
        """Initialize partition samplers for each actionable partition.

        If we init single continuous samplers and ignore constraints, we may
        need a way to avoid this for some cases. TODO: refine wiring here.
        """
        # cons = self.action_set.constraints
        # _, const_ids = cons.get_associated_features(self.part[0], return_constraint_ids=True)
        # self.constraints = [cons._map[id] for id in const_ids]
        self.constraints = self.action_set.constraints
        self.part_size = len(self.action_set)

    @staticmethod
    def from_values(x, action_set, rng, solver):
        """Create a PartitionSampler object for a partition of features."""
        assert len(x) > 0, "x should be non empty"
        assert len(x) == len(action_set), "x and action_set should have the same length"
        assert any(action_set.actionable), "at least one feature should be actionable"

        dim = "separable" if len(action_set) == 1 else "joint"
        discrete_ids = action_set.discrete
        part_type_ids = any(discrete_ids) + all(
            discrete_ids
        )  # 0: continuous, 1: mixed, 2: discrete
        part_type = ("continuous", "mixed", "discrete")[part_type_ids]
        match (dim, part_type):
            case ("separable", "discrete"):
                out = PartitionSampler1DDiscrete(x, action_set, rng, solver)
            case ("separable", "continuous"):
                out = PartitionSampler1DContinuous(x, action_set, rng, solver)
            case ("joint", "discrete"):
                out = PartitionSamplerNDDiscrete(x, action_set, rng, solver)
            case ("joint", "continuous"):
                out = PartitionSamplerNDContinuous(x, action_set, rng, solver)
            case ("joint", "mixed"):
                out = PartitionSamplerNDMixed(x, action_set, rng, solver)
        return out

    @abstractmethod
    def sample(self, n, **kwargs):
        """Sample `n` points/actions for this partition."""
        
    def check_feasibility(self, samples, solver=DEFAULT_SOLVER):
        """Check if the samples are feasible under constraints using MIP.

        :param samples: n x d array of samples (not actions)
        :return: True if valid sample, False otherwise
        """
        mip_obj = BaseMIP(self.action_set, self.x)

        actions = samples - self.x

        out: List[bool] = []
        for a in actions:
            # add extra constraints to match action on group 'c'
            for i, a_i in enumerate(a):
                mip_obj.add_linear_constraint(
                    name=f"match_c[{i}]",
                    terms=[("c", i, 1.0)],
                    sense="E",
                    rhs=float(a_i),
                )

            mip_obj.solve_model()
            out.append(bool(mip_obj.solution_exists))

            # remove extra constraints
            for i in range(len(a)):
                mip_obj.delete_constraint(f"match_c[{i}]")

        return out

    def validate(self, xp):
        """Validate the input point based on actionability constraints."""
        raise NotImplementedError()

    def __check_rep__(self):
        """Check internal representation invariants for the sampler."""
        raise NotImplementedError()
        return True

    def __repr__(self):
        """Debug string with feature names for this sampler."""
        return f"<Sampler for features {self.action_set.names}>"


@dataclass
class PartitionSampler1DContinuous(PartitionSampler):
    """Sampler for a singleton continuous partition.

    Note: `self.action_set` is an `ActionElement` collection of size 1.
    """

    def __post_init__(self):
        """Initialize bounds for continuous singleton partition."""
        super().__post_init__()
        self.lb, self.ub = self.action_set[0].feasible_bound(self.x, return_actions=False)

    def sample(self, n, **kwargs):
        """Draw `n` samples uniformly between scalar bounds."""
        out = self.rng.uniform(self.lb, self.ub, n).reshape(-1, 1)
        return out


@dataclass
class PartitionSampler1DDiscrete(PartitionSampler):
    """Sampler for a singleton discrete partition."""

    def __post_init__(self):
        """Initialize reachable grid for discrete singleton partition."""
        super().__post_init__()
        self.values = self.action_set[0].reachable_grid(self.x, return_actions=False)

    def sample(self, n, **kwargs):
        """Sample `n` values from the discrete grid with replacement."""
        # right now returning with replacement
        # return acts if len(acts) <= n else acts[self.rng.choice(len(acts), n, replace=False)]
        out = self.rng.choice(self.values, n, replace=True).reshape(-1, 1)
        return out


@dataclass
class PartitionSamplerNDDiscrete(PartitionSampler):
    """Sampler for a discrete partition with more than one element."""

    # todo: build set of actions outside of loop
    def __post_init__(self):
        """Prepare state for multi-feature discrete partition sampling."""
        super().__post_init__()
        self.actions = None
        self.values = None

        if len(self.constraints._map) == 1 and isinstance(
            self.constraints._map[0], ThermometerEncoding
        ):
            self.sampler = self._sample_enumerate
        else:
            # for const_key in self.constraints._map:
            #     all_c_df = self.constraints.df
            #     const_df = all_c_df.loc[all_c_df['const_id'] == const_key, ['feature_idx']]

            self.sampler = self._sample_reject

    def sample(self, n, check=True, **kwargs):
        """Sample `n` actions and return stacked samples after feasibility check."""
        out = np.vstack(self.sampler(n))

        return out

    def _sample_reject(self, n):
        n_remaining = n
        out = []

        while n_remaining > 0:
            # TODO: check if append is working correctly with vstack in sample
            samples = np.hstack([self._sample_1d(j, n_remaining) for j in range(self.part_size)])
            keep = self.check_feasibility(samples)  # if check else np.ones(n_remaining, dtype=bool)
            out.append(samples[keep])
            n_remaining -= np.sum(keep)

        return out

    def _sample_enumerate(self, n):
        enum = ReachableSetEnumerationMIP(action_set=self.action_set, x=self.x, solver=self.solver)
        enum.enumerate()
        self.actions = np.array(enum.feasible_actions)
        self.values = self.x + self.actions
        out = self.rng.choice(self.values, n, replace=True)

        return out

    def _sample_1d(self, j, n):
        """Sample from the j-th action set."""
        acts = self.action_set[j].reachable_grid(self.x[j], relax=True, return_actions=False)
        return self.rng.choice(acts, n, replace=True).reshape(-1, 1).astype(np.float32)


@dataclass
class PartitionSamplerNDContinuous(PartitionSampler):
    """Sampler for a continuous partition with more than one element."""

    def __post_init__(self):
        """Compute per-feature bounds for continuous partitions."""
        super().__post_init__()
        bounds = np.array(
            [self.action_set[j].feasible_bound(self.x[j]) for j in range(self.part_size)]
        )
        self.lbs = bounds[:, 0]
        self.ubs = bounds[:, 1]

    def sample(self, n, check=True, **kwargs):
        """Sample `n` points uniformly within bounds; optionally check feasibility."""
        n_remaining = n
        out = []
        while n_remaining > 0:
            samples = self.rng.uniform(self.lbs, self.ubs, (n_remaining, self.part_size))
            keep = self.check_feasibility(samples) if check else np.ones(n_remaining, dtype=bool)
            out.append(samples[keep])
            n_remaining -= np.sum(keep)

        out = np.vstack(out)

        return out


@dataclass
class PartitionSamplerNDMixed(PartitionSampler):
    """Sampler for a partition with at least one discrete and one continuous element."""

    def __post_init__(self):
        """Initialize per-partition samplers for discrete and continuous parts."""
        super().__post_init__()
        self.disc_part = [i for i, a in enumerate(self.action_set) if a.discrete]
        self.cts_part = list(set(range(len(self.action_set))) - set(self.disc_part))
        disc_rng, cts_rng = self.rng.spawn(2)
        self.samplers = {
            "disc": PartitionSamplerNDDiscrete(
                self.x[self.disc_part], self.action_set[self.disc_part], disc_rng, self.solver
            ),
            "cts": PartitionSamplerNDContinuous(
                self.x[self.cts_part], self.action_set[self.cts_part], cts_rng, self.solver
            ),
        }

    def sample(self, n, **kwargs):
        """Sample `n` actions from discrete and continuous partitions and combine."""
        n_remaining = n
        out = []
        while n_remaining > 0:
            disc_samples = self.samplers["disc"].sample(n_remaining, check=False)
            cts_samples = self.samplers["cts"].sample(n_remaining, check=False)
            samples = np.hstack([disc_samples, cts_samples])
            keep = self.check_feasibility(samples)
            out.append(samples[keep])
            n_remaining -= np.sum(keep)
        return out
