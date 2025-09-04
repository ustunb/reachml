"""Content-addressable reachable-set database for feature vectors.

This module provides `ReachableSetDatabase`, which generates, stores, and
retrieves reachable sets keyed by a stable hash of the rounded feature vector.
"""

import hashlib
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .action_set import ActionSet
from .reachable_set import EnumeratedReachableSet, ReachableSet, SampledReachableSet


class ReachableSetDatabase:
    """Generate, store, and retrieve reachable sets over a dataset.

    This database is content‑addressable: each reachable set is stored under a
    key derived from its rounded feature vector.

    Attributes:
        action_set: The `ActionSet` used to generate reachable sets.
        path: The HDF5 file backing this database.
        precision: Digits used to round vectors before hashing.
    """

    _PRECISION = 8
    _METADATA_ATTR_NAME = "metadata"
    _X_ATTR_NAME = "x"
    _STATS_ATTR_NAME = "stats"
    _STATS_KEYS = ["time", "n_points", "complete"]

    def __init__(self, action_set: ActionSet, path: str = None, **kwargs):
        """Initialize a reachable-set database.

        Args:
            action_set: Action set for generating reachable sets.
            path: Optional path to an HDF5 file; creates a temp file if None.
            **kwargs: Optional `precision` (int) and generation `method`.
        """
        assert isinstance(action_set, ActionSet)
        self._action_set = action_set

        # attach path
        f = Path(tempfile.mkstemp(suffix=".h5")[1]) if path is None else Path(path)
        f.parents[0].mkdir(parents=True, exist_ok=True)  # create directory
        try:
            with h5py.File(f, "a") as _:
                pass
        except FileNotFoundError as err:
            raise ValueError(f"Cannot write to database file: {f}") from err
        self._path = f

        # attach precision
        # TODO: When reading a db, check if the precision matches.
        precision = kwargs.get("precision", ReachableSetDatabase._PRECISION)
        self._precision = int(precision)

        # determine generation method
        default = "enumerate" if action_set.can_enumerate else "sample"  # default from action_set
        self._method = kwargs.get("method", default)

        self.RS = EnumeratedReachableSet if self._method == "enumerate" else SampledReachableSet

        return

    @property
    def action_set(self) -> ActionSet:
        """Action set used to generate reachable sets."""
        return self._action_set

    @property
    def path(self) -> Path:
        """Path to the underlying HDF5 database file."""
        return self._path

    @property
    def precision(self) -> int:
        """Number of rounding digits applied before hashing keys."""
        return self._precision

    @property
    def method(self) -> str:
        """Generation method: "enumerate" or "sample"."""
        return self._method

    def array_to_key(self, x: np.ndarray) -> str:
        """Compute a stable content hash key for a feature vector.

        Rounds `x` to `precision` digits using a float16/32 container before
        hashing to reduce sensitivity to tiny numeric differences.
        """
        float_dtype = np.float16 if self._precision <= 4 else np.float32
        b = np.array(x, dtype=float_dtype).round(self._precision).tobytes()
        return hashlib.sha256(b).hexdigest()

    def __len__(self) -> int:
        """Number of distinct points for which we have a reachable set."""
        out = 0
        with h5py.File(self.path, "r") as db:
            out = len(db)
        return out

    def keys(self) -> np.ndarray:
        """Return the list of feature vectors stored in the database."""
        out = []
        with h5py.File(self.path, "r") as backend:
            out = [backend[k].attrs[self._X_ATTR_NAME] for k in backend.keys()]
        out = np.array(out).reshape(1, -1) if len(out) == 1 else np.array(out)
        return out

    def __getitem__(self, x: Union[np.ndarray, pd.Series]) -> ReachableSet:
        """Fetch the reachable set for feature vector `x`.

        Args:
            x: Feature vector as `np.ndarray`, `pd.Series`, or list.

        Returns:
            A `ReachableSet` instance reconstructed from the database entry.
        """
        if isinstance(x, list):
            x = np.array(x)
        elif isinstance(x, pd.Series):
            x = x.values
        key = self.array_to_key(x)
        try:
            with h5py.File(self.path, "r") as db:
                args = dict(
                    zip(
                        self.RS._METADATA_KEYS,
                        db[key].attrs[self._METADATA_ATTR_NAME],
                        strict=False,
                    )
                )
                args.update({"time": db[key].attrs[self._STATS_ATTR_NAME][-1]})
                out = self.RS(self._action_set, x=x, values=db[key], **args)
        except KeyError as err:
            raise KeyError(
                f"point x={str(x)} with key={key} not found in database at {self.path}"
            ) from err
        return out

    def _store_reachable_set(self, db, key, x, reachable_set, final_time):
        """Store a reachable set and return summary statistics dict."""
        stats = {
            "n_points": len(reachable_set),
            "complete": reachable_set.complete,
            "time": final_time,
        }
        if key in db:  # delete existing entry (avoid error)
            del db[key]
        db.create_dataset(key, data=reachable_set.X)
        db[key].attrs[ReachableSetDatabase._X_ATTR_NAME] = x
        db[key].attrs[ReachableSetDatabase._METADATA_ATTR_NAME] = (
            reachable_set._get_metadata().astype(np.float32).values
        )
        db[key].attrs[ReachableSetDatabase._STATS_ATTR_NAME] = np.array(list(stats.values()))
        return stats

    def generate(
        self, X: Union[np.ndarray, pd.DataFrame], overwrite: bool = False, **kwargs
    ):
        """Generate reachable sets for each row in `X` and persist them.

        Args:
            X: Feature matrix (`np.ndarray` or `pd.DataFrame`).
            overwrite: If True, overwrite any existing entries for keys in `X`.
            **kwargs: Passed to `ReachableSet` constructors and `.generate()`; for
                sampling, includes `resp_thresh`, `n`, `seed`, and `solver`.

        Returns:
            `pd.DataFrame` with summary statistics (time, n_points, complete).
        """
        # Note: duplicates per unique mutable pattern are handled efficiently.
        if isinstance(X, pd.DataFrame):
            X = X.values
        assert X.ndim == 2 and X.shape[0] > 0 and X.shape[1] == len(self.action_set), (
            f"X should be 2D with {len(self.action_set)} columns"
        )
        assert np.isfinite(X).all()

        def flatten(xss):
            return [x for xs in xss for x in xs]

        mutable = sorted(flatten(self.action_set.actionable_partition))  # actionable or targeted
        immutable = list(set(range(len(self.action_set))) - set(mutable))
        U = np.unique(X, axis=0)
        _, types, types_to_x = np.unique(
            U[:, mutable], axis=0, return_index=True, return_inverse=True
        )
        siblings = {i: np.flatnonzero(i == types_to_x) for i in range(len(types))}

        if self._method == "sample":
            init_seed_seq = np.random.SeedSequence(kwargs.get("seed", None))
            seed_seqs = init_seed_seq.spawn(U.shape[0])
        else:
            seed_seqs = [None] * U.shape[0]

        out = []
        with h5py.File(self.path, "a") as db:
            for _unique_mutable_idx, sib_idxs in tqdm(siblings.items()):
                x = U[sib_idxs[0]]
                key = self.array_to_key(x)
                seed_seq = seed_seqs[sib_idxs[0]]
                kwargs["seed"] = seed_seq

                new_entries = []
                if overwrite or key not in db:
                    start_time = time.time()
                    reachable_set = self.RS(self.action_set, x, **kwargs)
                    reachable_set.generate(**kwargs)
                    final_time = time.time() - start_time
                    new_entries.append((key, x, reachable_set, final_time))
                else:
                    reachable_set = self[x]

                for s in sib_idxs[1:]:
                    key = self.array_to_key(U[s])
                    seed_seq = seed_seqs[s]
                    kwargs["seed"] = seed_seq

                    if overwrite or key not in db:
                        start_time = time.time()
                        reachable_set = self._gen_sibling_reachable_set(
                            U[s], reachable_set, immutable, **kwargs
                        )
                        final_time = time.time() - start_time
                        new_entries.append((key, U[s], reachable_set, final_time))

                out += [self._store_reachable_set(db, *entry) for entry in new_entries]

        # update summary statistics
        out = pd.DataFrame(out) if out else pd.DataFrame(columns=self._STATS_KEYS)
        return out

    def _gen_sibling_reachable_set(self, x, sib_rs, immutable, **kwargs):
        """Construct a sibling reachable set for `x` from an existing one.

        For "enumerate", copy and align immutable feature values in-place.
        For "sample", reseed, reset, and resample.

        Args:
            x: Feature vector for the sibling point.
            sib_rs: ReachableSet of a sibling point.
            immutable: List of immutable feature indices.
            **kwargs: Passed through to reachable-set constructors/generation.
        """
        # Perhaps move values = to enumerate
        R = self.RS(self.action_set, x, values=sib_rs.X, **kwargs)
        # R.add(sib_rs.X, actions=False) # copy the reachable set

        if self.method == "enumerate":
            R.X[:, immutable] = x[immutable]
            R._complete = True
        else:
            R = deepcopy(sib_rs)
            R.seed = kwargs.get("seed", R.seed)
            R.x = x
            R.reset()
            R.generate(**kwargs)

        return R
