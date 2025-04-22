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
    """
    Container class to generate, store, and retrieve a collection of reachable sets over a dataset.

    The database is content-addressable so the feature vectors are keys themselves.

    Attrs:
        action_set ActionSet: Action set.
        path str: Path to the database.
        precision int: Digits of precision.
    """

    _PRECISION = 4
    _METADATA_ATTR_NAME = "metadata"
    _X_ATTR_NAME = "x"
    _STATS_ATTR_NAME = "stats"
    _STATS_KEYS = ["time", "n_points", "complete"]

    def __init__(self, action_set: ActionSet, path: str = None, **kwargs):
        """
        :param action_set:
        :param path:
        """
        assert isinstance(action_set, ActionSet)
        self._action_set = action_set

        # attach path
        f = Path(tempfile.mkstemp(suffix=".h5")[1]) if path is None else Path(path)
        f.parents[0].mkdir(parents=True, exist_ok=True)  # create directory
        try:
            with h5py.File(f, "a") as _:
                pass
        except FileNotFoundError:
            raise ValueError(f"Cannot write to database file: {f}")
        self._path = f

        # attach precision
        # TODO: When reading a db, check if the precision matches.
        precision = kwargs.get("precision", ReachableSetDatabase._PRECISION)
        self._precision = int(precision)

        # determine generation method
        default = (
            "enumerate" if action_set.can_enumerate else "sample"
        )  # default from action_set
        self._method = kwargs.get("method", default)

        self.RS = (
            EnumeratedReachableSet
            if self._method == "enumerate"
            else SampledReachableSet
        )

        return

    @property
    def action_set(self) -> ActionSet:
        return self._action_set

    @property
    def path(self) -> Path:
        return self._path

    @property
    def precision(self) -> int:
        return self._precision

    @property
    def method(self) -> str:
        return self._method

    def array_to_key(self, x: np.ndarray) -> str:
        float_dtype = np.float16 if self._precision <= 4 else np.float32
        b = np.array(x, dtype=float_dtype).round(self._precision).tobytes()
        return hashlib.sha256(b).hexdigest()

    def __len__(self) -> int:
        """number of distinct points for which we have a reachable set"""
        out = 0
        with h5py.File(self.path, "r") as db:
            out = len(db)
        return out

    def keys(self) -> np.ndarray:
        out = []
        with h5py.File(self.path, "r") as backend:
            out = [backend[k].attrs[self._X_ATTR_NAME] for k in backend.keys()]
        out = np.array(out).reshape(1, -1) if len(out) == 1 else np.array(out)
        return out

    def __getitem__(self, x: Union[np.ndarray, pd.Series]) -> ReachableSet:
        """
        Fetches the reachable set for feature vector x
        :param x numpy.ndarray: Feature vector
        :return:
        """
        if isinstance(x, list):
            x = np.array(x)
        elif isinstance(x, pd.Series):
            x = x.values
        key = self.array_to_key(x)
        try:
            with h5py.File(self.path, "r") as db:
                args = dict(
                    zip(self.RS._METADATA_KEYS, db[key].attrs[self._METADATA_ATTR_NAME])
                )
                args.update({"time": db[key].attrs[self._STATS_ATTR_NAME][-1]})
                out = self.RS(self._action_set, x=x, values=db[key], **args)
        except KeyError:
            raise KeyError(
                f"point `x={str(x)}` with `key = {key}` not found in database at `{self.path}`."
            )
        return out

    def _store_reachable_set(self, db, key, x, reachable_set, final_time):
        """stores reachable set in database and returns summary statistics"""
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
        db[key].attrs[ReachableSetDatabase._STATS_ATTR_NAME] = np.array(
            list(stats.values())
        )
        return stats

    def generate(
        self, X: Union[np.ndarray, pd.DataFrame], overwrite: bool = False, **kwargs
    ):
        """
        Generate reachable sets for each feature vector in X
        :param X: feature matrix (np.array or pd.DataFrame)
        :param overwrite: whether to overwrite existing entries
        :param kwargs: additional arguments to pass to ReachableSet.generate. For sampling:
            - resp_thresh: responsiveness lower bound (epsilon in the paper)
            - n: number of samples (overrides thresholds)
        :return: pd.DataFrame of summary statistics about the reachable sets
        """
        # todo: replace with reachable_set = ReachableSet.generate()
        # todo: make sure we have iid samples for each point / only do the duplicate trick for continuous cases
        if isinstance(X, pd.DataFrame):
            X = X.values
        assert X.ndim == 2 and X.shape[0] > 0 and X.shape[1] == len(self.action_set), (
            f"X should be 2D with {len(self.action_set)} columns"
        )
        assert np.isfinite(X).all()

        flatten = lambda xss: [x for xs in xss for x in xs]
        mutable = sorted(
            flatten(self.action_set.actionable_partition)
        )  # actionable or targeted
        immutable = list(set(range(len(self.action_set))) - set(mutable))
        U = np.unique(X, axis=0)
        _, types, types_to_x = np.unique(
            U[:, mutable], axis=0, return_index=True, return_inverse=True
        )
        siblings = {i: np.flatnonzero(i == types_to_x) for i in range(len(types))}

        out = []
        with h5py.File(self.path, "a") as db:
            for unique_mutable_idx, sib_idxs in tqdm(siblings.items()):
                x = U[sib_idxs[0]]
                key = self.array_to_key(x)

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
        """returns a sibling ReachableSet by making a deepcopy and:
        if self._method = "enumerate", we change immutable feature values of sib_rs.X to match x (no extra computation)
        if self._method = "sample", we sample again

        Args:
            x (np.ndarray): Feature vector
            sib_rs (ReachabeSet): ReachableSet of a sibling point (of x)
            immutable (list): List of immutable feature indices
        """
        # Perhaps move values = to enumerate
        R = self.RS(self.action_set, x, values=sib_rs.X, **kwargs)
        # R.add(sib_rs.X, actions=False) # copy the reachable set

        if self.method == "enumerate":
            R.X[:, immutable] = x[immutable]
            R._complete = True
        else:
            R = deepcopy(sib_rs)
            R.x = x
            R.reset()
            R.generate(**kwargs)

        return R
