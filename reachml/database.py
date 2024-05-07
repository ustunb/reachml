import numpy as np
import pandas as pd
import time
import hashlib
from typing import Union, Generic
from pathlib import Path
import tempfile
import h5py
from tqdm.auto import tqdm
from . import ActionSet, ReachableSetEnumerator, ReachableSet

# fmt:off
def _array_to_key(array: np.ndarray, precision: int) -> str:
    array_as_bytes = array.astype(np.float32).round(precision).tobytes()
    return hashlib.sha256(array_as_bytes).hexdigest()

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
        f.parents[0].mkdir(parents=True, exist_ok=True) # create directory
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

    def array_to_key(self, x: np.ndarray) -> str:
        b = np.array(x, dtype = np.float32).round(self._precision).tobytes()
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
        out = np.array(out).reshape(1, -1) if len(out) <= 1 else np.array(out)
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
                args = dict(zip(ReachableSet._METADATA_KEYS, db[key].attrs[self._METADATA_ATTR_NAME]))
                out = ReachableSet(self._action_set, values=db[key], **args)
        except KeyError:
            raise KeyError(f"point `x={str(x)}` with `key = {key}` not found in database at `{self.path}`.")
        return out

    def generate(self, X: Union[np.ndarray, pd.DataFrame], overwrite: bool=False):
        """
        Generate reachable sets for each feature vector in X
        :param X: feature matrix (np.array or pd.DataFrame)
        :param keys: unique keys for the feature vectors.
        :return: pd.DataFrame of summary statistics about the reachable sets
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        assert X.ndim == 2 and X.shape[0] > 0 and X.shape[1] == len(self.action_set), f"X should be 2D with {len(self.action_set)} columns"
        assert np.isfinite(X).all()

        out = []
        U = np.unique(X, axis=0)
        keys = [self.array_to_key(u) for u in U]
        with h5py.File(self.path, "a") as db:
            for x, key in tqdm(list(zip(U, keys))):
                # todo: if key in db then check if complete, if not enumerate remaining items
                if overwrite or key not in db:
                    start_time = time.time()
                    enumerator = ReachableSetEnumerator(action_set=self.action_set, x=x)
                    reachable_set = enumerator.enumerate()
                    stats = {
                        "n_points": len(reachable_set),
                        "complete": reachable_set.complete,
                        "time": time.time() - start_time,
                        }
                    out.append(stats)
                    assert set(stats.keys()) == set(self._STATS_KEYS)
                    if key in db:
                        del db[key]
                    db.create_dataset(key, data=reachable_set.X)
                    db[key].attrs[self._X_ATTR_NAME] = x
                    db[key].attrs[self._METADATA_ATTR_NAME] = reachable_set._get_metadata().values
                    db[key].attrs[self._STATS_ATTR_NAME] = np.array(list(stats.values()))

       # todo: consolidate over immutable
        # flatten = lambda xss: [x for xs in xss for x in xs]
        # mutable = sorted(flatten(self.action_set.actionable_partition)) #actionable or targetted
        # immutable = list(set(range(len(self.action_set))) - set(mutable))
        # U, types, types_to_x = np.unique(X[:, mutable], axis =0, return_index = True, return_inverse = True)
        # siblings = {t: np.flatnonzero(t == types_to_x) for t in types}
        # out = []
        # with h5py.File(self.path, "a") as db:
        #     # enumerate for first case
        #     for t, u in tqdm(list(zip(types, U))):
        #         x = X[t]
        #         key = self.array_to_key(x)
        #         sibling_indices = siblings.get(t)
        #         if overwrite or key not in db:
        #             start_time = time.time()
        #             enumerator = ReachableSetEnumerator(action_set=self.action_set, x=x)
        #             reachable_set = enumerator.enumerate()
        #             stats = {
        #                 "n_points": len(reachable_set),
        #                 "complete": reachable_set.complete,
        #                 "time": time.time() - start_time,
        #                 }
        #             out.append(stats)
        #             assert set(stats.keys()) == set(self._STATS_KEYS)
        #             if key in db:
        #                 del db[key]
        #             db.create_dataset(key, data=reachable_set.X)
        #             db[key].attrs[self._X_ATTR_NAME] = x
        #             db[key].attrs[self._METADATA_ATTR_NAME] = reachable_set._get_metadata().values
        #             db[key].attrs[self._STATS_ATTR_NAME] = np.array(list(stats.values()))
        #         elif len(sibling_indices) >= 2:
        #             reachable_set = self[x]
        #
        #         for s in sibling_indices[1:]:
        #             key = self.array_to_key(x)
        #             if overwrite or key not in db:
        #                 start_time = time.time()
        #                 reachable_set.X[:, immutable] = X[s, immutable]
        #                 stats = {
        #                     "n_points": len(reachable_set),
        #                     "complete": reachable_set.complete,
        #                     "time": time.time() - start_time,
        #                     }
        #                 out.append(stats)
        #                 assert set(stats.keys()) == set(self._STATS_KEYS)
        #                 if key in db:
        #                     del db[key]
        #                 db.create_dataset(key, data=reachable_set.X)
        #                 db[key].attrs[self._X_ATTR_NAME] = x
        #                 db[key].attrs[self._METADATA_ATTR_NAME] = reachable_set._get_metadata().values
        #                 db[key].attrs[self._STATS_ATTR_NAME] = np.array(list(stats.values()))

        # update summary statistics
        out = pd.DataFrame(out) if out else pd.DataFrame(columns = self._STATS_KEYS)
        return out

    def audit(self, X = Union[np.ndarray], clf = Generic, target = 1, scaler = None, include_target = True) -> pd.DataFrame:

        if isinstance(X, pd.DataFrame):
            raw_index = X.index.tolist()
            X = X.values
        else:
            raw_index = list(range(X.shape[0]))

        assert X.ndim == 2 and X.shape[0] > 0 and X.shape[1] == len(self.action_set), f"X should be 2D with {len(self.action_set)} columns"
        assert np.isfinite(X).all()

        if scaler is None:
            rescale = lambda x: x
        else:
            reformat = lambda x: x.reshape(1, -1) if x.ndim == 1 else x
            rescale = lambda x: scaler.transform(reformat(x))

        U, distinct_idx = np.unique(X, axis = 0, return_inverse = True)
        H = clf.predict(rescale(U)).flatten()
        all_idx = np.flatnonzero(np.arange(U.shape[0]))
        target_idx = np.flatnonzero(np.equal(H, target))
        audit_idx = all_idx if include_target else np.setdiff1d(np.arange(U.shape[0]), target_idx)

        # solve recourse problem
        n_iterations = len(H) if include_target else len(audit_idx)
        output = []
        pbar = tqdm(total=n_iterations) ## stop tqdm from playing badly in ipython notebook.
        for idx in audit_idx:
            x = U[idx, :]
            fx = H[idx]
            fxp = fx
            R = self[x]
            S = clf.predict(rescale(R.X))
            feasible_idx = np.equal(S, target)
            n_feasible = np.sum(feasible_idx)
            feasible = n_feasible > 0
            output.append({
                "idx": idx,
                "yhat": fx > 0,
                "yhat_post": fxp > 0,  # = f(x + a)
                "recourse": feasible,
                "n_reachable": len(R),
                "n_feasible": n_feasible,
                "complete": R.complete,
                "abstain": (R.complete == False) and not feasible,
                "recourse_scores": R.scores(point_mask = feasible_idx, max_score = len(R)),
                "reachable_scores": R.scores(max_score = len(R)),
                "immutability_scores": R.scores(max_score = len(R), invert = True),
                })
            pbar.update(1)
        pbar.close()

        # add in points that were not denied recourse
        df = pd.DataFrame(output)
        df = df.set_index('idx')

        # include unique points that attain desired label already
        df = df.reindex(range(U.shape[0]))

        # include duplicates of original points
        df = df.iloc[distinct_idx]
        df = df.reset_index(drop = True)
        df.index = raw_index
        return df
