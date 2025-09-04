"""Auditing responsiveness over reachable sets.

Provides `ResponsivenessAuditor` to evaluate whether points have feasible
recourse under a model by querying reachable sets from a database.
"""

import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta, binom
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

from .database import EnumeratedReachableSet, ReachableSetDatabase


class ResponsivenessAuditor:
    """Auditor object for responsiveness verification."""

    def __init__(self, database: ReachableSetDatabase, clf: Any, **kwargs):
        """Initialize the auditor with a database and classifier.

        Args:
            database: Reachable set database.
            clf: Classifier or predictor implementing `predict`.
            **kwargs: Reserved for future options.
        """
        self._clf = clf
        self._db = database
        self._audit = None

    @property
    def db(self):
        """Reachable set database backing this auditor."""
        return self._db

    @property
    def clf(self):
        """Classifier or predictor used for evaluation."""
        return self._clf

    @property
    def audit(self):
        """Most recent audit results DataFrame and metadata."""
        if self._audit is None:
            raise ValueError("No audit has been conducted yet")

        return self._audit

    @property
    def complete(self):
        """Whether an audit has been run and saved."""
        return False if self._audit is None else True

    @audit.setter
    def audit(self, value):
        self._audit = value

    def __repr__(self):
        """Debug summary string for the auditor."""
        return f"<ResponsivenessAuditor<database={self.db}, complete={self.complete}>"

    def __call__(self, X, y, target=1, population="all", save=True, overwrite=True, **kwargs):
        """Run an audit of responsiveness over the dataset `X`.

        Args:
            X: Array or DataFrame of features.
            y: True labels aligned with `X`.
            target: Target label considered as achieving recourse.
            population: Subset to audit ("all", "target", or others).
            save: If True, store results on this instance.
            overwrite: If True, overwrite previous results.
            **kwargs: Optional parameters (e.g., feature subset).

        Returns:
            DataFrame with audit outcomes per input row.
        """
        if isinstance(X, pd.DataFrame):
            raw_index = X.index.tolist()
            X = X.values
        else:
            raw_index = list(range(X.shape[0]))

        assert X.ndim == 2 and X.shape[0] > 0 and X.shape[1] == len(self.db.action_set), (
            f"X should be 2D with {len(self.db.action_set)} columns"
        )
        assert np.isfinite(X).all()

        include_target = population in ("all", "target")

        if kwargs.get("features") is not None:
            feats = kwargs["features"]
        else:
            feats = np.arange(X.shape[1])

        U, distinct_idx = np.unique(X, axis=0, return_inverse=True)
        H = self.clf.predict(U[:, feats]).flatten()
        all_idx = np.arange(U.shape[0])
        target_idx = np.flatnonzero(np.equal(H, target))
        audit_idx = all_idx if include_target else np.setdiff1d(np.arange(U.shape[0]), target_idx)

        # solve recourse problem
        n_iterations = len(H) if include_target else len(audit_idx)
        output = []
        pbar = tqdm(total=n_iterations)  ## stop tqdm from playing badly in ipython notebook.

        for idx in audit_idx:
            start_time = time.time()
            x = U[idx, :]
            fx = H[idx]
            R = self.db[x]
            S = self.clf.predict(R.X[:, feats])
            feasible_idx = np.equal(S, target)
            n_feasible = np.sum(feasible_idx)
            feasible = n_feasible > 0
            out = {
                "idx": idx,
                "clf_pred": fx > 0,
                "recourse": feasible,
                "rset_size": len(R),
                "n_feasible": n_feasible,
                "resp_prop": n_feasible / len(R),
                "complete": bool(R.complete),
                # "abstain": (R.complete == False) and not feasible,  # not used for now
                "rset_method": "enumerated" if isinstance(R, EnumeratedReachableSet) else "sampled",
                "rset_time": R.time,
            }

            if out["rset_method"] == "sampled":
                out.update(self._sampling_audit_stats(R, n_feasible, **kwargs))

            final_time = time.time() - start_time
            out["audit_time"] = final_time

            output.append(out)
            pbar.update(1)
        pbar.close()

        # add in points that were not denied recourse
        df = pd.DataFrame(output)
        df = df.set_index("idx")

        # include unique points that attain desired label already
        df = df.reindex(range(U.shape[0]))

        # include duplicates of original points
        df = df.iloc[distinct_idx]
        df = df.reset_index(drop=True)
        df.index = raw_index

        df["target"] = target
        df["y"] = y

        if save:
            update = self._audit is None or overwrite
            if update:
                self._audit = {
                    "df": df,
                    "target": target,
                    "population": population,
                }

        df["model"] = (
            self.clf.steps[-1][-1].__class__.__name__
            if isinstance(self.clf, Pipeline)
            else self.clf.__class__.__name__
        )

        return df

    def _sampling_audit_stats(self, R, n_feasible, alpha=0.05, **kwargs):
        """Compute sampling-based uncertainty metrics for a sampled reachable set.

        Args:
            R: Reachable set object with sampling metadata.
            n_feasible: Number of feasible points observed in `R`.
            alpha: Significance level for intervals and thresholds.
            **kwargs: Additional options (unused).

        Returns:
            Dict with epsilon threshold, minimum samples, p-value, and Clopper–Pearson upper bound.
        """
        out = {
            "eps": R.resp_thresh,
            "min_samp": np.ceil(np.log(alpha) / np.log(1 - R.resp_thresh)),
            "p_val": binom.cdf(n_feasible, len(R), R.resp_thresh),
            "cp_int_ub": beta.ppf(1 - alpha, n_feasible + 1, len(R) - n_feasible)
            if n_feasible < len(R)
            else 1.0,
        }

        return out
