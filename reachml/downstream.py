"""
This file contains classes and functions to represent and manipulate a
data generating distribution
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from reachml.action_set import ActionSet
tfd = tfp.distributions

import numpy as np
from typing import Any, Dict, Optional, List, Literal


CombinerT = Literal["sum", "mixture"]


class DownstreamSampler:
    """
    Samples downstream effects for (X, A) pairs.
    Keeps samples within ActionSet bounds and enforces monotonic directions.
    Each variable can be modeled as:
      • a single distribution, or
      • a combination of multiple component distributions combined by:
          - "sum":     Y = sum_k w_k * S_k
          - "mixture": Y ~ Mixture( w_k, S_k )
    You can add components incrementally with `add_component(...)`.

    Notes
    -----
    - Component params can be scalars, vectors of length m (rows in X),
      or callables f(x_col, a_col) -> tensorlike.
    - For "sum", weights default to 1.0 if omitted.
    - For "mixture", weights are required (they will be normalized per variable).
    """

    def __init__(self, action_set: ActionSet):
        if tfd is None:
            raise ImportError("TensorFlow Probability required. pip install tensorflow-probability")
        self.action_set = action_set
        self._lb = np.asarray(action_set.lb, float)
        self._ub = np.asarray(action_set.ub, float)
        self._inc_mask, self._dec_mask = self._direction_masks()

        # Registry schema:
        # var_name -> {
        #    "combiner": "sum" | "mixture",
        #    "components": [
        #        {"dist_name": str, "params": dict, "weight": Optional[float]}
        #    ]
        # }
        self._registry: Dict[str, Dict[str, Any]] = {}

    # ---------------------------
    # Registry / configuration API
    # ---------------------------
    def register_distribution(
        self,
        var_name: str,
        dist_name: str,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Backwards-compatible: define a single distribution for a variable.
        Equivalent to clear_components(var), set_combiner(var,"sum"), add_component(...).

        Args:
            var_name: feature name (must exist in ActionSet)
            dist_name: string in tfp.distributions, e.g., "Normal", "Laplace"
            params: optional dict of kwargs (may contain callables)
        """
        self.clear_components(var_name)
        self.set_combiner(var_name, "sum")
        self.add_component(var_name, dist_name, params=params or {}, weight=1.0)

    def add_component(
        self,
        var_name: str,
        dist_name: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        weight: Optional[float] = None
    ):
        """
        Add a component distribution to a variable's combination model.

        Args:
            var_name: feature name (must exist in ActionSet)
            dist_name: string name in tfp.distributions.* (e.g., "Normal", "Laplace")
            params: kwargs dict for that distribution (can include callables)
            weight: for "sum": optional scalar multiplier (defaults to 1.0)
                    for "mixture": REQUIRED (acts as mixture weight, will be normalized)
        """
        if var_name not in self.action_set._names:
            raise KeyError(f"Variable '{var_name}' not found in ActionSet.")
        if not hasattr(tfd, dist_name):
            raise ValueError(f"Distribution not in library: '{dist_name}'")

        if var_name not in self._registry:
            self._registry[var_name] = {"combiner": "sum", "components": []}

        comb = self._registry[var_name]["combiner"]
        if comb == "mixture" and weight is None:
            raise ValueError("Mixture combiner requires 'weight' for each component.")

        self._registry[var_name]["components"].append(
            {"dist_name": dist_name, "params": params or {}, "weight": (1.0 if weight is None else float(weight))}
        )

    def set_combiner(self, var_name: str, combiner: CombinerT):
        """
        Set how components for `var_name` are combined: 'sum' or 'mixture'.
        """
        if var_name not in self.action_set._names:
            raise KeyError(f"Variable '{var_name}' not found in ActionSet.")
        if var_name not in self._registry:
            self._registry[var_name] = {"combiner": combiner, "components": []}
        else:
            self._registry[var_name]["combiner"] = combiner

    def clear_components(self, var_name: str):
        """
        Remove all components for a variable (does not remove the entry).
        """
        if var_name not in self.action_set._names:
            raise KeyError(f"Variable '{var_name}' not found in ActionSet.")
        self._registry[var_name] = {"combiner": "sum", "components": []}

    # ---------------------------
    # Sampling
    # ---------------------------
    def sample(
    self,
    X: np.ndarray,
    A: np.ndarray,
    *,
    n: int = 1,
    outcome: str = "delta",
    monotone_vs: str = "original",
    x0: Optional[np.ndarray] = None,
    filter_constraints: bool = True,
    seed: Optional[int] = None,
    dtype: str = "float32",
    clip_to_bounds: bool = True
    ):
        assert X.shape == A.shape, "X and A must have the same shape"
        m, d = X.shape
        assert d == len(self.action_set)

        # ---- NEW: validate outcome early to ensure expected error path ----
        if outcome not in ("delta", "point"):
            raise ValueError("outcome must be 'delta' or 'point'")

        tf_dtype = getattr(tf, dtype)
        X_tf = tf.convert_to_tensor(X, dtype=tf_dtype)
        A_tf = tf.convert_to_tensor(A, dtype=tf_dtype)

        # ---- NEW: simple seed arg for TFP ----
        seed_arg = int(seed) if seed is not None else None

        samples = np.zeros((n, m, d), dtype=float)

        for j, var in enumerate(self.action_set._names):
            entry = self._registry.get(var, None)

            if not entry or len(entry["components"]) == 0:
                samples[:, :, j] = np.repeat(X[:, j][None, :], n, axis=0)
                continue

            comb: CombinerT = entry["combiner"]
            comps: List[Dict[str, Any]] = entry["components"]

            def _as_vector(t):
                t = tf.convert_to_tensor(t, dtype=tf_dtype)
                if t.shape.rank == 0:
                    return tf.broadcast_to(t, (m,))
                return t

            tf_components = []
            weights = []
            for c in comps:
                dist_name = c["dist_name"]
                params = c.get("params", {})
                dist_cls = getattr(tfd, dist_name)

                eval_params = {}
                for k, v in params.items():
                    pv = v(X_tf[:, j], A_tf[:, j]) if callable(v) else v
                    eval_params[k] = _as_vector(pv)

                tf_components.append(dist_cls(**eval_params))
                weights.append(c.get("weight", 1.0))

            weights = tf.convert_to_tensor(weights, dtype=tf_dtype)
            K = len(tf_components)

            if comb == "sum":
                s_total = tf.zeros((n, m), dtype=tf_dtype)
                for k in range(K):
                    s_k = tf_components[k].sample(sample_shape=(n,), seed=seed_arg)  # <-- seed_arg (int or None)
                    if s_k.shape.rank == 1:
                        s_k = tf.broadcast_to(s_k[:, None], (n, m))
                    s_total = s_total + weights[k] * s_k
                s_final = s_total

            elif comb == "mixture":
                probs = weights / tf.reduce_sum(weights)
                cat = tfd.Categorical(probs=probs)

                idx = cat.sample(sample_shape=(n, m), seed=seed_arg)  # <-- seed_arg

                s_stack = []
                for k in range(K):
                    s_k = tf_components[k].sample(sample_shape=(n,), seed=seed_arg)  # <-- seed_arg
                    if s_k.shape.rank == 1:
                        s_k = tf.broadcast_to(s_k[:, None], (n, m))
                    s_stack.append(s_k)
                S = tf.stack(s_stack, axis=0)              # [K, n, m]
                one_hot = tf.one_hot(idx, depth=K, dtype=tf_dtype)  # [n, m, K]
                one_hot_T = tf.transpose(one_hot, perm=[2, 0, 1])   # [K, n, m]
                s_final = tf.reduce_sum(S * one_hot_T, axis=0)      # [n, m]

            else:
                raise ValueError(f"Unknown combiner: {comb}")

            s_np = s_final.numpy()
            if outcome == "delta":
                samples[:, :, j] = s_np + X[:, j][None, :]
            else:  # outcome == "point"
                samples[:, :, j] = s_np

        Y = samples.reshape(n * m, d)

        # (1) clip to bounds
        if clip_to_bounds:
            Y = np.minimum(np.maximum(Y, self._lb), self._ub)

        # (2) enforce monotonicity
        if monotone_vs == "original":
            if x0 is None:
                raise ValueError("x0 required when monotone_vs='original'")
            base = np.repeat(x0[None, :], n * m, axis=0)
        elif monotone_vs == "base":
            base = np.repeat(X, repeats=n, axis=0)
        else:
            raise ValueError("monotone_vs must be 'original' or 'base'")

        if self._inc_mask.any():
            Y[:, self._inc_mask] = np.maximum(Y[:, self._inc_mask], base[:, self._inc_mask])
        if self._dec_mask.any():
            Y[:, self._dec_mask] = np.minimum(Y[:, self._dec_mask], base[:, self._dec_mask])

        # (3) filter constraint violations
        if filter_constraints:
            rep = self.action_set.validate(Y, warn=False, return_df=True)
            mask = np.all(rep.values, axis=1)
            Y = Y[mask]

        if Y.shape[0] == n * m:
            return Y.reshape(n, m, d)
        return Y  # ragged if filtered

    # ---------------------------
    # Internal helpers
    # ---------------------------
    def _direction_masks(self):
        sd = np.array(self.action_set.step_direction, dtype=object)
        def inc(s): return isinstance(s, str) and s.lower().startswith(("inc", "nondec", "up"))
        def dec(s): return isinstance(s, str) and s.lower().startswith(("dec", "noninc", "down"))
        inc_mask = np.array([inc(v) for v in sd], dtype=bool)
        dec_mask = np.array([dec(v) for v in sd], dtype=bool)
        return inc_mask, dec_mask
