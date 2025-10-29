import numpy as np
import pandas as pd
import pytest

try:
    import tensorflow as tf
    import tensorflow_probability as tfp
    TFP_AVAILABLE = True
except Exception:
    TFP_AVAILABLE = False

from typing import Optional, Any, Dict

from reachml.downstream import DownstreamSampler
from reachml.action_set import ActionSet

pytestmark = pytest.mark.skipif(not TFP_AVAILABLE, reason="tensorflow-probability is required")

# -----------------------------
# Helpers
# -----------------------------
def _mk_arrays_from_df(X_df: pd.DataFrame):
    X = X_df.values.astype(np.float32)
    A = np.zeros_like(X, dtype=np.float32)  # action increments (can be anything; sampler doesn't use A directly except param callables)
    return X, A

def _default_register_all(sampler: DownstreamSampler, dist="Normal", params: Optional[Dict[str, Any]] = None):
    for name in sampler.action_set._names:
        sampler.register_distribution(name, dist, params or {"loc": 0.0, "scale": 0.0})

# -----------------------------
# Registration tests
# -----------------------------
def test_register_distribution_rejects_unknown_var(discrete_test_case):
    X_df, Aset = discrete_test_case["X"], discrete_test_case["A"]
    sampler = DownstreamSampler(Aset)
    with pytest.raises(KeyError):
        sampler.register_distribution("not_in_actionset", "Normal")

def test_register_distribution_rejects_unknown_dist(discrete_test_case):
    X_df, Aset = discrete_test_case["X"], discrete_test_case["A"]
    sampler = DownstreamSampler(Aset)
    with pytest.raises(ValueError, match="Distribution not in library"):
        sampler.register_distribution(Aset._names[0], "NotADistribution")

def test_register_distribution_ok(discrete_test_case):
    X_df, Aset = discrete_test_case["X"], discrete_test_case["A"]
    sampler = DownstreamSampler(Aset)
    sampler.register_distribution(Aset._names[0], "Normal", {"loc": 0.0, "scale": 1.0})
    assert Aset._names[0] in sampler._registry

# -----------------------------
# Shape & outcome tests
# -----------------------------
@pytest.mark.parametrize("outcome", ["delta", "point"])
def test_sample_shape_matches_n_m_d(discrete_test_case, outcome):
    X_df, Aset = discrete_test_case["X"], discrete_test_case["A"]
    X, A = _mk_arrays_from_df(X_df)

    sampler = DownstreamSampler(Aset)
    # deterministic: scale 0 so we can assert exact shapes without randomness
    _default_register_all(sampler, "Normal", {"loc": 0.0, "scale": 0.0})

    out = sampler.sample(X, A, n=5, outcome=outcome, monotone_vs="base", filter_constraints=False, dtype="float32")
    assert out.shape == (5, X.shape[0], X.shape[1])

def test_invalid_outcome_raises(discrete_test_case):
    X_df, Aset = discrete_test_case["X"], discrete_test_case["A"]
    X, A = _mk_arrays_from_df(X_df)
    sampler = DownstreamSampler(Aset)
    _default_register_all(sampler, "Normal", {"loc": 0.0, "scale": 0.0})
    with pytest.raises(ValueError, match="outcome must be 'delta' or 'point'"):
        sampler.sample(X, A, n=1, outcome="weird", monotone_vs="base", filter_constraints=False)

# -----------------------------
# Monotonicity tests
# -----------------------------
def test_monotonicity_vs_base_increasing(dataset_actionset_2d):
    # boolean_2d, boolean_2d_immutable, boolean_2d_monotonic (parametrized)
    X_df, Aset = dataset_actionset_2d["X"], dataset_actionset_2d["A"]
    X, A = _mk_arrays_from_df(X_df)

    # Force samples below base to see that monotonicity lifts them up
    sampler = DownstreamSampler(Aset)
    # point outcome: always propose -1 for every var; monotonicity should raise to base when increasing
    _default_register_all(sampler, "Normal", {"loc": -1.0, "scale": 0.0})

    Y = sampler.sample(X, A, n=2, outcome="point", monotone_vs="base", filter_constraints=False)
    Y_flat = Y.reshape(-1, Y.shape[-1])

    inc_mask, _ = sampler._direction_masks()
    if inc_mask.any():
        base = np.repeat(X, repeats=2, axis=0)
        assert np.all(Y_flat[:, inc_mask] >= base[:, inc_mask])

def test_monotonicity_vs_base_decreasing(discrete_test_case):
    # find a 1d decreasing case
    name = None
    if isinstance(discrete_test_case, dict):
        # crude guard; only run on decreasing 1d cases to avoid false positives
        name = next((k for k in ["int_1d_decreasing", "uint_1d_decreasing", "boolean_1d_decreasing"] if k in str(discrete_test_case)), None)
    if not name:
        pytest.skip("Not a decreasing test case")

    X_df, Aset = discrete_test_case["X"], discrete_test_case["A"]
    X, A = _mk_arrays_from_df(X_df)

    sampler = DownstreamSampler(Aset)
    # propose very large value; monotone decreasing should clamp down to base
    _default_register_all(sampler, "Normal", {"loc": 999.0, "scale": 0.0})

    Y = sampler.sample(X, A, n=3, outcome="point", monotone_vs="base", filter_constraints=False)
    Y_flat = Y.reshape(-1, Y.shape[-1])

    _, dec_mask = sampler._direction_masks()
    if dec_mask.any():
        base = np.repeat(X, repeats=3, axis=0)
        assert np.all(Y_flat[:, dec_mask] <= base[:, dec_mask])

def test_monotonicity_vs_original_requires_x0(discrete_test_case):
    X_df, Aset = discrete_test_case["X"], discrete_test_case["A"]
    X, A = _mk_arrays_from_df(X_df)

    sampler = DownstreamSampler(Aset)
    _default_register_all(sampler, "Normal", {"loc": 0.0, "scale": 0.0})

    with pytest.raises(ValueError, match="x0 required when monotone_vs='original'"):
        sampler.sample(X, A, n=1, outcome="point", monotone_vs="original", x0=None, filter_constraints=False)

def test_monotonicity_vs_original_works_with_x0(discrete_test_case):
    X_df, Aset = discrete_test_case["X"], discrete_test_case["A"]
    X, A = _mk_arrays_from_df(X_df)

    sampler = DownstreamSampler(Aset)
    # Propose -1 so increasing variables must be >= x0
    _default_register_all(sampler, "Normal", {"loc": -1.0, "scale": 0.0})

    # Choose x0 slightly below X to make the effect observable
    x0 = (X - 0.5).astype(np.float32)

    Y = sampler.sample(X, A, n=1, outcome="point", monotone_vs="original", x0=x0, filter_constraints=False)
    inc_mask, dec_mask = sampler._direction_masks()

    Y2d = Y.reshape(-1, Y.shape[-1])
    x0_rep = np.repeat(x0, repeats=1, axis=0)
    if inc_mask.any():
        assert np.all(Y2d[:, inc_mask] >= x0_rep[:, inc_mask])
    if dec_mask.any():
        assert np.all(Y2d[:, dec_mask] <= x0_rep[:, dec_mask])

# -----------------------------
# Bounds clipping tests
# -----------------------------
def test_bounds_are_respected(discrete_test_case):
    X_df, Aset = discrete_test_case["X"], discrete_test_case["A"]
    X, A = _mk_arrays_from_df(X_df)

    sampler = DownstreamSampler(Aset)
    # Intentionally propose extreme values that exceed bounds
    _default_register_all(sampler, "Normal", {"loc": 1e6, "scale": 0.0})

    Y = sampler.sample(X, A, n=2, outcome="point", monotone_vs="base", filter_constraints=False)
    Y_flat = Y.reshape(-1, Y.shape[-1])

    lb = np.asarray(Aset.lb, float)
    ub = np.asarray(Aset.ub, float)
    assert np.all(Y_flat <= ub + 1e-7)
    assert np.all(Y_flat >= lb - 1e-7)

# -----------------------------
# Param callables & delta/point behavior
# -----------------------------
def test_callable_params_receive_x_and_a(discrete_test_case):
    X_df, Aset = discrete_test_case["X"], discrete_test_case["A"]
    X, A = _mk_arrays_from_df(X_df)

    sampler = DownstreamSampler(Aset)

    # Make 'point' samples equal to X + A via loc callable; scale 0 for determinism
    for j, name in enumerate(Aset._names):
        sampler.register_distribution(
            name,
            "Normal",
            {
                "loc": (lambda x_col, a_col: x_col + a_col),
                "scale": 0.0,
            },
        )

    Y_point = sampler.sample(X, A, n=1, outcome="point", monotone_vs="base", filter_constraints=False).squeeze(0)
    # With A all zeros, Y_point should equal X (subject to monotonicity clipping which is neutral here)
    assert np.allclose(Y_point, X, atol=1e-7)

def test_outcome_delta_vs_point(discrete_test_case):
    X_df, Aset = discrete_test_case["X"], discrete_test_case["A"]
    X, A = _mk_arrays_from_df(X_df)

    sampler = DownstreamSampler(Aset)
    # Always sample constant s=2
    _default_register_all(sampler, "Normal", {"loc": 2.0, "scale": 0.0})

    Y_point = sampler.sample(X, A, n=1, outcome="point", monotone_vs="base", filter_constraints=False, clip_to_bounds=False).squeeze(0)
    Y_delta = sampler.sample(X, A, n=1, outcome="delta", monotone_vs="base", filter_constraints=False, clip_to_bounds=False).squeeze(0)

    assert np.allclose(Y_point, np.full_like(X, 2.0))
    assert np.allclose(Y_delta, X + 2.0)


# -----------------------------
# Constraint filtering (monkeypatch validate)
# -----------------------------
def test_filter_constraints_applies_mask(monkeypatch, discrete_test_case):
    X_df, Aset = discrete_test_case["X"], discrete_test_case["A"]
    X, A = _mk_arrays_from_df(X_df)

    sampler = DownstreamSampler(Aset)
    # Deterministic samples: Y equals X to simplify
    _default_register_all(sampler, "Normal", {"loc": 0.0, "scale": 0.0})

    # Fake validate: mark every other row invalid
    def fake_validate(Y, warn=False, return_df=True):
        mask = np.ones((Y.shape[0],), dtype=bool)
        mask[::2] = False  # invalidate rows 0,2,4,...
        # Return a DataFrame of booleans across d columns (all same mask), sampler uses all-rows all-True across columns
        d = Y.shape[1]
        M = np.tile(mask[:, None], (1, d))
        return pd.DataFrame(M)

    monkeypatch.setattr(Aset, "validate", fake_validate, raising=True)

    out = sampler.sample(X, A, n=2, outcome="point", monotone_vs="base", filter_constraints=True)
    # Ragged: half filtered out
    assert out.ndim == 2  # ragged return path
    # expected remaining rows = (n*m)/2 rounded down
    expected = (2 * X.shape[0]) // 2
    assert out.shape[0] == expected
    assert out.shape[1] == X.shape[1]

def test_add_component_sum_combiner_constant_components(discrete_test_case):
    """Sum-combiner: Y = sum_k w_k * S_k, check exact constant result."""
    X_df, Aset = discrete_test_case["X"], discrete_test_case["A"]
    X, A = _mk_arrays_from_df(X_df)
    sampler = DownstreamSampler(Aset)

    vname = Aset._names[0]
    sampler.set_combiner(vname, "sum")
    sampler.add_component(vname, "Normal", params={"loc": 1.0, "scale": 0.0}, weight=2.0)   # 2*1 = 2
    sampler.add_component(vname, "Normal", params={"loc": 3.0, "scale": 0.0}, weight=0.5)  # 0.5*3 = 1.5

    # Disable clamping to isolate combiner math
    sampler._inc_mask = np.zeros(len(Aset), dtype=bool)
    sampler._dec_mask = np.zeros(len(Aset), dtype=bool)

    Y = sampler.sample(X, A, n=1, outcome="point", monotone_vs="base",
                       filter_constraints=False, clip_to_bounds=False).squeeze(0)

    expected = 3.5
    assert np.allclose(Y[:, 0], expected)
    if Y.shape[1] > 1:
        # Unconfigured vars fall back to identity (X)
        assert np.allclose(Y[:, 1:], X[:, 1:])

def test_mixture_requires_weights(discrete_test_case):
    """Mixture combiner should require weights for each component."""
    X_df, Aset = discrete_test_case["X"], discrete_test_case["A"]
    sampler = DownstreamSampler(Aset)
    vname = Aset._names[0]

    sampler.set_combiner(vname, "mixture")
    sampler.add_component(vname, "Normal", params={"loc": 0.0, "scale": 0.0}, weight=0.5)
    with pytest.raises(ValueError, match="Mixture combiner requires 'weight'"):
        sampler.add_component(vname, "Normal", params={"loc": 1.0, "scale": 0.0})

def test_mixture_empirical_weights(discrete_test_case):
    """Mixture: sample many times and check empirical proportion ~ weights."""
    X_df, Aset = discrete_test_case["X"], discrete_test_case["A"]
    X, A = _mk_arrays_from_df(X_df)
    sampler = DownstreamSampler(Aset)

    vname = Aset._names[0]
    sampler.set_combiner(vname, "mixture")
    sampler.add_component(vname, "Normal", params={"loc": -5.0, "scale": 0.0}, weight=0.2)
    sampler.add_component(vname, "Normal", params={"loc": 5.0,  "scale": 0.0}, weight=0.8)

    # Disable clamping to keep point masses intact
    sampler._inc_mask = np.zeros(len(Aset), dtype=bool)
    sampler._dec_mask = np.zeros(len(Aset), dtype=bool)

    n = 5000
    Y = sampler.sample(X, A, n=n, outcome="point", monotone_vs="base",
                       filter_constraints=False, clip_to_bounds=False, seed=123)
    first_col = Y[..., 0].reshape(-1)
    prop_pos = np.mean(first_col == 5.0)
    prop_neg = np.mean(first_col == -5.0)
    assert abs(prop_pos - 0.8) < 0.05
    assert abs(prop_neg - 0.2) < 0.05

def test_clear_components_and_backcompat_register(discrete_test_case):
    """clear_components empties; register_distribution restores single-component 'sum'."""
    X_df, Aset = discrete_test_case["X"], discrete_test_case["A"]
    sampler = DownstreamSampler(Aset)
    vname = Aset._names[0]

    sampler.set_combiner(vname, "sum")
    sampler.add_component(vname, "Normal", params={"loc": 1.0, "scale": 0.0}, weight=1.0)
    sampler.add_component(vname, "Normal", params={"loc": 2.0, "scale": 0.0}, weight=1.0)
    assert len(sampler._registry[vname]["components"]) == 2

    sampler.clear_components(vname)
    assert len(sampler._registry[vname]["components"]) == 0
    assert sampler._registry[vname]["combiner"] == "sum"

    sampler.register_distribution(vname, "Normal", {"loc": 3.0, "scale": 0.0})
    reg = sampler._registry[vname]
    assert reg["combiner"] == "sum"
    assert len(reg["components"]) == 1
    assert reg["components"][0]["dist_name"] == "Normal"

def test_incremental_addition_changes_output_under_sum(discrete_test_case):
    """Adding a new component under 'sum' should shift output by exactly its weighted loc."""
    X_df, Aset = discrete_test_case["X"], discrete_test_case["A"]
    X, A = _mk_arrays_from_df(X_df)
    sampler = DownstreamSampler(Aset)

    vname = Aset._names[0]
    sampler.set_combiner(vname, "sum")
    sampler.add_component(vname, "Normal", params={"loc": 1.0, "scale": 0.0}, weight=1.0)

    sampler._inc_mask = np.zeros(len(Aset), dtype=bool)
    sampler._dec_mask = np.zeros(len(Aset), dtype=bool)

    Y1 = sampler.sample(X, A, n=1, outcome="point", monotone_vs="base",
                        filter_constraints=False, clip_to_bounds=False, seed=7).squeeze(0)

    sampler.add_component(vname, "Normal", params={"loc": 2.0, "scale": 0.0}, weight=3.0)
    Y2 = sampler.sample(X, A, n=1, outcome="point", monotone_vs="base",
                        filter_constraints=False, clip_to_bounds=False, seed=7).squeeze(0)

    diff = Y2[:, 0] - Y1[:, 0]
    assert np.allclose(diff, 6.0)