import numpy as np
from stats_utils import paired_bootstrap_diff


def test_clear_positive_difference_is_significant():
    method = [1.0, 1.1, 0.9, 1.2, 1.0]
    base = [0.0, 0.1, -0.1, 0.2, 0.0]
    r = paired_bootstrap_diff(method, base, n_boot=5000, seed=0)
    assert r["mean_diff"] > 0.5
    assert r["ci_low"] > 0.0            # CI excludes 0
    assert r["p_value"] < 0.05


def test_no_difference_is_not_significant():
    x = [0.0, 0.1, -0.1, 0.2, -0.05]
    r = paired_bootstrap_diff(x, list(x), n_boot=5000, seed=0)
    assert abs(r["mean_diff"]) < 1e-9
    assert r["p_value"] > 0.05


def test_bootstrap_ci_has_real_spread_and_brackets_mean():
    # per-seed diffs vary: [0.5, 1.5, -0.2, 2.0, 0.8] => mean_diff = 0.92
    method = [1.5, 2.5, 0.8, 3.0, 1.8]
    base =   [1.0, 1.0, 1.0, 1.0, 1.0]
    r = paired_bootstrap_diff(method, base, n_boot=5000, seed=0)
    # non-degenerate CI: strictly brackets the mean diff with real width
    assert r["ci_low"] < r["mean_diff"] < r["ci_high"]
    assert (r["ci_high"] - r["ci_low"]) > 0.1
    # p-value is a sane probability in (0,1]
    assert 0.0 < r["p_value"] <= 1.0
