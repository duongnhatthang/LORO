import numpy as np


def paired_bootstrap_diff(method_per_seed, baseline_per_seed, n_boot=10000, seed=0):
    """Paired per-seed difference with a bootstrap CI and a two-sided bootstrap p-value.

    Inputs are equal-length lists indexed by the SAME seed (paired). Returns mean
    difference, 95% percentile CI over resampled seed-differences, and a two-sided
    p-value (fraction of bootstrap means on the opposite side of 0, doubled).
    """
    d = np.asarray(method_per_seed, float) - np.asarray(baseline_per_seed, float)
    n = d.shape[0]
    rng = np.random.default_rng(seed)
    boot = np.array([np.mean(d[rng.integers(0, n, n)]) for _ in range(n_boot)])
    mean_diff = float(np.mean(d))
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    frac_le0 = float(np.mean(boot <= 0.0))
    frac_ge0 = float(np.mean(boot >= 0.0))
    p_value = min(1.0, 2.0 * min(frac_le0, frac_ge0))
    p_value = max(p_value, 1.0 / n_boot)
    return {"mean_diff": mean_diff, "ci_low": float(ci_low),
            "ci_high": float(ci_high), "p_value": p_value}
