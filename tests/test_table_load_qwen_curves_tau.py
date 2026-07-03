"""
Test that extract_cumulative_rewards_table.load_qwen_curves averages exactly
n_pretrain_eps episodes (the τ warm-start window), not a wider window.

This closes the coverage gap found in the final F5 review: the Task-5 test
only covered notebook_utils; this test exercises the TABLE code path directly.
"""
import pickle
import numpy as np
import pytest

import extract_cumulative_rewards_table as tbl


class _FakeEpisode:
    def __init__(self, r):
        self._r = r

    def compute_return(self):
        return self._r


class _FakeDataset:
    def __init__(self, episodes):
        self.episodes = episodes


def _make_stub_pkl(tmp_path, returns):
    """Write a fake MDPDataset-like object with known per-episode returns."""
    ds = _FakeDataset([_FakeEpisode(r) for r in returns])
    p = tmp_path / "stub.pkl"
    with open(p, "wb") as f:
        pickle.dump(ds, f)
    return str(p)


def test_table_load_qwen_curves_uses_n_pretrain_eps(monkeypatch, tmp_path):
    """
    With 5 episodes (returns 1,2,3,100,200) and n_pretrain_eps=3,
    the constant must equal mean([1,2,3])=2.0, not mean of all 5 (61.2).
    """
    stub_path = _make_stub_pkl(tmp_path, [1.0, 2.0, 3.0, 100.0, 200.0])

    # Monkeypatch resolve_llm_path so both 7B and 32B resolve to the stub
    monkeypatch.setattr(tbl, "resolve_llm_path", lambda path, env_base, size_key="7B": stub_path)

    Qwen_7B, Qwen_32B = tbl.load_qwen_curves("CartPole-v1", n_episodes=6, n_pretrain_eps=3)

    expected = np.mean([1.0, 2.0, 3.0])   # mean of EXACTLY τ=3 episodes
    wrong_all = np.mean([1.0, 2.0, 3.0, 100.0, 200.0])

    assert abs(float(Qwen_7B[0]) - expected) < 1e-9, (
        f"Qwen_7B constant should be {expected} (mean of tau=3), got {Qwen_7B[0]}"
    )
    assert abs(float(Qwen_32B[0]) - expected) < 1e-9, (
        f"Qwen_32B constant should be {expected} (mean of tau=3), got {Qwen_32B[0]}"
    )
    # Explicitly confirm we are NOT averaging all episodes (the pre-fix behaviour)
    assert abs(float(Qwen_7B[0]) - wrong_all) > 1.0, (
        "Got mean-of-all-episodes instead of mean-of-tau — bug not fixed"
    )


def test_table_load_qwen_curves_default_n_pretrain_eps_is_10():
    """The default τ for the table path must be 10, matching get_pretrain_defaults."""
    import inspect
    sig = inspect.signature(tbl.load_qwen_curves)
    assert sig.parameters["n_pretrain_eps"].default == 10, (
        f"Expected default n_pretrain_eps=10, got {sig.parameters['n_pretrain_eps'].default}"
    )
