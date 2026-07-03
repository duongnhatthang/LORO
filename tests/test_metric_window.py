import numpy as np
from vis_utils import processing_offline_online_data


def test_constant_and_per_episode_have_identical_auc():
    n_pre, n_eps, n_exp = 3, 6, 2
    per_ep = [10.0, 20.0, 30.0]          # real warm-start returns for the tau episodes
    const = float(np.mean(per_ep))
    online_cache = {f"m_{i}": [1.0, 2.0, 3.0] for i in range(n_exp)}  # online part (len n_eps-n_pre)

    avg_const, std_const = processing_offline_online_data(
        avg_offline_returns=[const] * n_eps, online_cache=online_cache,
        n_pretrain_eps=n_pre, online_cache_key="m", n_exp=n_exp, n_episodes=n_eps,
        use_iqm=False, warmstart_mode="constant",
    )
    avg_pe, std_pe = processing_offline_online_data(
        avg_offline_returns=[const] * n_eps, online_cache=online_cache,
        n_pretrain_eps=n_pre, online_cache_key="m", n_exp=n_exp, n_episodes=n_eps,
        use_iqm=False, warmstart_mode="per_episode", warmstart_per_episode=per_ep,
    )
    assert abs(np.sum(avg_const) - np.sum(avg_pe)) < 1e-9    # AUC identical by construction
    assert np.allclose(std_const[:n_pre], 0.0)               # constant line has no band
    assert np.any(std_pe[:n_pre] > 0.0)                      # per-episode shows real spread


def test_load_qwen_curves_default_tau_is_10():
    import inspect
    from notebook_utils import load_qwen_curves
    sig = inspect.signature(load_qwen_curves)
    assert sig.parameters["n_pretrain_eps"].default == 10
