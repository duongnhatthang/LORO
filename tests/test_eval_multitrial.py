import inspect
import utils


def test_rollout_uses_multitrial_eval_for_all_envs():
    src = inspect.getsource(utils.rollout_and_eval)
    # The old single-trial path (EnvironmentEvaluator(..., n_trials=1)) must be gone.
    assert "n_trials=1" not in src
    assert "EVAL_N_TRIALS" in src


def test_eval_n_trials_constant_is_at_least_20():
    assert utils.EVAL_N_TRIALS >= 20
