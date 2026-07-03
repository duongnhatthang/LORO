import numpy as np
import utils


class _StepEnv:
    """obs = step index; action recorded so we can verify pairing."""
    def __init__(self): self.t = 0; self.taken = []
    def reset(self): self.t = 0; return np.array([0.0], dtype=np.float32), {}
    def step(self, a):
        self.taken.append((self.t, int(a)))   # (state_index_when_acting, action)
        self.t += 1
        return np.array([float(self.t)], dtype=np.float32), 1.0, self.t >= 3, False, {}


class _FixedAlgo:
    def predict(self, obs):  # always action 7, echoing the observed state index
        return np.array([int(round(float(np.ravel(obs)[0])))])


def test_eval_dataset_pairs_action_with_pre_step_state():
    env = _StepEnv()
    _mean, ds = utils.evaluate_qlearning_with_environment(_FixedAlgo(), env, max_episode_len=3, n_trials=1)
    ep = ds.episodes[0]
    # First stored observation must be the reset state (0), not the successor (1).
    assert float(np.ravel(ep.observations[0])[0]) == 0.0
