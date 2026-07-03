import numpy as np
import pytest


class ScriptedEnv:
    """Deterministic 3-step env: state = step index (0,1,2,3). Reward = state after step."""
    def __init__(self):
        self._t = 0

    def reset(self):
        self._t = 0
        return self._t, {}

    def step(self, action):
        self._t += 1
        obs = self._t
        reward = float(self._t)          # reward for taking `action` in the PRE-step state
        done = self._t >= 3
        info = {"action_taken_in_state": self._t - 1}
        return obs, reward, done, info


class ScriptedAgent:
    """Chooses action = current observation (so we can detect state/action misalignment)."""
    def act(self, observation):
        return int(observation)

    def assign_reward(self, r):
        pass

    def terminate_episode(self, train=False):
        return {}


@pytest.fixture
def scripted_env():
    return ScriptedEnv()


@pytest.fixture
def scripted_agent():
    return ScriptedAgent()
