import numpy as np
from tests.conftest import ScriptedEnv, ScriptedAgent
from llm_main import collect_episode


def test_action_paired_with_pre_step_state():
    obs, acts, rews, terms = collect_episode(
        ScriptedEnv(), ScriptedAgent(), env_name="CartPole-v0",
        max_episode_len=200, eps=0.0,
    )
    # ScriptedAgent picks action == observed state. So action[k] must equal obs[k].
    assert acts == [int(o) for o in obs]
    # First observation is the reset state 0 (included), not the successor.
    assert obs[0] == 0
    # 3 steps -> 3 stored transitions; last terminal is 1.
    assert len(obs) == 3 and terms[-1] == 1
