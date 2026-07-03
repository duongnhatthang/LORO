import json

import llm_main
from tests.conftest import ScriptedEnv, ScriptedAgent


def test_timing_log_records_parse_stats(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    stats = {"n_steps": 10, "n_parse_failures": 3, "parse_failure_rate": 0.3}
    llm_main.llm_write_timing_log(1.5, {"env": "X"}, "TS123", parse_stats=stats)
    with open(tmp_path / "logs" / "llm_timing_log_TS123.json") as f:
        data = json.load(f)
    assert data["parse_stats"] == stats


def test_timing_log_parse_stats_optional(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    llm_main.llm_write_timing_log(1.0, {}, "TS0")
    with open(tmp_path / "logs" / "llm_timing_log_TS0.json") as f:
        data = json.load(f)
    assert "parse_stats" not in data


def test_rollout_returns_parse_stats(monkeypatch):
    # d3rlpy seeding expects a gym env with action_space; ScriptedEnv has none.
    monkeypatch.setattr(llm_main.d3rlpy, "seed", lambda *a, **k: None)
    monkeypatch.setattr(llm_main.d3rlpy.envs, "seed_env", lambda *a, **k: None)
    hp = {
        "seed": 0, "env": "CartPole-v0", "max_episode_len": 200,
        "eps": 0.0, "n_episodes": 2, "SFT": False,
    }
    dataset, parse_stats = llm_main.rollout(ScriptedAgent(), ScriptedEnv(), hp)
    assert parse_stats["n_steps"] == 6            # 2 episodes x 3 scripted steps
    assert parse_stats["n_parse_failures"] == 0   # scripted agent never falls back
    assert parse_stats["parse_failure_rate"] == 0.0
