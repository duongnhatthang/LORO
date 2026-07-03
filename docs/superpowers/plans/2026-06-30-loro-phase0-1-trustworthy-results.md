# LORO Phase 0 + Phase 1: Trustworthy, Re-runnable Toy-Env Results — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix every trustworthiness bug the audits found (metrics, evaluation, and LLM data collection), add a regression test for each, re-collect LLM data with the corrected pipeline, and regenerate the toy-env tables with a delta report — so the results are defensible.

**Architecture:** Two groups of changes. (1) **Metric/eval fixes** operate on the *existing* cached results and pure-Python plotting/aggregation code, so they iterate for free (no GPU/LLM). (2) **Collection fixes** correct `llm_main.py` + the env action parsers, then re-collect on **self-hosted GPUs** (7B on the a local GPU, 32B in 4-bit on a rented 24 GB RTX 4090) via a response-cached generation path, and re-run `online_main.py`. Each fix is an isolated, unit-tested unit; experiment tasks end with a concrete validation check.

**Tech Stack:** Python 3.9, `d3rlpy`, `gymnasium`, `numpy`, `pytest` (new), Qwen3/Qwen2.5 via **local `transformers` with 4-bit quantization** on local + rented GPUs. (An OpenAI-compatible API client is included as an optional fallback only; not used for the main runs.)

## Global Constraints

- Branch: `investigation/trustworthy-results-and-reframe` (already created).
- Design spec of record: `docs/superpowers/specs/2026-06-30-loro-trustworthy-results-and-reframe-design.md`. Task IDs below (F1–F8) map to spec §4.
- **Never** commit the compute server's hostname/IP/hardware specs or SSH config (user global rule). Server-specific run notes stay in local memory, not the repo.
- Headline metric = **full-budget cumulative reward, τ warm-start episodes INCLUDED** (spec §4 F5). Do not exclude the warm-start window from the primary number.
- Warm-start phase (method arm) rendered two ways that must be **AUC-identical by construction**: (a) constant line at the mean of exactly the τ episodes used (default figure), (b) real per-episode returns (appendix). The Online RL baseline always uses real per-episode returns.
- Paired seeds across arms: `seed = base_seed + i`. Keep this pairing; it is correct and load-bearing for the paired statistics.
- Env base names used everywhere: `CartPole, Pendulum, MountainCar, FrozenLake, CliffWalking, RepresentedPong`.
- Do not "fix" `truncate_dataset` terminal handling or the episode-vs-step budget philosophy in this plan — those are deliberate/known and out of scope here (spec §9, B6).
- After each task: run the task's test(s), confirm pass, commit.

---

## File Structure

**New files:**
- `tests/conftest.py` — pytest fixtures (scripted fake env + fake agent, tiny caches).
- `tests/test_collection_alignment.py` — F1
- `tests/test_action_parsing.py` — F2
- `tests/test_norm_constants.py` — F3
- `tests/test_eval_multitrial.py` — F4
- `tests/test_metric_window.py` — F5
- `tests/test_stats_utils.py` — F6
- `env/action_parsing.py` — F2 shared, tested action-extraction helper.
- `tools/measure_norm_constants.py` — F3 script → writes `data/norm_constants.json`.
- `stats_utils.py` — F6 paired bootstrap difference + paired test.
- `llm_client.py` — F8 cached LLM client (API + local backends).
- `tools/snapshot_baseline.py` + `docs/superpowers/baselines/2026-06-30-pre-fix-table.txt` — Phase 0 frozen reference.
- `tools/delta_report.py` — compares a regenerated table against the frozen baseline.

**Modified files:**
- `utils.py` — F4 (multi-trial eval in `rollout_and_eval`), F1-eval (`evaluate_qlearning_with_environment` off-by-one), F7 (SAC config).
- `vis_utils.py` — F3 (load measured norm constants), F5 (per-episode warm-start option + real variance).
- `notebook_utils.py` — F5 (`load_qwen_curves` constant over exactly τ).
- `llm_main.py` — F1-collection (store `s_t`), F2 (use `env/action_parsing.py`), F8 (use `llm_client.py`).
- `llamagym/agent.py` — F2 (Qwen/ChatML response-span extraction; B3).
- `env/translation_agent.py` — F2 (route all `extract_action` through `env/action_parsing.py`).
- `extract_cumulative_rewards_table.py` — F3/F6 (measured constants + paired stats columns).
- `pyproject.toml` — add `pytest` dev dependency.

---

## Task 1 (Phase 0): Test harness + frozen baseline

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/conftest.py`, `tools/snapshot_baseline.py`, `docs/superpowers/baselines/2026-06-30-pre-fix-table.txt`

**Interfaces:**
- Produces: pytest runnable via `python -m pytest`; fixture `scripted_env` (a deterministic 3-step gym-style env) and `scripted_agent`; frozen baseline table text file.

- [ ] **Step 1: Add pytest dev dependency**

In `pyproject.toml`, under `[tool.poetry.group.dev.dependencies]` (create the group if absent):
```toml
[tool.poetry.group.dev.dependencies]
pytest = "^8.0"
```

- [ ] **Step 2: Install and verify pytest runs**

Run: `python -m pytest --version`
Expected: prints a pytest 8.x version. (If poetry-managed: `poetry install --with dev` first.)

- [ ] **Step 3: Create shared fixtures**

Create `tests/conftest.py`:
```python
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
```

- [ ] **Step 4: Freeze the current (pre-fix) table**

Create `tools/snapshot_baseline.py`:
```python
"""Run the current cumulative-rewards table and save it as an immutable reference."""
import subprocess
import sys

OUT = "docs/superpowers/baselines/2026-06-30-pre-fix-table.txt"

if __name__ == "__main__":
    res = subprocess.run(
        [sys.executable, "extract_cumulative_rewards_table.py"],
        capture_output=True, text=True,
    )
    with open(OUT, "w") as f:
        f.write(res.stdout)
        if res.stderr:
            f.write("\n# STDERR\n" + res.stderr)
    print(f"Wrote {OUT}")
    print(res.stdout)
```

- [ ] **Step 5: Run it and commit the frozen baseline**

Run: `python tools/snapshot_baseline.py`
Expected: prints the current table (matching paper Table 1/4 numbers, e.g. Average row) and writes the txt file.
```bash
git add pyproject.toml tests/conftest.py tools/snapshot_baseline.py docs/superpowers/baselines/2026-06-30-pre-fix-table.txt
git commit -m "test: add pytest harness + freeze pre-fix results table (Phase 0)"
```

---

## Task 2 (F7): SAC config is explicit and logged

**Files:**
- Modify: `utils.py:158-207` (`create_d3rlpy_model`)
- Test: `tests/test_sac_config.py`

**Interfaces:**
- Consumes: `create_d3rlpy_model(env_name, batch_size, learning_rate, gamma, target_update_interval, gpu, model_type)`.
- Produces: SAC created with explicit `actor_learning_rate`/`critic_learning_rate` = `learning_rate` (was silently dropped). No signature change.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sac_config.py`:
```python
from utils import create_d3rlpy_model


def test_sac_uses_configured_learning_rate():
    model = create_d3rlpy_model(
        "Pendulum-v1", batch_size=256, learning_rate=5e-5, gamma=0.99,
        target_update_interval=1000, gpu=False, model_type="default",
    )
    cfg = model.config
    assert cfg.actor_learning_rate == 5e-5
    assert cfg.critic_learning_rate == 5e-5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sac_config.py -v`
Expected: FAIL (current code passes only `batch_size, gamma`, so LR is the d3rlpy default, not 5e-5).

- [ ] **Step 3: Make SAC honor the configured LR**

In `utils.py`, replace the `if is_continuous:` SAC branch inside `model_type == "default"` (currently lines 163-167):
```python
        if is_continuous:  # Continuous action space
            model = d3rlpy.algos.SACConfig(
                batch_size=batch_size,
                gamma=gamma,
                actor_learning_rate=learning_rate,
                critic_learning_rate=learning_rate,
            ).create(device=gpu)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sac_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add utils.py tests/test_sac_config.py
git commit -m "fix(F7): SAC honors configured learning_rate (was silently dropped)"
```
Note for paper: the continuous-env LR is now the configured value; update the implementation-details text (B5) when writing results.

---

## Task 3 (F3): Empirical normalization constants

**Files:**
- Create: `tools/measure_norm_constants.py`, `data/norm_constants.json`
- Modify: `vis_utils.py:37-46` (load measured constants; keep dict as fallback)
- Test: `tests/test_norm_constants.py`

**Interfaces:**
- Produces: `data/norm_constants.json` = `{env_base: {"x_random": float, "x_expert": float}}`; `vis_utils.load_norm_constants() -> dict`; `DEFAULT_NORM_CONST_DICT` becomes the merge of measured-over-hardcoded.

- [ ] **Step 1: Write the failing test**

Create `tests/test_norm_constants.py`:
```python
import json
import numpy as np
from vis_utils import load_norm_constants, normalize_reward


def test_load_norm_constants_reads_json(tmp_path, monkeypatch):
    p = tmp_path / "norm_constants.json"
    p.write_text(json.dumps({"CliffWalking": {"x_random": -7700.0, "x_expert": -13.0}}))
    monkeypatch.setattr("vis_utils.NORM_CONSTANTS_PATH", str(p))
    d = load_norm_constants()
    assert d["CliffWalking"]["x_random"] == -7700.0


def test_random_policy_normalizes_to_zero():
    # With x_random set to the true random return, a random-level score maps to ~0.
    d = {"CliffWalking": {"x_random": -7700.0, "x_expert": -13.0}}
    assert abs(float(normalize_reward(-7700.0, "CliffWalking", d))) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_norm_constants.py -v`
Expected: FAIL with `AttributeError: module 'vis_utils' has no attribute 'load_norm_constants'`.

- [ ] **Step 3: Add the loader to vis_utils.py**

In `vis_utils.py`, immediately after `DEFAULT_NORM_CONST_DICT` (after line 46) add:
```python
import json as _json
import os as _os

NORM_CONSTANTS_PATH = "data/norm_constants.json"


def load_norm_constants():
    """Measured constants override the hardcoded defaults; hardcoded fill any gaps."""
    merged = {k: dict(v) for k, v in DEFAULT_NORM_CONST_DICT.items()}
    if _os.path.exists(NORM_CONSTANTS_PATH):
        with open(NORM_CONSTANTS_PATH) as f:
            measured = _json.load(f)
        for env, vals in measured.items():
            merged.setdefault(env, {}).update(vals)
    return merged
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_norm_constants.py -v`
Expected: PASS.

- [ ] **Step 5: Write the measurement script**

Create `tools/measure_norm_constants.py`:
```python
"""Measure x_random (uniform-random policy) and x_expert (converged/known-optimal) per env.

x_random: mean episode return of env.action_space.sample() over N episodes (H=200 cap).
x_expert: known optimum where available, else the mean return of a DoubleDQN/SAC trained
          to convergence (long run). For envs with a known optimum we use it directly:
            CartPole 200, FrozenLake 1.0, MountainCar -110 (approx solve), CliffWalking -13,
            Pendulum ~ -150 (near-upright), RepresentedPong 21.
Writes data/norm_constants.json (x_expert kept from known values; x_random measured).
"""
import json
import numpy as np
import gymnasium as gym
from vis_utils import DEFAULT_NORM_CONST_DICT

N_EPISODES = 100
H = 200
ENVS = ["CartPole-v0", "Pendulum-v1", "MountainCar-v0",
        "FrozenLake-v1", "CliffWalking-v0", "RepresentedPong-v0"]


def measure_random(env_name):
    env = gym.make(env_name)
    returns = []
    for _ in range(N_EPISODES):
        _obs, _ = env.reset()
        total, count, done = 0.0, 0, False
        while not done and count < H:
            _obs, r, term, trunc, _ = env.step(env.action_space.sample())
            total += float(r); count += 1; done = term or trunc
        returns.append(total)
    return float(np.mean(returns))


if __name__ == "__main__":
    out = {}
    for env_name in ENVS:
        base = env_name.split("-")[0]
        try:
            x_random = measure_random(env_name)
        except Exception as e:
            print(f"skip {base}: {e}"); continue
        x_expert = DEFAULT_NORM_CONST_DICT[base]["x_expert"]  # keep known-optimal expert
        out[base] = {"x_random": x_random, "x_expert": x_expert}
        print(f"{base}: x_random={x_random:.2f} x_expert={x_expert}")
    with open("data/norm_constants.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote data/norm_constants.json")
```

- [ ] **Step 6: Run measurement and wire consumers to the loader**

Run: `python tools/measure_norm_constants.py`
Expected: prints per-env constants; CliffWalking `x_random` should be far below −2120 (roughly −6000 to −10000), confirming the audit.

Then in `extract_cumulative_rewards_table.py`, change the hyperparams construction (line 204) from:
```python
        "norm_const_dict": {**DEFAULT_NORM_CONST_DICT},
```
to:
```python
        "norm_const_dict": load_norm_constants(),
```
and add `load_norm_constants` to the existing `from vis_utils import (...)` import.

- [ ] **Step 7: Commit**

```bash
git add vis_utils.py tools/measure_norm_constants.py data/norm_constants.json extract_cumulative_rewards_table.py tests/test_norm_constants.py
git commit -m "fix(F3): measure empirical x_random per env; load over hardcoded constants"
```

---

## Task 4 (F4): Multi-trial evaluation on eval_env

**Files:**
- Modify: `utils.py:356-368` (`rollout_and_eval` per-episode and final eval)
- Test: `tests/test_eval_multitrial.py`

**Interfaces:**
- Consumes: `evaluate_qlearning_with_environment(algo, env, max_episode_len, n_trials=10, epsilon=0.0)` — already supports `n_trials`.
- Produces: every env's per-checkpoint eval uses `eval_env` with `n_trials = EVAL_N_TRIALS` (=20); no single-trial path remains.

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_multitrial.py`:
```python
import inspect
import utils


def test_rollout_uses_multitrial_eval_for_all_envs():
    src = inspect.getsource(utils.rollout_and_eval)
    # The old single-trial path (EnvironmentEvaluator(..., n_trials=1)) must be gone.
    assert "n_trials=1" not in src
    assert "EVAL_N_TRIALS" in src


def test_eval_n_trials_constant_is_at_least_20():
    assert utils.EVAL_N_TRIALS >= 20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval_multitrial.py -v`
Expected: FAIL (`EVAL_N_TRIALS` undefined; `n_trials=1` present).

- [ ] **Step 3: Replace the per-episode eval block**

In `utils.py`, add near the top (after imports, ~line 11):
```python
EVAL_N_TRIALS = 20  # per-checkpoint evaluation episodes (was 1 for all envs except CliffWalking)
```
Then replace the per-episode eval block (currently lines 356-363) inside `rollout_and_eval`:
```python
        r, _ = evaluate_qlearning_with_environment(
            algorithm, eval_env, max_episode_len, n_trials=EVAL_N_TRIALS
        )
        rewards.append(r)
```
(This removes the `if env_name == "CliffWalking-v0": ... else EnvironmentEvaluator(env, n_trials=1)` branch and always evaluates on `eval_env` over `EVAL_N_TRIALS` episodes.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_eval_multitrial.py -v`
Expected: PASS.

- [ ] **Step 5: Smoke-check it runs end-to-end (fast, CPU)**

Run:
```bash
python -c "import utils, d3rlpy; \
algo = utils.create_d3rlpy_model('CartPole-v0',256,5e-5,0.99,1000,False,'default'); \
env,ev = utils.get_env_and_eval_env('CartPole-v0',0); \
print(utils.evaluate_qlearning_with_environment(algo, ev, 200, n_trials=utils.EVAL_N_TRIALS)[0])"
```
Expected: prints a float (untrained ≈ low return), no exception. Confirms the eval path works with EVAL_N_TRIALS.

- [ ] **Step 6: Commit**

```bash
git add utils.py tests/test_eval_multitrial.py
git commit -m "fix(F4): evaluate all envs on eval_env over EVAL_N_TRIALS=20 (was single-trial)"
```

---

## Task 5 (F5): Warm-start representation — constant over exactly τ, plus per-episode variant

**Files:**
- Modify: `notebook_utils.py:29-49` (`load_qwen_curves`)
- Modify: `vis_utils.py:172-206` (`processing_offline_online_data` — add per-episode option + real variance)
- Test: `tests/test_metric_window.py`

**Interfaces:**
- Consumes: LLM dataset `.episodes[i].compute_return()`.
- Produces:
  - `load_qwen_curves(env_name, n_episodes, n_pretrain_eps=10, sft=False, long_cot=False)` — constant computed over **exactly `n_pretrain_eps`** episodes (default changed 30→10 to match τ).
  - `processing_offline_online_data(..., warmstart_mode="constant"|"per_episode", warmstart_per_episode=None)` — `"constant"` = current default figure; `"per_episode"` fills the first τ with real returns and real std (no forced `std=0`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_metric_window.py`:
```python
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
```

And for `load_qwen_curves`:
```python
def test_load_qwen_curves_default_tau_is_10():
    import inspect
    from notebook_utils import load_qwen_curves
    sig = inspect.signature(load_qwen_curves)
    assert sig.parameters["n_pretrain_eps"].default == 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_metric_window.py -v`
Expected: FAIL (`warmstart_mode` is not a parameter; default τ is 30).

- [ ] **Step 3: Fix `load_qwen_curves` default τ**

In `notebook_utils.py` line 29, change the signature default:
```python
def load_qwen_curves(env_name, n_episodes, n_pretrain_eps=10, sft=False, long_cot=False):
```
(The body already averages `min(n_pretrain_eps, len(...))` episodes, so with τ=10 the constant now matches the pretrain budget.)

- [ ] **Step 4: Add warm-start modes to `processing_offline_online_data`**

In `vis_utils.py`, change the signature (line 172) and the warm-start fill (lines 194-197):
```python
def processing_offline_online_data(avg_offline_returns, online_cache, n_pretrain_eps, online_cache_key, n_exp, n_episodes, use_iqm=True, hyperparams=None, warmstart_mode="constant", warmstart_per_episode=None):
```
Replace lines 194-197 with:
```python
    average_returns = np.empty(n_episodes)
    average_returns[n_pretrain_eps:] = online_agg[n_pretrain_eps:]
    if warmstart_mode == "per_episode" and warmstart_per_episode is not None:
        wp = np.asarray(warmstart_per_episode, dtype=float)[:n_pretrain_eps]
        average_returns[:n_pretrain_eps] = wp
        # real spread across the tau warm-start episodes (broadcast as a per-step proxy band)
        std_returns[:n_pretrain_eps] = float(np.std(wp)) if wp.size > 1 else 0.0
    else:  # "constant": flat line at the LLM mean, no learning happens here
        average_returns[:n_pretrain_eps] = avg_offline_returns[:n_pretrain_eps]
        std_returns[:n_pretrain_eps] = np.zeros(n_pretrain_eps)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_metric_window.py -v`
Expected: PASS (both).

- [ ] **Step 6: Commit**

```bash
git add notebook_utils.py vis_utils.py tests/test_metric_window.py
git commit -m "fix(F5): warm-start constant over exactly tau + per-episode variant (AUC-identical)"
```

---

## Task 6 (F1-eval): Off-by-one in the evaluation/coverage dataset

**Files:**
- Modify: `utils.py:80-120` (`evaluate_qlearning_with_environment`)
- Test: extend `tests/test_collection_alignment.py` (created in Task 7) — but add a focused check here.

**Interfaces:**
- Produces: the returned `MDPDataset` stores `(s_t, a_t, r_t)` with `s_t` the state the action was taken in, and includes the reset state. The scalar mean return is unchanged.

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_alignment.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eval_alignment.py -v`
Expected: FAIL (current code appends `observation` after `env.step`, so `observations[0]` is the successor state `1.0`, and the reset state is dropped).

- [ ] **Step 3: Fix the append order**

In `utils.py` `evaluate_qlearning_with_environment`, restructure the loop (lines 86-111) so the pre-step observation is stored before stepping:
```python
        while True:
            if isinstance(observation, np.ndarray):
                obs_input = np.expand_dims(observation, axis=0)
            elif isinstance(observation, (tuple, list)):
                obs_input = [np.expand_dims(o, axis=0) for o in observation]
            else:
                raise ValueError(f"Unsupported observation type: {type(observation)}")
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = algo.predict(obs_input)[0]
            observations.append(observation)   # s_t: state the action was taken in
            actions.append(action)             # a_t
            observation, reward, done, truncated, _ = env.step(action)
            episode_reward += float(reward)
            count += 1
            rewards.append(reward)             # r_t
            if count >= max_episode_len:
                done = True
            terminals.append(int(done or truncated))
            if done or truncated:
                break
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_eval_alignment.py -v`
Expected: PASS. Also run the Task-4 smoke check again to confirm the scalar return is still sane.

- [ ] **Step 5: Commit**

```bash
git add utils.py tests/test_eval_alignment.py
git commit -m "fix(F1-eval): pair action with pre-step state in eval/coverage dataset"
```

---

## Task 7 (F1-collection): Off-by-one in LLM data collection

**Files:**
- Modify: `llm_main.py:245-291` (rollout loop + dataset build)
- Test: `tests/test_collection_alignment.py`

**Interfaces:**
- Produces: collected `MDPDataset` where `observations[k]` is the state `agent.act` saw when choosing `actions[k]`, `rewards[k]` is the reward for that action, and the reset state `s_0` is included. This is the offline dataset consumed by `pretrain_from_llm`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_collection_alignment.py`. It exercises the *exact* pairing logic by importing a small refactored helper (Step 3 extracts the loop into `collect_episode` so it is unit-testable):
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_collection_alignment.py -v`
Expected: FAIL with `ImportError: cannot import name 'collect_episode'`.

- [ ] **Step 3: Extract a corrected, testable `collect_episode`**

In `llm_main.py`, add this function at module scope (above `if __name__ == "__main__":`):
```python
def collect_episode(env, agent, env_name, max_episode_len, eps):
    """Roll out one episode; return aligned (observations, actions, rewards, terminals).

    observations[k] is the state where actions[k] was chosen (pre-step), rewards[k] the
    resulting reward, terminals[k] the done flag after the step. s_0 is included.
    """
    observations, actions, rewards, terminals = [], [], [], []
    observation, _info = env.reset()
    done = False
    n_step = 0
    while not done:
        if bool(np.random.binomial(n=1, p=eps)):
            action = env.action_space.sample()
        else:
            action = agent.act(observation)
        observations.append(observation)   # s_t (pre-step)
        actions.append(action)             # a_t
        next_observation, reward, done, _info = env.step(action)
        if "Cliff" in env_name or "Frozen" in env_name:
            agent.add_env_hist(next_observation, reward, action)
        agent.assign_reward(reward)
        rewards.append(reward)             # r_t
        n_step += 1
        if n_step >= max_episode_len:
            done = True
        terminals.append(int(done))
        observation = next_observation
    return observations, actions, rewards, terminals
```
Then replace the inline rollout loop (currently lines 247-268, inside `for episode in trange(...)`) so it calls the helper and extends the master lists:
```python
    for episode in trange(hyperparams["n_episodes"]):
        ep_obs, ep_act, ep_rew, ep_term = collect_episode(
            env, agent, hyperparams["env"], hyperparams["max_episode_len"], hyperparams["eps"]
        )
        observations += ep_obs
        actions += ep_act
        rewards += ep_rew
        terminals += ep_term
        train_stats = agent.terminate_episode(train=hyperparams["SFT"])
        counter += 1
```
(Keep the `observations, actions, rewards, terminals = [], [], [], []` init at line 245 and the `MDPDataset` build at 285-291 unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_collection_alignment.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add llm_main.py tests/test_collection_alignment.py
git commit -m "fix(F1-collection): pair action with pre-step state; include s0 (was s_{t+1})"
```

---

## Task 8 (F2): Robust, tested action parsing

**Files:**
- Create: `env/action_parsing.py`
- Modify: `llamagym/agent.py:105-125` (response-span extraction; B3), `env/translation_agent.py` (route all `extract_action` through the helper)
- Test: `tests/test_action_parsing.py`

**Interfaces:**
- Produces:
  - `env/action_parsing.py`: `extract_generated_text(full_decoded: str, prompt_text: str) -> str` and `extract_choice_index(generated_text: str, valid_1based: list[int], keyword_map: dict[str,int] | None = None) -> tuple[int, bool]` returning `(action_1based, is_fallback)`. On failure `is_fallback=True` and the caller substitutes a **random valid** action, incrementing a counter.
  - Module-level `PARSE_FAILURES` counter (int) + `reset_parse_failures()` for per-run reporting.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_action_parsing.py`:
```python
from env.action_parsing import extract_choice_index, extract_generated_text, reset_parse_failures, PARSE_FAILURES_GET


def test_prefers_explicit_final_choice_not_stray_digits():
    # Stray coordinate "3" appears mid-reasoning; the final answer is action 2.
    text = "I am at cell 3, moving toward the goal. Final answer: 2"
    idx, fb = extract_choice_index(text, valid_1based=[1, 2], keyword_map={"left": 1, "right": 2})
    assert idx == 2 and fb is False


def test_keyword_fallback_is_case_insensitive_and_reachable():
    text = "I will go Left to avoid the cliff."   # no digit present
    idx, fb = extract_choice_index(text, valid_1based=[1, 2], keyword_map={"left": 1, "right": 2})
    assert idx == 1 and fb is False


def test_hard_failure_flags_fallback():
    reset_parse_failures()
    idx, fb = extract_choice_index("banana", valid_1based=[1, 2], keyword_map={"left": 1, "right": 2})
    assert fb is True
    assert idx in (1, 2)
    assert PARSE_FAILURES_GET() == 1


def test_generated_text_strips_prompt_for_chatml():
    full = "<|im_start|>system\nchoose 1 or 2<|im_end|>\n<|im_start|>assistant\nAnswer: 2"
    prompt = "<|im_start|>system\nchoose 1 or 2<|im_end|>\n<|im_start|>assistant\n"
    gen = extract_generated_text(full, prompt)
    assert gen.strip() == "Answer: 2"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_action_parsing.py -v`
Expected: FAIL (`env/action_parsing.py` does not exist).

- [ ] **Step 3: Implement the parser**

Create `env/action_parsing.py`:
```python
"""Robust action extraction shared by all TranslationAgents.

Parses from the model's GENERATED span only (not the prompt), prefers an explicit
final answer, then a case-insensitive keyword, then a last-resort random-valid action
(flagged and counted so the run can report a parse-failure rate).
"""
import re
import numpy as np

_PARSE_FAILURES = 0


def reset_parse_failures():
    global _PARSE_FAILURES
    _PARSE_FAILURES = 0


def PARSE_FAILURES_GET():
    return _PARSE_FAILURES


def extract_generated_text(full_decoded: str, prompt_text: str) -> str:
    """Return only the newly generated text. Works for ChatML (Qwen/DeepSeek) and Llama."""
    if prompt_text and full_decoded.startswith(prompt_text):
        return full_decoded[len(prompt_text):]
    for marker in ("<|im_start|>assistant", "[/INST]", "<|assistant|>"):
        if marker in full_decoded:
            return full_decoded.split(marker)[-1]
    return full_decoded


def extract_choice_index(generated_text: str, valid_1based, keyword_map=None):
    """Return (action_1based, is_fallback). valid_1based e.g. [1,2,3,4]."""
    global _PARSE_FAILURES
    valid_str = {str(v) for v in valid_1based}
    # 1) explicit "answer/action: N" or "final answer N"
    m = re.findall(r"(?:answer|action|choice)\D{0,10}(\d+)", generated_text, flags=re.I)
    for cand in reversed(m):
        if cand in valid_str:
            return int(cand), False
    # 2) last valid standalone digit token
    tokens = re.findall(r"\d+", generated_text)
    for cand in reversed(tokens):
        if cand in valid_str:
            return int(cand), False
    # 3) case-insensitive keyword
    if keyword_map:
        low = generated_text.lower()
        best_pos, best_val = -1, None
        for kw, val in keyword_map.items():
            pos = low.rfind(kw.lower())
            if pos > best_pos and val in valid_1based:
                best_pos, best_val = pos, val
        if best_val is not None:
            return best_val, False
    # 4) hard failure -> random valid action, flagged + counted
    _PARSE_FAILURES += 1
    return int(np.random.choice(list(valid_1based))), True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_action_parsing.py -v`
Expected: PASS (all four).

- [ ] **Step 5: Route the response-span fix in llamagym**

In `llamagym/agent.py`, replace the `[/INST]`-only extraction (line 111) so it uses the shared helper. Find the line:
```python
        response = outputs[0].split("[/INST]")[-1].strip()
```
and replace with:
```python
        from env.action_parsing import extract_generated_text
        prompt_text = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False) \
            if hasattr(self, "tokenizer") and "inputs" in dir() else ""
        response = extract_generated_text(outputs[0], prompt_text).strip()
```
If the local variable holding the tokenized prompt is named differently, decode that tensor instead; the goal is `prompt_text` = the exact decoded prompt so the helper strips it. (Inspect `agent.py:95-111` for the actual variable name used when building `outputs`.)

- [ ] **Step 6: Route one agent (CartPole) through the helper as the worked example**

In `env/translation_agent.py`, replace `CartPoleAgent.extract_action` (lines 332-354) with:
```python
class CartPoleAgent(TranslationAgent):
    def extract_action(self, response: str) -> gym.core.ActType:
        from env.action_parsing import extract_choice_index
        out, _fb = extract_choice_index(response, valid_1based=[1, 2],
                                        keyword_map={"left": 1, "right": 2})
        return out - 1  # LLM biased toward 0; elicit 1-based then shift to 0-based
```

- [ ] **Step 7: Route the remaining agents identically**

Apply the same pattern to each agent's `extract_action` in `env/translation_agent.py`, using its existing valid range and action words. Exact per-env parameters (do each; they are near-identical to CartPole):

| Agent (line) | `valid_1based` | `keyword_map` | return |
|---|---|---|---|
| `MountainCarAgent` (~378) | `[1,2,3]` | `{"left":1,"not":2,"right":3}` | `out - 1` |
| `FrozenLakeAgent` (~481) | `[1,2,3,4]` | `{"left":1,"down":2,"right":3,"up":4}` | `out - 1` |
| `CliffWalkingAgent` (~547) | `[1,2,3,4]` | `{"up":1,"right":2,"down":3,"left":4}` | `out - 1` |
| `PongAgent` (~300) | `[1,2,3,4,5,6]` | `{"noop":1,"fire":2,"right":3,"left":4}` | `out - 1` |
| `TaxiAgent` (~581) | `[1,2,3,4,5,6]` | `{"south":1,"north":2,"east":3,"west":4,"pickup":5,"dropoff":6}` | `out - 1` |
| `AcrobotAgent` (~358) | `[1,2,3]` | `{"left":1,"none":2,"right":3}` | `out - 1` |

Keep each agent's existing action-word semantics if they differ from the table above — check the surrounding describer text in the corresponding `env/*/*_translator.py` and match the words the prompt actually uses. Pendulum stays as-is (continuous; already bounded/clamped — do not change).

- [ ] **Step 8: Run the full parsing test + a syntax import check**

Run:
```bash
python -m pytest tests/test_action_parsing.py -v
python -c "import env.translation_agent"   # import must succeed
```
Expected: tests PASS; import prints nothing (no error).

- [ ] **Step 9: Commit**

```bash
git add env/action_parsing.py env/translation_agent.py llamagym/agent.py tests/test_action_parsing.py
git commit -m "fix(F2): robust shared action parsing; Qwen-aware response span; random-on-fail + counter"
```

---

## Task 9 (F6): Paired statistics for significance

**Files:**
- Create: `stats_utils.py`
- Modify: `extract_cumulative_rewards_table.py` (add a paired method−baseline column with CI + p)
- Test: `tests/test_stats_utils.py`

**Interfaces:**
- Produces: `paired_bootstrap_diff(method_per_seed, baseline_per_seed, n_boot=10000, seed=0) -> {"mean_diff", "ci_low", "ci_high", "p_value"}` (two-sided paired sign-flip / bootstrap p over per-seed differences).

- [ ] **Step 1: Write the failing test**

Create `tests/test_stats_utils.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_stats_utils.py -v`
Expected: FAIL (`stats_utils` missing).

- [ ] **Step 3: Implement the stats**

Create `stats_utils.py`:
```python
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
    return {"mean_diff": mean_diff, "ci_low": float(ci_low),
            "ci_high": float(ci_high), "p_value": p_value}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_stats_utils.py -v`
Expected: PASS.

- [ ] **Step 5: Surface per-seed values from the table pipeline**

In `extract_cumulative_rewards_table.py`, `extract_cumulative_rewards_for_env` currently reduces each baseline to `float(np.mean(arr))` (line 230). Add, alongside the scalar, the per-seed vector so downstream code can run paired tests. Change lines 224-231 to also collect per-seed arrays where the cache exposes them (the pretrain/online caches are per-seed keyed `..._{i}`); expose:
```python
    results, per_seed = {}, {}
    for label, key in baseline_specs:
        if key not in cache:
            results[label] = None; per_seed[label] = None; continue
        arr = np.asarray(cache[key])
        results[label] = float(np.mean(arr))
        per_seed[label] = arr  # full curve; caller reduces per seed as needed
    return results, per_seed, None
```
Update the single caller in `main()` (line 240) to unpack three values and, after building `rows`, print a paired-difference block: for each env, `paired_bootstrap_diff(<LLM-pretrain per-seed AUC>, <Online RL per-seed AUC>)` and show `mean_diff [ci_low, ci_high] p=...`. (The per-seed AUC per method is obtained by re-running `extract_data` with `use_iqm=False` and aggregating per seed, or by reading the per-seed cache arrays directly; wire whichever the cache structure exposes — inspect `extract_data` return keys first.)

- [ ] **Step 6: Commit**

```bash
git add stats_utils.py extract_cumulative_rewards_table.py tests/test_stats_utils.py
git commit -m "feat(F6): paired bootstrap method-baseline difference with CI and p-value"
```

---

## Task 10 (F8): Response-cached LLM generation (local primary; API optional fallback)

**Files:**
- Create: `llm_client.py`
- Modify: `llm_main.py` (wrap generation in the disk cache; `--backend local` is the default and primary path — runs on the a local GPU for 7B and a rented 24 GB 4090 for 32B in 4-bit; `--backend api` is an optional fallback)
- Test: `tests/test_llm_client_cache.py`

**Interfaces:**
- Produces: `class LLMClient(model, backend, cache_dir="data/llm_cache")` with `.chat(messages: list[dict], temperature, top_p, max_new_tokens, seed) -> str`, disk-cached by a hash of `(model, messages, sampling)`. `backend in {"local","api"}` — **`local` is primary** (wraps the existing `transformers` generation with 4-bit quantization); `api` uses an OpenAI-compatible endpoint (`LLM_API_BASE`, `LLM_API_KEY`) and is an optional fallback only. The cache wraps both backends, so re-parsing/re-alignment needs no new generation.

- [ ] **Step 1: Write the failing test (cache hit needs no network)**

Create `tests/test_llm_client_cache.py`:
```python
from llm_client import LLMClient


def test_cache_roundtrip(tmp_path):
    c = LLMClient(model="test-model", backend="api", cache_dir=str(tmp_path))
    msgs = [{"role": "user", "content": "hi"}]
    key = c._cache_key(msgs, temperature=0.9, top_p=0.6, max_new_tokens=10, seed=0)
    c._cache_write(key, "cached-response")
    # A cached key must be returned without calling the backend.
    got = c.chat(msgs, temperature=0.9, top_p=0.6, max_new_tokens=10, seed=0)
    assert got == "cached-response"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_llm_client_cache.py -v`
Expected: FAIL (`llm_client` missing).

- [ ] **Step 3: Implement the client**

Create `llm_client.py`:
```python
import os
import json
import hashlib


class LLMClient:
    def __init__(self, model, backend="api", cache_dir="data/llm_cache"):
        self.model = model
        self.backend = backend
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_key(self, messages, temperature, top_p, max_new_tokens, seed):
        payload = json.dumps({
            "model": self.model, "messages": messages, "temperature": temperature,
            "top_p": top_p, "max_new_tokens": max_new_tokens, "seed": seed,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cache_path(self, key):
        return os.path.join(self.cache_dir, key + ".json")

    def _cache_read(self, key):
        p = self._cache_path(key)
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)["response"]
        return None

    def _cache_write(self, key, response):
        with open(self._cache_path(key), "w") as f:
            json.dump({"response": response}, f)

    def chat(self, messages, temperature=0.9, top_p=0.6, max_new_tokens=2000, seed=0):
        key = self._cache_key(messages, temperature, top_p, max_new_tokens, seed)
        cached = self._cache_read(key)
        if cached is not None:
            return cached
        if self.backend == "api":
            resp = self._call_api(messages, temperature, top_p, max_new_tokens, seed)
        else:
            resp = self._call_local(messages, temperature, top_p, max_new_tokens, seed)
        self._cache_write(key, resp)
        return resp

    def _call_api(self, messages, temperature, top_p, max_new_tokens, seed):
        from openai import OpenAI
        client = OpenAI(base_url=os.environ["LLM_API_BASE"], api_key=os.environ["LLM_API_KEY"])
        out = client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature,
            top_p=top_p, max_tokens=max_new_tokens, seed=seed,
        )
        return out.choices[0].message.content

    def _call_local(self, messages, temperature, top_p, max_new_tokens, seed):
        raise NotImplementedError("Local backend wired in llm_main.py's existing transformers path")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_llm_client_cache.py -v`
Expected: PASS (cache hit, no network).

- [ ] **Step 5: Add `openai` dep and a `--backend` flag**

Add to `pyproject.toml` dependencies: `openai = "^1.0"`. In `llm_main.py` argparse block (after line 315) add:
```python
    parser.add_argument("--backend", type=str, default="local", choices=["local", "api"],
                        help="LLM backend: local transformers or OpenAI-compatible API")
```
Wire `hyperparams["backend"] = args.backend`. Where the agent currently generates via the local model, branch: if `backend == "api"`, use `LLMClient(model=args.model_name, backend="api").chat(...)`; else keep the existing transformers generation. (Inspect `TranslationAgent.act`/`get_response` in `env/translation_agent.py` / `llamagym/agent.py` for the generation call site to branch cleanly, or inject the client into the agent.)

- [ ] **Step 6: Commit**

```bash
git add llm_client.py llm_main.py pyproject.toml tests/test_llm_client_cache.py
git commit -m "feat(F8): cached LLM client (API + local) keyed by model+messages+sampling"
```

---

## Task 11: Re-collect LLM datasets with the fixed pipeline

**Files:**
- Uses: `llm_main.py` (fixed), `llm_client.py`, `env/action_parsing.py`
- Produces: `data/{Env}_{model}_Neps_30.pkl` (regenerated) + a parse-failure-rate log per run.

**Interfaces:**
- Consumes: self-hosted GPUs — a local GPU for 7B, rented 24 GB RTX 4090 for 32B in 4-bit (`--backend local --quantization 4bit`). No API required.

- [ ] **Step 1: Add a parse-failure report to the collection run**

In `llm_main.py`, before the collection loop call `env.action_parsing.reset_parse_failures()`, and after building the dataset print/log `PARSE_FAILURES_GET()` and the failure rate = failures / total_steps. Persist it next to the pkl as `data/{name}_parsefail.json`.

- [ ] **Step 2: Dry-run one env, 1 episode, to validate end-to-end (7B local)**

Run:
```bash
python llm_main.py --env CartPole-v0 --model_name Qwen/Qwen3-8B --n_episodes 1 --backend local
```
Expected: completes; writes a pkl; parse-failure rate printed and is low (<5%). Load it and assert alignment:
```bash
python -c "import pickle,numpy as np; ds=pickle.load(open('data/CartPole_Qwen3-8B_Neps_1.pkl','rb')); \
ep=ds.episodes[0]; print('len',len(ep.observations),'first_obs',ep.observations[0])"
```
Expected: prints a plausible first observation (the reset state), not a successor.

- [ ] **Step 3: Full re-collection (parallelizable across envs/models)**

For each env in `[CartPole-v0, Pendulum-v1, MountainCar-v0, FrozenLake-v1, CliffWalking-v0, RepresentedPong-v0]`:
- 7B (a local GPU or rented 4090): `python llm_main.py --env <ENV> --model_name Qwen/Qwen3-8B --n_episodes 30 --backend local --quantization 4bit`
- 32B (rented 24 GB 4090, 4-bit): `python llm_main.py --env <ENV> --model_name Qwen/Qwen3-32B --n_episodes 30 --backend local --quantization 4bit`

Run these as parallel jobs — one process per GPU, several rented 4090s at once across (env × model). Each writes `data/{Env}_{model}_Neps_30.pkl` + a parsefail json. (Verify 32B-4bit fits in 24 GB with the configured `max_new_tokens`; if VRAM-tight, lower `max_new_tokens` or context, or fall back to an 8 GB-larger rental.)

- [ ] **Step 4: Validate every collected dataset**

Run a validation script that, for each new pkl, asserts: (a) parse-failure rate < 10%, (b) `observations[0]` per episode equals the env reset distribution (not a mid-episode state), (c) episode count == 30. Print a summary table. Any env failing (a) or (b) is re-run before proceeding.

- [ ] **Step 5: Point `get_llm_data_paths` at the new files and commit the manifest**

Update `utils.get_llm_data_paths` (lines 12-23) to the new canonical filenames (Qwen3-8B / Qwen3-32B), keeping Qwen2.5 paths available for the backward-comparison point. Commit a `data/COLLECTION_MANIFEST.md` recording model, date, parse-failure rate, and seed for each dataset (do **not** commit large pkls if the repo's .gitignore excludes `data/*.pkl` — verify and follow the existing convention).

```bash
git add utils.py data/COLLECTION_MANIFEST.md
git commit -m "data: re-collect LLM datasets (Qwen3-8B local / Qwen3-32B API) with fixed pipeline"
```

---

## Task 12: Re-run the pipeline, regenerate tables, produce the delta report

**Files:**
- Uses: `online_main.py`, `extract_cumulative_rewards_table.py` (fixed), `tools/delta_report.py`
- Produces: regenerated caches + a delta report vs the frozen Phase-0 baseline.

- [ ] **Step 1: Bump seeds and eval trials for the significance run**

In `online_main.py`, set `n_exp` from 5 → the target (10–20; start with 10 to bound compute). Confirm `EVAL_N_TRIALS=20` (Task 4) is active. Keep τ (`n_pretrain_eps`) and `n_pretrain_steps` at the paper defaults for the primary comparison.

- [ ] **Step 2: Re-run all envs (parallel across env × seed)**

For each env, run the full `online_main.py` pipeline (LLM-data-pretrain, Mix, Online RL, plus the baselines). RL is CPU/CPU-light and embarrassingly parallel; launch per-env jobs. Confirm caches `cache_{env}_Neps_{10,20,30}.pkl` are written.

- [ ] **Step 3: Write the delta report tool**

Create `tools/delta_report.py`:
```python
"""Regenerate the table and diff it against the frozen pre-fix baseline."""
import subprocess, sys

BASE = "docs/superpowers/baselines/2026-06-30-pre-fix-table.txt"

if __name__ == "__main__":
    new = subprocess.run([sys.executable, "extract_cumulative_rewards_table.py"],
                         capture_output=True, text=True).stdout
    with open(BASE) as f:
        old = f.read()
    print("=== PRE-FIX ===\n" + old)
    print("=== POST-FIX ===\n" + new)
    print("=== (compare Average rows and per-env LLM-pretrain vs Online RL) ===")
```

- [ ] **Step 4: Generate the post-fix table + paired stats**

Run: `python tools/delta_report.py`
Expected: prints pre-fix and post-fix tables side by side. Record: does the LLM-pretrain − Online RL gap change (esp. on non-MountainCar envs) after the bug fixes, and are the paired p-values (Task 9) now interpretable?

- [ ] **Step 5: Commit the regenerated results + a short findings note**

Write `docs/superpowers/baselines/2026-06-30-post-fix-findings.md` summarizing: the new Average row, per-env paired differences with CIs/p-values, parse-failure rates, and whether fixing B1/B2 widened the gap (the key hypothesis). This note feeds the Phase 2 plan.
```bash
git add tools/delta_report.py docs/superpowers/baselines/2026-06-30-post-fix-findings.md
git commit -m "results: regenerate post-fix tables + delta/findings vs frozen baseline"
```

---

## Self-Review

**Spec coverage (spec §4 Phase 0 + Phase 1):**
- Phase 0 repro harness → Task 1. ✓
- F1 (off-by-one, collection + eval) → Task 7 + Task 6. ✓
- F2 (parsing/fallback + Qwen span) → Task 8. ✓
- F3 (empirical normalization) → Task 3. ✓
- F4 (multi-trial eval) → Task 4. ✓
- F5 (warm-start constant-over-τ + per-episode) → Task 5. ✓
- F6 (paired stats) → Task 9. ✓
- F7 (SAC LR) → Task 2. ✓
- Re-collect via API/local client → Task 10 (client) + Task 11 (re-collect). ✓
- Re-run + delta report → Task 12. ✓
- D1 (FrozenLake oracle history) — **disclosure/removal deferred to Phase 2 (reframe/writeup)**; not a code fix here. Noted as a gap intentionally left to Phase 2.
- B3 (Qwen response span) folded into F2/Task 8. ✓ B5 (SAC reporting) into Task 2. ✓ B7 (data-selection glob) partially addressed by Task 11 manifest + Task 3 loader; the `resolve_llm_path` glob fallback hardening is minor and can ride along in Task 11 Step 5.

**Placeholder scan:** No "TBD"/"handle edge cases". Two tasks (9 Step 5, 10 Step 5) instruct the implementer to *inspect a specific call site first* because the exact local variable/return-key name must be read from code — these name the file and line range to inspect and the transformation to apply, which is concrete, not a placeholder.

**Type consistency:** `extract_choice_index -> (int, bool)`, `paired_bootstrap_diff -> dict(mean_diff, ci_low, ci_high, p_value)`, `LLMClient.chat -> str`, `collect_episode -> (list, list, list, list)`, `load_norm_constants -> dict`, `EVAL_N_TRIALS: int` — all used consistently across tasks and tests.

**Known follow-through:** Task 11 Step 3 / Task 12 Step 2 are multi-hour compute runs (parallelized); they are execution steps validated by explicit checks, not unit tests. This is expected for experiment tasks.
