#!/usr/bin/env python3
"""
Extract average episode reward for each of the six environments and each baseline
using extract_data from vis_utils, then print a table.

Baselines: LLM-pretrain, Online RL, Qwen 7B, Qwen 32B, Random, Mix w/o pretrain,
           Pretrain w/ Online RL data, Pretrain w/ random policy data.

Usage (run from project root with the same conda/env as the notebooks):
  python extract_cumulative_rewards_table.py

Requires: cache_<Env>_Neps_10.pkl, cache_<Env>_Neps_20.pkl, cache_<Env>_Neps_30.pkl
(and on_policy_pretrain_exp / on_policy_pretrain_exp_rand) for each env.
Pendulum uses model_type ddpg (cache_*_ddpg.pkl). Envs missing caches are skipped.
"""

import os
import sys
import pickle
import numpy as np
import gymnasium as gym

# Add project root for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vis_utils import extract_data, DEFAULT_NORM_CONST_DICT, get_pretrain_defaults

# Six environments: (full env name, model type for cache suffix)
ENVS = [
    ("CartPole-v0", "default"),
    ("CliffWalking-v0", "default"),
    ("FrozenLake-v1", "default"),
    ("MountainCar-v0", "default"),
    ("Pendulum-v1", "default"),
    ("RepresentedPong-v0", "default"),
]

# Keep this aligned with visualization.ipynb plot_mix(model_size=...)
MIX_PRETRAIN_MODEL_SIZE = "32b"


def baseline_specs_for_model(model_type, mix_pretrain_model_size=MIX_PRETRAIN_MODEL_SIZE):
    """Baseline display names and cache keys; pretrain size/step depend on model (ddpg vs default)."""
    if mix_pretrain_model_size not in ("7b", "32b"):
        raise ValueError("mix_pretrain_model_size must be '7b' or '32b'")
    pretrain_size, pretrain_step = get_pretrain_defaults(model_type)
    return [
        ("LLM-pretrain", f"mean_pretrain_{mix_pretrain_model_size}_{pretrain_step}_pre_{pretrain_size}"),
        ("Online RL", "mean_onl"),
        ("Qwen 7B", "Qwen_7B"),
        ("Qwen 32B", "Qwen_32B"),
        ("Random", "mean_random"),
        ("Mix w/o pretrain", f"mean_mix_{mix_pretrain_model_size}_pre_{pretrain_size}"),
        ("Pretrain w/ Online RL data", f"mean_on_pol_pretrain_{pretrain_step}_pre_{pretrain_size}"),
        ("Pretrain w/ random policy data", f"mean_rand_pretrain_{pretrain_step}_pre_{pretrain_size}"),
    ]


def n_episodes_for_env(env_name):
    base = env_name.split("-")[0]
    if base in ["CartPole", "FrozenLake"]:
        return 150
    if base in ["Pendulum", "CliffWalking", "RepresentedPong"]:
        return 200
    if base == "MountainCar":
        return 300
    return 200


def make_env(env_name):
    if "Represented" in env_name:
        from env.atari.represented_atari_game import GymCompatWrapper2
        return GymCompatWrapper2(gym.make(env_name))
    try:
        raw = gym.make(env_name)
        if isinstance(raw.observation_space, gym.spaces.Discrete):
            from online_main import OneHotWrapper
            return OneHotWrapper(raw)
    except Exception:
        pass
    return gym.make(env_name)


def get_random_mean_reward(env_name, n_episodes, max_episode_len=200, n_trials=30, seed=42069):
    """Evaluate random policy and return constant curve (mean per episode)."""
    try:
        env = make_env(env_name)
        env.reset(seed=seed)
        random_rewards = []
        for _ in range(n_trials):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            count = 0
            while not done:
                action = env.action_space.sample()
                obs, reward, done, _, _ = env.step(action)
                total_reward += reward
                count += 1
                if count >= max_episode_len:
                    break
            random_rewards.append(total_reward)
        mean_val = float(np.mean(random_rewards))
        return np.ones(n_episodes) * mean_val
    except Exception as e:
        # Fallback: use x_random from norm const as placeholder so extract_data can run
        env_base = env_name.split("-")[0]
        norm = DEFAULT_NORM_CONST_DICT.get(env_base, {}).get("x_random", 0.0)
        return np.ones(n_episodes) * norm


def resolve_llm_path(path, env_base, size_key="7B"):
    """If path does not exist, try to find a matching pkl in data/."""
    if path is None:
        return None
    if os.path.isfile(path):
        return path
    import glob
    # e.g. data/CartPole_Qwen2.5-7B-Instruct_Neps_30.pkl -> CartPole*7B*Neps_30*.pkl
    pattern = os.path.join("data", f"{env_base}_*{size_key}*_Neps_30*.pkl")
    candidates = glob.glob(pattern)
    if candidates:
        return candidates[0]
    pattern2 = os.path.join("data", f"{env_base}_*{size_key}*_Neps_30.pkl")
    candidates2 = glob.glob(pattern2)
    return candidates2[0] if candidates2 else None


def load_qwen_curves(env_name, n_episodes, sft=False, long_cot=False):
    """Load Qwen 7B and 32 B constant curves (mean reward per episode)."""
    try:
        from utils import get_llm_data_paths
    except Exception:
        return np.ones(n_episodes) * 0.0, np.ones(n_episodes) * 0.0
    path_7b, path_32b = get_llm_data_paths(env_name, sft=sft, long_cot=long_cot)
    env_base = env_name.split("-")[0]
    path_7b = resolve_llm_path(path_7b, env_base, "7B")
    path_32b = resolve_llm_path(path_32b, env_base, "32B")
    Qwen_7B = np.ones(n_episodes) * 0.0  # fallback
    Qwen_32B = np.ones(n_episodes) * 0.0
    if path_7b:
        try:
            with open(path_7b, "rb") as f:
                ds = pickle.load(f)
            rewards = [ds.episodes[i].compute_return() for i in range(min(30, len(ds.episodes)))]
            Qwen_7B = np.ones(n_episodes) * np.mean(rewards)
        except Exception as e:
            print(f"  Warning: could not load Qwen 7B for {env_base}: {e}", file=sys.stderr)
    if path_32b:
        try:
            with open(path_32b, "rb") as f:
                ds = pickle.load(f)
            rewards = [ds.episodes[i].compute_return() for i in range(min(30, len(ds.episodes)))]
            Qwen_32B = np.ones(n_episodes) * np.mean(rewards)
        except Exception as e:
            print(f"  Warning: could not load Qwen 32B for {env_base}: {e}", file=sys.stderr)
    return Qwen_7B, Qwen_32B


def load_ds_curves_if_frozenlake(env_name, n_episodes):
    """Load DS 7B / 14 B only for FrozenLake (optional)."""
    env_base = env_name.split("-")[0]
    if env_base != "FrozenLake":
        return None, None
    path_7b = "data/FrozenLake_DeepSeek-R1-Distill-Qwen-7B_Neps_30_20250502071148.pkl"
    path_14b = "data/FrozenLake_DeepSeek-R1-Distill-Qwen-14B_Neps_30_20250502084016.pkl"
    for p in [path_7b, "data/FrozenLake_DeepSeek-R1-Distill-Qwen-7B_Neps_30.pkl"]:
        if os.path.isfile(p):
            path_7b = p
            break
    for p in [path_14b, "data/FrozenLake_DeepSeek-R1-Distill-Qwen-14B_Neps_30.pkl"]:
        if os.path.isfile(p):
            path_14b = p
            break
    DS_7B, DS_14B = None, None
    try:
        if os.path.isfile(path_7b):
            with open(path_7b, "rb") as f:
                ds = pickle.load(f)
            rewards = [ds.episodes[i].compute_return() for i in range(min(30, len(ds.episodes)))]
            DS_7B = np.ones(n_episodes) * np.mean(rewards)
    except Exception:
        pass
    try:
        if os.path.isfile(path_14b):
            with open(path_14b, "rb") as f:
                ds = pickle.load(f)
            rewards = [ds.episodes[i].compute_return() for i in range(min(30, len(ds.episodes)))]
            DS_14B = np.ones(n_episodes) * np.mean(rewards)
    except Exception:
        pass
    return DS_7B, DS_14B


def extract_cumulative_rewards_for_env(env_name, model_type, n_exp=5, max_episode_len=200):
    """Run extract_data for one environment and return dict of baseline_name -> cumulative reward."""
    env_base = env_name.split("-")[0]
    n_episodes = n_episodes_for_env(env_name)
    hyperparams = {
        "env": env_name,
        "n_exp": n_exp,
        "n_episodes": n_episodes,
        "max_episode_len": max_episode_len,
        "norm_const_dict": {**DEFAULT_NORM_CONST_DICT},
        "model": model_type,
    }
    Qwen_7B, Qwen_32B = load_qwen_curves(env_name, n_episodes)
    DS_7B, DS_14B = load_ds_curves_if_frozenlake(env_name, n_episodes)
    if DS_7B is None:
        DS_7B = np.ones(n_episodes) * 0.0
    if DS_14B is None:
        DS_14B = np.ones(n_episodes) * 0.0
    mean_random = get_random_mean_reward(env_name, n_episodes, max_episode_len=max_episode_len)
    try:
        cache = extract_data(
            hyperparams, Qwen_7B, Qwen_32B, DS_7B, DS_14B, mean_random,
            model_type=model_type, use_iqm=True
        )
    except FileNotFoundError as e:
        return None, str(e)
    except Exception as e:
        return None, str(e)
    baseline_specs = baseline_specs_for_model(model_type, MIX_PRETRAIN_MODEL_SIZE)
    results = {}
    for label, key in baseline_specs:
        if key not in cache:
            results[label] = None  # N/A
            continue
        arr = np.asarray(cache[key])
        results[label] = float(np.mean(arr))
    return results, None


def main():
    # Collect (env_display_name -> { baseline -> value })
    env_display_names = [e[0].split("-")[0] for e in ENVS]
    rows = []
    for (env_name, model_type) in ENVS:
        env_display = env_name.split("-")[0]
        res, err = extract_cumulative_rewards_for_env(env_name, model_type)
        if err is not None:
            print(f"Skip {env_display}: {err}", file=sys.stderr)
            rows.append((env_display, None))
            continue
        rows.append((env_display, res))
    # Build table: columns = baselines, rows = environments (average episode reward)
    baseline_names = [s[0] for s in baseline_specs_for_model("default", MIX_PRETRAIN_MODEL_SIZE)]
    env_col_width = max(len("Environment"), max(len(b) for b in env_display_names)) + 2
    # Per-column width: at least 12 for numbers, or full header length
    col_widths = [max(12, len(b)) for b in baseline_names]
    header = "Environment".ljust(env_col_width) + "".join(b.ljust(col_widths[i]) for i, b in enumerate(baseline_names))
    sep_len = len(header)
    print(header)
    print("-" * sep_len)
    for env_display, res in rows:
        if res is None:
            line = env_display.ljust(env_col_width) + "".join("N/A".ljust(col_widths[i]) for i in range(len(baseline_names)))
        else:
            parts = [env_display.ljust(env_col_width)]
            for i, label in enumerate(baseline_names):
                val = res.get(label)
                if val is None:
                    s = "N/A"
                else:
                    s = f"{val:.2f}"
                parts.append(s.ljust(col_widths[i]))
            line = "".join(parts)
        print(line)
    # Average row: mean over all environments per baseline (only over valid numeric values)
    avg_row = {}
    for label in baseline_names:
        vals = [res.get(label) for _, res in rows if res is not None and res.get(label) is not None]
        avg_row[label] = float(np.mean(vals)) if vals else None
    parts = ["Average".ljust(env_col_width)]
    for i, label in enumerate(baseline_names):
        val = avg_row.get(label)
        s = f"{val:.2f}" if val is not None else "N/A"
        parts.append(s.ljust(col_widths[i]))
    print("".join(parts))
    # Also print a markdown-style table
    print("\n--- Markdown table ---\n")
    md_header = "| Environment | " + " | ".join(baseline_names) + " |"
    md_sep = "|" + "---|" * (len(baseline_names) + 1)
    print(md_header)
    print(md_sep)
    for env_display, res in rows:
        if res is None:
            cells = ["N/A"] * len(baseline_names)
        else:
            cells = [f"{res.get(label, 'N/A'):.2f}" if res.get(label) is not None else "N/A" for label in baseline_names]
        print("|", env_display, "|", " | ".join(cells), "|")
    # Average row in markdown table
    avg_cells = [f"{avg_row[label]:.2f}" if avg_row.get(label) is not None else "N/A" for label in baseline_names]
    print("|", "Average", "|", " | ".join(avg_cells), "|")
    return 0


if __name__ == "__main__":
    sys.exit(main())
