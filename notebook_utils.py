"""
Shared helpers for visualization.ipynb and coverage_vis.ipynb.
Import these instead of re-declaring env creation, data loading, and random baseline.
"""

import pickle
import numpy as np
import gymnasium as gym

from utils import get_llm_data_paths


def make_env(env_name):
    """Create env and eval_env with correct wrapper (RepresentedPong, Discrete, or default)."""
    if "Represented" in env_name:
        from env.atari.represented_atari_game import GymCompatWrapper2
        env = GymCompatWrapper2(gym.make(env_name))
        eval_env = GymCompatWrapper2(gym.make(env_name))
    elif isinstance(gym.make(env_name).observation_space, gym.spaces.Discrete):
        from online_main import OneHotWrapper
        env = OneHotWrapper(gym.make(env_name))
        eval_env = OneHotWrapper(gym.make(env_name))
    else:
        env = gym.make(env_name)
        eval_env = gym.make(env_name)
    return env, eval_env


def load_qwen_curves(env_name, n_episodes, n_pretrain_eps=10, sft=False, long_cot=False):
    """
    Load Qwen 7B and 32B datasets and return constant per-episode curves (mean reward).
    Returns: Qwen_7B, Qwen_32B (each is a list of length n_episodes).
    When path_32b is None (e.g. SFT mode), Qwen_32B is set to zeros.
    """
    path_7b, path_32b = get_llm_data_paths(env_name, sft=sft, long_cot=long_cot)
    with open(path_7b, "rb") as f:
        ds_7b = pickle.load(f)
    n_load = min(n_pretrain_eps, len(ds_7b.episodes))
    Qwen_7B_rewards = [ds_7b.episodes[i].compute_return() for i in range(n_load)]
    Qwen_7B = n_episodes * [np.mean(Qwen_7B_rewards)]
    if path_32b is None:
        Qwen_32B = n_episodes * [0.0]
    else:
        with open(path_32b, "rb") as f:
            ds_32b = pickle.load(f)
        n_load_32 = min(n_pretrain_eps, len(ds_32b.episodes))
        Qwen_32B_rewards = [ds_32b.episodes[i].compute_return() for i in range(n_load_32)]
        Qwen_32B = n_episodes * [np.mean(Qwen_32B_rewards)]
    return Qwen_7B, Qwen_32B


def load_ds_and_sft_curves(hyperparams):
    """
    Load DS 7B/14B and optionally SFT (Qwen_7B_SFT) curves when data exists.
    hyperparams must have: env, n_episodes, and optionally n_pretrain_eps (default 30), n_online_eps (default 0).
    Returns: DS_7B, DS_14B, Qwen_7B_SFT (any can be None if load fails).
    """
    env_name = hyperparams["env"].split("-")[0]
    n_episodes = hyperparams["n_episodes"]
    n_pretrain_eps = hyperparams.get("n_pretrain_eps", 30)
    n_online_eps = hyperparams.get("n_online_eps", n_episodes - n_pretrain_eps)

    Qwen_7B_SFT = None
    try:
        if env_name == "Pendulum":
            path_SFT = f"data/{env_name}_Qwen2.5-7B-Instruct_Neps_{n_pretrain_eps}_20250422210707SFT.pkl"
        elif env_name == "CliffWalking":
            path_SFT = f"data/{env_name}_Qwen2.5-7B-Instruct_Neps_{n_pretrain_eps}_20250505135458SFT.pkl"
        elif env_name == "FrozenLake":
            path_SFT = f"data/{env_name}_Qwen2.5-7B-Instruct_Neps_{n_pretrain_eps}_20250501013018SFT.pkl"
        else:
            raise FileNotFoundError("No SFT path for this env")
        with open(path_SFT, "rb") as f:
            ds_sft = pickle.load(f)
        Qwen_7B_SFT_rewards = [ds_sft.episodes[i].compute_return() for i in range(n_pretrain_eps)]
        Qwen_7B_SFT = Qwen_7B_SFT_rewards + n_online_eps * [np.mean(Qwen_7B_SFT_rewards)]
    except Exception:
        pass

    DS_7B, DS_14B = None, None
    try:
        path_DS_7B = f"data/{env_name}_DeepSeek-R1-Distill-Qwen-7B_Neps_{n_pretrain_eps}_20250502071148.pkl"
        path_DS_14B = f"data/{env_name}_DeepSeek-R1-Distill-Qwen-14B_Neps_{n_pretrain_eps}_20250502084016.pkl"
        with open(path_DS_7B, "rb") as f:
            ds_7b = pickle.load(f)
        with open(path_DS_14B, "rb") as f:
            ds_14b = pickle.load(f)
        DS_7B_rewards = [ds_7b.episodes[i].compute_return() for i in range(n_pretrain_eps)]
        DS_14B_rewards = [ds_14b.episodes[i].compute_return() for i in range(n_pretrain_eps)]
        DS_7B = n_episodes * [np.mean(DS_7B_rewards)]
        DS_14B = n_episodes * [np.mean(DS_14B_rewards)]
    except Exception:
        pass

    return DS_7B, DS_14B, Qwen_7B_SFT


def get_mean_random(env, hyperparams, n_trials=30):
    """
    Evaluate random policy for n_trials episodes and return constant curve of length n_episodes.
    """
    n_episodes = hyperparams["n_episodes"]
    max_episode_len = hyperparams.get("max_episode_len", 200)
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
    return np.ones(n_episodes) * np.mean(random_rewards)
