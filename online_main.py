import gymnasium as gym
import numpy as np
import d3rlpy
import pickle
import argparse
import time
import json
import os
from tqdm import trange
from datetime import datetime
from utils import *


def online_training_with_pretrain(hyperparams, explorer, seed, n_pretrain_steps, n_pretrain_eps, n_online_eps):
    dqn = create_d3rlpy_model(hyperparams["env"], hyperparams["batch_size"], hyperparams["learning_rate"], hyperparams["gamma"], hyperparams["target_update_interval"], hyperparams["gpu"], hyperparams["awac"])
    tmp_env, _ = get_env_and_eval_env(hyperparams["env"], seed)
    # Initialize empty FIFO buffer
    temp_buffer = d3rlpy.dataset.ReplayBuffer(
        buffer=d3rlpy.dataset.FIFOBuffer(limit=hyperparams["buffer_size"]),
        env=tmp_env,
    )
    eps_rewards_offline, _, temp_buffer = rollout_and_eval(hyperparams["max_episode_len"], hyperparams["env"], explorer, dqn, temp_buffer, n_pretrain_eps, seed)
    offline_dataset, buffer = buffer_to_dataset(temp_buffer, hyperparams["buffer_size"], n_pretrain_eps, tmp_env)

    dqn.fit(
        buffer,
        n_steps=n_pretrain_steps,
        n_steps_per_epoch=hyperparams["n_steps_per_epoch"],
    )

    eps_rewards_online, final_eval_dataset, _ = rollout_and_eval(hyperparams["max_episode_len"], hyperparams["env"], explorer, dqn, buffer, n_online_eps, seed)
    eps_rewards = eps_rewards_offline + eps_rewards_online
    return eps_rewards, final_eval_dataset, offline_dataset


def online_training_rand(hyperparams, explorer, seed, n_pretrain_steps, n_pretrain_eps, n_online_eps):
    """
    Same as online_training_with_pretrain, but with random actions for the first n_pretrain_eps episodes.
    """
    dqn = create_d3rlpy_model(hyperparams["env"], hyperparams["batch_size"], hyperparams["learning_rate"], hyperparams["gamma"], hyperparams["target_update_interval"], hyperparams["gpu"], hyperparams["awac"])
    random_policy = create_random_model(hyperparams["env"], hyperparams["gpu"])

    tmp_env, _ = get_env_and_eval_env(hyperparams["env"], seed)
    # Initialize empty FIFO buffer
    temp_buffer = d3rlpy.dataset.ReplayBuffer(
        buffer=d3rlpy.dataset.FIFOBuffer(limit=hyperparams["buffer_size"]),
        env=tmp_env,
    )
    eps_rewards_offline, _, temp_buffer = rollout_and_eval(hyperparams["max_episode_len"], hyperparams["env"], explorer, random_policy, temp_buffer, n_pretrain_eps, seed)
    offline_dataset, buffer = buffer_to_dataset(temp_buffer, hyperparams["buffer_size"], n_pretrain_eps, tmp_env)


    dqn.fit(
        buffer,
        n_steps=n_pretrain_steps,
        n_steps_per_epoch=hyperparams["n_steps_per_epoch"],
    )

    eps_rewards_online, final_eval_dataset, _ = rollout_and_eval(hyperparams["max_episode_len"], hyperparams["env"], explorer, dqn, buffer, n_online_eps, seed)
    eps_rewards = eps_rewards_offline + eps_rewards_online
    return eps_rewards, final_eval_dataset, offline_dataset


def run_exp(
    n_pretrain_steps, n_pretrain_eps, n_online_eps, cache, hyperparams, explorer, online_training_fn
):
    for i in range(hyperparams["n_exp"]):
        cache[
            f'pretrain_{n_pretrain_eps}_eps_{n_pretrain_steps}_steps_{i}'
        ], cache[
            f'pretrain_{n_pretrain_eps}_eps_{n_pretrain_steps}_steps_{i}_dataset'
        ], cache[
            f'pretrain_{n_pretrain_eps}_eps_{n_pretrain_steps}_steps_{i}_offline_dataset'
        ] = online_training_fn(hyperparams, explorer, hyperparams["seed"]+i, n_pretrain_steps, n_pretrain_eps, n_online_eps)
    return cache

def run_exp_and_save(
    hyperparams, explorer, is_rand=True
):
    if is_rand:
        online_training_fn = online_training_rand
        suffix = "_rand"
    else:
        online_training_fn = online_training_with_pretrain
        suffix = ""
    n_episodes = hyperparams["n_online_eps"] + hyperparams["n_pretrain_eps"]
    cache = {}
    for n_pretrain_steps in [1000, 3000]:
        for n_pretrain_eps in [10, 20, 30]:
            n_online_eps = n_episodes - n_pretrain_eps
            cache = run_exp(n_pretrain_steps, n_pretrain_eps, n_online_eps, cache, hyperparams, explorer, online_training_fn)

    with open(
        f'data/cache_{hyperparams["env"].split("-")[0]}_on_policy_pretrain_exp{suffix}.pkl',
        "wb",
    ) as file:
        pickle.dump(cache, file)

# def online_training(env, eval_env, hyperparams, explorer=None, model=None):
#     # Load model
#     if model is not None:
#         dqn = model
#     else:
#         dqn = create_d3rlpy_model(hyperparams["env"], hyperparams["batch_size"], hyperparams["learning_rate"], hyperparams["gamma"], hyperparams["target_update_interval"], hyperparams["gpu"])

#     # Initialize empty FIFO buffer
#     buffer = d3rlpy.dataset.ReplayBuffer(
#         buffer=d3rlpy.dataset.FIFOBuffer(limit=hyperparams["buffer_size"]),
#         env=env,
#     )

#     n_pretrain_eps = hyperparams["n_pretrain_eps"]
#     # Load and merge offline data with type-checking
#     if hyperparams["data_path"] is not None:
#         buffer, n_pretrain_eps = load_dataset_to_buffer(buffer, n_pretrain_eps, hyperparams["data_path"], hyperparams["env"])

#     n_rollouts = n_pretrain_eps + hyperparams["n_online_eps"]
#     return rollout_and_eval(hyperparams["max_episode_len"], hyperparams["env"], explorer, dqn, buffer, n_rollouts, hyperparams["seed"])


def _online_training_from_pretrained_model(hyperparams, explorer, cache, model_size, suffix):
    model = pretrain_from_llm(hyperparams["data_path"], hyperparams, model_size, suffix)
    # assert hyperparams["data_path"] is None, "data_path should be None when training from a pretrained model"
    # assert hyperparams["n_pretrain_eps"] == 0, "n_pretrain_eps should be 0 when training from a pretrained model"
    tmp_env, _ = get_env_and_eval_env(hyperparams["env"], hyperparams["seed"]) # just use tmp_env to initialize the buffer

    # Initialize empty FIFO buffer
    buffer = d3rlpy.dataset.ReplayBuffer(
        buffer=d3rlpy.dataset.FIFOBuffer(limit=hyperparams["buffer_size"]),
        env=tmp_env,
    )

    n_rollouts = hyperparams["n_online_eps"] # Only rollout for n_online_eps since the model was pretrained on n_pretrain_eps
    for i in range(hyperparams["n_exp"]):
        cache[f"pretrain_{model_size}_{hyperparams['n_pretrain_steps']}_{i}"], cache[f"pretrain_{model_size}_{hyperparams['n_pretrain_steps']}_{i}_dataset"], _ = rollout_and_eval(hyperparams["max_episode_len"], hyperparams["env"], explorer, model, buffer, n_rollouts, hyperparams["seed"]+i)
    return cache


def _online_training_from_scratch(hyperparams, explorer, cache):
    # assert hyperparams["data_path"] is None, "data_path should be None when training from scratch"
    # assert hyperparams["n_pretrain_eps"] > 0, "n_pretrain_eps should be larger than 0 when training from scratch"
    tmp_env, _ = get_env_and_eval_env(hyperparams["env"], hyperparams["seed"]) # just use tmp_env to initialize the buffer
    model = create_d3rlpy_model(hyperparams["env"], hyperparams["batch_size"], hyperparams["learning_rate"], hyperparams["gamma"], hyperparams["target_update_interval"], hyperparams["gpu"], hyperparams["awac"])
    # Initialize empty FIFO buffer
    buffer = d3rlpy.dataset.ReplayBuffer(
        buffer=d3rlpy.dataset.FIFOBuffer(limit=hyperparams["buffer_size"]),
        env=tmp_env,
    )

    n_rollouts = hyperparams["n_pretrain_eps"] + hyperparams["n_online_eps"]
    for i in range(hyperparams["n_exp"]):
        cache[f"online_{i}"], cache[f"online_{i}_dataset"], _ = rollout_and_eval(hyperparams["max_episode_len"], hyperparams["env"], explorer, model, buffer, n_rollouts, hyperparams["seed"]+i)
    return cache


def _online_training_with_mixed_pretraining_data(hyperparams, explorer, cache):
    # assert hyperparams["data_path"] is not None, "data_path should not be None when training with mixed pretraining data"
    # assert hyperparams["n_pretrain_eps"] > 0, "n_pretrain_eps should be larger than 0 when training with mixed pretraining data"
    tmp_env, _ = get_env_and_eval_env(hyperparams["env"], hyperparams["seed"]) # just use tmp_env to initialize the buffer
    model = create_d3rlpy_model(hyperparams["env"], hyperparams["batch_size"], hyperparams["learning_rate"], hyperparams["gamma"], hyperparams["target_update_interval"], hyperparams["gpu"], hyperparams["awac"])

    # Initialize empty FIFO buffer
    buffer = d3rlpy.dataset.ReplayBuffer(
        buffer=d3rlpy.dataset.FIFOBuffer(limit=hyperparams["buffer_size"]),
        env=tmp_env,
    )

    # Load and merge offline data with type-checking
    buffer, _ = load_dataset_to_buffer(buffer, hyperparams["n_pretrain_eps"], hyperparams["data_path"], hyperparams["env"])
    n_rollouts = hyperparams["n_online_eps"] # Just mix the pretraining data with the online data, so only rollout for n_online_eps

    for i in range(hyperparams["n_exp"]):
        cache[f"mix_32b_{i}"], cache[f"mix_32b_{i}_dataset"], _ = rollout_and_eval(hyperparams["max_episode_len"], hyperparams["env"], explorer, model, buffer, n_rollouts, hyperparams["seed"]+i)
    return cache


def _online_training(hyperparams, explorer, cache, data_path, model_size, suffix, skip_from_scratch=False):
    hyperparams["data_path"] = data_path # Restore data path for mixed pretraining runs
    cache = _online_training_from_pretrained_model(hyperparams, explorer, cache, model_size, suffix)
    cache = _online_training_from_pretrained_model(hyperparams, explorer, cache, model_size, suffix)

    if not skip_from_scratch:
        cache = _online_training_from_scratch(hyperparams, explorer, cache)

    # Mixed pretraining and online data.
    cache = _online_training_with_mixed_pretraining_data(hyperparams, explorer, cache)
    return cache

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Online training with hyperparameter configuration")
    
    # Environment and training parameters
    parser.add_argument("--env", type=str, default="CliffWalking-v0", 
                       help="Environment name (e.g., CartPole-v0, MountainCar-v0, FrozenLake-v1, Pendulum-v1, CliffWalking-v0, RepresentedPong-v0)")
    parser.add_argument("--max_episode_len", type=int, default=200,
                       help="Maximum episode length")
    parser.add_argument("--n_online_eps", type=int, default=190,
                       help="Number of online episodes")
    parser.add_argument("--n_pretrain_eps", type=int, default=10,
                       help="Number of pretraining episodes")
    parser.add_argument("--seed", type=int, default=42069,
                       help="Random seed")
    parser.add_argument("--eps", type=float, default=0.1,
                       help="Epsilon for exploration (only applies to online training, not pretraining with LLM data)")
    parser.add_argument("--n_exp", type=int, default=5,
                       help="Number of experiments to run")
    
    # Model and training hyperparameters
    parser.add_argument("--gpu", action="store_true", default=True,
                       help="Use GPU for training with d3rlpy")
    parser.add_argument("--no-gpu", dest="gpu", action="store_false",
                       help="Disable GPU usage")
    parser.add_argument("--buffer_size", type=int, default=100000,
                       help="Replay buffer size")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Batch size (test smaller: 32, 64)")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("--target_update_interval", type=int, default=1000,
                       help="Target network update interval")
    
    # Data and model configuration
    parser.add_argument("--sft", action="store_true", default=False,
                       help="Use SFT data paths")
    parser.add_argument("--long_cot", action="store_true", default=False,
                       help="Use DeepSeek long CoT data paths")
    parser.add_argument("--n_pretrain_steps", type=int, default=1000,
                       help="Number of pretraining steps")
    parser.add_argument("--pretraining_exp", action="store_true", default=False,
                       help="Run pretraining experiments (run_exp_and_save calls)")
    parser.add_argument("--awac", action="store_true", default=False,
                       help="Using AWAC model")
    
    args = parser.parse_args()
    
    hyperparams = {
        "env": args.env,
        "max_episode_len": args.max_episode_len,
        "n_online_eps": args.n_online_eps,
        "n_pretrain_eps": args.n_pretrain_eps,
        "seed": args.seed,
        "eps": args.eps,
        "n_exp": args.n_exp,
        "gpu": args.gpu,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "target_update_interval": args.target_update_interval,
        "sft": args.sft,
        "long_cot": args.long_cot,
        "n_pretrain_steps": args.n_pretrain_steps,
        "pretraining_exp": args.pretraining_exp,
        "awac": args.awac,
    }

    # setup explorers
    explorer = d3rlpy.algos.ConstantEpsilonGreedy(hyperparams["eps"])
    path_7b, path_32b = get_llm_data_paths(hyperparams["env"], hyperparams["sft"], hyperparams["long_cot"])
    suffix = ""
    if hyperparams["sft"]:
        suffix = "SFT"
    elif hyperparams["long_cot"]:
        suffix = "LCOT"

    cache = {}
    timing_data = {}
    
    # Time _online_training for 32b model
    if path_32b is not None:
        print("Starting _online_training for 32b model...")
        start_time = time.time()
        cache = _online_training(hyperparams, explorer, cache, path_32b, "32b", suffix)
        end_time = time.time()
        timing_data['_online_training_32b'] = end_time - start_time
        print(f"_online_training for 32b model completed in {timing_data['_online_training_32b']:.2f} seconds")

    # Time _online_training for 7b model
    if path_7b is not None:
        if "online_0" in cache.keys():
            skip_from_scratch = True
        else:
            skip_from_scratch = False
        print("Starting _online_training for 7b model...")
        start_time = time.time()
        cache = _online_training(hyperparams, explorer, cache, path_7b, "7b", suffix, skip_from_scratch)
        end_time = time.time()
        timing_data['_online_training_7b'] = end_time - start_time
        print(f"_online_training for 7b model completed in {timing_data['_online_training_7b']:.2f} seconds")

    with open(
        f'data/cache_{hyperparams["env"].split("-")[0]}_Neps_{hyperparams["n_pretrain_eps"]}{suffix}.pkl',
        "wb",
    ) as file:
        pickle.dump(cache, file)

    # New experiments to test pretraining with online RL and Random data
    if hyperparams["pretraining_exp"]:
        print("Starting run_exp_and_save with random data...")
        start_time = time.time()
        run_exp_and_save(hyperparams, explorer, is_rand=True)
        end_time = time.time()
        timing_data['run_exp_and_save_rand'] = end_time - start_time
        print(f"run_exp_and_save with random data completed in {timing_data['run_exp_and_save_rand']:.2f} seconds")
        
        print("Starting run_exp_and_save with pretrain data...")
        start_time = time.time()
        run_exp_and_save(hyperparams, explorer, is_rand=False)
        end_time = time.time()
        timing_data['run_exp_and_save_pretrain'] = end_time - start_time
        print(f"run_exp_and_save with pretrain data completed in {timing_data['run_exp_and_save_pretrain']:.2f} seconds")
    else:
        print("Skipping pretraining experiments (pretraining_exp=False)")
    
    # Write timing log with hyperparameters
    write_timing_log(hyperparams, timing_data)
