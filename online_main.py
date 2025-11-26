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
from utils import (
    get_llm_data_paths, get_env_and_eval_env, create_d3rlpy_model, create_random_model,
    pretrain_from_llm, load_dataset_to_buffer, rollout_and_eval, buffer_to_dataset,
    write_timing_log, SB3_AVAILABLE, EpisodeRewardCallback, create_ppo_env,
    create_ppo_model, evaluate_ppo_policy, online_training_ppo_with_init_policy, online_training_ppo_rand
)


def online_training_with_init_policy(hyperparams, explorer, seed, n_pretrain_steps, n_pretrain_eps, n_online_eps):
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


def online_training_rand_policy(hyperparams, explorer, seed, n_pretrain_steps, n_pretrain_eps, n_online_eps):
    """
    Same as online_training_with_init_policy, but with random actions for the first n_pretrain_eps episodes.
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
        online_training_fn = online_training_rand_policy
        suffix = "_rand"
    else:
        online_training_fn = online_training_with_init_policy
        suffix = ""
    n_episodes = hyperparams["n_online_eps"] + hyperparams["n_pretrain_eps"]
    cache = {}
    for n_pretrain_steps in [1000, 3000]:
        for n_pretrain_eps in [10, 20, 30]:
            n_online_eps = n_episodes - n_pretrain_eps
            cache = run_exp(n_pretrain_steps, n_pretrain_eps, n_online_eps, cache, hyperparams, explorer, online_training_fn)

    with open(
        f'data/cache_{hyperparams["env"].split("-")[0]}_on_policy_pretrain_exp{suffix}{"_awac" if hyperparams["awac"] else ""}.pkl',
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


def _online_training_from_pretrained_model(hyperparams, explorer, cache, model_size, suffix, n_pretrain_steps):
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
        cache[f"pretrain_{model_size}_{n_pretrain_steps}_{i}"], cache[f"pretrain_{model_size}_{n_pretrain_steps}_{i}_dataset"], _ = rollout_and_eval(hyperparams["max_episode_len"], hyperparams["env"], explorer, model, buffer, n_rollouts, hyperparams["seed"]+i)
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


def _online_training_with_mixed_pretraining_data(hyperparams, explorer, cache, model_size):
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
        cache[f"mix_{model_size}_{i}"], cache[f"mix_{model_size}_{i}_dataset"], _ = rollout_and_eval(hyperparams["max_episode_len"], hyperparams["env"], explorer, model, buffer, n_rollouts, hyperparams["seed"]+i)
    return cache


def _online_training(hyperparams, explorer, cache, data_path, model_size, suffix, skip_from_scratch=False):
    hyperparams["data_path"] = data_path # Restore data path for mixed pretraining runs
    cache = _online_training_from_pretrained_model(hyperparams, explorer, cache, model_size, suffix, 1000)
    cache = _online_training_from_pretrained_model(hyperparams, explorer, cache, model_size, suffix, 3000)

    if not skip_from_scratch:
        cache = _online_training_from_scratch(hyperparams, explorer, cache)

    # Mixed pretraining and online data.
    cache = _online_training_with_mixed_pretraining_data(hyperparams, explorer, cache, model_size)
    return cache


# ============================================================================
# PPO Experiment Functions
# ============================================================================

def run_ppo_exp(n_pretrain_steps, n_pretrain_eps, n_online_eps, cache, hyperparams, online_training_fn):
    """
    Run PPO experiments similar to run_exp but for PPO algorithm.
    """
    for i in range(hyperparams["n_exp"]):
        cache[
            f'ppo_pretrain_{n_pretrain_eps}_eps_{n_pretrain_steps}_steps_{i}'
        ], cache[
            f'ppo_pretrain_{n_pretrain_eps}_eps_{n_pretrain_steps}_steps_{i}_dataset'
        ], cache[
            f'ppo_pretrain_{n_pretrain_eps}_eps_{n_pretrain_steps}_steps_{i}_offline_dataset'
        ] = online_training_fn(hyperparams, hyperparams["seed"]+i, n_pretrain_steps, n_pretrain_eps, n_online_eps)
    return cache


def run_ppo_exp_and_save(hyperparams, is_rand=True):
    """
    Run PPO experiments to test pretraining with online RL and Random data.
    """
    if not SB3_AVAILABLE:
        print("Warning: stable-baselines3 not installed. Skipping PPO experiments.")
        return
    
    if is_rand:
        online_training_fn = online_training_ppo_rand
        suffix = "_ppo_rand"
    else:
        online_training_fn = online_training_ppo_with_init_policy
        suffix = "_ppo"
    
    n_episodes = hyperparams["n_online_eps"] + hyperparams["n_pretrain_eps"]
    cache = {}
    
    for n_pretrain_steps in [1000, 3000]:
        for n_pretrain_eps in [10, 20, 30]:
            n_online_eps = n_episodes - n_pretrain_eps
            cache = run_ppo_exp(n_pretrain_steps, n_pretrain_eps, n_online_eps, cache, hyperparams, online_training_fn)
    
    with open(
        f'data/cache_{hyperparams["env"].split("-")[0]}_on_policy_pretrain_exp{suffix}{"_awac" if hyperparams["awac"] else ""}.pkl',
        "wb",
    ) as file:
        pickle.dump(cache, file)


def _online_training_ppo_from_scratch(hyperparams, cache):
    """
    Train PPO from scratch (no pretraining).
    """
    if not SB3_AVAILABLE:
        raise ImportError("stable-baselines3 is required for PPO training.")
    
    n_rollouts = hyperparams["n_pretrain_eps"] + hyperparams["n_online_eps"]
    
    for i in range(hyperparams["n_exp"]):
        seed = hyperparams["seed"] + i
        np.random.seed(seed)
        
        env = create_ppo_env(hyperparams["env"], seed)
        eval_env = create_ppo_env(hyperparams["env"], seed)
        
        hyperparams_with_seed = {**hyperparams, "seed": seed}
        ppo_model = create_ppo_model(env, hyperparams_with_seed)
        
        total_timesteps = n_rollouts * hyperparams["max_episode_len"]
        
        callback = EpisodeRewardCallback()
        ppo_model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False,
        )
        
        eps_rewards = callback.get_episode_rewards()[:n_rollouts]
        
        # Fill remaining if needed
        while len(eps_rewards) < n_rollouts:
            mean_reward, _, _ = evaluate_ppo_policy(ppo_model, eval_env, n_eval_episodes=1, max_episode_len=hyperparams["max_episode_len"])
            eps_rewards.append(mean_reward)
        
        _, _, final_eval_dataset = evaluate_ppo_policy(
            ppo_model, eval_env,
            n_eval_episodes=10,
            max_episode_len=hyperparams["max_episode_len"]
        )
        
        cache[f"ppo_online_{i}"] = eps_rewards
        cache[f"ppo_online_{i}_dataset"] = final_eval_dataset
        
        env.close()
        eval_env.close()
    
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
    parser.add_argument("--awac", action="store_true", default=False,
                       help="Using AWAC model")
    parser.add_argument("--n_steps_per_epoch", type=int, default=200,
                       help="Number of steps per epoch for training")
    parser.add_argument("--online_exp", action="store_true", default=False,
                       help="Run the main fine-tune experiments")
    parser.add_argument("--online_rand", action="store_true", default=False,
                       help="Run the random and online fine-tune experiments")
    parser.add_argument("--ppo", action="store_true", default=False,
                       help="Run PPO experiments (requires stable-baselines3)")
    parser.add_argument("--ppo_n_steps", type=int, default=2048,
                       help="Number of steps per PPO update")
    parser.add_argument("--ppo_n_epochs", type=int, default=10,
                       help="Number of PPO epochs per update")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                       help="GAE lambda for PPO")
    parser.add_argument("--clip_range", type=float, default=0.2,
                       help="Clipping range for PPO")
    parser.add_argument("--ent_coef", type=float, default=0.0,
                       help="Entropy coefficient for PPO")
    parser.add_argument("--vf_coef", type=float, default=0.5,
                       help="Value function coefficient for PPO")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                       help="Max gradient norm for PPO")
    
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
        "awac": args.awac,
        "n_steps_per_epoch": args.n_steps_per_epoch,
        "online_exp": args.online_exp,
        "online_rand": args.online_rand,
        # PPO-specific hyperparameters
        "ppo": args.ppo,
        "ppo_n_steps": args.ppo_n_steps,
        "ppo_n_epochs": args.ppo_n_epochs,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
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
    if hyperparams["online_exp"] and path_32b is not None:
        print("Starting _online_training for 32b model...")
        start_time = time.time()
        cache = _online_training(hyperparams, explorer, cache, path_32b, "32b", suffix)
        end_time = time.time()
        timing_data['_online_training_32b'] = end_time - start_time
        print(f"_online_training for 32b model completed in {timing_data['_online_training_32b']:.2f} seconds")
    elif not hyperparams["online_exp"]:
        print("Skipping the main fine-tune experiments (online_exp=False)")

    # Time _online_training for 7b model
    if hyperparams["online_exp"] and path_7b is not None:
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
        f'data/cache_{hyperparams["env"].split("-")[0]}_Neps_{hyperparams["n_pretrain_eps"]}{suffix}{"_awac" if hyperparams["awac"] else ""}.pkl',
        "wb",
    ) as file:
        pickle.dump(cache, file)

    # New experiments to test pretraining with online RL and Random data
    if hyperparams["online_rand"]:
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
        print("Skipping random and online fine-tune experiments (online_rand=False)")
    
    # PPO experiments
    if hyperparams["ppo"]:
        if not SB3_AVAILABLE:
            print("Error: stable-baselines3 not installed. Skipping PPO experiments.")
            print("Install with: pip install stable-baselines3")
        else:
            print("=" * 60)
            print("Starting PPO experiments...")
            print("=" * 60)
            
            # PPO from scratch
            print("Starting PPO training from scratch...")
            start_time = time.time()
            ppo_cache = {}
            # Missing training PPO from pretrained model
            ppo_cache = _online_training_ppo_from_scratch(hyperparams, ppo_cache)
            end_time = time.time()
            timing_data['ppo_from_scratch'] = end_time - start_time
            print(f"PPO from scratch completed in {timing_data['ppo_from_scratch']:.2f} seconds")
            
            # PPO with pretrain
            print("Starting PPO run_ppo_exp_and_save with pretrain data...")
            start_time = time.time()
            run_ppo_exp_and_save(hyperparams, is_rand=False)
            end_time = time.time()
            timing_data['ppo_pretrain'] = end_time - start_time
            print(f"PPO pretrain experiments completed in {timing_data['ppo_pretrain']:.2f} seconds")
            
            # PPO with random warmup
            print("Starting PPO run_ppo_exp_and_save with random data...")
            start_time = time.time()
            run_ppo_exp_and_save(hyperparams, is_rand=True)
            end_time = time.time()
            timing_data['ppo_rand'] = end_time - start_time
            print(f"PPO random experiments completed in {timing_data['ppo_rand']:.2f} seconds")
            
            # Save PPO cache
            with open(
                f'data/cache_{hyperparams["env"].split("-")[0]}_ppo_Neps_{hyperparams["n_pretrain_eps"]}.pkl',
                "wb",
            ) as file:
                pickle.dump(ppo_cache, file)
            print(f"PPO cache saved to data/cache_{hyperparams['env'].split('-')[0]}_ppo_Neps_{hyperparams['n_pretrain_eps']}.pkl")
    else:
        print("Skipping PPO experiments (ppo=False)")
    
    # Write timing log with hyperparameters
    write_timing_log(hyperparams, timing_data)
