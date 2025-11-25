#!/usr/bin/env python3
"""
Script to read cache from online_main.py and edit it by running the 
_online_training_with_mixed_pretraining_data function.

This script loads existing cache files, runs the mixed pretraining function,
and saves the updated cache back to disk.
"""

import argparse
import pickle
import time
import d3rlpy
from online_main import _online_training_with_mixed_pretraining_data
from utils import get_llm_data_paths


def load_cache(env_name, n_pretrain_eps, suffix=""):
    """Load existing cache file for the environment."""
    cache_filename = f'data/cache_{env_name}_Neps_{n_pretrain_eps}{suffix}.pkl'
    try:
        with open(cache_filename, "rb") as file:
            cache = pickle.load(file)
        print(f"Loaded cache from {cache_filename}")
        return cache
    except FileNotFoundError:
        print(f"Cache file {cache_filename} not found. Creating new cache.")
        return {}


def save_cache(cache, env_name, n_pretrain_eps, suffix=""):
    """Save cache to file."""
    cache_filename = f'data/cache_{env_name}_Neps_{n_pretrain_eps}{suffix}.pkl'
    with open(cache_filename, "wb") as file:
        pickle.dump(cache, file)
    print(f"Saved cache to {cache_filename}")


def run_mixed_pretraining_for_env(hyperparams):
    """
    Run mixed pretraining for a specific environment.
    
    Args:
        hyperparams: Dictionary containing all hyperparameters
    """
    env_name = hyperparams["env"].split("-")[0]
    print(f"\n{'='*60}")
    print(f"Running mixed pretraining for {env_name}")
    print(f"{'='*60}")
    
    # Determine suffix for file naming
    suffix = ""
    if hyperparams["sft"]:
        suffix = "SFT"
    elif hyperparams["long_cot"]:
        suffix = "LCOT"
    
    # Get data paths
    path_7b, path_32b = get_llm_data_paths(hyperparams["env"], hyperparams["sft"], hyperparams["long_cot"])
    
    # Set up explorer
    explorer = d3rlpy.algos.ConstantEpsilonGreedy(hyperparams["eps"])
    
    # Load existing cache
    cache = load_cache(env_name, hyperparams["n_pretrain_eps"], suffix)
    
    # Run mixed pretraining for 7B model if data exists
    if path_7b is not None:
        print(f"Running mixed pretraining with 7B model data: {path_7b}")
        hyperparams["data_path"] = path_7b
        start_time = time.time()
        cache = _online_training_with_mixed_pretraining_data(hyperparams, explorer, cache, "7b")
        end_time = time.time()
        print(f"7B mixed pretraining completed in {end_time - start_time:.2f} seconds")
    
    # Run mixed pretraining for 32B model if data exists
    if path_32b is not None:
        print(f"Running mixed pretraining with 32B model data: {path_32b}")
        hyperparams["data_path"] = path_32b
        start_time = time.time()
        cache = _online_training_with_mixed_pretraining_data(hyperparams, explorer, cache, "32b")
        end_time = time.time()
        print(f"32B mixed pretraining completed in {end_time - start_time:.2f} seconds")
    
    # Save updated cache
    save_cache(cache, env_name, hyperparams["n_pretrain_eps"], suffix)
    
    print(f"Completed mixed pretraining for {env_name}")
    return cache


def main():
    parser = argparse.ArgumentParser(description="Run mixed pretraining for environments")
    
    # Environment and training parameters
    parser.add_argument("--env", type=str, default="CliffWalking-v0", 
                       help="Environment name (e.g., CartPole-v0, MountainCar-v0, FrozenLake-v1, Pendulum-v1, CliffWalking-v0, RepresentedPong-v0)")
    parser.add_argument("--max_episode_len", type=int, default=200,
                       help="Maximum episode length")
    parser.add_argument("--n_online_eps", type=int, default=190,
                       help="Number of online episodes")
    parser.add_argument("--n_pretrain_eps", type=int, default=30,
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
    
    args = parser.parse_args()
    
    # Create hyperparams dictionary (same structure as online_main.py)
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
    }
    
    # Validate environment name
    valid_envs = ["CartPole-v0", "CliffWalking-v0", "FrozenLake-v1", "MountainCar-v0", "Pendulum-v1", "RepresentedPong-v0"]
    if args.env not in valid_envs:
        print(f"Error: Environment '{args.env}' not supported.")
        print(f"Valid environments: {', '.join(valid_envs)}")
        return 1
    
    # Run mixed pretraining
    try:
        cache = run_mixed_pretraining_for_env(hyperparams)
        print(f"\nSuccessfully completed mixed pretraining for {args.env}")
        return 0
    except Exception as e:
        print(f"Error running mixed pretraining for {args.env}: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
