import d3rlpy
import argparse
from utils import *

hyperparams = {
    "env": "FrozenLake-v1",  # "CartPole-v0", Pendulum-v1, "MountainCar-v0", "FrozenLake-v1", "CliffWalking-v0", "RepresentedPong-v0"
    "seed": 42069,
    # "n_episodes": 10,  # 5000,
    # "max_episode_len": 200,  # Around 10h per 100k steps in Leviathan server
    # "eps": 0.0,  # epsilon for exploration
    # "n_exp": 5,
    "n_pretrain_eps": 10,
    # "n_online_eps": 140,
    "gpu": True,  # True if use GPU to train with d3rlpy
    # "buffer_size": 100000,  # Test with 100k, 200k, 500k. 1M might be too much
    # "data_path": None,  #'data/CartPole_Qwen2.5-7B-Instruct_Neps_10_20250406040150.pkl',
    # "model_path": None,  #'d3rlpy_loss/DoubleDQN_online_20250331153346/model_600000.d3',
    "batch_size": 256,  # Test smaller batch size: 32, 64. May be noisier
    "learning_rate": 5e-5,
    "gamma": 0.99,
    "target_update_interval": 200,  # Test with 1k, 2k, 5k
    "n_steps_per_epoch": 200,
    "n_pretrain_steps": 1000,
    "sft": False,  # Set to True to use SFT data paths
    "long_cot": False,  # Set to True to use DeepSeek long CoT data paths
}


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Pretrain from LLM data')
    parser.add_argument('--env', type=str, default=hyperparams["env"], 
                        help='Environment name (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=hyperparams["seed"], 
                        help='Random seed (default: %(default)s)')
    parser.add_argument('--n_pretrain_eps', type=int, default=hyperparams["n_pretrain_eps"], 
                        help='Number of pretrain episodes (default: %(default)s)')
    parser.add_argument('--n_pretrain_steps', type=int, default=hyperparams["n_pretrain_steps"], 
                        help='Number of pretrain steps (default: %(default)s)')
    parser.add_argument('--sft', action='store_true', default=hyperparams["sft"], 
                        help='Use SFT data paths (default: %(default)s)')
    parser.add_argument('--long_cot', action='store_true', default=hyperparams["long_cot"], 
                        help='Use long CoT data paths (default: %(default)s)')
    
    args = parser.parse_args()
    
    # Update hyperparams with parsed arguments
    hyperparams["env"] = args.env
    hyperparams["seed"] = args.seed
    hyperparams["n_pretrain_eps"] = args.n_pretrain_eps
    hyperparams["n_pretrain_steps"] = args.n_pretrain_steps
    hyperparams["sft"] = args.sft
    hyperparams["long_cot"] = args.long_cot
    
    # n_pretrain_eps = hyperparams["n_pretrain_eps"]  # 10
    # n_online_eps = hyperparams["n_online_eps"]
    # n_exp = hyperparams["n_exp"]
    # hyperparams["target_update_interval"] = (
    #     200  # For pretraining, leave the original value of 1000 for online training
    # )

    # fix seed
    d3rlpy.seed(hyperparams["seed"])

    path_7b, path_32b = get_llm_data_paths(hyperparams["env"], hyperparams["sft"], hyperparams["long_cot"])
    suffix = ""
    if hyperparams["sft"]:
        suffix = "SFT"
    elif hyperparams["long_cot"]:
        suffix = "LCOT"
    if path_7b is not None:
        pretrain_7b_dqn = pretrain_from_llm(path_7b, hyperparams, "7b", suffix)
    if path_32b is not None:
        pretrain_32b_dqn = pretrain_from_llm(path_32b, hyperparams, "32b", suffix)
