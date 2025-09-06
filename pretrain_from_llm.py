import gymnasium as gym
import numpy as np
import d3rlpy
import pickle
import argparse


def get_llm_data_paths(env, sft=False, long_cot=False):
    env_name = env.split("-")[0]
    if env_name == "Pendulum":
        path_7b = f"data/{env_name}_Qwen2.5-7B-Instruct_Neps_30_20250413234248.pkl"
    elif env_name == "CliffWalking":
        if sft:
            path_7b = f"data/{env_name}_Qwen2.5-7B-Instruct_Neps_30_20250505135458SFT.pkl"  # CliffWalking SFT
        else:
            path_7b = f"data/{env_name}_Qwen2.5-7B-Instruct_Neps_30_20250502104658.pkl"  # CliffWalking
    elif env_name == "FrozenLake":
        if sft:
            path_7b = f"data/{env_name}_Qwen2.5-7B-Instruct_Neps_30_20250501013018SFT.pkl"  # FrozenLake SFT
        elif long_cot:
            path_7b = f"data/{env_name}_DeepSeek-R1-Distill-Qwen-7B_Neps_30_20250502071148.pkl"  # FrozenLake DS 7b
        else:
            path_7b = f"data/{env_name}_Qwen2.5-7B-Instruct_Neps_30_20250430010558.pkl"  # FrozenLake
    elif env_name == "CartPole":
        path_7b = f"data/{env_name}_Qwen2.5-7B-Instruct_Neps_30_20250410211529.pkl"
    elif env_name == "MountainCar":
        path_7b = f"data/{env_name}_Qwen2.5-7B-Instruct_Neps_30_20250415095844.pkl"
    elif env_name == "RepresentedPong":
        path_7b = f"data/{env_name}_Qwen2.5-7B-Instruct_Neps_30_20250420103044.pkl"

    if env_name == "Pendulum":
        path_32b = f"data/{env_name}_Qwen2.5-32B-Instruct_Neps_30_20250414014508.pkl"
    elif env_name == "CliffWalking":
        if sft:
            path_32b = None  # For the SFT experiment
        else:
            path_32b = f"data/{env_name}_Qwen2.5-32B-Instruct_Neps_30_20250506015247.pkl"  # CliffWalking
    elif env_name == "FrozenLake":
        if sft:
            path_32b = None  # For the SFT experiment
        elif long_cot:
            path_32b = f"data/{env_name}_DeepSeek-R1-Distill-Qwen-14B_Neps_30_20250502084016.pkl"  # FrozenLake DS 14b
        else:
            path_32b = f"data/{env_name}_Qwen2.5-32B-Instruct_Neps_30_20250429223843.pkl"  # FrozenLake
    elif env_name == "CartPole":
        path_32b = f"data/{env_name}_Qwen2.5-32B-Instruct_Neps_30_20250412032827.pkl"
    elif env_name == "MountainCar":
        path_32b = f"data/{env_name}_Qwen2.5-32B-Instruct_Neps_30_20250413081613.pkl"
    elif env_name == "RepresentedPong":
        path_32b = f"data/{env_name}_Qwen2.5-32B-Instruct_Neps_30_20250420162547.pkl"
    return path_7b, path_32b


def ensure_numpy_array(x, hyperparams=hyperparams):
    if isinstance(gym.make(hyperparams["env"]).observation_space, gym.spaces.Discrete):
        # Convert discrete observations to one-hot encoded arrays
        return np.eye(gym.make(hyperparams["env"]).observation_space.n)[x]
    if not isinstance(x, np.ndarray):
        return np.array([x])
    return x


def get_new_dataset(dataset, n_eps):
    observations, actions, rewards, terminals = [], [], [], []
    # Extract observations, actions, rewards, and terminals from the original dataset
    for episode in dataset.episodes[:n_eps]:
        observations += [
            ensure_numpy_array(o) for o in episode.observations
        ]  # Ensure observations are numpy arrays
        actions += [a for a in episode.actions]
        rewards += [r for r in episode.rewards]
        terminals += [0] * (len(episode.rewards) - 1) + [
            1
        ]  # Add terminal flag for the last step
    dataset_new = d3rlpy.dataset.MDPDataset(
        observations=np.array(observations),
        actions=np.array(actions),
        rewards=np.array(rewards),
        terminals=np.array(terminals),
    )
    # Verify the lengths of the extracted lists
    print(f"Number of episodes: {len(dataset_new.episodes)}")
    return dataset_new


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
    
    n_pretrain_eps = hyperparams["n_pretrain_eps"]  # 10
    # n_online_eps = hyperparams["n_online_eps"]
    # n_exp = hyperparams["n_exp"]
    # hyperparams["target_update_interval"] = (
    #     200  # For pretraining, leave the original value of 1000 for online training
    # )

    # fix seed
    d3rlpy.seed(hyperparams["seed"])

    path_7b, path_32b = get_llm_data_paths(hyperparams["env"], hyperparams["sft"], hyperparams["long_cot"])
    suffix = ""
    if "SFT" in path_7b:
        suffix = "SFT"
    elif "DeepSeek" in path_7b:
        suffix = "DS"
    if path_7b is not None:
        with open(path_7b, "rb") as file:
            Qwen_7B_dataset = pickle.load(file)
        Qwen_7B_dataset_new = get_new_dataset(Qwen_7B_dataset, n_pretrain_eps)

        # Determine the algorithm based on the action space
        if isinstance(
            gym.make(hyperparams["env"]).action_space, gym.spaces.Box
        ):  # Continuous action space
            pretrain_7b_dqn = d3rlpy.algos.SACConfig(
                batch_size=hyperparams["batch_size"],
                gamma=hyperparams["gamma"],
            ).create(device=hyperparams["gpu"])
        else:  # Discrete action space
            pretrain_7b_dqn = d3rlpy.algos.DoubleDQNConfig(
                batch_size=hyperparams["batch_size"],
                learning_rate=hyperparams["learning_rate"],
                gamma=hyperparams["gamma"],
                target_update_interval=hyperparams["target_update_interval"],
            ).create(device=hyperparams["gpu"])
        # start offline training
        pretrain_7b_dqn.fit(
            Qwen_7B_dataset_new,
            n_steps=hyperparams["n_pretrain_steps"],
            n_steps_per_epoch=hyperparams["n_steps_per_epoch"],
        )
        with open(
            f'models/{hyperparams["env"].split("-")[0]}_ddqn_pretrain_7b_{hyperparams["n_pretrain_steps"]}_steps_{hyperparams["n_pretrain_eps"]}{suffix}.pkl',
            "wb",
        ) as file:
            pickle.dump(pretrain_7b_dqn, file)
    if path_32b is not None:
        with open(path_32b, "rb") as file:
            Qwen_32B_dataset = pickle.load(file)

        # Create the new dataset with the specified number of episodes
        Qwen_32B_dataset_new = get_new_dataset(Qwen_32B_dataset, n_pretrain_eps)

        # Determine the algorithm based on the action space
        if isinstance(
            gym.make(hyperparams["env"]).action_space, gym.spaces.Box
        ):  # Continuous action space
            pretrain_32b_dqn = d3rlpy.algos.SACConfig(
                batch_size=hyperparams["batch_size"],
                gamma=hyperparams["gamma"],
            ).create(device=hyperparams["gpu"])
        else:  # Discrete action space
            pretrain_32b_dqn = d3rlpy.algos.DoubleDQNConfig(
                batch_size=hyperparams["batch_size"],
                learning_rate=hyperparams["learning_rate"],
                gamma=hyperparams["gamma"],
                target_update_interval=hyperparams["target_update_interval"],
            ).create(device=hyperparams["gpu"])

        # start offline training
        pretrain_32b_dqn.fit(
            Qwen_32B_dataset_new,
            n_steps=hyperparams["n_pretrain_steps"],
            n_steps_per_epoch=hyperparams["n_steps_per_epoch"],
        )
        with open(
            f'models/{hyperparams["env"].split("-")[0]}_ddqn_pretrain_32b_{hyperparams["n_pretrain_steps"]}_steps_{hyperparams["n_pretrain_eps"]}{suffix}.pkl',
            "wb",
        ) as file:
            pickle.dump(pretrain_32b_dqn, file)
