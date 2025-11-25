import gymnasium as gym
import numpy as np
import d3rlpy
import pickle
from tqdm import trange
from datetime import datetime

from env.atari.represented_atari_game import GymCompatWrapper2
from d3rlpy.metrics import EnvironmentEvaluator

from online_main import OneHotWrapper, evaluate_qlearning_with_environment
from vis_utils import reformat_on_policy_pretrain_cache

from utils import *

def buffer_to_dataset(temp_buffer, buffer_size, n_pretrain_eps, env):
    # Truncate the buffer to the size of n_pretrain_eps
    buffer = d3rlpy.dataset.ReplayBuffer(
        buffer=d3rlpy.dataset.FIFOBuffer(limit=buffer_size),
        env=env,
    )
    for episode in temp_buffer.episodes:
        if len(episode) > 0 and hasattr(episode, "rewards"):
            buffer.append_episode(episode)
        else:
            print(f"Skipping invalid episode: {episode}")
        if len(buffer.episodes) >= n_pretrain_eps:
            break
    # Convert the ReplayBuffer to an MDPDataset by extracting data manually
    # Since ReplayBuffer doesn't have to_mdp_dataset(), we need to create it manually
    observations, actions, rewards, terminals = [], [], [], []
    
    # Extract data from the buffer's episodes
    #TODO: .extend here stores list of of np.array. May need to fix this
    if hasattr(buffer, 'episodes') and len(buffer.episodes) > 0 and buffer.transition_count > 0:
        for episode in buffer.episodes:
            if len(episode) > 0 and hasattr(episode, "rewards"):
                observations.extend(episode.observations)
                actions.extend(episode.actions)
                rewards.extend(episode.rewards)
                # Add terminal flags: 0 for all steps except the last one
                terminals.extend([0] * (len(episode.rewards) - 1) + [1])
    
    # Create MDPDataset only if we have data
    if len(observations) > 0:
        offline_dataset = d3rlpy.dataset.MDPDataset(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            terminals=np.array(terminals),
        )
    else:
        # Create empty dataset if no data is available
        offline_dataset = d3rlpy.dataset.MDPDataset(
            observations=np.array([]),
            actions=np.array([]),
            rewards=np.array([]),
            terminals=np.array([]),
        )
    return offline_dataset, buffer

def online_training_with_pretrain(hyperparams, explorer, seed, n_pretrain_steps, n_pretrain_eps, n_online_eps):
    dqn = create_d3rlpy_model(hyperparams["env"], hyperparams["batch_size"], hyperparams["learning_rate"], hyperparams["gamma"], hyperparams["target_update_interval"], hyperparams["gpu"], hyperparams["awac"])
    tmp_env, _ = get_env_and_eval_env(hyperparams["env"], seed)
    # Initialize empty FIFO buffer
    temp_buffer = d3rlpy.dataset.ReplayBuffer(
        buffer=d3rlpy.dataset.FIFOBuffer(limit=hyperparams["buffer_size"]),
        env=tmp_env,
    )

    # eps_rewards = []
    # timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # for _ in trange(hyperparams["n_pretrain_eps"]):
    #     dqn.fit_online(
    #         env=env,
    #         buffer=temp_buffer,
    #         explorer=explorer,
    #         n_steps=hyperparams["max_episode_len"],
    #         experiment_name=f"{timestamp}_online_training",
    #     )
    #     if hyperparams["env"] == "CliffWalking-v0":
    #         r, _ = evaluate_qlearning_with_environment(
    #             dqn, eval_env, hyperparams["max_episode_len"]
    #         )
    #     else:
    #         env_evaluator = EnvironmentEvaluator(env, n_trials=1)
    #         r = env_evaluator(dqn, dataset=None)
    #     eps_rewards.append(r)
    #     # break if we have collected enough offline data. If not, the buffer will collect hyperparams["n_pretrain_eps"]*hyperparams["max_episode_len"] transitions
    #     if len(temp_buffer.episodes) >= hyperparams["n_pretrain_eps"]:
    #         break

    eps_rewards_offline, _, temp_buffer = rollout_and_eval(hyperparams["max_episode_len"], hyperparams["env"], explorer, dqn, temp_buffer, n_pretrain_eps, seed)
    # # Truncate the buffer to the size of hyperparams["n_pretrain_eps"]
    # buffer = d3rlpy.dataset.ReplayBuffer(
    #     buffer=d3rlpy.dataset.FIFOBuffer(limit=hyperparams["buffer_size"]),
    #     env=env,
    # )
    # for episode in temp_buffer.episodes:
    #     if len(episode) > 0 and hasattr(episode, "rewards"):
    #         buffer.append_episode(episode)
    #     else:
    #         print(f"Skipping invalid episode: {episode}")
    #     if len(buffer.episodes) >= hyperparams["n_pretrain_eps"]:
    #         break
    # # Convert the ReplayBuffer to an MDPDataset by extracting data manually
    # # Since ReplayBuffer doesn't have to_mdp_dataset(), we need to create it manually
    # observations, actions, rewards, terminals = [], [], [], []
    
    # # Extract data from the buffer's episodes
    # #TODO: .extend here stores list of of np.array. May need to fix this
    # if hasattr(buffer, 'episodes') and len(buffer.episodes) > 0 and buffer.transition_count > 0:
    #     for episode in buffer.episodes:
    #         if len(episode) > 0 and hasattr(episode, "rewards"):
    #             observations.extend(episode.observations)
    #             actions.extend(episode.actions)
    #             rewards.extend(episode.rewards)
    #             # Add terminal flags: 0 for all steps except the last one
    #             terminals.extend([0] * (len(episode.rewards) - 1) + [1])
    
    # # Create MDPDataset only if we have data
    # if len(observations) > 0:
    #     offline_dataset = d3rlpy.dataset.MDPDataset(
    #         observations=np.array(observations),
    #         actions=np.array(actions),
    #         rewards=np.array(rewards),
    #         terminals=np.array(terminals),
    #     )
    # else:
    #     # Create empty dataset if no data is available
    #     offline_dataset = d3rlpy.dataset.MDPDataset(
    #         observations=np.array([]),
    #         actions=np.array([]),
    #         rewards=np.array([]),
    #         terminals=np.array([]),
    #     )

    offline_dataset, buffer = buffer_to_dataset(temp_buffer, hyperparams["buffer_size"], n_pretrain_eps, tmp_env)

    dqn.fit(
        buffer,
        n_steps=n_pretrain_steps,
        n_steps_per_epoch=hyperparams["n_steps_per_epoch"],
    )

    # for _ in trange(hyperparams["n_online_eps"]):
    #     dqn.fit_online(
    #         env=env,
    #         buffer=buffer,
    #         explorer=explorer,
    #         n_steps=hyperparams["max_episode_len"],
    #         experiment_name=f"{timestamp}_online_training",
    #     )
    #     if hyperparams["env"] == "CliffWalking-v0":
    #         r, _ = evaluate_qlearning_with_environment(
    #             dqn, eval_env, hyperparams["max_episode_len"]
    #         )
    #     else:
    #         env_evaluator = EnvironmentEvaluator(env, n_trials=1)
    #         r = env_evaluator(dqn, dataset=None)
    #     eps_rewards.append(r)
    
    # # Evaluate the final policy on the evaluation environment
    # _, dataset = evaluate_qlearning_with_environment(
    #     dqn, eval_env, hyperparams["max_episode_len"]
    # )
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

    # eps_rewards = []
    # timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # observations, actions, pretrain_rewards, terminals = [], [], [], []
    # for _ in range(hyperparams["n_pretrain_eps"]):
    #     env.reset()
    #     done = False
    #     eps_reward = 0
    #     count = 0
    #     while not done:
    #         action = env.action_space.sample()
    #         observation, reward, done, _, info = env.step(action)
    #         if count >= hyperparams["max_episode_len"]:
    #             done = True
    #         observations.append(observation)
    #         actions.append(action)
    #         pretrain_rewards.append(reward)
    #         terminals.append(int(done))
    #         eps_reward += reward
    #         count += 1
    #     eps_rewards.append(eps_reward)
    # offline_dataset = d3rlpy.dataset.MDPDataset(
    #     observations=np.array(observations),
    #     actions=np.array(actions),
    #     rewards=np.array(pretrain_rewards),
    #     terminals=np.array(terminals),
    # )
    # for episode in offline_dataset.episodes:
    #     if len(episode) > 0 and hasattr(episode, "rewards"):
    #         buffer.append_episode(episode)
    #     else:
    #         print(f"Skipping invalid episode: {episode}")

    dqn.fit(
        buffer,
        n_steps=n_pretrain_steps,
        n_steps_per_epoch=hyperparams["n_steps_per_epoch"],
    )

    # for _ in trange(hyperparams["n_online_eps"]):
    #     dqn.fit_online(
    #         env=env,
    #         buffer=buffer,
    #         explorer=explorer,
    #         n_steps=hyperparams["max_episode_len"],
    #         experiment_name=f"{timestamp}_online_training",
    #     )
    #     if hyperparams["env"] == "CliffWalking-v0":
    #         r, _ = evaluate_qlearning_with_environment(
    #             dqn, eval_env, hyperparams["max_episode_len"]
    #         )
    #     else:
    #         env_evaluator = EnvironmentEvaluator(env, n_trials=1)
    #         r = env_evaluator(dqn, dataset=None)
    #     eps_rewards.append(r)
    
    # # Evaluate the final policy on the evaluation environment
    # _, dataset = evaluate_qlearning_with_environment(
    #     dqn, eval_env, hyperparams["max_episode_len"]
    # )
    # return eps_rewards, dataset, pretrain_dataset
    eps_rewards_online, final_eval_dataset, _ = rollout_and_eval(hyperparams["max_episode_len"], hyperparams["env"], explorer, dqn, buffer, n_online_eps, seed)
    eps_rewards = eps_rewards_offline + eps_rewards_online
    return eps_rewards, final_eval_dataset, offline_dataset


def run_exp(
    n_pretrain_steps, n_pretrain_eps, n_online_eps, cache, hyperparams, explorer, online_training_fn
):
    # hyperparams["n_pretrain_steps"] = n_pretrain_steps # pretrain steps on offline dataset (not collecting new data)
    # hyperparams["n_pretrain_eps"] = n_pretrain_eps # offline dataset collection
    # hyperparams["n_online_eps"] = hyperparams["n_episodes"] - hyperparams["n_pretrain_eps"]
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


if __name__ == "__main__":
    hyperparams = {
        "env": "RepresentedPong-v0",  # "CartPole-v0", "MountainCar-v0", "FrozenLake-v1", "CliffWalking-v0", "RepresentedPong-v0"
        # "n_episodes": 200,  # Keep this here to use in the run_exp_rand and run_exp_split functions
        "n_online_eps": 170,  # 10-5990 for mountainCar, 30-120 for CartPole, 30-120 for FrozenLake
        "n_pretrain_eps": 30,
        "seed": 42069,
        "max_episode_len": 200,  # Around 10h per 100k steps in Leviathan server
        "eps": 0.1,  # epsilon for exploration. Only apply to online training. Do not use it for pretraining with LLM collected data.
        "n_exp": 5,
        "gpu": True,  # True if use GPU to train with d3rlpy
        "buffer_size": 100000,  # Test with 100k, 200k, 500k. 1M might be too much
        # "data_path": None,  #'data/CartPole_Qwen2.5-7B-Instruct_Neps_10_20250406040150.pkl',
        # "model_path": None,  #'d3rlpy_loss/DoubleDQN_online_20250331153346/model_600000.d3',
        "batch_size": 256,  # Test smaller batch size: 32, 64. May be noisier
        "learning_rate": 5e-5,
        "gamma": 0.99,
        "target_update_interval": 1000,  # Test with 1k, 2k, 5k
        "n_steps_per_epoch": 200,
        # "n_pretrain_steps": 1000,
        "awac": False,
    }

    # setup explorers
    explorer = d3rlpy.algos.ConstantEpsilonGreedy(hyperparams["eps"])
    run_exp_and_save(hyperparams, explorer, is_rand=True)
    run_exp_and_save(hyperparams, explorer, is_rand=False)
