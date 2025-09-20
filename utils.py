import gymnasium as gym
import numpy as np
import d3rlpy
from env.atari.represented_atari_game import GymCompatWrapper2
import pickle
from datetime import datetime
from tqdm import trange
from d3rlpy.metrics import EnvironmentEvaluator
import os
import json

def get_llm_data_paths(env, sft=False, long_cot=False):
    env_name = env.split("-")[0]
    path_7b = f"data/{env_name}_Qwen2.5-7B-Instruct_Neps_30.pkl"
    path_32b = f"data/{env_name}_Qwen2.5-32B-Instruct_Neps_30.pkl"
    if env_name == "Pendulum" or env_name == "CliffWalking" or env_name == "FrozenLake":
        if sft:
            path_7b = f"data/{env_name}_Qwen2.5-7B-Instruct_Neps_30SFT.pkl"
            path_32b = None
    if env_name == "FrozenLake" and long_cot:
        path_7b = f"data/{env_name}_DeepSeek-R1-Distill-Qwen-7B_Neps_30.pkl"  # FrozenLake DS 7b
        path_32b = f"data/{env_name}_DeepSeek-R1-Distill-Qwen-14B_Neps_30.pkl"  # FrozenLake DS 14b
    return path_7b, path_32b


class OneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(
            env.observation_space, gym.spaces.Discrete
        ), "Only Discrete observation spaces are supported."
        self.n = env.observation_space.n
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.n,), dtype=np.float32
        )

    def observation(self, obs):
        one_hot = np.zeros((self.n,), dtype=np.float32)
        one_hot[obs] = 1.0
        return one_hot


def evaluate_qlearning_with_environment(
    algo,
    env,
    max_episode_len,
    n_trials: int = 10,
    epsilon: float = 0.0,
) -> float:
    """
    From d3rlpy.metrics.utility.evaluate_with_environment
    Modified because the original code bugged out on CliffWalking-v0. The episode never end.
    Also return the dataset of the episodes.

    Returns average environment score.

    .. code-block:: python

        import gym

        from d3rlpy.algos import DQN
        from d3rlpy.metrics.utility import evaluate_with_environment

        env = gym.make('CartPole-v0')

        cql = CQL()

        mean_episode_return = evaluate_with_environment(cql, env)


    Args:
        alg: algorithm object.
        env: gym-styled environment.
        n_trials: the number of trials.
        epsilon: noise factor for epsilon-greedy policy.

    Returns:
        average score.
    """
    episode_rewards = []
    observations, actions, rewards, terminals = [], [], [], []
    for _ in range(n_trials):
        observation, _ = env.reset()
        episode_reward = 0.0
        count = 0
        while True:
            # take action
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                if isinstance(observation, np.ndarray):
                    observation = np.expand_dims(observation, axis=0)
                elif isinstance(observation, (tuple, list)):
                    observation = [np.expand_dims(o, axis=0) for o in observation]
                else:
                    raise ValueError(
                        f"Unsupported observation type: {type(observation)}"
                    )
                action = algo.predict(observation)[0]
            observation, reward, done, truncated, _ = env.step(action)
            episode_reward += float(reward)
            count += 1
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            if count >= max_episode_len:
                done = True
            terminals.append(int(done or truncated))

            if done or truncated:
                break
        episode_rewards.append(episode_reward)

    dataset = d3rlpy.dataset.MDPDataset(
        observations=np.array(observations),
        actions=np.array(actions),
        rewards=np.array(rewards),
        terminals=np.array(terminals),
    )
    return float(np.mean(episode_rewards)), dataset


def ensure_numpy_array(x, env_name):
    if isinstance(gym.make(env_name).observation_space, gym.spaces.Discrete):
        # Convert discrete observations to one-hot encoded arrays
        return np.eye(gym.make(env_name).observation_space.n)[x]
    if not isinstance(x, np.ndarray):
        return np.array([x])
    return x


def truncate_dataset(dataset, n_eps, env_name):
    """
    Truncate the dataset to the specified number of episodes.
    """
    observations, actions, rewards, terminals = [], [], [], []
    # Extract observations, actions, rewards, and terminals from the original dataset
    for episode in dataset.episodes[:n_eps]:
        observations += [
            ensure_numpy_array(o, env_name) for o in episode.observations
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


def create_d3rlpy_model(env_name, batch_size, learning_rate, gamma, target_update_interval, gpu, awac=False):
    if not awac:
        # Determine the algorithm based on the action space
        if isinstance(
            gym.make(env_name).action_space, gym.spaces.Box
        ):  # Continuous action space
            model = d3rlpy.algos.SACConfig(
                batch_size=batch_size,
                gamma=gamma,
            ).create(device=gpu)
        else:  # Discrete action space
            model = d3rlpy.algos.DoubleDQNConfig(
                batch_size=batch_size,
                learning_rate=learning_rate,
                gamma=gamma,
                target_update_interval=target_update_interval,
            ).create(device=gpu)
    else:
        # AWAC only works with continuous action spaces
        if isinstance(
            gym.make(env_name).action_space, gym.spaces.Box
        ):  # Continuous action space
            model = d3rlpy.algos.AWACConfig(
                batch_size=batch_size,
                gamma=gamma,
            ).create(device=gpu)
        else:  # Discrete action space - fall back to DoubleDQN
            print(f"Warning: AWAC is not compatible with discrete action spaces. Using DoubleDQN instead for {env_name}")
            model = d3rlpy.algos.DoubleDQNConfig(
                batch_size=batch_size,
                learning_rate=learning_rate,
                gamma=gamma,
                target_update_interval=target_update_interval,
            ).create(device=gpu)
    return model


def create_random_model(env_name, gpu):
    # Determine the algorithm based on the action space
    if isinstance(
        gym.make(env_name).action_space, gym.spaces.Box
    ):  # Continuous action space
        model = d3rlpy.algos.RandomPolicyConfig().create(device=gpu)
    else:  # Discrete action space
        model = d3rlpy.algos.DiscreteRandomPolicyConfig().create(device=gpu)
    return model


def pretrain_from_llm(dataset_path, hyperparams, model_size, suffix):
    with open(dataset_path, "rb") as file:
        dataset = pickle.load(file)
    dataset_new = truncate_dataset(dataset, hyperparams["n_pretrain_eps"], hyperparams["env"])

    model = create_d3rlpy_model(hyperparams["env"], hyperparams["batch_size"], hyperparams["learning_rate"], hyperparams["gamma"], hyperparams["target_update_interval"], hyperparams["gpu"], hyperparams["awac"])
    # start offline training
    model.fit(
        dataset_new,
        n_steps=hyperparams["n_pretrain_steps"],
        n_steps_per_epoch=hyperparams["n_steps_per_epoch"],
    )
    with open(
        f'models/{hyperparams["env"].split("-")[0]}_ddqn_pretrain_{model_size}_{hyperparams["n_pretrain_steps"]}_steps_{hyperparams["n_pretrain_eps"]}{suffix}.pkl',
        "wb",
    ) as file:
        pickle.dump(model, file)
    return model



def get_env_and_eval_env(env_name, seed):
    # d3rlpy supports both Gym and Gymnasium
    if "Represented" in env_name:
        env = GymCompatWrapper2(gym.make(env_name))
        eval_env = GymCompatWrapper2(gym.make(env_name))
    elif isinstance(
        gym.make(env_name).observation_space, gym.spaces.Discrete
    ):
        env = OneHotWrapper(gym.make(env_name))
        eval_env = OneHotWrapper(gym.make(env_name))
    else:
        env = gym.make(env_name)
        eval_env = gym.make(env_name)
    # fix seed
    d3rlpy.seed(seed)
    d3rlpy.envs.seed_env(env, seed)
    d3rlpy.envs.seed_env(eval_env, seed)
    np.random.seed(seed)
    return env, eval_env


def load_dataset_to_buffer(buffer, n_pretrain_eps, data_path, env_name):
    try:
        n_pretrain_eps_temp = n_pretrain_eps
        # Load dataset with proper validation
        with open(data_path, "rb") as f:
            dataset = pickle.load(f)
        dataset_new = truncate_dataset(dataset, n_pretrain_eps, env_name)

        # Append episodes with transition validation
        for episode in dataset_new.episodes:
            if len(episode) > 0 and hasattr(episode, "rewards"):
                buffer.append_episode(episode)
                n_pretrain_eps_temp -= 1
            else:
                print(f"Skipping invalid episode: {episode}")
            if n_pretrain_eps_temp <= 0:
                break
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")

    # Configure training with safety checks
    if buffer.transition_count == 0:
        print("Empty buffer")
    return buffer, n_pretrain_eps_temp


def rollout_and_eval(max_episode_len, env_name, explorer, algorithm, buffer, n_rollouts, seed):
    rewards = []
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    env, eval_env = get_env_and_eval_env(env_name, seed)
    
    # Check if this is a random policy (which doesn't need training)
    is_random_policy = isinstance(algorithm, (d3rlpy.algos.RandomPolicy, d3rlpy.algos.DiscreteRandomPolicy))
    
    for _ in trange(n_rollouts):
        if not is_random_policy:
            # Only call fit_online for trainable algorithms
            algorithm.fit_online(
                env=env,
                buffer=buffer,
                explorer=explorer,
                n_steps=max_episode_len,
                experiment_name=f"{timestamp}_online_training",
            )
        else:
            # For random policies, collect data and add to buffer
            # Run one episode to collect data
            observation, _ = env.reset()
            episode_observations = [observation]
            episode_actions = []
            episode_rewards = []
            episode_terminals = []
            count = 0
            
            while True:
                # Random policies don't need exploration, they're already random
                if isinstance(observation, np.ndarray):
                    observation_input = np.expand_dims(observation, axis=0)
                elif isinstance(observation, (tuple, list)):
                    observation_input = [np.expand_dims(o, axis=0) for o in observation]
                else:
                    observation_input = observation
                
                action = algorithm.predict(observation_input)[0]
                observation, reward, done, truncated, _ = env.step(action)
                
                episode_observations.append(observation)
                episode_actions.append(action)
                episode_rewards.append(reward)
                count += 1
                
                if count >= max_episode_len:
                    done = True
                
                episode_terminals.append(int(done or truncated))
                
                if done or truncated:
                    break
            
            # Create MDPDataset from the collected data and add to buffer
            # Convert to numpy arrays and create dataset
            episode_dataset = d3rlpy.dataset.MDPDataset(
                observations=np.array(episode_observations),
                actions=np.array(episode_actions),
                rewards=np.array(episode_rewards),
                terminals=np.array(episode_terminals),
            )
            
            # Add episodes from the dataset to the buffer
            for episode in episode_dataset.episodes:
                if len(episode) > 0 and hasattr(episode, "rewards"):
                    buffer.append_episode(episode)
        
        if env_name == "CliffWalking-v0":
            r, _ = evaluate_qlearning_with_environment(
                algorithm, eval_env, max_episode_len
            )
        else:
            env_evaluator = EnvironmentEvaluator(env, n_trials=1)
            r = env_evaluator(algorithm, dataset=None)
        rewards.append(r)
    
    # Evaluate the final policy on the evaluation environment
    _, dataset = evaluate_qlearning_with_environment(
        algorithm, eval_env, max_episode_len
    )
    return rewards, dataset, buffer


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


def write_timing_log(hyperparams, timing_data, log_filename=None):
    """
    Write timing information and hyperparameters to a log file.
    
    Args:
        hyperparams: Dictionary containing hyperparameters
        timing_data: Dictionary containing timing information
        log_filename: Optional custom filename. If None, generates timestamp-based name.
    """
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Generate timestamp-based filename if not provided
    if log_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f'logs/timing_log_{timestamp}.json'
    
    # Prepare log data
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'hyperparameters': hyperparams,
        'timing_data': timing_data
    }
    
    # Write to file
    with open(log_filename, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"Timing log written to: {log_filename}")
    return log_filename