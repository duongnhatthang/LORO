import os
import time
import json
from tqdm import trange
from datetime import datetime
import argparse

from transformers import AutoTokenizer
from peft import LoraConfig
from trl import AutoModelForCausalLMWithValueHead

import gymnasium as gym
import numpy as np
import d3rlpy
import pickle

from env.translation_agent import SpaceInvadersAgent, PongAgent
from env import classic_control, toy_text, translation_agent
from env.atari.represented_atari_game import GymCompatWrapper


def get_agent(model, tokenizer, device, hyperparams):
    """
    Returns an instance of the Agent with the provided model, tokenizer, device, and hyperparameters.
    """
    parser = argparse.ArgumentParser(description="Place holder args to init stuff.")
    parser.add_argument(
        "--is_only_local_obs",
        type=int,
        default=1,
        help="Whether only taking local observations, if is_only_local_obs = 1, only using local obs",
    )
    parser.add_argument(
        "--max_episode_len",
        type=int,
        default=hyperparams["max_episode_len"],
        help="The maximum number of steps in an episode",
    )
    args = parser.parse_args()
    # Create the agent with the specified parameters
    if hyperparams["env"] == "RepresentedSpaceInvaders-v0":
        print("Creating SpaceInvadersAgent")
        agent = SpaceInvadersAgent(
            model=model,
            tokenizer=tokenizer,
            device=device,
            generate_config_dict={
                key: value
                for key, value in hyperparams.items()
                if key.startswith("generate/")
            },
            ppo_config_dict={
                "batch_size": hyperparams["batch_size"],
                "mini_batch_size": hyperparams["batch_size"],
            },
        )
    elif hyperparams["env"] == "RepresentedPong-v0":
        print("Creating PongAgent")
        agent = PongAgent(
            model=model,
            tokenizer=tokenizer,
            device=device,
            generate_config_dict={
                key: value
                for key, value in hyperparams.items()
                if key.startswith("generate/")
            },
            ppo_config_dict={
                "batch_size": hyperparams["batch_size"],
                "mini_batch_size": hyperparams["batch_size"],
            },
        )
    elif hyperparams["env"] == "CartPole-v0":
        print("Creating PongAgent")
        agent = translation_agent.CartPoleAgent(
            model=model,
            tokenizer=tokenizer,
            device=device,
            generate_config_dict={
                key: value
                for key, value in hyperparams.items()
                if key.startswith("generate/")
            },
            ppo_config_dict={
                "batch_size": hyperparams["batch_size"],
                "mini_batch_size": hyperparams["batch_size"],
            },
            obs_translator=classic_control.cartpole_translator.ObsTranslator(),
            game_describer=classic_control.cartpole_translator.GameDescriber(args),
        )
    elif hyperparams["env"] == "Acrobot-v0":
        print("Creating AcrobotAgent")
        agent = translation_agent.AcrobotAgent(
            model=model,
            tokenizer=tokenizer,
            device=device,
            generate_config_dict={
                key: value
                for key, value in hyperparams.items()
                if key.startswith("generate/")
            },
            ppo_config_dict={
                "batch_size": hyperparams["batch_size"],
                "mini_batch_size": hyperparams["batch_size"],
            },
            obs_translator=classic_control.acrobot_translator.ObsTranslator(),
            game_describer=classic_control.acrobot_translator.GameDescriber(args),
        )
    elif hyperparams["env"] == "MountainCar-v0":
        print("Creating MountainCarAgent")
        agent = translation_agent.MountainCarAgent(
            model=model,
            tokenizer=tokenizer,
            device=device,
            generate_config_dict={
                key: value
                for key, value in hyperparams.items()
                if key.startswith("generate/")
            },
            ppo_config_dict={
                "batch_size": hyperparams["batch_size"],
                "mini_batch_size": hyperparams["batch_size"],
            },
            obs_translator=classic_control.mountaincar_translator.ObsTranslator(),
            game_describer=classic_control.mountaincar_translator.GameDescriber(args),
        )
    elif hyperparams["env"] == "FrozenLake-v1":
        print("Creating FrozenLakeAgent")
        agent = translation_agent.FrozenLakeAgent(
            model=model,
            tokenizer=tokenizer,
            device=device,
            generate_config_dict={
                key: value
                for key, value in hyperparams.items()
                if key.startswith("generate/")
            },
            ppo_config_dict={
                "batch_size": hyperparams["batch_size"],
                "mini_batch_size": hyperparams["batch_size"],
            },
            obs_translator=toy_text.frozenlake_translator.ObsTranslator(),
            game_describer=toy_text.frozenlake_translator.GameDescriber(args),
        )
    elif hyperparams["env"] == "CliffWalking-v0":
        print("Creating CliffWalkingAgent")
        agent = translation_agent.CliffWalkingAgent(
            model=model,
            tokenizer=tokenizer,
            device=device,
            generate_config_dict={
                key: value
                for key, value in hyperparams.items()
                if key.startswith("generate/")
            },
            ppo_config_dict={
                "batch_size": hyperparams["batch_size"],
                "mini_batch_size": hyperparams["batch_size"],
            },
            obs_translator=toy_text.cliffwalking_translator.ObsTranslator(),
            game_describer=toy_text.cliffwalking_translator.GameDescriber(args),
        )
    elif hyperparams["env"] == "Taxi-v3":
        print("Creating TaxiAgent")
        agent = translation_agent.TaxiAgent(
            model=model,
            tokenizer=tokenizer,
            device=device,
            generate_config_dict={
                key: value
                for key, value in hyperparams.items()
                if key.startswith("generate/")
            },
            ppo_config_dict={
                "batch_size": hyperparams["batch_size"],
                "mini_batch_size": hyperparams["batch_size"],
            },
            obs_translator=toy_text.taxi_translator.ObsTranslator(),
            game_describer=toy_text.taxi_translator.GameDescriber(args),
        )
    elif hyperparams["env"] == "Pendulum-v1":
        print("Creating PendulumAgent")
        agent = translation_agent.PendulumAgent(
            model=model,
            tokenizer=tokenizer,
            device=device,
            generate_config_dict={
                key: value
                for key, value in hyperparams.items()
                if key.startswith("generate/")
            },
            ppo_config_dict={
                "batch_size": hyperparams["batch_size"],
                "mini_batch_size": hyperparams["batch_size"],
            },
            obs_translator=classic_control.pendulum_translator.ObsTranslator(),
            game_describer=classic_control.pendulum_translator.GameDescriber(args),
        )
    else:
        assert (
            False
        ), f"Environment {hyperparams['env']} is not supported. Please provide a valid environment."
    return agent

def llm_write_timing_log(rollout_time, hyperparams, timestamp):
    """
    Write timing information and hyperparameters to a log file with timestamp.
    
    Args:
        rollout_time (float): Time taken for rollout in seconds
        hyperparams (dict): Dictionary containing hyperparameters
        timestamp (str): Timestamp string for unique file naming
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Create log data
    log_data = {
        "timestamp": timestamp,
        "rollout_time_seconds": rollout_time,
        "rollout_time_minutes": rollout_time / 60.0,
        "rollout_time_hours": rollout_time / 3600.0,
        "hyperparameters": hyperparams
    }
    
    # Create unique log filename with timestamp
    log_filename = f"logs/llm_timing_log_{timestamp}.json"
    
    # Write to log file
    with open(log_filename, "w") as f:
        json.dump(log_data, f, indent=2)
    
    print(f"Timing log written to: {log_filename}")
    print(f"Rollout completed in {rollout_time:.2f} seconds ({rollout_time/60:.2f} minutes)")

def rollout(agent, env, hyperparams):
    d3rlpy.seed(hyperparams["seed"])
    d3rlpy.envs.seed_env(env, hyperparams["seed"])
    np.random.seed(hyperparams["seed"])

    observations, actions, rewards, terminals = [], [], [], []
    counter = 0
    for episode in trange(hyperparams["n_episodes"]):
        observation, info = env.reset()
        done = False
        n_step = 0
        while not done:
            rand = bool(np.random.binomial(n=1, p=hyperparams["eps"]))
            if rand:
                action = env.action_space.sample()
            else:
                action = agent.act(observation)
            # wandb.log({"action": action})
            observation, reward, done, info = env.step(action)
            if "Cliff" in hyperparams["env"] or "Frozen" in hyperparams["env"]:
                agent.add_env_hist(observation, reward, action)
            agent.assign_reward(reward)
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            n_step += 1
            if n_step >= hyperparams["max_episode_len"]:
                done = True
            terminals.append(int(done))
            print(n_step, observation, action, reward)
        # episode_stats = {
        #     "episode": episode,
        #     "sum_return": sum(agent.current_episode_rewards),
        #     "message_ct": len(agent.current_episode_messages),
        #     "episode_messages": agent.current_episode_messages,
        # }
        train_stats = agent.terminate_episode(train=hyperparams["SFT"])
        # episode_stats.update(train_stats)
        # wandb.log(episode_stats)
        if counter > 0 and counter % int(hyperparams["n_episodes"] % 100) == 0:
            print(
                f"Episode {counter}, sum return: {sum(agent.current_episode_rewards)}"
            )
        counter += 1

    dataset = d3rlpy.dataset.MDPDataset(
        observations=np.array(observations),
        actions=np.array(actions),
        rewards=np.array(rewards),
        terminals=np.array(terminals),
    )
    return dataset

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="LORO Training Script")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model name to use for training") #"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    parser.add_argument("--env", type=str, default="CliffWalking-v0",
                        help="Environment to train on") # "CartPole-v0", "MountainCar-v0", "FrozenLake-v1", "CliffWalking-v0", Pendulum-v1, "Taxi-v3", "RepresentedPong-v0"
    parser.add_argument("--n_episodes", type=int, default=30,
                        help="Number of episodes to run")
    parser.add_argument("--max_episode_len", type=int, default=200,
                        help="Maximum episode length")
    parser.add_argument("--SFT", action="store_true", default=False,
                        help="Whether to use supervised fine-tuning")
    parser.add_argument("--seed", type=int, default=42069,
                        help="Random seed for reproducibility")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for training")
    parser.add_argument("--eps", type=float, default=0.0,
                        help="Epsilon for exploration")
    parser.add_argument("--load_in_8bit", action="store_true", default=True,
                        help="Whether to load model in 8-bit")
    
    args = parser.parse_args()
    
    hyperparams = {
        "model_name": args.model_name,
        "env": args.env,
        "lora/target_modules": [
            "q_proj",
            "up_proj",
            "o_proj",
            "k_proj",
            "down_proj",
            "gate_proj",
            "v_proj",
        ],
        "lora/r": 8,
        "lora/lora_alpha": 16,
        "lora/lora_dropout": 0.05,
        "lora/bias": "none",
        "lora/task_type": "CAUSAL_LM",
        "load_in_8bit": args.load_in_8bit,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "n_episodes": args.n_episodes,
        "generate/max_new_tokens": 2000,
        "generate/do_sample": True,
        "generate/top_p": 0.6,
        "generate/top_k": 0,
        "generate/temperature": 0.9,
        "max_episode_len": args.max_episode_len,
        "eps": args.eps,
        "SFT": args.SFT,
    }
    # wandb_run = wandb.init(project=os.environ.get("WANDB_PROJECT"), config=hyperparams)
    device = "cuda"
    HF_TOKEN = os.environ.get("HF_TOKEN")

    lora_config = LoraConfig(
        **{
            key.split("/")[-1]: value
            for key, value in hyperparams.items()
            if key.startswith("lora/")
        }
    )
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        pretrained_model_name_or_path=hyperparams["model_name"],
        peft_config=lora_config,
        load_in_8bit=hyperparams["load_in_8bit"],
        token=HF_TOKEN,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(hyperparams["model_name"], token=HF_TOKEN)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model.pretrained_model.resize_token_embeddings(len(tokenizer))

    agent = get_agent(model, tokenizer, device, hyperparams)
    if "Represented" in hyperparams["env"]:
        env = gym.make(hyperparams["env"])
    else:
        env = GymCompatWrapper(gym.make(hyperparams["env"]))

    # Generate timestamp for unique file naming
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Time the rollout process
    print("Starting rollout...")
    start_time = time.time()
    dataset = rollout(agent, env, hyperparams)
    end_time = time.time()
    rollout_time = end_time - start_time
    
    # Write timing log
    llm_write_timing_log(rollout_time, hyperparams, timestamp)
    if hyperparams["SFT"]:
        is_SFT = "SFT"
    else:
        is_SFT = ""
    with open(
        "data/"
        + hyperparams["env"].split("-")[0]
        + "_"
        + hyperparams["model_name"].split("/")[-1]
        + "_Neps_"
        + str(hyperparams["n_episodes"])
        # + "_" # Comment out timestamp to standardize the data path. Use this for debug if needed.
        # + timestamp
        + is_SFT
        + ".pkl",
        "wb",
    ) as file:
        pickle.dump(dataset, file)
