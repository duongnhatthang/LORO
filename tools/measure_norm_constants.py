"""Measure x_random (uniform-random policy) and x_expert (converged/known-optimal) per env.

x_random: mean episode return of env.action_space.sample() over N episodes (H=200 cap).
x_expert: known optimum where available, else the mean return of a DoubleDQN/SAC trained
          to convergence (long run). For envs with a known optimum we use it directly:
            CartPole 200, FrozenLake 1.0, MountainCar -110 (approx solve), CliffWalking -13,
            Pendulum ~ -150 (near-upright), RepresentedPong 21.
Writes data/norm_constants.json (x_expert kept from known values; x_random measured).
"""
import json
import os
import sys
import numpy as np

# Ensure project root is on sys.path so vis_utils is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from vis_utils import DEFAULT_NORM_CONST_DICT

N_EPISODES = 100
H = 200
ENVS = ["CartPole-v0", "Pendulum-v1", "MountainCar-v0",
        "FrozenLake-v1", "CliffWalking-v0", "RepresentedPong-v0"]


def measure_random(env_name):
    env = gym.make(env_name)
    returns = []
    for _ in range(N_EPISODES):
        _obs, _ = env.reset()
        total, count, done = 0.0, 0, False
        while not done and count < H:
            _obs, r, term, trunc, _ = env.step(env.action_space.sample())
            total += float(r); count += 1; done = term or trunc
        returns.append(total)
    return float(np.mean(returns))


if __name__ == "__main__":
    out = {}
    for env_name in ENVS:
        base = env_name.split("-")[0]
        try:
            x_random = measure_random(env_name)
        except Exception as e:
            print(f"skip {base}: {e}"); continue
        x_expert = DEFAULT_NORM_CONST_DICT[base]["x_expert"]  # keep known-optimal expert
        out[base] = {"x_random": x_random, "x_expert": x_expert}
        print(f"{base}: x_random={x_random:.2f} x_expert={x_expert}")
    _out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "norm_constants.json")
    with open(_out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {_out_path}")
