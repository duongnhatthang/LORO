# Improving the Data-efficiency of Reinforcement Learning by Warm-starting with LLM

This repository is the official implementation of [Improving the Data-efficiency of Reinforcement Learning by Warm-starting with LLM](placeholder). 

![](https://github.com/duongnhatthang/LlamaGym/blob/main/figs/loro.png)

## Requirements

### Prerequisites
- Python 3.9+
- CUDA-compatible GPU (recommended for LLM inference and RL training)
- Hugging Face account (for accessing LLM models)

### Installation

To install requirements:

```bash
pip install -r requirements.txt
pip install git+https://github.com/mila-iqia/atari-representation-learning.git
pip install d4rl
```

Install Mujoco (required for some environments): https://gist.github.com/saratrajput/60b1310fe9d9df664f9983b38b50d5da

### Environment Variables

Set up the following environment variables before running the scripts:

```bash
export HF_TOKEN="your_huggingface_token"  # Required for accessing Hugging Face models
export WANDB_PROJECT="your_wandb_project"  # Optional: for experiment tracking
```

## Supported Environments

The following OpenAI Gym environments are supported:
- `CartPole-v0` - Classic control balance task
- `Pendulum-v1` - Classic control swing-up task
- `MountainCar-v0` - Classic control car climbing task
- `FrozenLake-v1` - Toy text grid navigation
- `CliffWalking-v0` - Toy text cliff avoidance
- `RepresentedPong-v0` - Atari Pong with RAM extracted representations

## Project Structure

```
LORO/
├── llm_main.py              # LLM data collection script
├── pretrain_from_llm.py     # Offline pretraining from LLM data
├── online_main.py           # Online fine-tuning experiments
├── run_mixed_pretraining.py # Mixed pretraining experiments
├── on_policy_pretrain_exp.py # On-policy (pure RL) baseline
├── main.sh                  # Full pipeline script (LLM + fine-tuning)
├── run_all_main.sh          # Batch experiments across all environments
├── run_all_mixed_pretraining.sh # Batch mixed pretraining experiments
├── utils.py                 # Utility functions
├── vis_utils.py             # Visualization utilities
├── visualization.ipynb      # Results visualization notebook
├── coverage_vis.ipynb       # Coverage visualization notebook
├── env/                     # Environment wrappers and translators
│   ├── classic_control/     # CartPole, Pendulum, MountainCar
│   ├── toy_text/            # FrozenLake, CliffWalking
│   └── atari/               # Atari environments
├── data/                    # Collected datasets and cache files
├── models/                  # Saved model checkpoints
├── figs/                    # Generated figures
└── logs/                    # Training logs
```

## Training

### Step 1: Collect Data from LLM

Use the LLM to collect trajectory data for a specific environment:

```bash
python llm_main.py --env CartPole-v0 --model_name Qwen/Qwen2.5-7B-Instruct --n_episodes 30
```

**Available arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `Qwen/Qwen2.5-7B-Instruct` | Hugging Face model name |
| `--env` | `CliffWalking-v0` | Environment name |
| `--n_episodes` | `30` | Number of episodes to collect |
| `--max_episode_len` | `200` | Maximum steps per episode |
| `--SFT` | `False` | Enable supervised fine-tuning |
| `--seed` | `42069` | Random seed |
| `--batch_size` | `1` | Batch size for LLM inference |
| `--eps` | `0.0` | Epsilon for exploration |
| `--quantization` | `none` | Quantization: `none`, `4bit`, or `8bit` |

The collected data will be saved to `data/{env}_{model}_Neps_{n_episodes}.pkl`.

### Step 2: Pretrain from LLM Data (Optional)

For standalone offline pretraining from collected LLM data:

```bash
python pretrain_from_llm.py
```

Note: To use data collected from other LLMs, configure the `get_llm_data_paths()` function in the script to specify the data file paths for 7B and 32B model data to be used in the pre-train and fine-tuning steps below.

### Step 3: Online Fine-tuning

Run the main fine-tuning experiments that combine LLM pretraining with online RL:

```bash
python online_main.py --env CartPole-v0 --n_pretrain_eps 10 --n_online_eps 190 --online_exp
```

**Available arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--env` | `CliffWalking-v0` | Environment name |
| `--n_pretrain_eps` | `10` | Number of pretraining episodes |
| `--n_online_eps` | `190` | Number of online episodes |
| `--n_pretrain_steps` | `1000` | Number of pretraining gradient steps |
| `--max_episode_len` | `200` | Maximum steps per episode |
| `--seed` | `42069` | Random seed |
| `--eps` | `0.1` | Epsilon for exploration |
| `--n_exp` | `5` | Number of experiment repetitions |
| `--gpu` / `--no-gpu` | `True` | Enable/disable GPU |
| `--buffer_size` | `100000` | Replay buffer size |
| `--batch_size` | `256` | Training batch size |
| `--learning_rate` | `5e-5` | Learning rate |
| `--gamma` | `0.99` | Discount factor |
| `--target_update_interval` | `1000` | Target network update frequency |
| `--model` | `default` | Model type: `default` (SAC/DoubleDQN), `awac`, or `ddpg` |
| `--sft` | `False` | Use SFT data paths |
| `--long_cot` | `False` | Use DeepSeek long CoT data paths |
| `--online_exp` | `False` | Run main fine-tune experiments |
| `--online_rand` | `False` | Run random/online baseline experiments |

### Step 4: Mixed Pretraining (Optional)

Run mixed pretraining experiments that combine LLM and online data:

```bash
python run_mixed_pretraining.py --env CartPole-v0 --n_pretrain_eps 30
```

This script loads existing cache files, runs mixed pretraining, and saves updated results.

### Baseline: Pure RL Training

To collect on-policy (pure RL) data and run baseline experiments:

```bash
python on_policy_pretrain_exp.py
```

## Running Batch Experiments

For convenience, shell scripts are provided to run the complete pipeline or batch experiments across all environments.

### Full Pipeline: `main.sh`

The `main.sh` script runs the complete LORO pipeline (LLM data collection + online fine-tuning) for a single environment:

```bash
# Run full pipeline for CartPole
./main.sh --env CartPole-v0

# Run with custom LLM models
./main.sh --env CartPole-v0 --model_name_1 Qwen/Qwen2.5-7B-Instruct --model_name_2 Qwen/Qwen2.5-32B-Instruct

# Skip LLM data collection (use existing data)
./main.sh --env CartPole-v0 --model_name_1 none --model_name_2 none
```

### Batch Experiments: `run_all_main.sh`

Run experiments across all environments automatically:

```bash
# Run all experiments for all environments (default)
./run_all_main.sh

# Run for a specific environment with custom parameters
./run_all_main.sh --env CartPole-v0 --n_pretrain_eps_1 20

# Run all experiments in parallel (faster, requires more resources)
./run_all_main.sh --parallel

# Show all available options
./run_all_main.sh --help
```

**Key arguments:**
| Argument | Description |
|----------|-------------|
| `--env` | Run for specific environment only |
| `--model_name_1/2` | LLM models to use (or `none` to skip) |
| `--n_pretrain_eps_1/2/3` | Pretraining episodes for each run (default: 10, 20, 30) |
| `--n_online_eps_1/2/3` | Online episodes for each run |
| `--parallel` | Run experiments in parallel |
| `--online_exp` | Run main fine-tune experiments |
| `--online_rand` | Run random/online baseline experiments |

### Mixed Pretraining Batch: `run_all_mixed_pretraining.sh`

Run mixed pretraining experiments (combining LLM and online data) across all environments:

```bash
# Run all mixed pretraining experiments (default)
./run_all_mixed_pretraining.sh

# Run for a specific environment
./run_all_mixed_pretraining.sh --env CartPole-v0 --n_pretrain_eps 20

# Use SFT data paths
./run_all_mixed_pretraining.sh --sft

# Show all available options
./run_all_mixed_pretraining.sh --help
```

**Default experiment configurations:**
| Environment | Online Episodes | Pretrain Episodes |
|-------------|-----------------|-------------------|
| CartPole-v0 | 140, 130, 120 | 10, 20, 30 |
| FrozenLake-v1 | 140, 130, 120 | 10, 20, 30 |
| MountainCar-v0 | 290, 280, 270 | 10, 20, 30 |
| Pendulum-v1 | 190, 180, 170 | 10, 20, 30 |
| RepresentedPong-v0 | 190, 180, 170 | 10, 20, 30 |
| CliffWalking-v0 | 190, 180, 170 | 10, 20, 30 |

## Evaluation

To visualize the results, use the provided Jupyter notebooks:

1. **Main Results**: Open and run `visualization.ipynb` to reproduce the figures shown in the paper.
2. **Coverage Analysis**: Open and run `coverage_vis.ipynb` to visualize state-action coverage.

The notebooks load cached experiment results from `data/cache_*.pkl` files and generate plots saved to `figs/`.

## Results

Our model achieves the following performance on six OpenAI Gym environments:

![](https://github.com/duongnhatthang/LlamaGym/blob/main/figs/main_results.png)

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Try using quantization with `--quantization 4bit` or `--quantization 8bit` when running `llm_main.py`.

2. **HF_TOKEN not set**: Ensure you have set the Hugging Face token:
   ```bash
   export HF_TOKEN="your_token_here"
   ```

3. **Missing Atari ROMs**: Install ROMs with:
   ```bash
   pip install autorom
   AutoROM --accept-license
   ```

4. **Mujoco installation issues**: Follow the detailed guide at https://gist.github.com/saratrajput/60b1310fe9d9df664f9983b38b50d5da

## Abstract

We investigate the usage of Large Language Models (LLMs) in collecting high-quality data to warm-start Reinforcement Learning (RL) algorithms for learning in Markov Decision Processes (MDPs). Specifically, we leverage the in-context decision-making capability of LLMs, to generate an "offline" dataset that sufficiently covers state-actions visited by some good policy, then use an off-the-shelf RL algorithm to further explore the environment and fine-tune its policy, in a black-box manner. By pretraining with LLM collected data, the learning algorithm can both converge to an optimal policy and have a high sample efficiency thanks to the good data coverage collected by the LLM. On multiple OpenAI Gym environments, such as CartPole and Pendulum, given the same environment interaction budget, we empirically demonstrate that LLM-pretrain outperforms baseline algorithms such as pure LLM-based policies, pure RL, and a naive combination of the two.

## Contributing

[Apache 2.0](https://github.com/duongnhatthang/LlamaGym/blob/main/LICENSE)

## Acknowledgments

The code is referencing this [repo](https://github.com/KhoomeiK/LlamaGym). The environment descriptions are referenced from this [repo](https://github.com/mail-ecnu/Text-Gym-Agents).

## Citation

```bibtex
@article{loro2025,
  title={Improving the Data-efficiency of Reinforcement Learning by Warm-starting with LLM},
  author={Placeholder},
  journal={Placeholder},
  year={2025}
}
```
