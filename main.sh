#!/bin/bash

# Main script to run both llm_main.py and online_main.py in sequence
# This script provides a unified interface for running the complete LORO pipeline

set -e  # Exit on any error

# Default values
MODEL_NAME_1="Qwen/Qwen2.5-7B-Instruct"
MODEL_NAME_2="Qwen/Qwen2.5-32B-Instruct"
ENV="CliffWalking-v0"
N_EPISODES=30
MAX_EPISODE_LEN=200
SFT=false
SEED=42069
BATCH_SIZE=1
EPS=0.0
QUANTIZATION="4bit"
N_ONLINE_EPS_1=190
N_ONLINE_EPS_2=180
N_ONLINE_EPS_3=170
N_PRETRAIN_EPS_1=10
N_PRETRAIN_EPS_2=20
N_PRETRAIN_EPS_3=30
N_EXP=5
GPU=true
BUFFER_SIZE=100000
LEARNING_RATE=5e-5
GAMMA=0.99
TARGET_UPDATE_INTERVAL=1000
N_PRETRAIN_STEPS=1000
LONG_COT=false
AWAC=false
N_STEPS_PER_EPOCH=200
ONLINE_EXP=true
ONLINE_RAND=true

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --model_name_1 MODEL         First model name for LLM, or 'none' to skip (default: $MODEL_NAME_1)"
    echo "  --model_name_2 MODEL         Second model name for LLM, or 'none' to skip (default: $MODEL_NAME_2)"
    echo "  --env ENV                    Environment name (default: $ENV)"
    echo "  --n_episodes N               Number of episodes for LLM training (default: $N_EPISODES)"
    echo "  --max_episode_len N          Maximum episode length (default: $MAX_EPISODE_LEN)"
    echo "  --SFT                        Use supervised fine-tuning (default: false)"
    echo "  --seed N                     Random seed (default: $SEED)"
    echo "  --batch_size N               Batch size (default: $BATCH_SIZE)"
    echo "  --eps FLOAT                  Epsilon for exploration (default: $EPS)"
    echo "  --quantization METHOD        Quantization method: none, 4bit, or 8bit (default: $QUANTIZATION)"
    echo "  --n_online_eps_1 N           Number of online episodes for run 1 (default: $N_ONLINE_EPS_1)"
    echo "  --n_online_eps_2 N           Number of online episodes for run 2 (default: $N_ONLINE_EPS_2)"
    echo "  --n_online_eps_3 N           Number of online episodes for run 3 (default: $N_ONLINE_EPS_3)"
    echo "  --n_pretrain_eps_1 N         Number of pretraining episodes for run 1 (default: $N_PRETRAIN_EPS_1)"
    echo "  --n_pretrain_eps_2 N         Number of pretraining episodes for run 2 (default: $N_PRETRAIN_EPS_2)"
    echo "  --n_pretrain_eps_3 N         Number of pretraining episodes for run 3 (default: $N_PRETRAIN_EPS_3)"
    echo "  --n_exp N                    Number of experiments (default: $N_EXP)"
    echo "  --no-gpu                     Disable GPU usage (default: enabled)"
    echo "  --buffer_size N              Replay buffer size (default: $BUFFER_SIZE)"
    echo "  --learning_rate FLOAT        Learning rate (default: $LEARNING_RATE)"
    echo "  --gamma FLOAT                Discount factor (default: $GAMMA)"
    echo "  --target_update_interval N   Target network update interval (default: $TARGET_UPDATE_INTERVAL)"
    echo "  --n_pretrain_steps N         Number of pretraining steps (default: $N_PRETRAIN_STEPS)"
    echo "  --long_cot                   Use DeepSeek long CoT data paths (default: false)"
    echo "  --awac                       Use AWAC model instead of SAC or DoubleDQN (default: false)"
    echo "  --n_steps_per_epoch N        Number of steps per epoch for training (default: $N_STEPS_PER_EPOCH)"
    echo "  --online_exp                 Run the main fine-tune experiments (default: true)"
    echo "  --online_rand                Run the random and online fine-tune experiments (default: true)"
    echo "  --help                       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --env CartPole-v0 --n_episodes 50"
    echo "  $0 --env CliffWalking-v0 --SFT --n_online_eps 100"
    echo "  $0 --model_name_1 deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --model_name_2 Qwen/Qwen2.5-7B-Instruct --env MountainCar-v0"
    echo "  $0 --model_name_1 Qwen/Qwen2.5-7B-Instruct --model_name_2 none --env CartPole-v0"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name_1)
            MODEL_NAME_1="$2"
            shift 2
            ;;
        --model_name_2)
            MODEL_NAME_2="$2"
            shift 2
            ;;
        --env)
            ENV="$2"
            shift 2
            ;;
        --n_episodes)
            N_EPISODES="$2"
            shift 2
            ;;
        --max_episode_len)
            MAX_EPISODE_LEN="$2"
            shift 2
            ;;
        --SFT)
            SFT=true
            shift
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --eps)
            EPS="$2"
            shift 2
            ;;
        --quantization)
            QUANTIZATION="$2"
            shift 2
            ;;
        --n_online_eps_1)
            N_ONLINE_EPS_1="$2"
            shift 2
            ;;
        --n_online_eps_2)
            N_ONLINE_EPS_2="$2"
            shift 2
            ;;
        --n_online_eps_3)
            N_ONLINE_EPS_3="$2"
            shift 2
            ;;
        --n_pretrain_eps_1)
            N_PRETRAIN_EPS_1="$2"
            shift 2
            ;;
        --n_pretrain_eps_2)
            N_PRETRAIN_EPS_2="$2"
            shift 2
            ;;
        --n_pretrain_eps_3)
            N_PRETRAIN_EPS_3="$2"
            shift 2
            ;;
        --n_exp)
            N_EXP="$2"
            shift 2
            ;;
        --no-gpu)
            GPU=false
            shift
            ;;
        --buffer_size)
            BUFFER_SIZE="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --gamma)
            GAMMA="$2"
            shift 2
            ;;
        --target_update_interval)
            TARGET_UPDATE_INTERVAL="$2"
            shift 2
            ;;
        --n_pretrain_steps)
            N_PRETRAIN_STEPS="$2"
            shift 2
            ;;
        --long_cot)
            LONG_COT=true
            shift
            ;;
        --awac)
            AWAC=true
            shift
            ;;
        --n_steps_per_epoch)
            N_STEPS_PER_EPOCH="$2"
            shift 2
            ;;
        --online_exp)
            ONLINE_EXP=true
            shift
            ;;
        --online_rand)
            ONLINE_RAND=true
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Create data directory if it doesn't exist
mkdir -p data
mkdir -p logs

echo "=========================================="
echo "LORO Pipeline Execution"
echo "=========================================="
echo "Environment: $ENV"
echo "Model 1: $MODEL_NAME_1"
echo "Model 2: $MODEL_NAME_2"
echo "LLM Episodes: $N_EPISODES"
echo "Online Episodes 1: $N_ONLINE_EPS_1"
echo "Online Episodes 2: $N_ONLINE_EPS_2"
echo "Online Episodes 3: $N_ONLINE_EPS_3"
echo "Pretrain Episodes 1: $N_PRETRAIN_EPS_1"
echo "Pretrain Episodes 2: $N_PRETRAIN_EPS_2"
echo "Pretrain Episodes 3: $N_PRETRAIN_EPS_3"
echo "Seed: $SEED"
echo "SFT: $SFT"
echo "GPU: $GPU"
echo "AWAC: $AWAC"
echo "Online Exp: $ONLINE_EXP"
echo "Online Rand: $ONLINE_RAND"
echo "=========================================="

# Build base arguments for llm_main.py (without model_name)
LLM_BASE_ARGS="--env $ENV --n_episodes $N_EPISODES --max_episode_len $MAX_EPISODE_LEN --seed $SEED --batch_size $BATCH_SIZE --eps $EPS"
if [ "$SFT" = true ]; then
    LLM_BASE_ARGS="$LLM_BASE_ARGS --SFT"
fi
LLM_BASE_ARGS="$LLM_BASE_ARGS --quantization $QUANTIZATION"

# Build arguments for online_main.py (will be customized for each run)
ONLINE_BASE_ARGS="--env $ENV --max_episode_len $MAX_EPISODE_LEN --seed $SEED --eps $EPS --n_exp $N_EXP --buffer_size $BUFFER_SIZE --batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE --gamma $GAMMA --target_update_interval $TARGET_UPDATE_INTERVAL --n_pretrain_steps $N_PRETRAIN_STEPS --n_steps_per_epoch $N_STEPS_PER_EPOCH"
if [ "$GPU" = true ]; then
    ONLINE_BASE_ARGS="$ONLINE_BASE_ARGS --gpu"
fi
if [ "$SFT" = true ]; then
    ONLINE_BASE_ARGS="$ONLINE_BASE_ARGS --sft"
fi
if [ "$LONG_COT" = true ]; then
    ONLINE_BASE_ARGS="$ONLINE_BASE_ARGS --long_cot"
fi
if [ "$AWAC" = true ]; then
    ONLINE_BASE_ARGS="$ONLINE_BASE_ARGS --awac"
fi
if [ "$ONLINE_EXP" = true ]; then
    ONLINE_BASE_ARGS="$ONLINE_BASE_ARGS --online_exp"
fi
if [ "$ONLINE_RAND" = true ]; then
    ONLINE_BASE_ARGS="$ONLINE_BASE_ARGS --online_rand"
fi

# Step 1: Run LLM training with Model 1 (skip if MODEL_NAME_1 is "none")
if [ "$MODEL_NAME_1" != "none" ]; then
    echo "Step 1: Running LLM training with Model 1..."
    echo "Command: python llm_main.py --model_name $MODEL_NAME_1 $LLM_BASE_ARGS"
    echo ""

    python llm_main.py --model_name "$MODEL_NAME_1" $LLM_BASE_ARGS

    if [ $? -ne 0 ]; then
        echo "Error: LLM training with Model 1 failed!"
        exit 1
    fi

    echo ""
    echo "LLM training with Model 1 completed successfully!"
    echo ""
else
    echo "Step 1: Skipping LLM training with Model 1 (MODEL_NAME_1 is set to 'none')"
    echo ""
fi

# Step 2: Run LLM training with Model 2 (skip if MODEL_NAME_2 is "none")
if [ "$MODEL_NAME_2" != "none" ]; then
    echo "Step 2: Running LLM training with Model 2..."
    echo "Command: python llm_main.py --model_name $MODEL_NAME_2 $LLM_BASE_ARGS"
    echo ""

    python llm_main.py --model_name "$MODEL_NAME_2" $LLM_BASE_ARGS

    if [ $? -ne 0 ]; then
        echo "Error: LLM training with Model 2 failed!"
        exit 1
    fi

    echo ""
    echo "LLM training with Model 2 completed successfully!"
    echo ""
else
    echo "Step 2: Skipping LLM training with Model 2 (MODEL_NAME_2 is set to 'none')"
    echo ""
fi

# Step 3: Run online training with first pair
echo "Step 3: Running online training with first pair..."
ONLINE_ARGS_1="$ONLINE_BASE_ARGS --n_online_eps $N_ONLINE_EPS_1 --n_pretrain_eps $N_PRETRAIN_EPS_1"

echo "Command: python online_main.py $ONLINE_ARGS_1"
echo ""

python online_main.py $ONLINE_ARGS_1

if [ $? -ne 0 ]; then
    echo "Error: Online training with first pair failed!"
    exit 1
fi

echo ""
echo "Online training with first pair completed successfully!"
echo ""

# Step 4: Run online training with second pair
echo "Step 4: Running online training with second pair..."
ONLINE_ARGS_2="$ONLINE_BASE_ARGS --n_online_eps $N_ONLINE_EPS_2 --n_pretrain_eps $N_PRETRAIN_EPS_2"

echo "Command: python online_main.py $ONLINE_ARGS_2"
echo ""

python online_main.py $ONLINE_ARGS_2

if [ $? -ne 0 ]; then
    echo "Error: Online training with second pair failed!"
    exit 1
fi

echo ""
echo "Online training with second pair completed successfully!"
echo ""

# Step 5: Run online training with third pair
echo "Step 5: Running online training with third pair..."
ONLINE_ARGS_3="$ONLINE_BASE_ARGS --n_online_eps $N_ONLINE_EPS_3 --n_pretrain_eps $N_PRETRAIN_EPS_3"

echo "Command: python online_main.py $ONLINE_ARGS_3"
echo ""

python online_main.py $ONLINE_ARGS_3

if [ $? -ne 0 ]; then
    echo "Error: Online training with third pair failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "LORO Pipeline completed successfully!"
echo "=========================================="
echo ""
echo "Generated files:"
if [ "$MODEL_NAME_1" != "none" ]; then
    echo "- LLM dataset (Model 1): data/${ENV%%-*}_${MODEL_NAME_1##*/}_Neps_${N_EPISODES}$([ "$SFT" = true ] && echo "SFT" || echo "").pkl"
fi
if [ "$MODEL_NAME_2" != "none" ]; then
    echo "- LLM dataset (Model 2): data/${ENV%%-*}_${MODEL_NAME_2##*/}_Neps_${N_EPISODES}$([ "$SFT" = true ] && echo "SFT" || echo "").pkl"
fi
echo "- Online results (Run 1): data/cache_${ENV%%-*}_Neps_${N_PRETRAIN_EPS_1}$([ "$SFT" = true ] && echo "SFT" || [ "$LONG_COT" = true ] && echo "LCOT" || echo "").pkl"
echo "- Online results (Run 2): data/cache_${ENV%%-*}_Neps_${N_PRETRAIN_EPS_2}$([ "$SFT" = true ] && echo "SFT" || [ "$LONG_COT" = true ] && echo "LCOT" || echo "").pkl"
echo "- Online results (Run 3): data/cache_${ENV%%-*}_Neps_${N_PRETRAIN_EPS_3}$([ "$SFT" = true ] && echo "SFT" || [ "$LONG_COT" = true ] && echo "LCOT" || echo "").pkl"
echo "- Timing logs: logs/"
echo ""
