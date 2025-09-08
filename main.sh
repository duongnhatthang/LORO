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
LOAD_IN_8BIT=true
N_ONLINE_EPS=190
N_PRETRAIN_EPS=10
N_EXP=5
GPU=true
BUFFER_SIZE=100000
LEARNING_RATE=5e-5
GAMMA=0.99
TARGET_UPDATE_INTERVAL=1000
N_PRETRAIN_STEPS=1000
PRETRAINING_EXP=false
LONG_COT=false
AWAC=false

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --model_name_1 MODEL         First model name for LLM (default: $MODEL_NAME_1)"
    echo "  --model_name_2 MODEL         Second model name for LLM (default: $MODEL_NAME_2)"
    echo "  --env ENV                    Environment name (default: $ENV)"
    echo "  --n_episodes N               Number of episodes for LLM training (default: $N_EPISODES)"
    echo "  --max_episode_len N          Maximum episode length (default: $MAX_EPISODE_LEN)"
    echo "  --SFT                        Use supervised fine-tuning (default: false)"
    echo "  --seed N                     Random seed (default: $SEED)"
    echo "  --batch_size N               Batch size (default: $BATCH_SIZE)"
    echo "  --eps FLOAT                  Epsilon for exploration (default: $EPS)"
    echo "  --no-8bit                    Disable 8-bit loading (default: enabled)"
    echo "  --n_online_eps N             Number of online episodes (default: $N_ONLINE_EPS)"
    echo "  --n_pretrain_eps N           Number of pretraining episodes (default: $N_PRETRAIN_EPS)"
    echo "  --n_exp N                    Number of experiments (default: $N_EXP)"
    echo "  --no-gpu                     Disable GPU usage (default: enabled)"
    echo "  --buffer_size N              Replay buffer size (default: $BUFFER_SIZE)"
    echo "  --learning_rate FLOAT        Learning rate (default: $LEARNING_RATE)"
    echo "  --gamma FLOAT                Discount factor (default: $GAMMA)"
    echo "  --target_update_interval N   Target network update interval (default: $TARGET_UPDATE_INTERVAL)"
    echo "  --n_pretrain_steps N         Number of pretraining steps (default: $N_PRETRAIN_STEPS)"
    echo "  --pretraining_exp            Run pretraining experiments (default: false)"
    echo "  --long_cot                   Use DeepSeek long CoT data paths (default: false)"
    echo "  --awac                       Use AWAC model instead of SAC or DoubleDQN (default: false)"
    echo "  --help                       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --env CartPole-v0 --n_episodes 50"
    echo "  $0 --env CliffWalking-v0 --SFT --n_online_eps 100"
    echo "  $0 --model_name_1 deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --model_name_2 Qwen/Qwen2.5-7B-Instruct --env MountainCar-v0"
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
        --no-8bit)
            LOAD_IN_8BIT=false
            shift
            ;;
        --n_online_eps)
            N_ONLINE_EPS="$2"
            shift 2
            ;;
        --n_pretrain_eps)
            N_PRETRAIN_EPS="$2"
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
        --pretraining_exp)
            PRETRAINING_EXP=true
            shift
            ;;
        --long_cot)
            LONG_COT=true
            shift
            ;;
        --awac)
            AWAC=true
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
echo "Online Episodes: $N_ONLINE_EPS"
echo "Pretrain Episodes: $N_PRETRAIN_EPS"
echo "Seed: $SEED"
echo "SFT: $SFT"
echo "GPU: $GPU"
echo "AWAC: $AWAC"
echo "=========================================="

# Build base arguments for llm_main.py (without model_name)
LLM_BASE_ARGS="--env $ENV --n_episodes $N_EPISODES --max_episode_len $MAX_EPISODE_LEN --seed $SEED --batch_size $BATCH_SIZE --eps $EPS"
if [ "$SFT" = true ]; then
    LLM_BASE_ARGS="$LLM_BASE_ARGS --SFT"
fi
if [ "$LOAD_IN_8BIT" = true ]; then
    LLM_BASE_ARGS="$LLM_BASE_ARGS --load_in_8bit true"
else
    LLM_BASE_ARGS="$LLM_BASE_ARGS --load_in_8bit false"
fi

# Build arguments for online_main.py
ONLINE_ARGS="--env $ENV --max_episode_len $MAX_EPISODE_LEN --n_online_eps $N_ONLINE_EPS --n_pretrain_eps $N_PRETRAIN_EPS --seed $SEED --eps $EPS --n_exp $N_EXP --buffer_size $BUFFER_SIZE --batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE --gamma $GAMMA --target_update_interval $TARGET_UPDATE_INTERVAL --n_pretrain_steps $N_PRETRAIN_STEPS"
if [ "$GPU" = true ]; then
    ONLINE_ARGS="$ONLINE_ARGS --gpu"
fi
if [ "$SFT" = true ]; then
    ONLINE_ARGS="$ONLINE_ARGS --sft"
fi
if [ "$PRETRAINING_EXP" = true ]; then
    ONLINE_ARGS="$ONLINE_ARGS --pretraining_exp"
fi
if [ "$LONG_COT" = true ]; then
    ONLINE_ARGS="$ONLINE_ARGS --long_cot"
fi
if [ "$AWAC" = true ]; then
    ONLINE_ARGS="$ONLINE_ARGS --awac"
fi

# Step 1: Run LLM training with Model 1
echo ""
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

# Step 2: Run LLM training with Model 2
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

# Step 3: Run online training
echo "Step 3: Running online training..."
echo "Command: python online_main.py $ONLINE_ARGS"
echo ""

python online_main.py $ONLINE_ARGS

if [ $? -ne 0 ]; then
    echo "Error: Online training failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "LORO Pipeline completed successfully!"
echo "=========================================="
echo ""
echo "Generated files:"
echo "- LLM dataset (Model 1): data/${ENV%%-*}_${MODEL_NAME_1##*/}_Neps_${N_EPISODES}$([ "$SFT" = true ] && echo "SFT" || echo "").pkl"
echo "- LLM dataset (Model 2): data/${ENV%%-*}_${MODEL_NAME_2##*/}_Neps_${N_EPISODES}$([ "$SFT" = true ] && echo "SFT" || echo "").pkl"
echo "- Online results: data/cache_${ENV%%-*}_Neps_${N_PRETRAIN_EPS}$([ "$SFT" = true ] && echo "SFT" || [ "$LONG_COT" = true ] && echo "LCOT" || echo "").pkl"
echo "- Timing logs: logs/"
echo ""
