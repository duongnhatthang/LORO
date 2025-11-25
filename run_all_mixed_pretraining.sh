#!/bin/bash

# Script to run mixed pretraining for all environments with customizable arguments
# This script allows you to customize all arguments that run_mixed_pretraining.py accepts

set -e  # Exit on any error

# Default values (can be overridden by command line arguments)
ENV=""
MAX_EPISODE_LEN=""
N_ONLINE_EPS=""
N_PRETRAIN_EPS=""
SEED=""
EPS=""
N_EXP=""
GPU=""
BUFFER_SIZE=""
BATCH_SIZE=""
LEARNING_RATE=""
GAMMA=""
TARGET_UPDATE_INTERVAL=""
SFT=""
LONG_COT=""
N_PRETRAIN_STEPS=""
AWAC=""
N_STEPS_PER_EPOCH=""
RUN_ALL_EXPERIMENTS=false

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Run mixed pretraining for all environments with customizable arguments."
    echo ""
    echo "Options (all arguments from run_mixed_pretraining.py are supported):"
    echo "  --env ENV                           Environment name (e.g., CartPole-v0, CliffWalking-v0, etc.)"
    echo "  --max_episode_len LEN               Maximum episode length (default: 200)"
    echo "  --n_online_eps EPS                  Number of online episodes (default: 190)"
    echo "  --n_pretrain_eps EPS                Number of pretraining episodes (default: 30)"
    echo "  --seed SEED                         Random seed (default: 42069)"
    echo "  --eps EPS                           Epsilon for exploration (default: 0.1)"
    echo "  --n_exp EXP                         Number of experiments to run (default: 5)"
    echo "  --gpu                               Use GPU for training (default: true)"
    echo "  --no-gpu                            Disable GPU usage"
    echo "  --buffer_size SIZE                  Replay buffer size (default: 100000)"
    echo "  --batch_size SIZE                   Batch size (default: 256)"
    echo "  --learning_rate RATE                Learning rate (default: 5e-5)"
    echo "  --gamma GAMMA                       Discount factor (default: 0.99)"
    echo "  --target_update_interval INTERVAL   Target network update interval (default: 1000)"
    echo "  --sft                               Use SFT data paths"
    echo "  --long_cot                          Use DeepSeek long CoT data paths"
    echo "  --n_pretrain_steps STEPS            Number of pretraining steps (default: 1000)"
    echo "  --awac                              Using AWAC model"
    echo "  --n_steps_per_epoch STEPS           Number of steps per epoch for training (default: 200)"
    echo ""
    echo "  --run_all_experiments               Run predefined experiments (DEFAULT)"
    echo "  --help, -h                          Show this help message"
    echo ""
    echo "Default behavior:"
    echo "  If no --env is specified, runs all experiments with predefined parameter combinations:"
    echo "  - CartPole-v0: n_online_eps=[140,130,120], n_pretrain_eps=[10,20,30]"
    echo "  - FrozenLake-v1: n_online_eps=[140,130,120], n_pretrain_eps=[10,20,30]"
    echo "  - MountainCar-v0: n_online_eps=[290,280,270], n_pretrain_eps=[10,20,30]"
    echo "  - Pendulum-v1: n_online_eps=[190,180,170], n_pretrain_eps=[10,20,30]"
    echo "  - RepresentedPong-v0: n_online_eps=[190,180,170], n_pretrain_eps=[10,20,30]"
    echo "  - CliffWalking-v0: n_online_eps=[190,180,170], n_pretrain_eps=[10,20,30]"
    echo ""
    echo "Examples:"
    echo "  $0                                  # Run all experiments (default)"
    echo "  $0 --env CartPole-v0 --n_pretrain_eps 20"
    echo "  $0 --env FrozenLake-v1 --sft --batch_size 128"
    echo "  $0 --env CliffWalking-v0 --learning_rate 1e-4 --n_exp 3"
    echo ""
    echo "Available environments:"
    echo "  CartPole-v0, CliffWalking-v0, FrozenLake-v1, MountainCar-v0, Pendulum-v1, RepresentedPong-v0"
}

# Function to build command arguments
build_args() {
    local exclude_eps_args=${1:-false}
    local args=""
    
    [ -n "$ENV" ] && args="$args --env $ENV"
    [ -n "$MAX_EPISODE_LEN" ] && args="$args --max_episode_len $MAX_EPISODE_LEN"
    
    # Only include n_online_eps and n_pretrain_eps if not excluding them
    if [ "$exclude_eps_args" = false ]; then
        [ -n "$N_ONLINE_EPS" ] && args="$args --n_online_eps $N_ONLINE_EPS"
        [ -n "$N_PRETRAIN_EPS" ] && args="$args --n_pretrain_eps $N_PRETRAIN_EPS"
    fi
    
    [ -n "$SEED" ] && args="$args --seed $SEED"
    [ -n "$EPS" ] && args="$args --eps $EPS"
    [ -n "$N_EXP" ] && args="$args --n_exp $N_EXP"
    [ -n "$GPU" ] && args="$args $GPU"
    [ -n "$BUFFER_SIZE" ] && args="$args --buffer_size $BUFFER_SIZE"
    [ -n "$BATCH_SIZE" ] && args="$args --batch_size $BATCH_SIZE"
    [ -n "$LEARNING_RATE" ] && args="$args --learning_rate $LEARNING_RATE"
    [ -n "$GAMMA" ] && args="$args --gamma $GAMMA"
    [ -n "$TARGET_UPDATE_INTERVAL" ] && args="$args --target_update_interval $TARGET_UPDATE_INTERVAL"
    [ -n "$SFT" ] && args="$args $SFT"
    [ -n "$LONG_COT" ] && args="$args $LONG_COT"
    [ -n "$N_PRETRAIN_STEPS" ] && args="$args --n_pretrain_steps $N_PRETRAIN_STEPS"
    [ -n "$AWAC" ] && args="$args $AWAC"
    [ -n "$N_STEPS_PER_EPOCH" ] && args="$args --n_steps_per_epoch $N_STEPS_PER_EPOCH"
    
    echo "$args"
}

# Function to run for all environments
run_all_environments() {
    local base_args="$1"
    local environments=("CartPole-v0" "CliffWalking-v0" "FrozenLake-v1" "MountainCar-v0" "Pendulum-v1" "RepresentedPong-v0")
    
    echo "Running mixed pretraining for all environments with arguments: $base_args"
    echo ""
    
    for env in "${environments[@]}"; do
        echo "Running: python run_mixed_pretraining.py --env $env $base_args"
        if ! python run_mixed_pretraining.py --env "$env" $base_args; then
            echo "Error: Failed to run mixed pretraining for $env"
            exit 1
        fi
        echo ""
    done
    
    echo "All environments completed successfully!"
}

# Function to run all experiments with specific parameter combinations
run_all_experiments() {
    local base_args=$(build_args true)  # Exclude n_online_eps and n_pretrain_eps from base_args
    
    echo "Running all experiments with predefined parameter combinations"
    echo "Base arguments: $base_args"
    echo ""
    
    # Define experiment configurations as arrays
    local environments=("CartPole-v0" "FrozenLake-v1" "MountainCar-v0" "Pendulum-v1" "RepresentedPong-v0" "CliffWalking-v0")
    local n_online_eps_configs=("140,130,120" "140,130,120" "290,280,270" "190,180,170" "190,180,170" "190,180,170")
    local n_pretrain_eps_configs=("10,20,30" "10,20,30" "10,20,30" "10,20,30" "10,20,30" "10,20,30")
    
    local total_experiments=0
    local completed_experiments=0
    
    # Count total experiments
    for i in "${!environments[@]}"; do
        local n_online_eps_array="${n_online_eps_configs[$i]}"
        local n_online_eps_list=($(echo "$n_online_eps_array" | tr ',' ' '))
        total_experiments=$((total_experiments + ${#n_online_eps_list[@]}))
    done
    
    echo "Total experiments to run: $total_experiments"
    echo ""
    
    # Run experiments
    for i in "${!environments[@]}"; do
        local env="${environments[$i]}"
        local n_online_eps_array="${n_online_eps_configs[$i]}"
        local n_pretrain_eps_array="${n_pretrain_eps_configs[$i]}"
        
        local n_online_eps_list=($(echo "$n_online_eps_array" | tr ',' ' '))
        local n_pretrain_eps_list=($(echo "$n_pretrain_eps_array" | tr ',' ' '))
        
        echo "Running experiments for $env:"
        
        for j in "${!n_online_eps_list[@]}"; do
            local n_online_eps="${n_online_eps_list[$j]}"
            local n_pretrain_eps="${n_pretrain_eps_list[$j]}"
            completed_experiments=$((completed_experiments + 1))
            
            echo "  [$completed_experiments/$total_experiments] $env: n_online_eps=$n_online_eps, n_pretrain_eps=$n_pretrain_eps"
            
            local cmd="python run_mixed_pretraining.py --env $env --n_online_eps $n_online_eps --n_pretrain_eps $n_pretrain_eps $base_args"
            echo "  Command: $cmd"
            
            if ! python run_mixed_pretraining.py --env "$env" --n_online_eps "$n_online_eps" --n_pretrain_eps "$n_pretrain_eps" $base_args; then
                echo "  Error: Failed to run experiment for $env with n_online_eps=$n_online_eps, n_pretrain_eps=$n_pretrain_eps"
                exit 1
            fi
            echo "  ✓ Completed"
            echo ""
        done
    done
    
    echo "All experiments completed successfully!"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_usage
            exit 0
            ;;
        --env)
            ENV="$2"
            shift 2
            ;;
        --max_episode_len)
            MAX_EPISODE_LEN="$2"
            shift 2
            ;;
        --n_online_eps)
            N_ONLINE_EPS="$2"
            shift 2
            ;;
        --n_pretrain_eps)
            N_PRETRAIN_EPS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --eps)
            EPS="$2"
            shift 2
            ;;
        --n_exp)
            N_EXP="$2"
            shift 2
            ;;
        --gpu)
            GPU="--gpu"
            shift
            ;;
        --no-gpu)
            GPU="--no-gpu"
            shift
            ;;
        --buffer_size)
            BUFFER_SIZE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
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
        --sft)
            SFT="--sft"
            shift
            ;;
        --long_cot)
            LONG_COT="--long_cot"
            shift
            ;;
        --n_pretrain_steps)
            N_PRETRAIN_STEPS="$2"
            shift 2
            ;;
        --awac)
            AWAC="--awac"
            shift
            ;;
        --n_steps_per_epoch)
            N_STEPS_PER_EPOCH="$2"
            shift 2
            ;;
        --run_all_experiments)
            RUN_ALL_EXPERIMENTS=true
            shift
            ;;
        *)
            echo "Error: Unknown option $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if Python script exists
if [ ! -f "run_mixed_pretraining.py" ]; then
    echo "Error: run_mixed_pretraining.py not found in current directory"
    exit 1
fi

# Determine which mode to run
if [ -n "$ENV" ]; then
    # If specific environment is provided, run only for that environment
    ARGS=$(build_args false)  # Include n_online_eps and n_pretrain_eps
    echo "Running mixed pretraining for $ENV with arguments: $ARGS"
    python run_mixed_pretraining.py $ARGS
elif [ "$RUN_ALL_EXPERIMENTS" = true ] || [ -z "$ENV" ]; then
    # Default behavior: run all experiments with predefined parameter combinations
    run_all_experiments
else
    # Run for all environments with default parameters
    ARGS=$(build_args false)  # Include n_online_eps and n_pretrain_eps
    run_all_environments "$ARGS"
fi
