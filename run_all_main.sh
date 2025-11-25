#!/bin/bash

# Script to run main.sh for all environments with customizable arguments
# This script allows you to customize all arguments that main.sh accepts

set -e  # Exit on any error

# Default values (can be overridden by command line arguments)
ENV=""
MODEL_NAME_1="none"
MODEL_NAME_2="none"
N_EPISODES=""
MAX_EPISODE_LEN=""
SFT=""
SEED=""
BATCH_SIZE=""
EPS=""
QUANTIZATION=""
N_ONLINE_EPS_1=""
N_ONLINE_EPS_2=""
N_ONLINE_EPS_3=""
N_PRETRAIN_EPS_1=""
N_PRETRAIN_EPS_2=""
N_PRETRAIN_EPS_3=""
N_EXP=""
GPU=""
BUFFER_SIZE=""
LEARNING_RATE=""
GAMMA=""
TARGET_UPDATE_INTERVAL=""
N_PRETRAIN_STEPS=""
LONG_COT=""
AWAC=""
N_STEPS_PER_EPOCH=""
ONLINE_EXP="false"
ONLINE_RAND=""
RUN_ALL_EXPERIMENTS=false
PARALLEL=false

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Run main.sh for all environments with customizable arguments."
    echo ""
    echo "Options (all arguments from main.sh are supported):"
    echo "  --env ENV                           Environment name (e.g., CartPole-v0, CliffWalking-v0, etc.)"
    echo "  --model_name_1 MODEL                First model name for LLM, or 'none' to skip (default: none)"
    echo "  --model_name_2 MODEL                Second model name for LLM, or 'none' to skip (default: none)"
    echo "  --n_episodes N                      Number of episodes for LLM training (default: 30)"
    echo "  --max_episode_len LEN               Maximum episode length (default: 200)"
    echo "  --SFT                               Use supervised fine-tuning (default: false)"
    echo "  --seed SEED                         Random seed (default: 42069)"
    echo "  --batch_size SIZE                   Batch size (default: 1)"
    echo "  --eps EPS                           Epsilon for exploration (default: 0.0)"
    echo "  --quantization METHOD               Quantization method: none, 4bit, or 8bit (default: 4bit)"
    echo "  --n_online_eps_1 N                  Number of online episodes for run 1 (default: varies by env)"
    echo "  --n_online_eps_2 N                  Number of online episodes for run 2 (default: varies by env)"
    echo "  --n_online_eps_3 N                  Number of online episodes for run 3 (default: varies by env)"
    echo "  --n_pretrain_eps_1 N                Number of pretraining episodes for run 1 (default: 10)"
    echo "  --n_pretrain_eps_2 N                Number of pretraining episodes for run 2 (default: 20)"
    echo "  --n_pretrain_eps_3 N                Number of pretraining episodes for run 3 (default: 30)"
    echo "  --n_exp N                           Number of experiments (default: 5)"
    echo "  --gpu                               Use GPU for training (default: true)"
    echo "  --no-gpu                            Disable GPU usage"
    echo "  --buffer_size SIZE                  Replay buffer size (default: 100000)"
    echo "  --learning_rate RATE                Learning rate (default: 5e-5)"
    echo "  --gamma GAMMA                       Discount factor (default: 0.99)"
    echo "  --target_update_interval INTERVAL   Target network update interval (default: 1000)"
    echo "  --n_pretrain_steps STEPS            Number of pretraining steps (default: 1000)"
    echo "  --long_cot                          Use DeepSeek long CoT data paths (default: false)"
    echo "  --awac                              Use AWAC model (default: false)"
    echo "  --n_steps_per_epoch STEPS           Number of steps per epoch for training (default: 200)"
    echo "  --online_exp                        Run the main fine-tune experiments (default: false)"
    echo "  --online_rand                       Run the random and online fine-tune experiments (default: true)"
    echo "  --parallel                          Run experiments in parallel (default: false)"
    echo ""
    echo "  --run_all_experiments               Run predefined experiments (DEFAULT)"
    echo "  --help, -h                          Show this help message"
    echo ""
    echo "Default behavior:"
    echo "  If no --env is specified, runs experiments for all environments (once per environment):"
    echo "  - CartPole-v0"
    echo "  - FrozenLake-v1"
    echo "  - MountainCar-v0"
    echo "  - Pendulum-v1"
    echo "  - RepresentedPong-v0"
    echo "  - CliffWalking-v0"
    echo ""
    echo "Examples:"
    echo "  $0                                  # Run all experiments (default)"
    echo "  $0 --env CartPole-v0 --n_pretrain_eps_1 20"
    echo "  $0 --env FrozenLake-v1 --SFT --batch_size 1"
    echo "  $0 --env CliffWalking-v0 --learning_rate 1e-4 --n_exp 3"
    echo "  $0 --parallel                       # Run all experiments in parallel"
    echo ""
    echo "Available environments:"
    echo "  CartPole-v0, CliffWalking-v0, FrozenLake-v1, MountainCar-v0, Pendulum-v1, RepresentedPong-v0"
}

# Function to build command arguments
build_args() {
    local exclude_eps_args=${1:-false}
    local args=""
    
    [ -n "$MODEL_NAME_1" ] && args="$args --model_name_1 $MODEL_NAME_1"
    [ -n "$MODEL_NAME_2" ] && args="$args --model_name_2 $MODEL_NAME_2"
    [ -n "$ENV" ] && args="$args --env $ENV"
    [ -n "$N_EPISODES" ] && args="$args --n_episodes $N_EPISODES"
    [ -n "$MAX_EPISODE_LEN" ] && args="$args --max_episode_len $MAX_EPISODE_LEN"
    
    # Only include n_online_eps and n_pretrain_eps if not excluding them
    if [ "$exclude_eps_args" = false ]; then
        [ -n "$N_ONLINE_EPS_1" ] && args="$args --n_online_eps_1 $N_ONLINE_EPS_1"
        [ -n "$N_ONLINE_EPS_2" ] && args="$args --n_online_eps_2 $N_ONLINE_EPS_2"
        [ -n "$N_ONLINE_EPS_3" ] && args="$args --n_online_eps_3 $N_ONLINE_EPS_3"
        [ -n "$N_PRETRAIN_EPS_1" ] && args="$args --n_pretrain_eps_1 $N_PRETRAIN_EPS_1"
        [ -n "$N_PRETRAIN_EPS_2" ] && args="$args --n_pretrain_eps_2 $N_PRETRAIN_EPS_2"
        [ -n "$N_PRETRAIN_EPS_3" ] && args="$args --n_pretrain_eps_3 $N_PRETRAIN_EPS_3"
    fi
    
    [ -n "$SFT" ] && args="$args $SFT"
    [ -n "$SEED" ] && args="$args --seed $SEED"
    [ -n "$BATCH_SIZE" ] && args="$args --batch_size $BATCH_SIZE"
    [ -n "$EPS" ] && args="$args --eps $EPS"
    [ -n "$QUANTIZATION" ] && args="$args --quantization $QUANTIZATION"
    [ -n "$N_EXP" ] && args="$args --n_exp $N_EXP"
    [ -n "$GPU" ] && args="$args $GPU"
    [ -n "$BUFFER_SIZE" ] && args="$args --buffer_size $BUFFER_SIZE"
    [ -n "$LEARNING_RATE" ] && args="$args --learning_rate $LEARNING_RATE"
    [ -n "$GAMMA" ] && args="$args --gamma $GAMMA"
    [ -n "$TARGET_UPDATE_INTERVAL" ] && args="$args --target_update_interval $TARGET_UPDATE_INTERVAL"
    [ -n "$N_PRETRAIN_STEPS" ] && args="$args --n_pretrain_steps $N_PRETRAIN_STEPS"
    [ -n "$LONG_COT" ] && args="$args $LONG_COT"
    [ -n "$AWAC" ] && args="$args $AWAC"
    [ -n "$N_STEPS_PER_EPOCH" ] && args="$args --n_steps_per_epoch $N_STEPS_PER_EPOCH"
    [ "$ONLINE_EXP" = "true" ] && args="$args --online_exp"
    [ "$ONLINE_RAND" = "true" ] && args="$args --online_rand"
    
    echo "$args"
}

# Function to run for all environments
run_all_environments() {
    local base_args="$1"
    local environments=("CartPole-v0" "CliffWalking-v0" "FrozenLake-v1" "MountainCar-v0" "Pendulum-v1" "RepresentedPong-v0")
    
    echo "Running main.sh for all environments with arguments: $base_args"
    echo ""
    
    for env in "${environments[@]}"; do
        echo "Running: ./main.sh --env $env $base_args"
        if ! ./main.sh --env "$env" $base_args; then
            echo "Error: Failed to run main.sh for $env"
            exit 1
        fi
        echo ""
    done
    
    echo "All environments completed successfully!"
}

# Function to run all experiments (once per environment)
run_all_experiments() {
    local base_args=$(build_args true)  # Exclude n_online_eps and n_pretrain_eps from base_args
    
    echo "Running experiments for all environments (once per environment)"
    echo "Base arguments: $base_args"
    echo ""
    
    # Define environments
    local environments=("CartPole-v0" "FrozenLake-v1" "MountainCar-v0" "Pendulum-v1" "RepresentedPong-v0" "CliffWalking-v0")
    
    local total_experiments=${#environments[@]}
    local completed_experiments=0
    
    echo "Total experiments to run: $total_experiments"
    echo ""
    
    # Array to store background job PIDs for parallel execution
    local pids=()
    local max_parallel_jobs=4  # Limit parallel jobs to avoid overwhelming the system
    
    # Run experiments
    for env in "${environments[@]}"; do
        completed_experiments=$((completed_experiments + 1))
        
        echo "[$completed_experiments/$total_experiments] Running experiment for $env"
        
        local cmd="./main.sh --env $env $base_args"
        echo "Command: $cmd"
        
        if [ "$PARALLEL" = true ]; then
            # Run in background for parallel execution
            (
                if ! ./main.sh --env "$env" $base_args; then
                    echo "Error: Failed to run experiment for $env"
                    exit 1
                fi
                echo "✓ Completed: $env"
            ) &
            
            local pid=$!
            pids+=($pid)
            
            # Wait if we've reached the maximum number of parallel jobs
            if [ ${#pids[@]} -ge $max_parallel_jobs ]; then
                wait ${pids[0]}
                pids=("${pids[@]:1}")  # Remove the first element
            fi
        else
            # Run sequentially
            if ! ./main.sh --env "$env" $base_args; then
                echo "Error: Failed to run experiment for $env"
                exit 1
            fi
            echo "✓ Completed"
        fi
        echo ""
    done
    
    # Wait for all background jobs to complete if running in parallel
    if [ "$PARALLEL" = true ]; then
        echo "Waiting for all parallel jobs to complete..."
        for pid in "${pids[@]}"; do
            wait $pid
        done
    fi
    
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
        --model_name_1)
            MODEL_NAME_1="$2"
            shift 2
            ;;
        --model_name_2)
            MODEL_NAME_2="$2"
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
            SFT="--SFT"
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
            LONG_COT="--long_cot"
            shift
            ;;
        --awac)
            AWAC="--awac"
            shift
            ;;
        --n_steps_per_epoch)
            N_STEPS_PER_EPOCH="$2"
            shift 2
            ;;
        --online_exp)
            ONLINE_EXP="true"
            shift
            ;;
        --online_rand)
            ONLINE_RAND="true"
            shift
            ;;
        --parallel)
            PARALLEL=true
            shift
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

# Check if main.sh exists
if [ ! -f "main.sh" ]; then
    echo "Error: main.sh not found in current directory"
    exit 1
fi

# Make main.sh executable if it isn't already
if [ ! -x "main.sh" ]; then
    chmod +x main.sh
fi

# Determine which mode to run
if [ -n "$ENV" ]; then
    # If specific environment is provided, run only for that environment
    ARGS=$(build_args false)  # Include n_online_eps and n_pretrain_eps
    echo "Running main.sh for $ENV with arguments: $ARGS"
    ./main.sh $ARGS
elif [ "$RUN_ALL_EXPERIMENTS" = true ] || [ -z "$ENV" ]; then
    # Default behavior: run all experiments with predefined parameter combinations
    run_all_experiments
else
    # Run for all environments with default parameters
    ARGS=$(build_args false)  # Include n_online_eps and n_pretrain_eps
    run_all_environments "$ARGS"
fi
