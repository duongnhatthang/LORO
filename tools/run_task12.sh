#!/usr/bin/env bash
# Task 12 — regenerate RL result caches on the FRESH LLM data, then rebuild the
# headline table. Run locally AFTER fresh 7B + 32B data is in data/ (both under
# the standardized names data/<Env>_Qwen2.5-{7B,32B}-Instruct_Neps_30.pkl).
#
# CPU is fine — the DQNs are tiny. Runs in the background for a few hours.
#   nohup bash tools/run_task12.sh > logs/task12.out 2>&1 &
#   tail -f logs/task12.out
#
# Knobs:  N_EXP (default 15; paper wants 10-20),  PY (python interpreter)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-/opt/miniconda3/envs/llamagym/bin/python}"
N_EXP="${N_EXP:-15}"
mkdir -p logs

# Discrete envs -> default Double-DQN. Each writes cache_<Env>_Neps_<tau>.pkl.
# tau in {10,20,30} = warm-start budget sweep the table's AUC-vs-budget needs.
DISCRETE=(CliffWalking-v0 FrozenLake-v1 CartPole-v0 MountainCar-v0 Taxi-v3)
for ENV in "${DISCRETE[@]}"; do
  for TAU in 10 20 30; do
    echo "==== $ENV  tau=$TAU  (n_exp=$N_EXP) ===="
    $PY online_main.py --env "$ENV" --online_exp --no-gpu \
      --n_exp "$N_EXP" --n_pretrain_eps "$TAU"
  done
done

# Pendulum -> continuous control, DDPG variant (cache_Pendulum_Neps_<tau>_ddpg.pkl).
for TAU in 10 20 30; do
  echo "==== Pendulum-v1  tau=$TAU  (ddpg, n_exp=$N_EXP) ===="
  $PY online_main.py --env Pendulum-v1 --online_exp --no-gpu \
    --n_exp "$N_EXP" --n_pretrain_eps "$TAU" --model ddpg
done

echo ""
echo "==== regenerate headline table (normalized cumulative reward) ===="
$PY extract_cumulative_rewards_table.py | tee "docs/superpowers/baselines/task12-post-fix-table.txt"
echo ""
echo "Compare against the frozen pre-fix baseline:"
echo "  docs/superpowers/baselines/2026-06-30-pre-fix-table.txt"
