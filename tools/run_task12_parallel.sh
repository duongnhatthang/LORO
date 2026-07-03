#!/usr/bin/env bash
# Task 12 (parallel) — same as run_task12.sh but runs the env × tau combos
# concurrently, one CPU core each. For multi-core boxes; the RL nets are tiny,
# so N single-threaded processes beat one process using N threads.
#
#   nohup bash tools/run_task12_parallel.sh > logs/task12.out 2>&1 &
#
# Knobs: N_EXP (default 15), JOBS (default 8 = max concurrent combos),
#        PY (python interpreter).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python}"
N_EXP="${N_EXP:-15}"
JOBS="${JOBS:-8}"
# One core per process so JOBS processes don't oversubscribe / thrash.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p logs/task12

emit_cmds() {
  for ENV in CliffWalking-v0 FrozenLake-v1 CartPole-v0 MountainCar-v0 Taxi-v3; do
    for TAU in 10 20 30; do
      echo "$PY online_main.py --env $ENV --online_exp --no-gpu --n_exp $N_EXP --n_pretrain_eps $TAU --n_online_eps ${N_ONLINE_EPS:-190} > logs/task12/${ENV}_tau${TAU}.log 2>&1"
    done
  done
  for TAU in 10 20 30; do
    echo "$PY online_main.py --env Pendulum-v1 --online_exp --no-gpu --n_exp $N_EXP --n_pretrain_eps $TAU --n_online_eps ${N_ONLINE_EPS:-190} --model ddpg > logs/task12/Pendulum-v1_tau${TAU}.log 2>&1"
  done
}

echo "== launching $(emit_cmds | wc -l | tr -d ' ') combos, up to $JOBS at a time (n_exp=$N_EXP) =="
emit_cmds | xargs -P "$JOBS" -I CMD bash -c CMD
echo "== all combos done =="

echo "== regenerate headline table =="
$PY extract_cumulative_rewards_table.py | tee "docs/superpowers/baselines/task12-post-fix-table.txt"
