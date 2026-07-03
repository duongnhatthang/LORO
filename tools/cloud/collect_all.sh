#!/usr/bin/env bash
# LORO — full re-collection sweep (run AFTER the smoke test passes).
# Run from anywhere; it cd's to the repo root itself.
# Usage:  tools/cloud/collect_all.sh MODEL_NAME [QUANT]
#   MODEL_NAME  e.g. Qwen/Qwen2.5-7B-Instruct   (or Qwen/Qwen2.5-32B-Instruct)
#   QUANT       none | 4bit | 8bit   (default: none; use 4bit for 32B on 24GB)
set -euo pipefail

cd "$(dirname "$0")/../.."            # tools/cloud -> repo root
source loro_env/bin/activate

MODEL="${1:?pass a model name, e.g. Qwen/Qwen2.5-7B-Instruct}"
QUANT="${2:-none}"
NEPS=30
SEED=42069

# Non-Atari envs (no ale_py/atariari needed thanks to the lazy-import fix).
# Pong is intentionally excluded — run it separately after installing the
# Atari stack (ale_py + atari-representation-learning).
ENVS=(CliffWalking-v0 FrozenLake-v1 CartPole-v0 MountainCar-v0 Taxi-v3 Pendulum-v1)

echo "== Collecting with $MODEL (quant=$QUANT, n_episodes=$NEPS) =="
for ENV in "${ENVS[@]}"; do
  echo ""
  echo "==================== $ENV ===================="
  python llm_main.py \
    --model_name "$MODEL" \
    --env "$ENV" \
    --n_episodes "$NEPS" \
    --max_episode_len 200 \
    --max_new_tokens "${MAX_NEW_TOKENS:-256}" \
    --quantization "$QUANT" \
    --backend local \
    --seed "$SEED"
done

echo ""
echo "== Done. Collected data: =="
ls -la data/*_Neps_${NEPS}.pkl
echo "== Parse-failure rates across this sweep: =="
for f in $(ls -t logs/llm_timing_log_*.json | head -${#ENVS[@]}); do
  python - "$f" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
hp = d.get("hyperparameters", {})
ps = d.get("parse_stats", {})
print(f"  {hp.get('env','?'):18s} rate={ps.get('parse_failure_rate', float('nan')):.4f} "
      f"({ps.get('n_parse_failures','?')}/{ps.get('n_steps','?')} steps)")
PY
done
