#!/usr/bin/env bash
# LORO — parallel collection against a running vLLM server (--backend api).
# Start the server first (tools/cloud/serve_vllm.sh) in another window.
#
# Usage:  tools/cloud/collect_all_api.sh MODEL_NAME
#   MODEL_NAME must MATCH the model the vLLM server is serving.
# Env knobs: NEPS (default 30), MAX_NEW_TOKENS (default 256),
#            LLM_API_BASE (default http://localhost:8000/v1)
set -euo pipefail
cd "$(dirname "$0")/../.."
source loro_env/bin/activate

# The API client needs the openai package (thin HTTP client).
python -c "import openai" 2>/dev/null || pip install -q openai

MODEL="${1:?pass the model name the vLLM server is serving}"
export LLM_API_BASE="${LLM_API_BASE:-http://localhost:8000/v1}"
export LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
NEPS="${NEPS:-30}"
SEED=42069

# All envs run CONCURRENTLY against the one server — vLLM batches the requests,
# so wall-clock ~= the slowest single env, not their sum.
ENVS=(CliffWalking-v0 FrozenLake-v1 CartPole-v0 MountainCar-v0 Taxi-v3 Pendulum-v1)

echo "== waiting for vLLM at $LLM_API_BASE =="
for i in $(seq 1 60); do
  if curl -sf "${LLM_API_BASE%/v1}/health" >/dev/null 2>&1; then echo "server up"; break; fi
  sleep 5
  [ "$i" = 60 ] && { echo "vLLM not reachable — is serve_vllm.sh running?"; exit 1; }
done

mkdir -p logs
pids=()
for ENV in "${ENVS[@]}"; do
  echo "launch $ENV -> logs/collect_${ENV}.out"
  python llm_main.py \
    --model_name "$MODEL" \
    --env "$ENV" \
    --n_episodes "$NEPS" \
    --max_episode_len 200 \
    --max_new_tokens "${MAX_NEW_TOKENS:-256}" \
    --backend api \
    --seed "$SEED" > "logs/collect_${ENV}.out" 2>&1 &
  pids+=($!)
done

echo "== waiting for ${#pids[@]} parallel collectors =="
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done

echo ""
echo "== done (fail=$fail). collected data: =="
ls -la data/*_Neps_${NEPS}.pkl || true
echo "== parse-failure rates: =="
for ENV in "${ENVS[@]}"; do
  grep -h "Action parsing:" "logs/collect_${ENV}.out" 2>/dev/null | tail -1 | sed "s/^/  ${ENV}: /" || true
done
