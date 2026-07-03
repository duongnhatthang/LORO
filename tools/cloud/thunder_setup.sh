#!/usr/bin/env bash
# LORO — cloud setup + smoke test (Thunder / any Ubuntu+CUDA box).
# Run from anywhere; it cd's to the repo root itself.
# Goal: reproduce the known-good stack, then prove the fixed collection
# pipeline works on ONE tiny run before spending real GPU hours.
set -euo pipefail

cd "$(dirname "$0")/../.."            # tools/cloud -> repo root
echo "== repo: $(pwd) =="
nvidia-smi --query-gpu=name,memory.total --format=csv || true

# --- 1. Environment (inherits the image's CUDA-enabled torch) ---------------
if [ ! -d loro_env ]; then
  python3 -m venv --system-site-packages loro_env
fi
# shellcheck disable=SC1091
source loro_env/bin/activate
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# Pinned to the versions proven to work locally (torch left to the image).
pip install -q \
  "transformers==4.38.2" "trl==0.7.11" "peft==0.7.1" "accelerate==0.21.0" \
  "gymnasium==0.29.1" "d3rlpy==2.6.1" "scipy==1.12.0" "numpy==1.26.4" tqdm openai
# For 32B 4-bit later (NOT needed for the 7B fp16 smoke test):
#   pip install -q bitsandbytes        # latest, matches the box's CUDA

# Qwen2.5 is NOT gated — HF_TOKEN is optional. Set it only if you hit auth:
#   export HF_TOKEN=hf_xxx

# --- 2. Smoke test: fixed pipeline, tiny run --------------------------------
# Skip with SKIP_SMOKE=1 (e.g. when the box is only serving the vLLM/API path
# and doesn't need the local HF weights downloaded/exercised).
if [ "${SKIP_SMOKE:-0}" = "1" ]; then
  echo "== SKIP_SMOKE=1 -> env built (loro_env ready); skipping HF smoke test =="
  exit 0
fi
# CliffWalking is the env where the collection bug hurt most (C2 finding),
# so it's the best validation target. Small + short so it finishes fast —
# this only validates the pipeline, not the data quality.
echo "== SMOKE TEST: Qwen2.5-7B x CliffWalking x 2 short episodes (fp16) =="
python llm_main.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --env CliffWalking-v0 \
  --n_episodes 2 \
  --max_episode_len 30 \
  --max_new_tokens 256 \
  --quantization none \
  --backend local \
  --seed 42069

echo ""
echo "== RESULT CHECKS =="
ls -la data/CliffWalking_Qwen2.5-7B-Instruct_Neps_2.pkl
echo "--- parse-failure rate + timing (newest log) ---"
cat "$(ls -t logs/llm_timing_log_*.json | head -1)"
