#!/usr/bin/env bash
# LORO — launch a vLLM OpenAI-compatible server for fast collection.
# Run in its OWN tmux window / terminal; leave it running while you collect.
#
# Usage:  tools/cloud/serve_vllm.sh [MODEL] [MAX_MODEL_LEN]
#   MODEL          served model name (default Qwen/Qwen2.5-7B-Instruct)
#                  For 32B on a 24-48GB card use a pre-quantized checkpoint,
#                  e.g. Qwen/Qwen2.5-32B-Instruct-AWQ.
#   MAX_MODEL_LEN  context window (default 8192; plenty for these envs)
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"
MAX_LEN="${2:-8192}"

# Isolated env: vLLM's recent torch/transformers must NOT mix with the pinned
# collection stack (trl==0.7.11 / transformers==4.38.2). No --system-site-packages.
if [ ! -x vllm_env/bin/vllm ]; then
  echo "== installing vLLM into vllm_env (one-time, a few minutes) =="
  python3 -m venv vllm_env
  vllm_env/bin/pip install -q -U pip
  vllm_env/bin/pip install -q vllm
fi
# ninja: FlashInfer JIT-compiles a CUDA kernel at startup and needs it.
vllm_env/bin/pip install -q ninja

# VLLM_USE_FLASHINFER_SAMPLER=0 -> use the built-in Torch sampler, avoiding the
#   FlashInfer JIT compile (which also needs nvcc; not always on the box).
export VLLM_USE_FLASHINFER_SAMPLER=0

# Optional knobs:
#   ENFORCE_EAGER=1  -> add --enforce-eager (skip CUDA-graph capture). Default 0
#                       now that the ninja issue is fixed; CUDA graphs are much
#                       faster. Set to 1 only if startup fails with a compile error.
#   SERVED_NAME=...  -> advertise a different model id than the checkpoint path
#                       (e.g. serve *-AWQ weights but report the canonical name so
#                       collection filenames match existing data).
EAGER_FLAG=""
[ "${ENFORCE_EAGER:-0}" = "1" ] && EAGER_FLAG="--enforce-eager"
SERVED_FLAG=""
[ -n "${SERVED_NAME:-}" ] && SERVED_FLAG="--served-model-name ${SERVED_NAME}"

echo "== serving $MODEL on http://0.0.0.0:8000/v1 (max_model_len=$MAX_LEN, eager=${ENFORCE_EAGER:-0}, served=${SERVED_NAME:-$MODEL}) =="
exec vllm_env/bin/vllm serve "$MODEL" \
  --host 0.0.0.0 --port 8000 \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization 0.90 \
  $EAGER_FLAG $SERVED_FLAG
