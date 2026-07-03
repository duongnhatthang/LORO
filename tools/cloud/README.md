# Cloud collection runbook (Thunder / Kaggle)

Re-collect LLM warm-start data with the **fixed** pipeline (F1 alignment,
F2 robust parsing, F4 multi-trial eval, Task-11 parse-failure logging).

**Strategy:** validate on ONE tiny run first, then sweep. Start with
**Qwen2.5-7B** (works on the pinned `transformers==4.38.2` / `trl==0.7.11`
stack and is the apples-to-apples "did the fixes help?" comparison). Qwen3
needs `transformers>=4.51`, which breaks `trl` — treat it as a separate
upgrade step, not a prerequisite for re-collection.

Known-good stack (torch inherited from the image): `transformers==4.38.2`,
`trl==0.7.11`, `peft==0.7.1`, `accelerate==0.21.0`, `gymnasium==0.29.1`,
`d3rlpy==2.6.1`, `scipy==1.12.0`, `numpy==1.26.4`.

---

## A. Thunder Compute (SSH box — primary, for debugging)

**1. Push code from your Mac** (no public git push — keeps private server specs
in the docs out of the public repo). Adapt the SSH target to your instance:

```bash
rsync -avz --progress \
  --exclude '.git' --exclude 'loro_env' --exclude 'models' --exclude 'figs' \
  --exclude 'logs' --exclude '__pycache__' --exclude '*.pkl' \
  -e "ssh -p <PORT>" \
  ~/Desktop/LORO/ <USER>@<HOST>:~/LORO/
```

(Or use Thunder's `tnr scp` / VS Code remote — anything that lands the repo
at `~/LORO`.)

**2. On the box — setup + smoke test:**

```bash
cd ~/LORO && bash tools/cloud/thunder_setup.sh
```

Confirm: run completes, a `data/CliffWalking_..._Neps_2.pkl` is written, and
the timing log shows a low `parse_failure_rate`. **Paste that log back** to
iterate on any real error.

**3. Full sweep — FAST path via vLLM (recommended).** The HF `model.generate`
path is ~15–18 s/step (a full 30-ep × 6-env sweep would take tens of hours).
vLLM gives ~5× faster decode **and** batches all envs against one server, so
the sweep drops to a few hours. Use two tmux windows:

```bash
# window 1 — start the server (leave running; installs vLLM in its own venv):
tools/cloud/serve_vllm.sh Qwen/Qwen2.5-7B-Instruct
#   32B on a 24-48GB card: serve a pre-quantized checkpoint instead, e.g.
#   tools/cloud/serve_vllm.sh Qwen/Qwen2.5-32B-Instruct-AWQ

# window 2 — collect all envs in parallel against it:
tools/cloud/collect_all_api.sh Qwen/Qwen2.5-7B-Instruct
```

Quick 1-env API check before the full parallel run:
```bash
source loro_env/bin/activate
export LLM_API_BASE=http://localhost:8000/v1 LLM_API_KEY=EMPTY
python llm_main.py --model_name Qwen/Qwen2.5-7B-Instruct --env CliffWalking-v0 \
  --n_episodes 2 --max_episode_len 30 --max_new_tokens 256 --backend api
```

**Slow fallback — HF, no server** (if vLLM won't install):
```bash
tools/cloud/collect_all.sh Qwen/Qwen2.5-7B-Instruct none      # 7B fp16
pip install bitsandbytes
tools/cloud/collect_all.sh Qwen/Qwen2.5-32B-Instruct 4bit     # 32B 4-bit
```

Run everything inside `tmux` so a dropped SSH connection doesn't kill it.

**4. Pull data back to your Mac:**

```bash
rsync -avz -e "ssh -p <PORT>" \
  <USER>@<HOST>:~/LORO/data/'*_Neps_30.pkl' ~/Desktop/LORO/data/
```

---

## B. Kaggle (free 30 GPU-hrs/week — bulk runner)

1. Zip the code (no data/models/figs) and upload as a **private** Kaggle
   Dataset (e.g. `loro-code`):
   ```bash
   cd ~/Desktop/LORO && zip -r /tmp/loro-code.zip . \
     -x '.git/*' 'data/*' 'models/*' 'figs/*' 'logs/*' '*.pyc' 'loro_env/*'
   ```
2. New Notebook → Settings: Accelerator **GPU T4 x2**, Internet **On**.
   Add the `loro-code` dataset (mounts at `/kaggle/input/loro-code`).
3. First cell:
   ```python
   !cp -r /kaggle/input/loro-code /kaggle/working/LORO
   %cd /kaggle/working/LORO
   !pip install -q "transformers==4.38.2" "trl==0.7.11" "peft==0.7.1" \
       "accelerate==0.21.0" "gymnasium==0.29.1" "d3rlpy==2.6.1" \
       "scipy==1.12.0" "numpy==1.26.4" tqdm
   ```
4. Collect (7B fits one T4; 32B-4bit uses both T4s via `device_map=auto`):
   ```python
   !python llm_main.py --model_name Qwen/Qwen2.5-7B-Instruct \
       --env CliffWalking-v0 --n_episodes 30 --quantization none --backend local
   ```
5. Data lands in `/kaggle/working/LORO/data/*.pkl` — download from the
   notebook's Output tab, or `Save Version` to persist.

Kaggle sessions cap at ~9–12h and the weekly quota is 30h; use it for the
free bulk once Thunder has proven the commands.
