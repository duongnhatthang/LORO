# LORO: Trustworthy Results, Reframe, and LLM-native MDP — Design

**Date:** 2026-06-30
**Branch:** `investigation/trustworthy-results-and-reframe`
**Paper:** "Sample-efficient Reinforcement Learning by Warm-starting with LLMs" (LORO), UAI 2026 resubmission
**Deadline:** ~1 month (next venue)

## 1. Problem statement

The paper has been rejected multiple times for two recurring reasons:

1. **Small / not statistically significant improvement over the pure Online RL baseline.**
2. **Only toy environments; no LLM-native MDPs** (reviewers specifically want text-native tasks like ALFWorld).

Goal of this work: (a) deep-clean the codebase so results are trustworthy, (b) diagnose and fix why the improvement looks small, (c) re-frame the central claim to what the data actually supports, and (d) add a text-native (LLM-native) MDP. Compute is bounded to ~$200/month.

## 2. Diagnosis (from three independent code audits, cross-verified against source)

Findings split into **measurement confounds** (the reported gap is partly artifact) and **genuine bugs that asymmetrically handicap the method** (fixing them should widen the true gap).

### 2.1 Measurement confounds
- **C1 — Single-trial evaluation.** All envs except CliffWalking evaluate each checkpoint on ONE episode (`utils.py:356-363`, `n_trials=1`). On stochastic envs this is the mechanical cause of "not significant." Non-CliffWalking eval also uses the *training* env, not `eval_env`.
- **C2 — Normalization constants (REVISED after empirical measurement, Task 3).** `r_random`/`r_expert` are hardcoded (`vis_utils.py:39-46`). Original claim was that CliffWalking `x_random=-2120` was ~4× wrong; **this is refuted**: a measured uniform-random policy returns ≈ **−2070** (the 200-step cap bounds it), essentially matching the hardcoded value. The −7729 figure from the audit was the **LLM's collection return**, not random — meaning the pre-fix LLM policy on CliffWalking was *worse than random* (normalized ≈ −2.7), which points at the collection bugs **B1/B2**, not the normalization. Net: constants are roughly correct, but we now derive them empirically (Task 3, `data/norm_constants.json`) so the axis is defensible and verifiable rather than hand-set. The "LLM worse than random on CliffWalking (pre-fix)" fact is itself evidence that F1/F2 should materially improve results there.
- **C3 — Table-1 scalar blends two quantities inconsistently.** Averages the full curve; for x<τ the method curve is a flat constant = LLM's raw collection reward (`vis_utils.py:194-196`, `notebook_utils.py:39-48`), while baselines show real per-episode returns. For CliffWalking this flips a real online-phase advantage (+168) into a reported deficit (−217).
- **C4 — Aggregate dominated by one env.** The entire +0.10 average improvement is carried by MountainCar (+0.52, where Online RL is stuck at the floor). On the other three measured envs the method is ~−0.04. This *is* the reviewers' complaint.
- **C5 — Fragile statistics.** IQM over 5 seeds keeps only the middle 3 (drops best+worst); ±1 bootstrap-SE bands are too tight and coarse for n=5. (Note: the prior "SE bug fix" commit `f34e5ad` was a genuine, correct fix — point estimate and error band now match.)

### 2.2 Genuine bugs handicapping the method
- **B1 — Off-by-one in LLM data collection** (`llm_main.py:258-264`). `observation` is overwritten by `env.step()` before being appended, so every stored transition is `(s_{t+1}, a_t, r_t)`; s₀ is dropped. Pretraining learns wrong state→action associations. This corrupts *only* the LLM-pretrain path (Online RL uses d3rlpy's own correct rollout), so it asymmetrically hurts the method. Mirrored in the eval/coverage path (`utils.py:100-108`), so coverage figures are also built on shifted pairs. **Hypothesis: fixing this widens the gap.**
- **B2 — Silent fixed-action fallback on parse failure** (`env/translation_agent.py`). The natural-language fallback is dead code (`"Left" in response.lower()` never matches a lowercased string), so any CoT parse failure collapses to a constant default action. Action is read as `digits[-1]` over a 2000-token CoT (any stray digit wins). No retry. Degrades LLM-policy quality and coverage by an unquantified amount.
- **B3 — `[/INST]` response split is a no-op for Qwen/DeepSeek** (`llamagym/agent.py:111`); `extract_action` scans the entire prompt+generation, including the action list in the system prompt.
- **B4 — Fine-tuning starts from an empty buffer** (`online_main.py:129-136`). LLM data is discarded after pretraining, so the warm-started net degrades immediately (the post-τ dip). Overfitting risk: 1000–3000 grad steps on ~10 short episodes.
- **B5 — SAC ignores configured LR/target-update** (`utils.py:164-167`); the paper's "lr 5e-5" is false for continuous envs. (Symmetric across arms, but a reporting error.)
- **B6 — Forced H=200 truncation treated as a true terminal** (e.g. `utils.py:337`, `pretrain_from_llm.py:91-93`), biasing bootstrap targets (symmetric, low priority).
- **B7 — Data-selection surface.** `pretrain_from_llm.py:7-48` hardcodes specific timestamped collections (several commented "bad env hist"/"Typo"); `extract_cumulative_rewards_table.py:112-126` has a glob fallback returning `candidates[0]`. Currently deterministic, but an un-audited selection surface.

### 2.3 Coverage-claim caveats (for honest reporting)
- **D1 — FrozenLake env-history injects hardcoded oracle hole coordinates** `[(1,1),(1,3),(2,3),(3,0)]` (`translation_agent.py:458`), not derivable from the reward signal, gated only on visitation. Only the two hard grid envs get a memory mechanism → inflates their optimal-policy coverage relative to a memoryless LLM. Must be disclosed or removed.
- **D2 — `--SFT` is reward-based PPO fine-tuning during collection**, not supervised fine-tuning; `*SFT.pkl` come from a reward-tuned LLM. The label is a misnomer; frame accordingly.

## 3. Reframed thesis

**Old (rejected):** LLM warm-starting improves sample efficiency over Online RL broadly.

**New:** *LLM-collected warm-start data rescues off-policy RL in hard-exploration / sparse-reward MDPs where from-scratch RL fails to leave the floor, at no cost to easy tasks — and this holds on text-native (LLM-native) MDPs, not just classic control.* The concentrability/coverage argument explains *why*: the LLM concentrates data on near-optimal state-actions that random exploration rarely reaches.

This is supported by the existing data (MountainCar, Pong) and is expected to strengthen once B1–B4 are fixed.

## 4. Scope of work (phased)

### Phase 0 — Reproducibility harness
- Snapshot current Table 1 / Table 4 numbers and the caches that produced them into `docs/superpowers/baselines/` as an immutable reference.
- `make_tables.py`: one command regenerates all tables/figures from caches, so every subsequent fix is measured against the frozen baseline (delta table).
- No behavior changes.

### Phase 1 — Correctness fixes (trustworthiness)
Each fix is an isolated, independently testable unit with a regression check on a tiny fixture:
- **F1** Fix off-by-one in collection (`llm_main.py`) and eval/coverage (`utils.py`): store `s_t` (pre-step) with `a_t`, `r_t`, `done`; include s₀. Add a unit test asserting `observations[i]` is the state where `actions[i]` was chosen on a scripted 3-step rollout.
- **F2** Robust action parsing: parse from the model's *generated* span only (Qwen/DeepSeek ChatML-aware), extract the action from a delimited tag or the first valid choice, fix the NL fallback (case-insensitive, actually reachable), and on hard failure resample once then fall back to a *random valid* action (logged + counted), not a fixed one. Emit a per-run parse-failure rate.
- **F3** Empirical normalization: compute `r_random` from an actual uniform-random policy (many episodes) and `r_expert` from a converged RL policy (or known optimum) per env; store in a versioned JSON. Optionally clip normalized values. Report the aggregate with and without MountainCar.
- **F4** Multi-trial evaluation: `n_trials ≥ 20` for all envs on a dedicated `eval_env`; separate eval seeds from training seeds.
- **F5** Fair full-budget metric, with BOTH warm-start representations (per advisor guidance). Primary metric is **full-budget cumulative reward / AUC over the entire episode budget, τ warm-start episodes INCLUDED** — honors the paper's high-labeling-cost / same-total-budget / cumulative-regret framing. The warm-start phase has no policy learning (the LLM is a stationary policy), so it is shown two ways that are **AUC-identical by construction**:
  - **(a) Constant-line version — DEFAULT figure (advisor's preference):** the method's first-τ episodes are a flat line at the mean LLM-collection reward, computed over **exactly the τ episodes actually spent** (this is the bug fix: today it averages `min(30,·)` episodes regardless of τ — `notebook_utils.py:38-48`). A constant cleanly signals "this reward comes from the LLM, not from learning."
  - **(b) Per-episode version — appendix/robustness:** the same τ episodes as real per-episode returns with real variance (replacing the `std=0` band, `vis_utils.py:197`).
  Because the constant = mean of those τ episodes, τ×mean = Σ returns, so both versions give the **same warm-start AUC / headline number**; only the variance/visual differs — reporting both is a robustness demonstration. The Online RL baseline keeps real per-episode returns throughout (it *is* learning in its first τ episodes). τ = episodes per the paper's LLM-query-budget definition; since collection episodes have variable length, **also report a steps-matched (labels = env transitions) robustness check** so "same budget" is unambiguous. Post-warm-start AUC is kept only as a **secondary diagnostic** to localize where the learning-speed gain arises.
- **F6** Statistics: 10–20 seeds; report paired per-seed differences (method − baseline) with a bootstrap CI and a paired significance test; keep IQM only as a secondary view.
- **F7** SAC honors configured LR/target-update (`utils.py`); fix reporting.
- Re-collect LLM data via API (Phase 1 infra) with the fixed pipeline; re-run all toy-env experiments; produce a delta table quantifying how much each fix moves the gap.

### Phase 2 — Significance + reframe
- **P1** Buffer warmup improvement: seed the online replay buffer with the LLM data (or a warmup phase) instead of discarding it — expected to remove the post-τ dip. Compare vs current discard and vs Mix.
- **P2** Add 1–2 more hard-exploration / sparse-reward envs where the thesis is strong (candidates: sparse FrozenLake-8x8, MountainCar variants, a MiniGrid sparse task). Keep the classic envs as "no-harm" evidence.
- **P3** Rewrite the claim, tables, and figures around the reframed thesis; disclose D1/D2 honestly (or remove the FrozenLake oracle history and re-measure).

### Phase 3 — LLM-native MDP (TextWorld first)
- **T1** TextWorld integration via a frozen text-embedding bridge: game observation text → frozen sentence-transformer embedding (e.g. `all-MiniLM-L6-v2`, 384-dim) → vector observation consumed by the existing d3rlpy DQN pipeline. Start with a **fixed, compact action set** TextWorld config (navigation + a few verbs) so standard DQN applies; note DRRN (state-action scoring for variable action sets) as a documented extension.
- **T2** LLM-as-policy on TextWorld via the same API path; collect warm-start data; run the full LORO pipeline (pretrain → online fine-tune) and all baselines.
- **T3** (Stretch, compute/time permitting) extend to BabyAI and/or ALFWorld reusing the embedding bridge.

## 5. Compute & infrastructure design

**Rented-GPU compute (decided).** LORO uses the LLM only for data collection / policy rollouts (except the SFT ablation, which fine-tunes the LLM and needs a training-capable GPU). **All LLM inference runs on self-hosted GPUs — local + rented — not a paid API.** (An OpenAI-compatible API client is kept in code as an optional fallback only; not used for the main runs.)

- **7B-class (collection + SFT ablation): a local GPU + rented 24 GB GPUs.** Qwen3-8B (and Qwen2.5-7B backward point) 4-bit inference and 7B QLoRA-SFT run on the self-hosted GPU (free) and/or rented RTX 4090/3090 (~$0.27/hr).
- **32B-class (collection only): rented 24 GB GPU.** Qwen3-32B in 4-bit (≈18–20 GB) fits on a single RTX 4090 (24 GB) for inference — run collection there via the local transformers path with `--quantization 4bit`. **No 32B SFT** (QLoRA-32B needs a 48 GB card, out of scope): the SFT ablation is 7B-only, as in the current paper — disclose this scope.
- **Parallelism:** rent **several 24 GB 4090s** and run collection **in parallel across (env × model × seed)** — the workload is embarrassingly parallel — to iterate faster.
- **Model set:** primary Qwen3-8B / Qwen3-32B (newer, stronger successors to Qwen2.5-7B/32B, same lineage → clean size axis); Qwen2.5-7B/32B kept as a backward-comparison point; one reasoning model (gpt-oss-20b or a DeepSeek-R1 distill) as a capability axis.
- **RL training + embeddings** run locally on a self-hosted GPU/CPU (tiny), parallelized across (env × seed × method); rent extra GPUs when seed/env sweeps are the bottleneck.
- **Raw-response caching** keyed by (model, prompt, sampling params): F1 (off-by-one storage fix) and F2 (re-parse) need **no new LLM generation** — they operate on cached responses — so most Phase-1 re-runs cost no GPU time. New generation only when prompt/sampling/#episodes change. Record temperature/top_p and a fixed generation seed.
- **Budget:** several parallel 24 GB rentals at ~$0.27/hr → a few-hundred GPU-hours/mo, comfortably within $200/mo. Student backup credits: GitHub Student Pack, GCP $300, Azure $200, Thunder Compute $20.
- **Server note:** a self-hosted GPU server is available for the tiny RL/embedding jobs. Its specifics are kept in local-only notes, out of the repo, per the user's global rule.

## 6. Experiment matrix (target)

- Envs: CartPole, Pendulum, MountainCar, FrozenLake, CliffWalking, RepresentedPong (existing) + ≥1 sparse env (Phase 2) + TextWorld (Phase 3).
- Methods: LLM-data-pretrain (fixed), Mix (Song et al. hybrid), Online RL, LLM-as-policy, Random; buffer-warmup variant (Phase 2).
- Models: Qwen3-8B, Qwen3-32B (+ Qwen2.5 backward point, + 1 reasoning model).
- Seeds: 10–20. Eval: ≥20 trials/checkpoint.
- Primary metric: full-budget normalized cumulative-reward/AUC over the entire episode budget (τ warm-start episodes included). Warm-start shown two AUC-identical ways: constant line (default figure) and real per-episode returns (appendix). Plus paired method−baseline difference with bootstrap CI and a paired test. Secondary/diagnostic: post-warm-start AUC (localizes learning-speed gain), steps-matched robustness check, IQM curves.

## 7. Success criteria
- Every audit finding B1–B7, C1–C5 is fixed or explicitly justified, each with a regression test or documented rationale.
- Regenerated toy-env results with rigorous stats; a clear, honest statement of where the method significantly helps (expected: hard-exploration/sparse envs) and where it's a wash.
- A working TextWorld LORO experiment with the full baseline set.
- A delta report attributing result changes to specific fixes.
- Total spend within $200/month.

## 8. Risks & mitigations
- *Fixing bugs doesn't widen the gap.* Then the reframe (hard-exploration regime + TextWorld) carries the paper; the cleanup still makes results trustworthy and defensible. Both are independently valuable.
- *TextWorld variable action space breaks fixed-action DQN.* Mitigate by starting with a compact fixed-action game config; DRRN as a fallback.
- *API rate limits / cost overrun.* Aggressive caching, start with 7B, budget-gate 32B and TextWorld runs.
- *Removing the FrozenLake oracle history hurts coverage numbers.* Acceptable — honesty is the point; report both.

## 9. Out of scope
- New offline-to-online RL algorithm design (paper is explicitly algorithm-agnostic).
- Large local LLM hosting / SFT at scale.
- Theoretical proofs for Assumption 2 (left as future work, unchanged).
