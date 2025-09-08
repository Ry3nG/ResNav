Unified AMR Navigation — Agent Development Notes

Scope
- This document captures conventions and guidance for evolving the agent stack without breaking interfaces or reproducibility.

Reward Architecture
- Single source of truth: All reward math lives in `amr_env/reward.py`.
- Stable schema: Env exposes a breakdown dict with
  - `raw`: unweighted terms (Phase I: progress, path, effort, sparse)
  - `weights`: mapping from term name to weight (keys must match `contrib` keys)
  - `contrib`: weighted contributions per term
  - `total`: scalar sum; `version`: schema tag (e.g., `rwd_v1`)
- Do not compute reward terms in the renderer or training loop. Consume the schema only.

Adding New Reward Terms
- Implement the raw term in `amr_env/reward.py` (`compute_terms`).
- Keep term names consistent across `raw`, `weights`, and `contrib`.
- Update `configs/reward/default.yaml` with a new weight key using the exact term name.
- The HUD and logs pick up new terms automatically.

Logging & Visualization
- Reward breakdown logging is enabled by default via `RewardTermsLoggingCallback`.
- HUD shows `R_total` and per-term contributions sorted by absolute value.
- Use `make render-*` and recorded videos to inspect behaviors; the HUD overlay is embedded in the frames.

SB3 Compatibility
- Reward returned to the algorithm must stay scalar.
- Expose auxiliary info via `info[...]` only (VecEnv friendly). Avoid changing Gym API signatures.

Performance & Determinism
- Avoid randomization inside reward functions; rely on env seeding.
- Keep compute O(1) per step. If adding complex terms, memoize via env-cached context.

Future Phases (Guidance)
- New terms candidates: clearance, yield/right-of-way, visibility/occlusion, social distancing, deadlock/no-progress.
- Prefer incremental schemas (`version`: rwd_v2, rwd_v3) when changing semantics to aid offline analysis.

