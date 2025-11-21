# Unified AMR Navigation

Residual SAC policy learns a corrective action on top of a Pure Pursuit baseline to navigate cluttered factory corridors. This README documents the final deliverable that accompanies the 11/21 submission and explains how to replay or extend the provided results.

## Final Deliverable Snapshot
- **Policy checkpoint**: `runs/demo_final/final/final_model.zip`
- **VecNormalize statistics**: `runs/demo_final/final/vecnorm_final.pkl`
- **Evaluation environment**: `configs/env/eval_hard.yaml`
- **Reference seed**: `20030413`
- **Reference rollout video**: `runs/eval_outputs/eval_hard_20030413_final.mp4`
- **Supporting metrics**: `runs/final_evaluation/` (contains JSON logs used in the report)

## Environment Setup
1. `cd ResNav`
2. `make setup` (creates/updates the `amr-nav` conda env from `environment.yml` and runs `pip install -e .`)
3. `conda activate amr-nav`
4. (Optional) `make amr` launches the interactive helper used during debugging; all commands below can be run manually.

## Final Evaluation Workflow
1. Ensure the `runs/demo_final/final` directory remains intact (contains both `final_model.zip` and `vecnorm_final.pkl`).
2. Use the provided helper to replay the submission trajectory:

```bash
python inference_demo.py --render --record runs/eval_outputs/eval_hard_20030413_final.mp4 --deterministic
```

`inference_demo.py` now defaults to:
- `--path runs/demo_final/final`
- `--env-cfg configs/env/eval_hard.yaml`
- `--seed 20030413`

Therefore a plain `python inference_demo.py --render` reproduces the same rollout that was submitted for grading. Pass `--record none` to skip writing the MP4 or provide a different filename/path.

3. To adjust evaluation parameters:
   - `--steps` controls rollout horizon (default 600).
   - `--deterministic` toggles greedy vs. stochastic actions.
   - `--robot-cfg` / `--reward-cfg` accept alternative YAMLs if you wish to perform ablations.

## Training & Batch Evaluation (Optional)
- Standard SAC training is still driven by `python train_demo.py`, which launches `training/train_sac.py` with `run.total_timesteps=2e5` and seed `20030413`.
- `batch_evaluation.py` remains available for large sweeps (edit its `MODEL_PATH` constant if you moved checkpoints). Typical usage: `python batch_evaluation.py --methods rl_agent --difficulties hard --seeds 20030413 42` for a quick sanity check, or append `--quick-test` to run the smallest smoke test.

## Repository Layout
- `amr_env/sim`: lightweight 2D simulator (dynamics, collision checking, LiDAR).
- `amr_env/gym`: `ResidualNavEnv`, observation builders, reward shaping.
- `amr_env/control` & `amr_env/planning`: classical controllers and path utilities; `pure_pursuit.py` defines the baseline tracker.
- `training`: SAC training loop, feature extractors, rollout utilities.
- `configs`: YAML configuration for environments, robots, rewards, and networks.
- `runs`: contains every experiment; the final submission artifacts reside in `runs/demo_final/final`.

## Troubleshooting Notes
- Use `SDL_VIDEODRIVER=dummy` if running headless and only recording videos.
- If `VecNormalize` stats are missing, re-copy `vecnorm_final.pkl` next to `final_model.zip`; the runner automatically picks it up.
- All scripts assume Python ≥ 3.10 (PEP 563-style annotations are used throughout).
