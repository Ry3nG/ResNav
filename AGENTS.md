# Repository Guidelines

## Project Structure & Module Organization
- `me5418_nav/`: Core Python package.
  - `envs/`: Gymnasium-compatible environment (`UnicycleNavEnv`, `EnvConfig`).
  - `controllers/`: Classical baselines (`pure_pursuit_apf.py`, `dwa.py`).
  - `sensors/`: LiDAR simulator.
  - `models/`: Unicycle robot model wrapper.
  - `viz/`: Pygame renderer.
- `scripts/`: Entry points and demos (e.g., `blockage_demo.py`).
- `logs/`: Run artifacts (safe to ignore in VCS).
- `environment.yml`: Conda environment for local development.

## Architecture Notes
- Two-grid convention:
  - Sensing grid: raw occupancy for LiDAR and rendering.
  - C-space grid: obstacles inflated by robot radius; used for feasibility and collisions.
- Controllers must sense on the sensing grid and check collisions on the C-space grid.

## Build, Test, and Development Commands
- Create env: `conda env create -f environment.yml && conda activate me5418-nav`.
- Run demo (DWA): `python scripts/blockage_demo.py --controller dwa`.
- Run demo (PP+APF): `python scripts/blockage_demo.py --controller ppapf`.
- Headless runs: add `--no-render` to avoid opening a Pygame window.
- Dev tip: scripts add the repo root to `sys.path`, so an editable install is optional. If desired: `pip install -e .`.

## Coding Style & Naming Conventions
- Python 3.10, 4‑space indentation, type hints encouraged.
- Naming: modules/functions `snake_case`; classes `PascalCase`; constants `UPPER_SNAKE_CASE` (see `constants.py`).
- Prefer dataclasses for configs (`PPAPFConfig`, `DWAConfig`, `EnvConfig`).
- Keep lines ≲ 100 chars; concise docstrings explaining intent and units.
- No linter configured; if you use one locally, `black` + `ruff` are fine, but do not reformat unrelated code.

## Testing Guidelines
- No test suite yet. Add `pytest` tests under `tests/` with files named `test_*.py`.
- Focus on small, deterministic checks:
  - LiDAR casting on a tiny synthetic grid.
  - DWA action respects limits and avoids obvious collisions on a C‑space grid.
  - Env step and termination flags for simple paths.
- Run: `pytest -q` (consider adding `pytest` to your env if not present).

## Commit & Pull Request Guidelines
- Commits: short, imperative, scoped prefixes when helpful. Examples:
  - `Env: use EDT-based C-space inflation`
  - `Controllers: tune DWA weights`
- PRs should include:
  - What/why summary, linked issues, and reproduction commands (e.g., demo invocations).
  - Screenshots or short screen captures for behavioral changes (renderer output).
  - Notes on grid usage: sensing grid for LiDAR; C‑space grid for collisions.

## Agent Development (RL)
- Observations: LiDAR ranges, (v, ω), path errors, and waypoint preview (see `UnicycleNavEnv`).
- Actions: continuous `[v, ω]`, bounds `[0.0, 1.5]` and `[-2.0, 2.0]` rad/s.
- Algorithm: PPO via Stable-Baselines3 (see `environment.yml`).
- Scripts: name training/eval as `scripts/train_*.py` and `scripts/eval_*.py`; save outputs under `logs/<exp>/<seed>/`.
- Repro: set Python/NumPy/SB3 seeds; prefer `--no-render` for speed.

## Scenarios & Evaluation
- Micro-benchmarks: Temporary Blockage, Occluded Merge, Narrow Counterflow.
- Metrics: success, collisions, deadlocks, completion time, path following.
- Keep scenario parameters aligned with `constants.py` (robot radius, LiDAR, grid resolution).

## Security & Configuration Tips
- Pygame requires a display; on headless servers use `--no-render`.
- Keep large artifacts out of Git; write run outputs to `logs/`.
- Two‑grid convention is mandatory for correctness and reproducibility.
