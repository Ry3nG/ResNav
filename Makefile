PY=python

.PHONY: setup train eval test lint fmt train-smoke train-full eval-best render-best eval-model render-model render-ckpt benchmark-ppo list-models

setup:
	@echo "[setup] Create environment via requirements.txt (or environment.yml)"

train:
	@echo "[train] Running PPO residual training (Phase I)"
	@$(PY) training/train_ppo.py

eval:
	@echo "[eval] Benchmark placeholder"
	@echo "Use: python eval/benchmark.py --model runs/best/best_model.zip --vecnorm runs/vecnorm.pkl"

test:
	@echo "[test] Running unit tests"
	@pytest -q

lint:
	@echo "[lint] (placeholder)"

fmt:
	@echo "[fmt] (placeholder)"

train-smoke:
	@echo "[train-smoke] 2 envs, 5k steps"
	@$(PY) training/train_ppo.py --n_envs 2 --timesteps 5000

train-full:
	@echo "[train-full] 16 envs, 1000000k steps"
	@$(PY) training/train_ppo.py --n_envs 16 --timesteps 10000000

# Parameterized targets - specify MODEL and VECNORM paths
eval-model:
	@echo "[eval-model] Evaluate specified model"
	@test -n "$(MODEL)" || (echo "Usage: make eval-model MODEL=path/to/model.zip VECNORM=path/to/vecnorm.pkl" && exit 1)
	@test -n "$(VECNORM)" || (echo "Usage: make eval-model MODEL=path/to/model.zip VECNORM=path/to/vecnorm.pkl" && exit 1)
	@$(eval MODEL_DIR := $(dir $(MODEL)))
	@$(eval TIMESTAMP_DIR := $(shell echo $(MODEL_DIR) | sed 's|^runs/||' | sed 's|/.*||'))
	@$(PY) eval/benchmark.py --agent ppo --episodes 100 --model $(MODEL) --vecnorm $(VECNORM) --csv runs/$(TIMESTAMP_DIR)/eval_$(shell basename $(MODEL) .zip).csv

render-model:
	@echo "[render-model] Record demo from specified model"
	@test -n "$(MODEL)" || (echo "Usage: make render-model MODEL=path/to/model.zip VECNORM=path/to/vecnorm.pkl [SEED=42]" && exit 1)
	@test -n "$(VECNORM)" || (echo "Usage: make render-model MODEL=path/to/model.zip VECNORM=path/to/vecnorm.pkl [SEED=42]" && exit 1)
	@$(eval MODEL_DIR := $(dir $(MODEL)))
	@$(eval TIMESTAMP_DIR := $(shell echo $(MODEL_DIR) | sed 's|^runs/||' | sed 's|/.*||'))
	@$(PY) training/rollout.py --record runs/$(TIMESTAMP_DIR)/demo_$(shell basename $(MODEL) .zip).mp4 --model $(MODEL) --vecnorm $(VECNORM) --steps 600 --deterministic --seed $(or $(SEED),$$RANDOM)

render-ckpt:
	@echo "[render-ckpt] Record demo from a checkpoint directory"
	@test -n "$(CKPT_DIR)" || (echo "Usage: make render-ckpt CKPT_DIR=runs/TIMESTAMP/checkpoints/ckpt_step_N [SEED=42]" && exit 1)
	@test -f "$(CKPT_DIR)/model.zip" || (echo "Missing $(CKPT_DIR)/model.zip" && exit 1)
	@test -f "$(CKPT_DIR)/vecnorm.pkl" || (echo "Missing $(CKPT_DIR)/vecnorm.pkl" && exit 1)
	@$(eval MODEL := $(CKPT_DIR)/model.zip)
	@$(eval VECNORM := $(CKPT_DIR)/vecnorm.pkl)
	@$(eval RUN_DIR := $(shell echo $(CKPT_DIR) | sed 's|/checkpoints/.*||'))
	@$(PY) training/rollout.py --record $(RUN_DIR)/demo_$(shell basename $(CKPT_DIR)).mp4 --model $(MODEL) --vecnorm $(VECNORM) --steps 600 --deterministic --seed $(or $(SEED),$$RANDOM)

# Legacy targets (deprecated - use eval-model/render-model instead)
eval-best:
	@echo "[DEPRECATED] Use: make eval-model MODEL=runs/TIMESTAMP/best/best_model.zip VECNORM=runs/TIMESTAMP/vecnorm.pkl"
	@echo "Available models:" && find runs -name "best_model.zip" 2>/dev/null | head -5

render-best:
	@echo "[DEPRECATED] Use: make render-model MODEL=runs/TIMESTAMP/best/best_model.zip VECNORM=runs/TIMESTAMP/vecnorm.pkl"
	@echo "Available models:" && find runs -name "best_model.zip" 2>/dev/null | head -5

render-dwa:
	@echo "[render-dwa] Visualize DWA baseline"
	@$(PY) training/rollout.py --render --agent dwa --steps 600 --dwa_cfg configs/control/dwa.yaml

render-pp:
	@echo "[render-pp] Visualize Pure Pursuit baseline"
	@$(PY) training/rollout.py --render --agent pp --steps 600

benchmark-all:
	@echo "[benchmark-all] Running pp, dwa, and ppo benchmarks (100 episodes each)"
	@$(PY) eval/benchmark.py --agent pp --episodes 100 --csv runs/bench_pp.csv
	@$(PY) eval/benchmark.py --agent dwa --episodes 100 --csv runs/bench_dwa.csv --dwa_cfg configs/control/dwa.yaml
	@echo "For PPO benchmark, specify model: make benchmark-ppo MODEL=runs/TIMESTAMP/best/best_model.zip VECNORM=runs/TIMESTAMP/vecnorm.pkl"

benchmark-ppo:
	@echo "[benchmark-ppo] Running PPO benchmark with specified model"
	@test -n "$(MODEL)" || (echo "Usage: make benchmark-ppo MODEL=path/to/model.zip VECNORM=path/to/vecnorm.pkl" && exit 1)
	@test -n "$(VECNORM)" || (echo "Usage: make benchmark-ppo MODEL=path/to/model.zip VECNORM=path/to/vecnorm.pkl" && exit 1)
	@$(eval MODEL_DIR := $(dir $(MODEL)))
	@$(eval TIMESTAMP_DIR := $(shell echo $(MODEL_DIR) | sed 's|^runs/||' | sed 's|/.*||'))
	@$(PY) eval/benchmark.py --agent ppo --episodes 100 --csv runs/$(TIMESTAMP_DIR)/bench_ppo_$(shell basename $(MODEL) .zip).csv --model $(MODEL) --vecnorm $(VECNORM)

list-models:
	@echo "[list-models] Available trained models:"
	@echo ""
	@for model in $$(find runs -name "best_model.zip" -path "*/best/*" 2>/dev/null | sort -r); do \
		dir=$$(dirname $$model | sed 's|/best||'); \
		vecnorm="$$dir/vecnorm.pkl"; \
		if [ -f "$$vecnorm" ]; then \
			echo "📁 $$dir"; \
			echo "   Model:   $$model"; \
			echo "   Vecnorm: $$vecnorm"; \
			echo "   📊 make eval-model MODEL=$$model VECNORM=$$vecnorm"; \
			echo "   🎬 make render-model MODEL=$$model VECNORM=$$vecnorm SEED=42"; \
			echo ""; \
		fi \
	done || echo "No models found"
