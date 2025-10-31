ENV_NAME=amr-nav

.PHONY: setup amr demo1031

setup:
	@echo "[setup] Ensure conda env '$(ENV_NAME)' and install package"
	@command -v conda >/dev/null 2>&1 || (echo "Conda not found. Install Miniconda/Anaconda first: https://docs.conda.io" && exit 1)
	@echo "[setup] Creating or updating environment.yml → $(ENV_NAME)"
	@if conda env list | grep -E '^$(ENV_NAME)[[:space:]]' >/dev/null 2>&1; then \
		echo "[setup] Environment exists → updating"; \
		conda env update -n $(ENV_NAME) -f environment.yml --prune; \
	else \
		echo "[setup] Environment not found → creating"; \
		conda env create -f environment.yml; \
	fi
	@echo "[setup] pip install -e . in $(ENV_NAME)"
	@conda run -n $(ENV_NAME) python -m pip install -e .
	@echo "[setup] Done. To activate later: 'conda activate $(ENV_NAME)'."

amr:
	@echo "[amr] Interactive launcher"
	@conda run -n $(ENV_NAME) --no-capture-output python tools/launcher.py

demo1031:
	@echo "[demo] Rollout runs/demo_1031 -> runs/demo_1031/outputs/demo.mp4"
	@conda run -n $(ENV_NAME) --no-capture-output \
		python training/rollout.py \
			--model 'runs/demo_1031/best' \
			--record 'runs/demo_1031/outputs/demo.mp4' \
			--steps 300 \
			--deterministic \
			--seed 20021213
