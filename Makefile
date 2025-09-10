ENV_NAME=amr-nav

.PHONY: setup amr

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
