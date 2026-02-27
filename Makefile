.PHONY: help setup setup-uv install download-data train-baseline train-radial demo demo-dl inference test lint format clean

help:
	@echo "Available commands:"
	@echo "  make setup              - Create conda environment and install dependencies"
	@echo "  make setup-uv           - Create uv venv (.venv) and install dependencies"
	@echo "  make install            - Install package in editable mode"
	@echo "  make download-data      - Download dataset from Kaggle"
	@echo "  make train-baseline     - Train baseline U-Net model"
	@echo "  make train-radial       - Train radial distortion model"
	@echo "  make demo               - Launch Gradio demo (ensemble mode)"
	@echo "  make demo-dl            - Launch Gradio demo (DL-only mode)"
	@echo "  make inference          - Run inference on test set"
	@echo "  make test               - Run pytest test suite"
	@echo "  make lint               - Run code quality checks (mypy, ruff)"
	@echo "  make format             - Format code with black and isort"
	@echo "  make clean              - Remove cache and temporary files"

setup:
	@echo "Creating conda environment..."
	conda env create -f environment.yml
	@echo "Environment created! Activate with: conda activate lens_correction"

setup-uv:
	uv venv
	uv pip install -r requirements.txt
	uv pip install -r requirements-dev.txt
	uv pip install -e .
	@echo "uv env ready. Activate with: .venv\\Scripts\\activate (Windows) or source .venv/bin/activate (Unix)"

install:
	pip install -e .
	pip install -r requirements-dev.txt

download-data:
	python scripts/download_data.py

train-baseline:
	python scripts/train.py --config configs/train_baseline.yaml

train-radial:
	python scripts/train.py --config configs/train_radial.yaml

train-advanced:
	python scripts/train.py --config configs/train_advanced.yaml

demo:
	python scripts/demo.py --ckpt outputs/models/radial_v1-epoch=04-val_ssim=0.7963.ckpt --config configs/train_radial.yaml --mode ensemble

demo-dl:
	python scripts/demo.py --ckpt outputs/models/radial_v1-epoch=04-val_ssim=0.7963.ckpt --config configs/train_radial.yaml --mode dl

inference:
	python scripts/inference.py --config configs/inference.yaml

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	@echo "Running mypy..."
	mypy src/
	@echo "Running ruff..."
	ruff check src/ tests/

format:
	@echo "Running black..."
	black src/ tests/ scripts/
	@echo "Running isort..."
	isort src/ tests/ scripts/

clean:
	@echo "Cleaning cache and temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/ 2>/dev/null || true
	@echo "Clean complete!"
