.PHONY: help setup install download-data train-baseline train-advanced inference test lint format clean

help:
	@echo "Available commands:"
	@echo "  make setup              - Create conda environment and install dependencies"
	@echo "  make install            - Install package in editable mode"
	@echo "  make download-data      - Download dataset from Kaggle"
	@echo "  make train-baseline     - Train baseline U-Net model"
	@echo "  make train-advanced     - Train advanced model"
	@echo "  make inference          - Run inference on test set"
	@echo "  make test               - Run pytest test suite"
	@echo "  make lint               - Run code quality checks (mypy, ruff)"
	@echo "  make format             - Format code with black and isort"
	@echo "  make clean              - Remove cache and temporary files"

setup:
	@echo "Creating conda environment..."
	conda env create -f environment.yml
	@echo "Environment created! Activate with: conda activate lens_correction"

install:
	pip install -e .
	pip install -r requirements-dev.txt

download-data:
	python scripts/download_data.py

train-baseline:
	python scripts/train.py --config configs/train_baseline.yaml

train-advanced:
	python scripts/train.py --config configs/train_advanced.yaml

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
