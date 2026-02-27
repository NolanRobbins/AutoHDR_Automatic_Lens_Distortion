# Getting Started with AutoHDR Lens Correction

This guide will help you get up and running with the lens correction project.

## ✅ What's Already Done

The complete project structure has been created with:

### 1. Project Structure
- **src/**: Production-ready Python package with type hints
  - `data/`: Dataset loading, transforms, utilities
  - `models/`: Swin-T TPS architecture, edge/perceptual loss functions, metrics
  - `training/`: PyTorch Lightning trainer
  - `inference/`: Prediction pipeline
  - `utils/`: Visualization, geometry, I/O helpers

- **configs/**: Training and inference configurations
  - `train_baseline.yaml`: Baseline U-Net setup (deprecated)
  - `train_advanced.yaml`: Advanced Swin-T + TPS model
  - `inference.yaml`: Test set inference settings

- **notebooks/**: Jupyter notebooks for development
  - `00_environment_setup.ipynb`: Verify setup (✅ Ready to run!)
  - `01_eda_and_analysis.ipynb`: Data exploration (after download)
  - `02_baseline_classical.ipynb`: Classical CV baseline
  - `03_dl_baseline_unet.ipynb`: Deep learning baseline
  - `04_advanced_training.ipynb`: Best model (Colab-compatible)
  - `05_inference_submission.ipynb`: Test set prediction
  - `06_results_analysis.ipynb`: Error analysis

- **tests/**: Unit tests (pytest)
- **Documentation**: README.md, REPORT.md, experiment tracking
- **Git**: Repository initialized with initial commit

### 2. Key Features

✅ **Type-safe code**: All functions have type hints (mypy compatible)
✅ **Modular design**: Easy to swap architectures, losses, transforms
✅ **Config-driven**: Change hyperparameters without code changes
✅ **Colab-ready**: Notebooks work on local or Google Colab
✅ **Test coverage**: Unit tests for models, data, losses
✅ **Documentation**: Comprehensive README and technical report template

---

## 🚀 Next Steps

### Step 1: Environment Setup

**Option A: uv (recommended, fast installs)**

```bash
# From project root (AutoHDR Project/)
uv venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Windows CMD
.venv\Scripts\activate.bat
# Linux/macOS
source .venv/bin/activate

uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt
uv pip install -e .
```

**Option B: Conda**

```bash
conda env create -f environment.yml
conda activate lens_correction
pip install -e .
```

**Verify (with either env active):**

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

**Expected output**: `PyTorch 2.x.x, CUDA: True`

**VS Code / Cursor**: Select the interpreter for this project: `.venv\Scripts\python.exe` (uv) or the conda env.

### Step 2: Verify Setup

Open and run [notebooks/00_environment_setup.ipynb](notebooks/00_environment_setup.ipynb):

```bash
# From project root
jupyter lab

# Or
code .  # Open in VS Code, then open the notebook
```

This notebook will:
- ✅ Check Python version
- ✅ Verify GPU availability (RTX 5060 Ti)
- ✅ Test all package imports
- ✅ Validate directory structure

### Step 3: Download Data

**Option A: Kaggle API** (Recommended)

```bash
# Install Kaggle API (if not in requirements)
pip install kaggle

# Set up credentials
# 1. Go to https://www.kaggle.com/settings
# 2. Create API token
# 3. Place kaggle.json in C:\Users\Robbinhood\.kaggle\

# Download data
cd "c:/Users/Robbinhood/OneDrive/Desktop/VS Code Projects/AutoHDR Project"
kaggle competitions download -c automatic-lens-correction
unzip automatic-lens-correction.zip -d data/

# Verify
dir data\train
dir data\test
```

**Option B: Manual Download**

1. Visit https://www.kaggle.com/competitions/automatic-lens-correction/data
2. Download train.zip and test.zip
3. Extract to `data/train/` and `data/test/`

**Option C: Using Script** (TODO: Create script)

```bash
python scripts/download_data.py
```

### Step 4: Exploratory Data Analysis

Once data is downloaded, run [notebooks/01_eda_and_analysis.ipynb](notebooks/01_eda_and_analysis.ipynb):

This will help you understand:
- Dataset size and characteristics
- Distortion types and severity
- Image resolution distribution
- Content analysis (interior vs exterior)
- Visualizations of distortion patterns

### Step 5: Start Development

Choose your workflow:

**Option A: Notebook-Based** (Recommended for exploration)
1. Work through notebooks 02-06 sequentially
2. Experiment, visualize, iterate quickly
3. Code gets transferred to src/ as it matures

**Option B: CLI-Based** (Production workflow)
```bash
# Train baseline
python scripts/train.py --config configs/train_baseline.yaml

# Train advanced
python scripts/train.py --config configs/train_advanced.yaml

# Run inference
python scripts/inference.py --config configs/inference.yaml
```

**Option C: Hybrid** (Best of both)
- Use notebooks for EDA and experimentation
- Use CLI scripts for final training runs
- Import src/ modules in notebooks: `from src.models import GeometryAwareUNet`

---

## 📊 Development Workflow

### Iterative Training Loop

1. **Train model** (local or Colab)
   ```bash
   python scripts/train.py --config configs/train_baseline.yaml
   ```

2. **Generate predictions** on test set
   ```bash
   python scripts/inference.py
   ```

3. **Upload to bounty.autohdr.com** → get score
   ```bash
   # Zip predictions
   cd outputs/predictions
   zip ../../submission.zip *.jpg

   # Upload via website or script
   python scripts/upload_to_bounty.py
   ```

4. **Download submission.csv**

5. **Submit to Kaggle leaderboard**

6. **Analyze results**, identify improvements, repeat

### Experiment Tracking

Log every experiment in [experiments/experiment_log.md](experiments/experiment_log.md):

```markdown
### Experiment 001: U-Net Baseline
**Date**: 2024-XX-XX
**Approach**: EfficientNet-B3 encoder, L1 + SSIM loss
**Results**:
- Bounty Score: XX/100
- PSNR: YY dB
- Observations: [Your insights]
**Next Steps**: Try geometric loss
```

Also use Weights & Biases for automated tracking:
```python
# In training script
import wandb
wandb.init(project="lens-correction", name="unet-baseline")
```

---

## 💡 Tips for Success

### For Interview Assessment

1. **Document your thinking**: Clear narratives in notebooks show thought process
2. **Code quality matters**: Type hints, docstrings, tests demonstrate professionalism
3. **Iterate systematically**: Baseline → Ablation → Advanced shows rigor
4. **Visualize results**: Plots and comparisons make findings clear
5. **Be honest about limitations**: Acknowledge what didn't work and why

### GPU Optimization

Your RTX 5060 Ti (16GB) is powerful! Optimize with:
- **Mixed precision**: `precision="16-mixed"` in trainer (2× speedup)
- **Batch size**: Start with 8, increase if memory allows
- **Gradient accumulation**: Effective batch size = batch_size × accumulate_grad_batches
- **Pin memory**: `pin_memory=True` in DataLoader (faster CPU→GPU transfer)

### Time Management

Estimated timeline:
- ✅ Setup (1-2 hours) - **DONE**
- EDA (3-4 hours)
- Classical baseline (2 hours)
- DL baseline (4-6 hours training)
- Advanced model (8-12 hours training)
- Analysis & polish (4-6 hours)

**Total**: ~30-40 hours for complete project

### Troubleshooting

**Issue**: CUDA out of memory
**Solution**: Reduce batch size, use gradient accumulation, or lower image resolution

**Issue**: Training too slow
**Solution**: Use mixed precision, or switch to Colab with A100

**Issue**: Poor results
**Solution**: Check data loading (visualize batches), verify loss computation, adjust weights

---

## 🎯 Success Criteria

For a strong submission:

✅ **Technical**:
- Model achieves >80/100 on bounty.autohdr.com
- Clean, modular code with type hints
- Comprehensive testing

✅ **Engineering**:
- Reproducible results (fixed seeds, version control)
- Clear documentation (README, REPORT.md)
- Systematic experimentation (logged in experiment_log.md)

✅ **Communication**:
- Clear narrative in notebooks
- Thoughtful analysis of results
- Honest assessment of limitations and future work

---

## 📚 Resources

**Project Files**:
- [README.md](README.md): Complete project overview
- [REPORT.md](REPORT.md): Technical report template
- [experiments/experiment_log.md](experiments/experiment_log.md): Experiment tracking

**External Resources**:
- Kaggle Competition: https://www.kaggle.com/competitions/automatic-lens-correction
- Bounty Scoring: https://bounty.autohdr.com
- Discord: https://discord.gg/yf4Ky2VXg

**Technical References**:
- U-Net paper: https://arxiv.org/abs/1505.04597
- EfficientNet: https://arxiv.org/abs/1905.11946
- Lens distortion models: Brown-Conrady model

---

## ❓ FAQ

**Q: Can I use Colab instead of local GPU?**
A: Yes! All notebooks are Colab-compatible. See notebook 04 for Colab-specific setup.

**Q: How long does training take?**
A: Baseline (~3-4 hours on RTX 5060 Ti), Advanced (~8-12 hours). Colab A100 is 3-5× faster.

**Q: What if I don't have enough disk space?**
A: Dataset is ~50GB. You can use external drive or stream data from cloud.

**Q: Can I modify the architecture?**
A: Absolutely! The modular design makes it easy. Change `configs/train_*.yaml` or edit `src/models/unet.py`.

**Q: How do I know if my approach is working?**
A: Compare to baseline scores. Classical CV ~20-30, Simple U-Net ~60, Advanced ~85+.

---

**Status**: ✅ Environment setup complete, ready for data download and EDA!

**Last Updated**: February 24, 2026
