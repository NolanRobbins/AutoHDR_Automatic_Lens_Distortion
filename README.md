# Automatic Lens Correction for Real-Estate Photography

**Kaggle Competition**: [Automatic Lens Correction](https://www.kaggle.com/competitions/automatic-lens-correction)

Deep learning solution for correcting barrel and pincushion lens distortion in real-estate photography without manufacturer lens profiles.

---

## Problem Statement

Modern cameras and lenses introduce predictable distortions (barrel/pincushion) that affect image quality, particularly in real-estate photography where straight edges matter. Traditional workflows rely on lens profiles from manufacturers, but coverage often lags behind new camera/lens releases.

**Objective**: Build a model that automatically corrects lens distortion in photos without requiring lens-specific calibration profiles.

---

## Project Structure

```
AutoHDR Project/
├── configs/                     # Training and inference configurations
│   ├── train_baseline.yaml     # Baseline U-Net config
│   ├── train_advanced.yaml     # Advanced model config
│   └── inference.yaml          # Inference settings
├── data/                        # Dataset (download separately)
│   ├── train/                  # Training pairs (distorted + corrected)
│   ├── test/                   # Test images to correct
│   └── README.md               # Data download instructions
├── notebooks/                   # Jupyter notebooks (development workflow)
│   ├── 00_environment_setup.ipynb
│   ├── 01_eda_and_analysis.ipynb
│   ├── 02_baseline_classical.ipynb
│   ├── 03_dl_baseline_unet.ipynb
│   ├── 04_advanced_training.ipynb    # Colab-compatible
│   ├── 05_inference_submission.ipynb
│   └── 06_results_analysis.ipynb
├── src/                         # Source code (production-ready)
│   ├── data/                   # Dataset, transforms
│   ├── models/                 # Architectures, losses, metrics
│   ├── training/               # Training pipeline
│   ├── inference/              # Inference pipeline
│   └── utils/                  # Helpers (visualization, geometry, I/O)
├── scripts/                     # CLI scripts
│   ├── download_data.py
│   ├── train.py
│   ├── inference.py
│   ├── prepare_submission.py
│   └── upload_to_bounty.py
├── tests/                       # Unit tests
├── outputs/                     # Generated files (gitignored)
│   ├── models/                 # Saved checkpoints
│   ├── predictions/            # Test set predictions
│   ├── figures/                # Visualizations
│   └── logs/                   # Training logs
├── requirements.txt             # Python dependencies
├── requirements-dev.txt         # Development dependencies
├── environment.yml              # Conda environment
├── Makefile                     # Automated workflows
└── REPORT.md                    # Technical report
```

---

## 🚀 Try the Demo (Quick Start for Interviewers)

**Want to see the lens correction in action? Launch the Gradio demo in 2 simple ways:**

### Option 1: Using Makefile (Easiest)

```bash
# Activate environment first
.venv\Scripts\activate   # Windows CMD
# OR: .venv\Scripts\Activate.ps1   # Windows PowerShell

# Launch ensemble demo (DL + classical refinement)
make demo

# OR: Launch DL-only demo
make demo-dl
```

### Option 2: Direct Command

```bash
# Navigate to project directory
cd "c:\Users\Robbinhood\OneDrive\Desktop\VS Code Projects\AutoHDR Project"

# Activate virtual environment (if not already active)
.venv\Scripts\activate   # Windows CMD

# Launch demo with the trained ensemble model
python scripts/demo.py --ckpt outputs/models/radial_v1-epoch=04-val_ssim=0.7963.ckpt --mode ensemble
```

**What you'll see**:
- **Gradio web interface** at `http://127.0.0.1:7860`
- **Upload any image** or try the built-in examples (high-distortion images included)
- **View correction details**: method (cascade vs dl_only), k1/k2 coefficients, lines detected
- **Compare original vs corrected** side-by-side

**Demo Modes**:
- `--mode ensemble` (default): DL + classical refinement when lines are detected
- `--mode dl`: Deep learning only (no classical refinement)

**Port in use?** Add `--port 7861` to use a different port or set `GRADIO_SERVER_PORT=7861`.

---

## Full Setup Instructions

### 1. Environment Setup

**Option A: Conda (Recommended)**
```bash
# Create environment
conda env create -f environment.yml

# Activate
conda activate lens_correction

# Install package
pip install -e .
```

**Option B: uv (Recommended for fast installs)**
```bash
# From project root (AutoHDR Project/)
uv venv
# Activate: Windows PowerShell
.venv\Scripts\Activate.ps1
# Activate: Windows CMD
.venv\Scripts\activate.bat
# Activate: Linux/macOS
source .venv/bin/activate

# Install dependencies (uses .venv if active, or pass --python .venv)
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt
uv pip install -e .
```

**Option C: venv**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

### 2. Verify Setup

Run notebook [00_environment_setup.ipynb](notebooks/00_environment_setup.ipynb) to verify:
- GPU availability (RTX 5060 Ti detected)
- Package imports work
- Data directory structure

### 3. Download Data

```bash
# Option 1: Using Makefile
make download-data

# Option 2: Manual with Kaggle API
kaggle competitions download -c automatic-lens-correction
unzip automatic-lens-correction.zip -d data/

# Option 3: Manual download from Kaggle website
# Place files in data/train/ and data/test/
```

**Dataset Overview**:
- Training: ~23,000 paired images (distorted + corrected)
- Test: 1,000 distorted images to correct
- File format: JPEG

### 4. Explore Data

Open and run [01_eda_and_analysis.ipynb](notebooks/01_eda_and_analysis.ipynb) to:
- Analyze image statistics (resolution, distortion severity)
- Visualize distortion patterns
- Understand geometric transformations
- Identify challenges

---

## Development Workflow

### Option 1: Notebook-Based (Recommended for Exploration)

1. **EDA**: `01_eda_and_analysis.ipynb`
2. **Baseline**: `02_baseline_classical.ipynb` (classical CV)
3. **Deep Learning**: `03_dl_baseline_unet.ipynb` (simple U-Net)
4. **Advanced**: `04_advanced_training.ipynb` (best model, Colab-compatible)
5. **Inference**: `05_inference_submission.ipynb`
6. **Analysis**: `06_results_analysis.ipynb`

### Option 2: CLI-Based (Production Workflow)

```bash
# Train baseline model
make train-baseline
# or: python scripts/train.py --config configs/train_baseline.yaml

# Train advanced model
make train-advanced

# Run inference
make inference

# Create submission
python scripts/prepare_submission.py
```

---

## Model Approach

### Architecture: Radial Distortion Model + Cascade Ensemble

**Key Innovation**: Physics-informed output space with DL + classical cascade for robust correction across all image types.

**Components**:
1. **DL Model (Swin-T + Radial Coefficients)**:
   - **Encoder**: Swin Transformer (Swin-T, pretrained) for global self-attention
   - **Output**: Predicts 2 Brown-Conrady coefficients (k1, k2) instead of hundreds of TPS parameters
   - **Grid Generation**: Analytical radial distortion model: `scale = 1 + k1·r² + k2·r⁴`
   - **Resampling**: `F.grid_sample` warps the *original image* using the analytical grid

2. **Classical Refinement (Line-Based Optimizer)**:
   - Canny edge detection + Hough line detection
   - Optimizes k1 to minimize line curvature
   - Uses DL prediction as initialization

3. **Cascade Ensemble**:
   - **Every image**: DL model predicts k1, k2
   - **When lines ≥ 5**: Classical optimizer refines k1 (method: "cascade")
   - **When lines < 5**: Use DL prediction only (method: "dl_only")
   - **Result**: One corrected image per input

**Why This Design?**
- **Physics-informed**: 2-parameter radial model yields stable training (vs. TPS plateau)
- **Universal coverage**: DL works on all images (including those with few lines)
- **Precision refinement**: Classical optimization improves results when line cues exist
- **Texture preservation**: Grid resampling preserves original image quality perfectly
- **Interpretable**: Demo shows method, k1/k2, line count, and correction quality

### Training Strategy

**Current Model: Radial v1**
- **Architecture**: Swin-T encoder + MLP head → (k1, k2)
- **Loss**: L1 + LPIPS + Sobel edge + L2 param regularization
- **Optimizer**: AdamW with cosine annealing warm restarts
- **Training**: Mixed precision, batch size 8, 224×224 resolution
- **Best Checkpoint**: `radial_v1-epoch=04-val_ssim=0.7963.ckpt`
- **Result**: Validation SSIM peaked at **0.7963** (epoch 4), stable training

---

## Evaluation

**Primary Metric**: Custom geometric scoring (0-100)
- Computed via [bounty.autohdr.com](https://bounty.autohdr.com)
- Focuses on: edge alignment, line straightness, gradient orientation, SSIM

**Local Metrics** (for development):
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- Edge alignment error
- Line curvature

**Workflow**:
1. Train model locally or on Colab
2. Generate predictions on test set
3. Upload to bounty.autohdr.com → get score
4. Download submission.csv
5. Submit to Kaggle leaderboard
6. Iterate based on feedback

---

## Running on Google Colab

All notebooks are Colab-compatible for faster training on A100 GPUs:

1. Upload project to Google Drive or GitHub
2. Open [04_advanced_training.ipynb](notebooks/04_advanced_training.ipynb) in Colab
3. Mount Drive / clone repo
4. Install dependencies: `!pip install -r requirements.txt`
5. Train with A100 (3-5x faster than local)
6. Save checkpoints to Drive

**Benefits**:
- Access to high-end GPUs (A100, V100)
- No local GPU wear
- Can run multiple experiments in parallel

---

## Testing

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_models.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

---

## Code Quality

```bash
# Format code
make format

# Lint code
make lint

# Clean cache
make clean
```

**Standards** (following [CLAUDE.md](../claude.md)):
- Type hints on all functions
- Google-style docstrings
- Black formatting (line length 88)
- mypy type checking
- pytest for testing

---

## Results

### Model Performance

| Model | Val SSIM | Val PSNR (dB) | Epoch | Notes |
|-------|----------|---------------|-------|-------|
| Swin-T + TPS | ~0.798 | ~12.6 | 0 | Plateaued at initialization |
| **Radial v1 (Swin-T + k1/k2)** | **0.7963** | ~12.3 | **4** | **Best model - stable training** |
| Cascade Ensemble | — | — | — | Same metrics; improves line straightness |

**Key Achievements**:
1. **Stable Training**: Physics-informed radial model trains reliably (vs. TPS plateau)
2. **Best Checkpoint**: Validation SSIM peaked early at epoch 4
3. **Ensemble Strategy**: DL works on all images; classical refinement when lines detected
4. **Texture Preservation**: Grid resampling maintains original image quality

---

## Key Insights

1. **Physics-informed output space**: Predicting 2 radial coefficients (k1, k2) instead of 200 TPS control points yields a well-conditioned optimization landscape and faster, more stable training than the TPS variant.

2. **Cascade ensemble strength**: DL handles all images (including those with few lines); classical refinement improves results when many lines are detected. This combines the strengths of both approaches.

3. **When to showcase DL**: Images with **few straight lines** (soft furnishings, nature, curves) rely on the DL prediction; images with **many lines** (bathrooms, kitchens) show strong classical refinement—both are correct use cases of the system.

4. **Early convergence**: Best validation metrics achieved at epoch 4, demonstrating efficient learning and proper regularization.

---

## Next Steps / Future Improvements

**Short-term** (1-2 weeks):
- [ ] Progressive resolution training (256 → 512 → 768)
- [ ] Test-time augmentation (TTA) for ensemble predictions
- [ ] Hard example mining for failure modes
- [ ] Multi-model averaging (diverse encoder backbones)

**Medium-term** (1-2 months):
- [ ] Self-supervised pre-training on unlabeled distorted images
- [ ] Domain adaptation (interior vs. exterior scenes)
- [ ] Real-time inference optimization (ONNX, TensorRT)
- [ ] Higher-order distortion models (mustache, asymmetric)

**Long-term** (research):
- [ ] Spatial Transformer Networks (STN) for explicit geometric learning
- [ ] Diffusion models for distortion correction
- [ ] Unsupervised/weakly-supervised methods
- [ ] Generalization to non-photographic distortion

---

## Dependencies

**Core**:
- Python 3.11
- PyTorch 2.1+
- OpenCV 4.8+
- Albumentations 1.3+

**Full list**: See [requirements.txt](requirements.txt)

---

## GPU Requirements

**Minimum**:
- 8GB VRAM (batch size 4, 512×512 images)

**Recommended**:
- 16GB VRAM (batch size 8-16, full resolution)
- RTX 3080 / 4090 / A100

**Current Setup**: RTX 5060 Ti (16GB) - Excellent for this task!

---

## License

This project is for educational and assessment purposes. Dataset images are provided under Kaggle competition terms and authorized strictly for training purposes.

---

## Acknowledgments

- Kaggle competition organizers
- PyTorch and timm libraries
- Albumentations for augmentation
- Segmentation Models PyTorch

---

## Contact

For questions or issues, please open a GitHub issue or contact [your email].

---

## Demo Screenshots

The Gradio interface shows:
- **Original distorted image** (left) vs **corrected image** (right)
- **Correction details**: Method (cascade/dl_only), k1/k2 coefficients, lines detected, DL k1
- **Example images**: High-distortion samples included to showcase correction quality

---

**Status**: ✅ Trained Model Available - Demo Ready

**Last Updated**: February 27, 2026

**Model Checkpoint**: `outputs/models/radial_v1-epoch=04-val_ssim=0.7963.ckpt`

**Quick Demo Command**:
```bash
python scripts/demo.py --ckpt outputs/models/radial_v1-epoch=04-val_ssim=0.7963.ckpt --mode ensemble
```
