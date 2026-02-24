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

## Quick Start

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

**Option B: venv**
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

### Architecture: Geometry-Aware U-Net

**Key Components**:
1. **Encoder**: EfficientNet-B4 (pretrained on ImageNet)
2. **Decoder**: U-Net with skip connections + attention (CBAM)
3. **Loss Function**: Multi-component geometric loss
   - Pixel accuracy (L1)
   - Edge alignment (Sobel-based)
   - Structural similarity (MS-SSIM)
   - Gradient orientation
   - Line straightness (optional)

**Why This Design?**
- Evaluation metric prioritizes **geometric accuracy** over photometric quality
- Real-estate images have strong structural features (lines, edges, corners)
- Pretrained encoder provides robust features
- Attention helps model focus on distorted regions

### Training Strategy

**Stage 1: Baseline (Quick Validation)**
- Simple U-Net with L1 + SSIM loss
- Train 50 epochs (~2-3 hours on RTX 5060 Ti)
- Target: Establish baseline performance

**Stage 2: Advanced (Best Performance)**
- Geometry-aware loss function
- Multi-task learning (optional param estimation)
- Longer training (70 epochs)
- Test-time augmentation (TTA)

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

*To be updated after training*

| Model | PSNR (dB) | SSIM | Bounty Score | Notes |
|-------|-----------|------|--------------|-------|
| Classical CV Baseline | - | - | ~20-30 | OpenCV calibration |
| U-Net Baseline | - | - | ~60 | Simple L1 loss |
| Geometry-Aware U-Net | - | - | **Target: 85+** | Multi-component loss |

---

## Key Insights

*To be documented after EDA and experimentation*

1. **Data Characteristics**: [TBD after EDA]
2. **Distortion Patterns**: [TBD]
3. **Model Performance**: [TBD]
4. **Failure Modes**: [TBD]

---

## Next Steps / Future Improvements

- [ ] Spatial Transformer Networks (STN) for explicit geometric learning
- [ ] Ensemble of diverse architectures
- [ ] Progressive resolution training (256 → 512 → 768)
- [ ] Self-supervised pre-training on unlabeled data
- [ ] Domain-specific fine-tuning (interior vs. exterior)

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

**Status**: 🚧 In Development

Last Updated: [Date]
