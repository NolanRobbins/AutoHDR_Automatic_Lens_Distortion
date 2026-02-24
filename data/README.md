# Data Directory

This directory contains the dataset for the Automatic Lens Correction competition.

## Directory Structure

```
data/
├── train/          # Training image pairs (~23,000 pairs)
│   ├── pair_0001/
│   │   ├── original.jpg    # Distorted image
│   │   └── generated.jpg   # Corrected ground truth
│   ├── pair_0002/
│   └── ...
├── test/           # Test images to correct (1,000 images)
│   ├── 0001.jpg
│   ├── 0002.jpg
│   └── ...
└── README.md       # This file
```

## Download Instructions

### Option 1: Kaggle API (Recommended)

```bash
# Install Kaggle API
pip install kaggle

# Set up credentials (if not already done)
# Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\Users\<username>\.kaggle\ (Windows)

# Download dataset
kaggle competitions download -c automatic-lens-correction

# Extract
unzip automatic-lens-correction.zip -d data/

# Verify
ls data/train/
ls data/test/
```

### Option 2: Manual Download

1. Go to https://www.kaggle.com/competitions/automatic-lens-correction/data
2. Download `train.zip` and `test.zip`
3. Extract both archives into this directory
4. Verify structure matches above

### Option 3: Using Project Script

```bash
python scripts/download_data.py
```

## Dataset Statistics

*To be filled after initial exploration*

**Training Set**:
- Total pairs: ~23,000
- Image format: JPEG
- Resolution range: [Min x Max]
- Average file size: [X] MB
- Total size: ~[Y] GB

**Test Set**:
- Total images: 1,000
- Image format: JPEG
- Resolution range: [Min x Max]
- Total size: ~[Z] GB

## Data Characteristics

*From EDA (01_eda_and_analysis.ipynb)*

**Distortion Types**:
- Barrel distortion: [X]%
- Pincushion distortion: [Y]%
- Mixed/complex: [Z]%

**Content Analysis**:
- Interior shots: [X]%
- Exterior shots: [Y]%
- Mixed scenes: [Z]%

**Quality**:
- Resolution: [Analysis]
- Compression artifacts: [Analysis]
- Color distribution: [Analysis]

## Data Splits

For training, we use stratified splits based on distortion severity:

```python
# 5-fold cross-validation
- Fold 1: Train on 80%, Val on 20%
- Fold 2: Train on 80%, Val on 20%
...

# Or single split
- Train: 85% (~19,550 pairs)
- Validation: 15% (~3,450 pairs)
- Test: 1,000 images (for competition submission)
```

## Usage

```python
from src.data import LensDistortionDataset, get_train_transforms

# Load training data
train_dataset = LensDistortionDataset(
    data_dir="data/train",
    split="train",
    transform=get_train_transforms(size=(512, 512))
)

# Load validation data
val_dataset = LensDistortionDataset(
    data_dir="data/train",
    split="val",
    transform=get_val_transforms(size=(512, 512))
)

# Load test data
test_dataset = LensDistortionDataset(
    data_dir="data/test",
    split="test",
    transform=get_test_transforms(size=(512, 512))
)
```

## Important Notes

⚠️ **Git Ignore**: Data files are gitignored due to large size. Always download fresh data.

⚠️ **Disk Space**: Ensure you have at least 50GB free space before downloading.

⚠️ **Authorization**: These images are authorized for training purposes only per Kaggle competition terms.

⚠️ **Ground Truth**: Test set ground truth is NOT available for download - it's used by the scoring service (bounty.autohdr.com).

## Troubleshooting

**Issue**: Kaggle API not authenticated
**Solution**:
```bash
# Go to https://www.kaggle.com/[username]/account
# Create new API token
# Download kaggle.json
# Move to ~/.kaggle/ (or C:\Users\<username>\.kaggle\)
```

**Issue**: Insufficient disk space
**Solution**: Free up space or use external drive for data directory

**Issue**: Corrupted downloads
**Solution**: Delete and re-download, verify checksums if available
