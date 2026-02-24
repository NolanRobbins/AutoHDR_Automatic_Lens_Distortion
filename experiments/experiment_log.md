# Experiment Log

Track all experiments here for reproducibility and learning.

## Template

```markdown
### Experiment [ID]: [Short Description]
**Date**: YYYY-MM-DD
**Goal**: [What are you trying to achieve/test?]
**Approach**: [Method, architecture, loss function, etc.]
**Configuration**:
- Model: [Architecture details]
- Loss: [Loss function and weights]
- Batch size: [X]
- Epochs: [Y]
- Learning rate: [Z]
- Other hyperparams: [...]

**Results**:
- Bounty Score: [X]/100
- PSNR: [Y] dB
- SSIM: [Z]
- Training time: [T] hours

**Observations**:
- [Key finding 1]
- [Key finding 2]
- [...]

**Next Steps**:
- [What to try next based on results]
```

---

## Experiments

### Experiment 001: Classical CV Baseline
**Date**: [TBD]
**Goal**: Establish baseline performance using traditional computer vision methods
**Approach**: OpenCV lens calibration with Hough line detection

**Configuration**:
- Method: Brown-Conrady distortion model
- Parameters: k1, k2, k3, p1, p2
- Optimization: Least squares to straighten detected lines

**Results**:
- Bounty Score: [TBD]/100
- Time: ~1 minute per image

**Observations**:
- [TBD after implementation]

**Next Steps**:
- Move to deep learning baseline

---

### Experiment 002: U-Net Baseline
**Date**: [TBD]
**Goal**: Establish deep learning baseline
**Approach**: Simple U-Net with L1 + SSIM loss

**Configuration**:
- Model: EfficientNet-B3 encoder + U-Net decoder
- Loss: L1 (w=1.0) + SSIM (w=1.0)
- Batch size: 8 (effective 16)
- Epochs: 50
- Learning rate: 1e-3
- Optimizer: AdamW (weight_decay=1e-4)
- Scheduler: Cosine annealing

**Results**:
- Bounty Score: [TBD]/100
- PSNR: [TBD] dB
- SSIM: [TBD]
- Training time: [TBD] hours on RTX 5060 Ti

**Observations**:
- [TBD]

**Next Steps**:
- Add geometric loss components

---

### Experiment 003: Geometry-Aware Loss
**Date**: [TBD]
**Goal**: Improve performance by adding edge and gradient losses
**Approach**: Multi-component geometric loss function

**Configuration**:
- Model: Same as Exp 002
- Loss: L1 (1.0) + Edge (2.0) + SSIM (1.0) + Gradient (1.5)
- Other params: Same as Exp 002

**Results**:
- Bounty Score: [TBD]/100
- Delta from baseline: [+X points]

**Observations**:
- [TBD]

**Next Steps**:
- Ablation study on loss components

---

### Experiment 004: Loss Ablation Study
**Date**: [TBD]
**Goal**: Determine contribution of each loss component
**Approach**: Systematically add loss components

**Variants**:
1. L1 only
2. L1 + Edge
3. L1 + Edge + SSIM
4. L1 + Edge + SSIM + Gradient
5. Full (all components)

**Results**:
| Variant | Bounty Score | PSNR | SSIM | Delta |
|---------|--------------|------|------|-------|
| 1. L1 only | [TBD] | [TBD] | [TBD] | - |
| 2. + Edge | [TBD] | [TBD] | [TBD] | [+X] |
| 3. + SSIM | [TBD] | [TBD] | [TBD] | [+Y] |
| 4. + Gradient | [TBD] | [TBD] | [TBD] | [+Z] |

**Observations**:
- Most impactful component: [TBD]
- [Other findings]

---

### Experiment 005: Advanced Architecture
**Date**: [TBD]
**Goal**: Improve model capacity
**Approach**: Larger encoder + attention mechanisms

**Configuration**:
- Model: EfficientNet-B4 + CBAM attention
- Loss: Best from Exp 004
- Batch size: 6 (effective 18)
- Epochs: 70
- Multi-stage training (warmup, full, fine-tune)

**Results**:
- Bounty Score: [TBD]/100
- Delta from Exp 004: [+X points]

**Observations**:
- [TBD]

---

### Experiment 006: Test-Time Augmentation
**Date**: [TBD]
**Goal**: Boost inference performance
**Approach**: Average predictions over multiple augmentations

**Configuration**:
- Base model: Best from Exp 005
- TTA transforms: None, HFlip, VFlip
- Ensemble method: Simple average

**Results**:
- Bounty Score: [TBD]/100
- Delta: [+X points]
- Inference time: [Xt slower]

**Observations**:
- [TBD]

---

## Summary Table

| Exp | Description | Bounty Score | PSNR | SSIM | Notes |
|-----|-------------|--------------|------|------|-------|
| 001 | Classical CV | [TBD] | - | - | Baseline |
| 002 | U-Net Baseline | [TBD] | [TBD] | [TBD] | First DL |
| 003 | Geometric Loss | [TBD] | [TBD] | [TBD] | Big improvement |
| 004 | Ablation | [TBD] | [TBD] | [TBD] | Analysis |
| 005 | Advanced Arch | [TBD] | [TBD] | [TBD] | Best single model |
| 006 | TTA | [TBD] | [TBD] | [TBD] | Final boost |

---

## Insights & Learnings

### What Worked Well
1. [TBD]
2. [TBD]

### What Didn't Work
1. [TBD]
2. [TBD]

### Future Directions
1. [TBD]
2. [TBD]
