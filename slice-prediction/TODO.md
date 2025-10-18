# TODO: Middleslice Prediction Experiments

## Current Status
- [x] Swin-UNETR baseline working
- [x] Dataset loader working (256×256 slices)
- [x] Training loop with W&B logging
- [x] Qualitative visualization working

## Immediate Tasks (This Week)

### 1. Add Wavelet Diffusion Models
- [ ] Create `models/` directory
- [ ] Implement `models/wavelet_diffusion_haar.py`
- [ ] Implement `models/wavelet_diffusion_db2.py`
- [ ] Test wavelet transform (2D DWT) on sample slice
- [ ] Integrate with training script

### 2. Train Core Models
- [ ] Swin-UNETR baseline (already working, just save best checkpoint)
- [ ] Fast-cWDM-Haar (~3-4 hours)
- [ ] Fast-cWDM-db2 (~3-4 hours)

### 3. Evaluation Scripts
- [ ] Create `evaluate.py` to calculate MSE and SSIM
- [ ] Run evaluation on all model predictions
- [ ] Save metrics to CSV files

### 4. Generate Paper Figure
- [ ] Create `generate_figure2.py`
- [ ] Select 1-2 best examples (clear tumor, low MSE)
- [ ] Generate comparison: [Z-1][Z+1][Swin][Haar][db2][GT][Error]
- [ ] Export as high-res PDF

## Later (For Final Paper)

### Additional Baselines
- [ ] Fast-DDPM (no wavelets) - optional comparison
- [ ] Direct L2 regression baseline - quick sanity check

### Downstream Segmentation
- [ ] Get/train Swin-UNETR segmentation model
- [ ] Run segmentation on reconstructed volumes
- [ ] Calculate Dice scores (WT, TC, ET)

### Efficiency Metrics
- [ ] Measure inference time (average over 100 samples)
- [ ] Measure GPU memory usage during inference
- [ ] Create timing comparison table

### Paper Writing
- [ ] Write Methods section 2.3 (2D slice reconstruction)
- [ ] Write Results section with tables
- [ ] Generate Figure 1 (method diagram)
- [ ] Update abstract with final numbers

## File Organization

```
middleslice-prediction/
├── train.py              ✓ Working
├── transforms.py         ✓ Working
├── logging_utils.py      ✓ Working
├── evaluate.py           ⚠️ Need to create
├── generate_figure2.py   ⚠️ Need to create
├── models/
│   ├── __init__.py
│   ├── wavelet_diffusion_haar.py  ⚠️ Need to create
│   └── wavelet_diffusion_db2.py   ⚠️ Need to create
├── results/
│   ├── swin/
│   │   ├── predictions/
│   │   └── metrics.csv
│   ├── haar/
│   │   ├── predictions/
│   │   └── metrics.csv
│   └── db2/
│       ├── predictions/
│       └── metrics.csv
└── checkpoints/
    ├── swin_best.pth
    ├── haar_best.pth
    └── db2_best.pth
```

## Expected Results

### Target Metrics (Rough Estimates)
- **Swin-UNETR:** MSE ~2.5-3.0, SSIM ~0.93-0.94
- **Fast-cWDM-Haar:** MSE ~2.2-2.5, SSIM ~0.94-0.95
- **Fast-cWDM-db2:** MSE ~2.0-2.3, SSIM ~0.94-0.96 (should be best)

### Timeline
- **Week 1:** Implement wavelet models, train all 3 models
- **Week 2:** Evaluation, figure generation, start writing
- **Week 3:** Add baselines if needed, polish paper

## Notes
- Keep image size at 256×256 for all models (fair comparison)
- Use MSE as primary metric (already implemented in loss)
- SSIM can be calculated post-hoc from predictions
- Focus on getting Haar and db2 working first, Swin is baseline