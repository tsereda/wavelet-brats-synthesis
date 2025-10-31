# Brain Tumor 3D Segmentation with MONAI

This repository contains Python scripts converted from the original Jupyter notebook for brain tumor 3D segmentation using MONAI (Medical Open Network for AI).

## Overview

The project implements a 3D segmentation pipeline for brain tumors using the BraTS (Brain Tumor Segmentation) dataset from the Medical Decathlon. The model segments three tumor sub-regions:

- **TC (Tumor Core)**: Necrotic and non-enhancing tumor core + enhancing tumor
- **WT (Whole Tumor)**: All tumor regions including edema
- **ET (Enhancing Tumor)**: Enhancing tumor structures

## Dataset

- **Source**: BraTS 2016 and 2017 datasets via Medical Decathlon
- **Modalities**: Multimodal MRI (FLAIR, T1w, T1gd, T2w)
- **Size**: 750 4D volumes (484 Training + 266 Testing)
- **Target**: Gliomas segmentation of necrotic/active tumor and edema

## Files Description

### Core Scripts

1. **`brats_segmentation_3d.py`** - Main training and evaluation script
   - Complete training pipeline
   - Model definition and training
   - Evaluation on original image spacing
   - Model conversion to ONNX format

2. **`onnx_inference.py`** - ONNX model inference and comparison
   - ONNX model loading and inference
   - Performance comparison between PyTorch and ONNX models
   - Visualization of model outputs

3. **`visualization_utils.py`** - Data visualization and analysis utilities
   - Dataset statistics analysis
   - Multi-modal image visualization
   - Ground truth label visualization
   - Multiple sample comparison

### Configuration Files

4. **`requirements.txt`** - Python dependencies
5. **`README.md`** - This documentation file

## Installation

1. Clone or download the repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Basic Training and Evaluation

Run the main training script:

```bash
python brats_segmentation_3d.py
```

This will:
- Download the BraTS dataset (if not already present)
- Train the SegResNet model for 300 epochs
- Evaluate the model on validation data
- Save the best model as both PyTorch (.pth) and ONNX (.onnx) formats

### 2. ONNX Inference

After training, run ONNX inference:

```bash
python onnx_inference.py
```

This will:
- Load the trained ONNX model
- Run inference on validation data
- Compare ONNX and PyTorch model performance
- Visualize model outputs

### 3. Data Visualization

Explore the dataset:

```bash
python visualization_utils.py
```

This will:
- Display dataset statistics
- Visualize multi-modal MRI images
- Show ground truth segmentation labels
- Compare multiple samples

## Configuration

### Environment Variables

- `MONAI_DATA_DIRECTORY`: Set this to specify where to store/cache the dataset
  ```bash
  export MONAI_DATA_DIRECTORY=/path/to/your/data
  ```

### Key Parameters

You can modify these parameters in the scripts:

- `max_epochs`: Number of training epochs (default: 300)
- `val_interval`: Validation frequency (default: every 2 epochs)
- `batch_size`: Training batch size (default: 1)
- `learning_rate`: Initial learning rate (default: 1e-4)

## Model Architecture

- **Network**: SegResNet (3D Segmentation Residual Network)
- **Input**: 4-channel 3D volumes (FLAIR, T1w, T1gd, T2w)
- **Output**: 3-channel probability maps (TC, WT, ET)
- **Loss Function**: Dice Loss
- **Optimizer**: Adam with Cosine Annealing LR scheduler

## Features

- **Multi-modal input**: Handles 4 MRI modalities simultaneously
- **Data augmentation**: Random flipping, intensity scaling, and shifting
- **Mixed precision training**: Automatic Mixed Precision (AMP) for faster training
- **Deterministic training**: Reproducible results with fixed random seeds
- **Sliding window inference**: Handles large 3D volumes efficiently
- **ONNX export**: Model conversion for deployment

## Hardware Requirements

- **GPU**: CUDA-compatible GPU recommended (model trains on CPU but very slowly)
- **Memory**: At least 8GB GPU memory for training
- **Storage**: ~50GB for dataset download and caching

## Expected Results

The model typically achieves:
- **Overall Dice Score**: ~0.79
- **Tumor Core (TC)**: ~0.84
- **Whole Tumor (WT)**: ~0.91
- **Enhancing Tumor (ET)**: ~0.62

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use smaller ROI size
2. **Dataset download fails**: Check internet connection and disk space
3. **ONNX warnings**: `pthread_setaffinity_np failed` warnings are normal and don't affect results

### Performance Tips

1. Use SSD storage for faster data loading
2. Increase `num_workers` in DataLoader for faster data preprocessing
3. Use `cache_rate > 0` to cache preprocessed data in memory

## License

Copyright (c) MONAI Consortium  
Licensed under the Apache License, Version 2.0

## References

- [MONAI Documentation](https://docs.monai.io/)
- [Medical Decathlon Dataset](http://medicaldecathlon.com/)
- [BraTS Challenge](https://www.med.upenn.edu/cbica/brats2020/)
- [SegResNet Paper](https://arxiv.org/abs/1810.11654)
