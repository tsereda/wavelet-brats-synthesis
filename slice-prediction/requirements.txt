# Core dependencies
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
opencv-python>=4.8.0

# Medical imaging
monai[nibabel,tqdm]>=1.3.0
nibabel>=5.0.0

# Wavelets
PyWavelets>=1.4.0

# Metrics and visualization
scikit-image>=0.21.0
matplotlib>=3.7.0
Pillow>=10.0.0

# Experiment tracking
wandb>=0.15.0

# Training utilities
einops>=0.7.0
tqdm>=4.65.0

# Optional (for faster data loading)
# opencv-python-headless>=4.8.0