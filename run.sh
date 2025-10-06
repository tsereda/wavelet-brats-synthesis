#!/bin/bash
set -e  # Exit on error

echo "========================================="
echo "Wavelet BRATS Synthesis - W&B Sweep Agent"
echo "========================================="

# Determine working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Working directory: $(pwd)"

# Install system dependencies
echo "[1/7] Installing system dependencies..."
apt-get update && apt-get install -y \
    p7zip-full \
    wget \
    git \
    unzip \
    || echo "System packages already installed"

# Set environment variables
export REPO_PATH="$(pwd)"
export PYTHONPATH="$(pwd)"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=UTF-8

echo "REPO_PATH: $REPO_PATH"
echo "PYTHONPATH: $PYTHONPATH"

# Install Python dependencies
echo "[2/7] Installing Python dependencies..."
pip install --no-cache-dir \
    pyyaml \
    torch \
    torchvision \
    tqdm \
    numpy \
    nibabel \
    wandb \
    matplotlib \
    blobfile \
    tensorboard \
    monai \
    pillow

# Verify critical files exist
echo "[3/7] Verifying repository structure..."
if [ ! -f "app/scripts/train.py" ]; then
    echo "ERROR: app/scripts/train.py not found!"
    echo "Current directory: $(pwd)"
    echo "Contents:"
    ls -la
    echo "Scripts directory:"
    ls -la app/scripts/ 2>/dev/null || echo "app/scripts/ directory not found!"
    exit 1
fi
echo "✓ app/scripts/train.py found"

# Setup directories
echo "[4/7] Setting up directories..."
mkdir -p ./datasets/BRATS2023/training
mkdir -p ./datasets/BRATS2023/validation
mkdir -p ./checkpoints
mkdir -p ./logs
mkdir -p ./wandb

# Prepare data
echo "[5/7] Preparing data..."
if [ ! -d "datasets/BRATS2023/training" ] || [ -z "$(ls -A datasets/BRATS2023/training)" ]; then
    echo "Extracting BRATS training data..."
    
    if [ -f "/data/ASNR-MICCAI-BraTS2023-GLI-MET-TrainingData.tar.gz" ]; then
        echo "Found training data archive, extracting..."
        7z x /data/ASNR-MICCAI-BraTS2023-GLI-MET-TrainingData.tar.gz -o.
        7z x ASNR-MICCAI-BraTS2023-GLI-MET-TrainingData.tar -o.
        mv ASNR-MICCAI-BraTS2023-GLI-MET-TrainingData/* datasets/BRATS2023/training/
        rm -f ASNR-MICCAI-BraTS2023-GLI-MET-TrainingData.tar*
    else
        echo "WARNING: Training data not found at /data/ASNR-MICCAI-BraTS2023-GLI-MET-TrainingData.tar.gz"
    fi
    
    if [ -f "/data/ASNR-MICCAI-BraTS2023-GLI-MET-ValidationData.tar.gz" ]; then
        echo "Found validation data archive, extracting..."
        7z x /data/ASNR-MICCAI-BraTS2023-GLI-MET-ValidationData.tar.gz -o.
        7z x ASNR-MICCAI-BraTS2023-GLI-MET-ValidationData.tar -o.
        mv ASNR-MICCAI-BraTS2023-GLI-MET-ValidationData/* datasets/BRATS2023/validation/
        rm -f ASNR-MICCAI-BraTS2023-GLI-MET-ValidationData.tar*
    else
        echo "WARNING: Validation data not found at /data/ASNR-MICCAI-BraTS2023-GLI-MET-ValidationData.tar.gz"
    fi
    
    # Clean hidden files
    find datasets/BRATS2023/training -name ".*" -delete 2>/dev/null || true
    find datasets/BRATS2023/validation -name ".*" -delete 2>/dev/null || true
    
    # Remove problematic patient if exists
    rm -rf datasets/BRATS2023/training/BraTS-MET-00232-000/ 2>/dev/null || true
else
    echo "BRATS data already prepared"
fi

# Setup checkpoints if available
if [ -f "/data/BraTSChekpoints.zip" ]; then
    echo "Extracting checkpoints..."
    unzip -o /data/BraTSChekpoints.zip -d ./
fi

echo "Training patients: $(ls datasets/BRATS2023/training/ 2>/dev/null | wc -l)"
echo "Validation patients: $(ls datasets/BRATS2023/validation/ 2>/dev/null | wc -l)"

# Verify environment
echo "[6/7] Verifying environment..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
    python -c "import torch; print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
fi
python -c "import wandb; print(f'W&B version: {wandb.__version__}')"

# Verify W&B authentication
echo "[7/7] Verifying W&B authentication..."
if [ -z "$WANDB_API_KEY" ]; then
    echo "WARNING: WANDB_API_KEY not set, attempting interactive login..."
    python -c "import wandb; wandb.login()"
else
    echo "✓ WANDB_API_KEY is set"
    python -c "import wandb; wandb.login()"
fi

echo "========================================="
echo "Setup complete!"
echo "========================================="

# Check for required environment variables
if [ -z "$SWEEP_ID" ]; then
    echo ""
    echo "WARNING: SWEEP_ID environment variable not set"
    echo "You have two options:"
    echo ""
    echo "1. Run with W&B sweep:"
    echo "   export SWEEP_ID='your-entity/your-project/sweep-id'"
    echo "   ./run.sh"
    echo ""
    echo "2. Run normal training:"
    echo "   python app/scripts/train.py --data_dir=./datasets/BRATS2023/training --contr=t1n --lr=1e-5"
    echo ""
    exit 1
fi

if [ -z "$WANDB_ENTITY" ]; then
    echo "WARNING: WANDB_ENTITY not set, using default from sweep config"
fi

if [ -z "$WANDB_PROJECT" ]; then
    echo "WARNING: WANDB_PROJECT not set, using default from sweep config"
fi

echo ""
echo "Starting W&B Sweep Agent..."
echo "Sweep ID: $SWEEP_ID"
echo "Entity: ${WANDB_ENTITY:-'(from sweep config)'}"
echo "Project: ${WANDB_PROJECT:-'(from sweep config)'}"
echo "========================================="
echo ""

# Launch W&B sweep agent
wandb agent "$SWEEP_ID"