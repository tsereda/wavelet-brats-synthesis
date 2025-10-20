#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

echo "========================================="
echo "Wavelet BRATS Synthesis - W&B Sweep Agent"
echo "========================================="

# Parse command line arguments
RESUME_RUN=""
CHECKPOINT_DIR="./checkpoints"

while [[ $# -gt 0 ]]; do
  case $1 in
    --resume_run)
      RESUME_RUN="$2"
      shift 2
      ;;
    --checkpoint_dir)
      CHECKPOINT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      shift
      ;;
  esac
done

# Determine working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Working directory: $(pwd)"
echo "Checkpoint directory: $CHECKPOINT_DIR"

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
# New environment variables for auto-resume
export LATEST_CHECKPOINT_FILE=""
export LATEST_CHECKPOINT_STEP=""

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
mkdir -p "$CHECKPOINT_DIR"
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

# Setup checkpoints
echo "[5.5/7] Setting up checkpoints..."

# Check if we need to resume from W&B
if [ -n "$RESUME_RUN" ]; then
    echo "🔄 Resuming from W&B run: $RESUME_RUN"
    
    # Download checkpoints from W&B run
    echo "Downloading checkpoints from W&B..."
    python3 << EOF
import wandb
import os
import sys

run_path = "$RESUME_RUN"
checkpoint_dir = "$CHECKPOINT_DIR"

try:
    # Initialize API
    api = wandb.Api()
    
    # Get the run
    print(f"Fetching run: {run_path}")
    run = api.run(run_path)
    
    # Download all checkpoint files
    checkpoint_files = [f for f in run.files() if f.name.endswith('.pt')]
    
    if not checkpoint_files:
        print("❌ No checkpoint files found in this run!")
        sys.exit(1)
    
    print(f"Found {len(checkpoint_files)} checkpoint files")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for file in checkpoint_files:
        print(f"   Downloading: {file.name}")
        file.download(root=checkpoint_dir, replace=True)
    
    print(f"✅ Downloaded {len(checkpoint_files)} checkpoints to {checkpoint_dir}")
    
except Exception as e:
    print(f"❌ Error downloading checkpoints: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to download checkpoints from W&B"
        exit 1
    fi
    
else
    # Check if checkpoint archive exists (fallback)
    if [ -f "/data/400kCheckpoints.zip" ]; then
        echo "Found checkpoint archive at /data/400kCheckpoints.zip"
        if [ ! -d "$CHECKPOINT_DIR" ] || [ -z "$(ls -A $CHECKPOINT_DIR/*.pt 2>/dev/null)" ]; then
            echo "Extracting checkpoints..."
            unzip -o /data/400kCheckpoints.zip -d ./
            
            # Copy checkpoints to the checkpoint directory
            for modality in t1n t1c t2w t2f; do
                if [ -d "${modality}" ]; then
                    echo "Copying ${modality} checkpoints..."
                    cp ${modality}/brats_*.pt "$CHECKPOINT_DIR/" 2>/dev/null || true
                fi
            done
            
            echo "✓ Checkpoints extracted to $CHECKPOINT_DIR"
        else
            echo "✓ Checkpoints already exist in $CHECKPOINT_DIR"
        fi

    else # <--- START OF NEW LOGIC: Find latest local checkpoint
        echo "Searching for the latest local checkpoint in $CHECKPOINT_DIR..."
        
        # Find all .pt files, strip their path, look for the step number (first large sequence of digits), 
        # and sort numerically to find the largest one.
        LATEST_STEP=$(find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name "*.pt" | \
                            while read f; do 
                                basename "$f" | grep -oP '\d+' | head -1; 
                            done 2>/dev/null | sort -rn | head -1)

        FULL_FILENAME=""
        if [ -n "$LATEST_STEP" ]; then
            # Find the full filename corresponding to the largest step number
            FULL_FILENAME=$(find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name "*${LATEST_STEP}*.pt" | head -1)
        fi

        if [ -n "$FULL_FILENAME" ]; then
            echo "✅ Found latest checkpoint: $(basename "$FULL_FILENAME") (Step: $LATEST_STEP)"
            # Export the checkpoint file and step as environment variables for the final training script
            export LATEST_CHECKPOINT_FILE="$FULL_FILENAME"
            export LATEST_CHECKPOINT_STEP="$LATEST_STEP"
            echo "Exported LATEST_CHECKPOINT_FILE and LATEST_CHECKPOINT_STEP."
        else
            echo "⚠️ No checkpoint files found in $CHECKPOINT_DIR. Starting from scratch."
        fi
    fi # <--- END OF NEW LOGIC
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
    # Modified to show how to use the exported variables for normal run
    if [ -n "$LATEST_CHECKPOINT_FILE" ]; then
        echo "   python app/scripts/train.py --data_dir=./datasets/BRATS2023/training --contr=t1n --lr=1e-5 \\"
        echo "     --resume_checkpoint=\"\$LATEST_CHECKPOINT_FILE\" --resume_step=\"\$LATEST_CHECKPOINT_STEP\""
    else
        echo "   python app/scripts/train.py --data_dir=./datasets/BRATS2023/training --contr=t1n --lr=1e-5"
    fi
    echo ""
    echo "3. Resume from W&B run:"
    echo "   ./run.sh --resume_run entity/project/run_id"
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

if [ -n "$RESUME_RUN" ]; then
    echo "Resumed from W&B run: $RESUME_RUN"
fi

# Inform the user about local checkpoint status
if [ -n "$LATEST_CHECKPOINT_FILE" ]; then
    echo "Local resume candidate: ${LATEST_CHECKPOINT_FILE} (Step: ${LATEST_CHECKPOINT_STEP})"
    echo "The W&B agent's runs will automatically use these local checkpoints if configured to resume."
fi

echo "========================================="
echo ""

# The W&B agent will execute the training command defined in your sweep YAML.
# To utilize the LATEST_CHECKPOINT_FILE and LATEST_CHECKPOINT_STEP, your 
# sweep config's command must refer to these environment variables.
wandb agent "$SWEEP_ID"