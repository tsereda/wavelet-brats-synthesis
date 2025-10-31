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

# ✅ FIXED: Set environment variables correctly
export REPO_PATH="$(pwd)"
# ✅ CRITICAL FIX: Add the app directory to PYTHONPATH so guided_diffusion can be imported
export PYTHONPATH="$(pwd):$(pwd)/app:${PYTHONPATH}"
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

# ✅ FIXED: Test the import to make sure it works
echo "[3.5/7] Testing Python imports..."
python -c "
import sys
print('Python path:')
for p in sys.path:
    print(f'  {p}')

try:
    from guided_diffusion import dist_util, logger
    print('✓ guided_diffusion import successful')
except ImportError as e:
    print(f'✗ guided_diffusion import failed: {e}')
    sys.exit(1)
"

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
    
    # Export RESUME_RUN so Python can access it
    export RESUME_RUN
    export CHECKPOINT_DIR
    
    # Download checkpoints from W&B and capture modality
    echo "Downloading checkpoints from W&B..."
    RESUME_MODALITY=$(python3 << 'EOF'
import wandb
import os
import sys

run_path = os.environ.get("RESUME_RUN")
checkpoint_dir = os.environ.get("CHECKPOINT_DIR")

# Add validation check
if not run_path:
    print(f"❌ RESUME_RUN environment variable is not set!", file=sys.stderr)
    sys.exit(1)

try:
    # Initialize API
    api = wandb.Api()
    
    # Get the run
    print(f"Fetching run: {run_path}", file=sys.stderr)
    run = api.run(run_path)
    
    # Extract modality from run config
    modality = run.config.get('contr', None)
    if modality:
        print(f"📋 Detected modality from run config: {modality}", file=sys.stderr)
        # Print modality to stdout for capture
        print(modality)
    else:
        print("⚠️  Could not detect modality from run config", file=sys.stderr)
    
    # Download all checkpoint files
    checkpoint_files = [f for f in run.files() if f.name.endswith('.pt')]
    
    if not checkpoint_files:
        print("❌ No checkpoint files found in this run!", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(checkpoint_files)} checkpoint files", file=sys.stderr)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for file in checkpoint_files:
        print(f"   Downloading: {file.name}", file=sys.stderr)
        file.download(root=checkpoint_dir, replace=True)
    
    print(f"✅ Downloaded {len(checkpoint_files)} checkpoints to {checkpoint_dir}", file=sys.stderr)
    
except Exception as e:
    print(f"❌ Error downloading checkpoints: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)
EOF
)
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to download checkpoints from W&B"
        exit 1
    fi
    
    # Export the captured modality
    if [ -n "$RESUME_MODALITY" ]; then
        export RESUME_MODALITY
        echo "✅ Will resume training for modality: $RESUME_MODALITY"
    fi
    
# elif [ -f "/data/400kCheckpoints.zip" ]; then
#     # Check if checkpoint archive exists (fallback)
#     echo "Found checkpoint archive at /data/400kCheckpoints.zip"
#     if [ ! -d "$CHECKPOINT_DIR" ] || [ -z "$(ls -A $CHECKPOINT_DIR/*.pt 2>/dev/null)" ]; then
#         echo "Extracting checkpoints..."
#         unzip -o /data/400kCheckpoints.zip -d ./
        
#         # Copy checkpoints to the checkpoint directory
#         for modality in t1n t1c t2w t2f; do
#             if [ -d "${modality}" ]; then
#                 echo "Copying ${modality} checkpoints..."
#                 cp ${modality}/brats_*.pt "$CHECKPOINT_DIR/" 2>/dev/null || true
#             fi
#         done
        
#         echo "✓ Checkpoints extracted to $CHECKPOINT_DIR"
#     else
#         echo "✓ Checkpoints already exist in $CHECKPOINT_DIR"
#     fi
fi

if [ -n "$RESUME_RUN" ]; then
    # Extract run ID from path (format: entity/project/run_id)
    RESUME_RUN_ID=$(echo "$RESUME_RUN" | rev | cut -d'/' -f1 | rev)
    
    echo "🔄 Setting W&B to resume run: $RESUME_RUN_ID"
    export WANDB_RUN_ID="$RESUME_RUN_ID"
    export WANDB_RESUME="allow"
fi

# ALWAYS detect latest checkpoint after any download/extraction operation
echo "Searching for the latest local checkpoint in $CHECKPOINT_DIR..."

# Find all MODEL checkpoint .pt files (exclude optimizer checkpoints starting with "opt_")
# and extract the step number (largest number in filename)
LATEST_STEP=$(find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name "*.pt" ! -name "opt_*" | \
                    while read f; do 
                        basename "$f" | grep -oP '\d+' | sort -rn | head -1
                    done 2>/dev/null | sort -rn | head -1)

FULL_FILENAME=""
if [ -n "$LATEST_STEP" ]; then
    # Find the full filename containing the largest step number (exclude optimizer checkpoints)
    FULL_FILENAME=$(find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name "*.pt" ! -name "opt_*" | \
                    grep -P "_${LATEST_STEP}\.pt$" | head -1)
    
    # Fallback if the above doesn't match
    if [ -z "$FULL_FILENAME" ]; then
        FULL_FILENAME=$(find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name "*${LATEST_STEP}*.pt" ! -name "opt_*" | head -1)
    fi
fi

if [ -n "$FULL_FILENAME" ]; then
    echo "✅ Found latest checkpoint: $(basename "$FULL_FILENAME") (Step: $LATEST_STEP)"
    export LATEST_CHECKPOINT_FILE="$FULL_FILENAME"
    export LATEST_CHECKPOINT_STEP="$LATEST_STEP"
    echo "Exported LATEST_CHECKPOINT_FILE and LATEST_CHECKPOINT_STEP."
else
    echo "⚠️ No checkpoint files found in $CHECKPOINT_DIR. Starting from scratch."
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

# Handle resume vs sweep logic
if [ -n "$RESUME_RUN" ]; then
    echo ""
    echo "🔄 RESUME MODE DETECTED"
    echo "Resuming W&B run: $RESUME_RUN"
    
    if [ -n "$LATEST_CHECKPOINT_FILE" ]; then
        echo "Using checkpoint: $LATEST_CHECKPOINT_FILE (Step: $LATEST_CHECKPOINT_STEP)"
    fi
    
    if [ -n "$RESUME_MODALITY" ]; then
        echo "Training modality: $RESUME_MODALITY"
    fi
    
    echo ""
    echo "Running DIRECT training (bypassing sweep agent)..."
    echo "========================================="
    
    # Use the detected modality or default to t2w
    TRAINING_MODALITY=${RESUME_MODALITY:-"t2w"}
    
    # ✅ CRITICAL FIX: Change directory to the project root and use proper path
    cd "$SCRIPT_DIR"
    
    # ✅ FIXED: Run training directly with PYTHONPATH properly set
    exec python -m app.scripts.train \
        --data_dir=./datasets/BRATS2023/training \
        --contr="$TRAINING_MODALITY" \
        --lr=1e-5 \
        --diffusion_steps=100 \
        --sample_schedule=sampled \
        --batch_size=2 \
        --num_workers=12 \
        --save_interval=5000 \
        --log_interval=100 \
        --special_checkpoint_steps="200000,400000,600000" \
        --resume_checkpoint="$LATEST_CHECKPOINT_FILE" \
        --resume_step="$LATEST_CHECKPOINT_STEP" \
        --wavelet=db2 \
        --save_to_wandb=true
fi

# Original sweep logic (only runs if no RESUME_RUN)
if [ -z "$SWEEP_ID" ]; then
    echo ""
    echo "WARNING: SWEEP_ID environment variable not set"
    echo "You have three options:"
    echo ""
    echo "1. Run with W&B sweep:"
    echo "   export SWEEP_ID='your-entity/your-project/sweep-id'"
    echo "   ./run.sh"
    echo ""
    echo "2. Run normal training:"
    if [ -n "$LATEST_CHECKPOINT_FILE" ]; then
        echo "   python -m app.scripts.train --data_dir=./datasets/BRATS2023/training --contr=t1n --lr=1e-5 \\"
        echo "     --resume_checkpoint=\"\$LATEST_CHECKPOINT_FILE\" --resume_step=\"\$LATEST_CHECKPOINT_STEP\""
    else
        echo "   python -m app.scripts.train --data_dir=./datasets/BRATS2023/training --contr=t1n --lr=1e-5"
    fi
    echo ""
    echo "3. Resume from W&B run:"
    echo "   ./run.sh --resume_run entity/project/run_id"
    echo ""
    exit 1
fi

# Rest of the sweep logic stays the same...
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

if [ -n "$LATEST_CHECKPOINT_FILE" ]; then
    echo "Local resume candidate: ${LATEST_CHECKPOINT_FILE} (Step: ${LATEST_CHECKPOINT_STEP})"
    echo "The W&B agent's runs will automatically use these local checkpoints if configured to resume."
fi

echo "========================================="
echo ""

wandb agent "$SWEEP_ID"