#!/bin/bash
# Checkpoint Management Helper for BraTS Training

set -e

CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"

show_usage() {
    cat << EOF
Checkpoint Management Helper for BraTS Training

Usage: $0 <command> [options]

Commands:
    list [modality]           List available checkpoints (optionally filter by modality)
    latest [modality]         Show the latest checkpoint (optionally for specific modality)
    info <checkpoint_file>    Show detailed information about a checkpoint
    resume <checkpoint_file>  Get resume command with proper arguments
    auto-resume [modality]    Auto-detect and show resume command for latest checkpoint
    clean-old <keep_n>        Keep only the N most recent checkpoints per modality
    verify                    Verify checkpoint directory structure

Options:
    --checkpoint-dir <path>   Override default checkpoint directory ($CHECKPOINT_DIR)

Examples:
    # List all checkpoints
    $0 list

    # List checkpoints for t1n modality
    $0 list t1n

    # Get resume command for latest checkpoint
    $0 auto-resume

    # Get resume command for latest t1c checkpoint
    $0 auto-resume t1c

    # Show info about a specific checkpoint
    $0 info checkpoints/brats_t1n_123456_sampled_100.pt

    # Clean up, keeping only 3 most recent checkpoints per modality
    $0 clean-old 3

EOF
}

list_checkpoints() {
    local modality=$1
    
    echo "📋 Checkpoints in $CHECKPOINT_DIR:"
    echo ""
    
    if [ -n "$modality" ]; then
        pattern="brats_${modality}_*.pt"
    else
        pattern="brats_*.pt"
    fi
    
    if ! ls "$CHECKPOINT_DIR"/$pattern 2>/dev/null; then
        echo "  No checkpoints found"
        return 1
    fi
    
    # List with details
    ls -lht "$CHECKPOINT_DIR"/$pattern | while read -r line; do
        filename=$(echo "$line" | awk '{print $NF}')
        size=$(echo "$line" | awk '{print $5}')
        date=$(echo "$line" | awk '{print $6, $7, $8}')
        
        basename=$(basename "$filename")
        
        # Extract info from filename
        if [[ $basename =~ brats_([a-z0-9]+)_([0-9]+)_([a-z]+)_([0-9]+)\.pt ]]; then
            mod="${BASH_REMATCH[1]}"
            step="${BASH_REMATCH[2]}"
            schedule="${BASH_REMATCH[3]}"
            timesteps="${BASH_REMATCH[4]}"
            
            echo "  📁 $basename"
            echo "     Modality: $mod | Step: $step | Schedule: $schedule | Timesteps: $timesteps"
            echo "     Size: $size | Date: $date"
            echo ""
        else
            echo "  📁 $basename (size: $size, date: $date)"
            echo ""
        fi
    done
}

show_latest() {
    local modality=$1
    
    if [ -n "$modality" ]; then
        pattern="brats_${modality}_*.pt"
        echo "🔍 Latest checkpoint for $modality:"
    else
        pattern="brats_*.pt"
        echo "🔍 Latest checkpoint:"
    fi
    
    latest=$(ls -t "$CHECKPOINT_DIR"/$pattern 2>/dev/null | head -1)
    
    if [ -z "$latest" ]; then
        echo "  ❌ No checkpoints found"
        return 1
    fi
    
    echo "  $latest"
    
    # Show file details
    ls -lh "$latest"
    
    # Extract step
    filename=$(basename "$latest")
    if [[ $filename =~ brats_[a-z0-9]+_([0-9]+)_ ]]; then
        step="${BASH_REMATCH[1]}"
        echo ""
        echo "  Resume step: $step"
        
        # Check for optimizer checkpoint
        opt_file="$CHECKPOINT_DIR/opt${step}.pt"
        if [ -f "$opt_file" ]; then
            echo "  ✓ Optimizer checkpoint: $opt_file"
        else
            echo "  ⚠️  Optimizer checkpoint not found: $opt_file"
        fi
    fi
}

show_info() {
    local checkpoint=$1
    
    if [ ! -f "$checkpoint" ]; then
        echo "❌ Checkpoint not found: $checkpoint"
        return 1
    fi
    
    echo "📊 Checkpoint Information:"
    echo "  File: $checkpoint"
    echo "  Size: $(ls -lh "$checkpoint" | awk '{print $5}')"
    echo "  Modified: $(ls -l "$checkpoint" | awk '{print $6, $7, $8}')"
    echo ""
    
    # Extract metadata from filename
    filename=$(basename "$checkpoint")
    if [[ $filename =~ brats_([a-z0-9]+)_([0-9]+)_([a-z]+)_([0-9]+)\.pt ]]; then
        echo "  Modality: ${BASH_REMATCH[1]}"
        echo "  Training Step: ${BASH_REMATCH[2]}"
        echo "  Schedule: ${BASH_REMATCH[3]}"
        echo "  Diffusion Steps: ${BASH_REMATCH[4]}"
    fi
    
    # Try to load with Python and show more details
    python3 << EOF
import torch
try:
    state_dict = torch.load("$checkpoint", map_location='cpu')
    if isinstance(state_dict, dict):
        print(f"\n  Keys in checkpoint: {len(state_dict.keys())}")
        print(f"  First few keys: {list(state_dict.keys())[:5]}")
    else:
        print(f"\n  Checkpoint type: {type(state_dict)}")
except Exception as e:
    print(f"\n  ⚠️  Could not load checkpoint: {e}")
EOF
}

get_resume_command() {
    local checkpoint=$1
    
    if [ ! -f "$checkpoint" ]; then
        echo "❌ Checkpoint not found: $checkpoint"
        return 1
    fi
    
    filename=$(basename "$checkpoint")
    
    # Extract info
    if [[ $filename =~ brats_([a-z0-9]+)_([0-9]+)_ ]]; then
        modality="${BASH_REMATCH[1]}"
        step="${BASH_REMATCH[2]}"
        
        echo "🚀 Resume Training Command:"
        echo ""
        echo "python app/scripts/train.py \\"
        echo "  --data_dir=./datasets/BRATS2023/training \\"
        echo "  --contr=$modality \\"
        echo "  --resume_checkpoint=\"$checkpoint\" \\"
        echo "  --resume_step=$step \\"
        echo "  --lr=1e-5"
        echo ""
        echo "Or with run.sh:"
        echo ""
        echo "./run.sh --resume_checkpoint=\"$checkpoint\" --resume_step=$step"
    else
        echo "⚠️  Could not extract step from filename"
        echo "Manual command:"
        echo "python app/scripts/train.py --resume_checkpoint=\"$checkpoint\""
    fi
}

auto_resume() {
    local modality=$1
    
    if [ -n "$modality" ]; then
        pattern="brats_${modality}_*.pt"
    else
        pattern="brats_*.pt"
    fi
    
    latest=$(ls -t "$CHECKPOINT_DIR"/$pattern 2>/dev/null | head -1)
    
    if [ -z "$latest" ]; then
        echo "❌ No checkpoints found"
        return 1
    fi
    
    echo "📌 Auto-detected latest checkpoint:"
    echo "  $latest"
    echo ""
    
    get_resume_command "$latest"
}

clean_old_checkpoints() {
    local keep_n=$1
    
    if [ -z "$keep_n" ] || [ "$keep_n" -lt 1 ]; then
        echo "❌ Invalid number. Usage: $0 clean-old <keep_n>"
        return 1
    fi
    
    echo "🧹 Cleaning old checkpoints (keeping $keep_n most recent per modality)..."
    echo ""
    
    for modality in t1n t1c t2w t2f; do
        echo "  Processing $modality..."
        
        # Get all checkpoints for this modality
        checkpoints=($(ls -t "$CHECKPOINT_DIR"/brats_${modality}_*.pt 2>/dev/null))
        
        total=${#checkpoints[@]}
        
        if [ $total -eq 0 ]; then
            echo "    No checkpoints found"
            continue
        fi
        
        if [ $total -le $keep_n ]; then
            echo "    ✓ Only $total checkpoint(s), nothing to clean"
            continue
        fi
        
        # Remove old ones
        to_remove=$((total - keep_n))
        echo "    Removing $to_remove old checkpoint(s)..."
        
        for ((i=keep_n; i<total; i++)); do
            checkpoint="${checkpoints[$i]}"
            echo "      Removing: $(basename "$checkpoint")"
            rm -f "$checkpoint"
            
            # Also remove corresponding optimizer checkpoint
            if [[ $(basename "$checkpoint") =~ brats_[a-z0-9]+_([0-9]+)_ ]]; then
                opt_file="$CHECKPOINT_DIR/opt${BASH_REMATCH[1]}.pt"
                if [ -f "$opt_file" ]; then
                    echo "      Removing: $(basename "$opt_file")"
                    rm -f "$opt_file"
                fi
            fi
        done
    done
    
    echo ""
    echo "✅ Cleanup complete"
}

verify_structure() {
    echo "🔍 Verifying checkpoint directory structure..."
    echo ""
    
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        echo "  ❌ Checkpoint directory not found: $CHECKPOINT_DIR"
        echo "  Creating directory..."
        mkdir -p "$CHECKPOINT_DIR"
        echo "  ✓ Created"
        return 0
    fi
    
    echo "  ✓ Checkpoint directory exists: $CHECKPOINT_DIR"
    
    # Count checkpoints
    model_count=$(ls "$CHECKPOINT_DIR"/brats_*.pt 2>/dev/null | wc -l)
    opt_count=$(ls "$CHECKPOINT_DIR"/opt*.pt 2>/dev/null | wc -l)
    
    echo "  Model checkpoints: $model_count"
    echo "  Optimizer checkpoints: $opt_count"
    echo ""
    
    # Check per modality
    for modality in t1n t1c t2w t2f; do
        count=$(ls "$CHECKPOINT_DIR"/brats_${modality}_*.pt 2>/dev/null | wc -l)
        if [ $count -gt 0 ]; then
            echo "  $modality: $count checkpoint(s)"
        fi
    done
}

# Main script
if [ $# -eq 0 ]; then
    show_usage
    exit 0
fi

# Parse checkpoint dir override
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

command=$1
shift

case $command in
    list)
        list_checkpoints "$@"
        ;;
    latest)
        show_latest "$@"
        ;;
    info)
        show_info "$@"
        ;;
    resume)
        get_resume_command "$@"
        ;;
    auto-resume)
        auto_resume "$@"
        ;;
    clean-old)
        clean_old_checkpoints "$@"
        ;;
    verify)
        verify_structure
        ;;
    *)
        echo "❌ Unknown command: $command"
        echo ""
        show_usage
        exit 1
        ;;
esac