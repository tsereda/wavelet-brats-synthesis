#!/usr/bin/env python3
"""
Generate Figure 2 for ISBI paper
Qualitative comparison of Swin, Haar, and db2 reconstructions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from train import BraTS2D5Dataset
from monai.networks.nets import SwinUNETR


def select_best_example(data_loader, model, device, num_candidates=50):
    """
    Find a good example slice for visualization
    Criteria: Clear tumor, good reconstruction quality
    """
    model.eval()
    
    candidates = []
    
    with torch.no_grad():
        for i, (inputs, targets, slice_indices) in enumerate(data_loader):
            if i >= num_candidates:
                break
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            
            # Take first sample from batch
            input_sample = inputs[0].cpu()
            target_sample = targets[0].cpu()
            output_sample = outputs[0].cpu()
            # Robust extraction in case dataset returns (slice_idx, patient_id)
            first_info = slice_indices[0]
            patient_id = None
            if isinstance(first_info, (list, tuple)):
                slice_idx = int(first_info[0])
                patient_id = str(first_info[1])
            else:
                try:
                    slice_idx = int(first_info.item())
                except Exception:
                    slice_idx = int(first_info)
            
            # Calculate MSE
            mse = torch.mean((output_sample - target_sample) ** 2).item()
            
            # Check if tumor is present (high intensity in T1ce)
            has_tumor = torch.max(target_sample[1]) > 0.3  # T1ce channel
            
            if has_tumor:
                candidates.append({
                    'input': input_sample,
                    'target': target_sample,
                    'output': output_sample,
                    'slice_idx': slice_idx,
                    'patient_id': patient_id,
                    'mse': mse,
                    'batch_idx': i
                })
    
    # Sort by MSE and pick a good middle example (not best, not worst)
    candidates.sort(key=lambda x: x['mse'])
    middle_idx = len(candidates) // 2
    
    return candidates[middle_idx] if candidates else None


def create_comparison_figure(examples, output_path):
    """
    Create figure comparing Swin, Haar, and db2
    
    Args:
        examples: dict with keys 'swin', 'haar', 'db2'
                  each containing input, target, output tensors
        output_path: where to save the figure
    """
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(4, 7, figure=fig, hspace=0.3, wspace=0.2)
    
    modalities = ['T1', 'T1ce', 'T2', 'FLAIR']
    model_names = ['Swin-UNETR', 'Fast-cWDM-Haar', 'Fast-cWDM-db2']
    
    # Assume all examples use same input/target
    example = examples['swin']
    inputs = example['input'].numpy()
    target = example['target'].numpy()
    
    for mod_idx, mod_name in enumerate(modalities):
        # Column 0: Z-1 (previous slice)
        ax = fig.add_subplot(gs[mod_idx, 0])
        ax.imshow(inputs[mod_idx], cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Input Z-1\n{mod_name}' if mod_idx == 0 else f'{mod_name}', 
                    fontsize=10)
        ax.axis('off')
        
        # Column 1: Z+1 (next slice)
        ax = fig.add_subplot(gs[mod_idx, 1])
        ax.imshow(inputs[mod_idx + 4], cmap='gray', vmin=0, vmax=1)
        ax.set_title('Input Z+1' if mod_idx == 0 else '', fontsize=10)
        ax.axis('off')
        
        # Columns 2-4: Model predictions (Swin, Haar, db2)
        for model_idx, (model_key, model_name) in enumerate(zip(
            ['swin', 'haar', 'db2'], model_names
        )):
            if model_key in examples:
                output = examples[model_key]['output'].numpy()
                
                ax = fig.add_subplot(gs[mod_idx, 2 + model_idx])
                ax.imshow(output[mod_idx], cmap='gray', vmin=0, vmax=1)
                ax.set_title(model_name if mod_idx == 0 else '', fontsize=10)
                ax.axis('off')
        
        # Column 5: Ground truth
        ax = fig.add_subplot(gs[mod_idx, 5])
        ax.imshow(target[mod_idx], cmap='gray', vmin=0, vmax=1)
        ax.set_title('Ground Truth' if mod_idx == 0 else '', fontsize=10)
        ax.axis('off')
        
        # Column 6: Error map (using best model - db2)
        if 'db2' in examples:
            output_db2 = examples['db2']['output'].numpy()
            error = np.abs(output_db2[mod_idx] - target[mod_idx])
            
            ax = fig.add_subplot(gs[mod_idx, 6])
            im = ax.imshow(error, cmap='hot', vmin=0, vmax=0.3)
            ax.set_title('|Error|' if mod_idx == 0 else '', fontsize=10)
            ax.axis('off')
            
            if mod_idx == 3:  # Add colorbar to last row
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Absolute Error', fontsize=8)
    
    # Add main title with metrics
    if 'db2' in examples:
        mse = examples['db2']['mse']
        slice_idx = examples['db2']['slice_idx']
        fig.suptitle(
            f'Middleslice Reconstruction Comparison (Slice {slice_idx}, MSE: {mse:.4f})',
            fontsize=14, fontweight='bold'
        )
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path}")
    plt.close()


def create_simple_comparison(swin_example, haar_example, db2_example, output_path):
    """
    Simplified version: just compare one modality (T1ce) across models
    """
    fig, axes = plt.subplots(2, 6, figsize=(15, 5))
    
    # Row 0: Images
    inputs = swin_example['input'].numpy()
    target = swin_example['target'].numpy()
    
    # T1ce only (index 1)
    mod_idx = 1
    
    axes[0, 0].imshow(inputs[mod_idx], cmap='gray')
    axes[0, 0].set_title('Input Z-1')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(inputs[mod_idx + 4], cmap='gray')
    axes[0, 1].set_title('Input Z+1')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(swin_example['output'].numpy()[mod_idx], cmap='gray')
    axes[0, 2].set_title(f"Swin\nMSE: {swin_example['mse']:.4f}")
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(haar_example['output'].numpy()[mod_idx], cmap='gray')
    axes[0, 3].set_title(f"Haar\nMSE: {haar_example['mse']:.4f}")
    axes[0, 3].axis('off')
    
    axes[0, 4].imshow(db2_example['output'].numpy()[mod_idx], cmap='gray')
    axes[0, 4].set_title(f"db2\nMSE: {db2_example['mse']:.4f}")
    axes[0, 4].axis('off')
    
    axes[0, 5].imshow(target[mod_idx], cmap='gray')
    axes[0, 5].set_title('Ground Truth')
    axes[0, 5].axis('off')
    
    # Row 1: Error maps
    axes[1, 0].axis('off')
    axes[1, 1].axis('off')
    
    error_swin = np.abs(swin_example['output'].numpy()[mod_idx] - target[mod_idx])
    axes[1, 2].imshow(error_swin, cmap='hot', vmin=0, vmax=0.3)
    axes[1, 2].set_title('Swin Error')
    axes[1, 2].axis('off')
    
    error_haar = np.abs(haar_example['output'].numpy()[mod_idx] - target[mod_idx])
    axes[1, 3].imshow(error_haar, cmap='hot', vmin=0, vmax=0.3)
    axes[1, 3].set_title('Haar Error')
    axes[1, 3].axis('off')
    
    error_db2 = np.abs(db2_example['output'].numpy()[mod_idx] - target[mod_idx])
    im = axes[1, 4].imshow(error_db2, cmap='hot', vmin=0, vmax=0.3)
    axes[1, 4].set_title('db2 Error')
    axes[1, 4].axis('off')
    
    axes[1, 5].axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[1, 4], fraction=0.046, pad=0.04)
    cbar.set_label('Absolute Error', fontsize=8)
    
    plt.suptitle(f'T1ce Reconstruction Comparison (Slice {swin_example["slice_idx"]})', 
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Simple comparison saved to {output_path}")
    plt.close()


def load_model(checkpoint_path, model_type, device):
    """Load trained model"""
    if model_type == 'swin':
        model = SwinUNETR(
            in_channels=8,
            out_channels=4,
            feature_size=24,
            spatial_dims=2
        ).to(device)
    else:
        # TODO: Implement wavelet models
        raise NotImplementedError(f"Model type {model_type} not yet implemented")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model


def get_args():
    parser = argparse.ArgumentParser(description="Generate Figure 2 for ISBI paper")
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to BraTS dataset')
    parser.add_argument('--swin_checkpoint', type=str, required=True,
                       help='Path to Swin-UNETR checkpoint')
    parser.add_argument('--haar_checkpoint', type=str, default=None,
                       help='Path to Haar checkpoint')
    parser.add_argument('--db2_checkpoint', type=str, default=None,
                       help='Path to db2 checkpoint')
    parser.add_argument('--output', type=str, default='./figures/figure2_qualitative.pdf',
                       help='Output path for figure')
    parser.add_argument('--simple', action='store_true',
                       help='Generate simplified version (one modality only)')
    parser.add_argument('--num_patients', type=int, default=10,
                       help='Number of patients to search for good examples')
    return parser.parse_args()


def main():
    args = get_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = BraTS2D5Dataset(
        data_dir=args.data_dir,
        image_size=(256, 256),
        spacing=(1.0, 1.0, 1.0),
        num_patients=args.num_patients
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )
    
    # Load models and generate examples
    examples = {}
    
    # Load Swin
    print(f"Loading Swin model from {args.swin_checkpoint}...")
    swin_model = load_model(args.swin_checkpoint, 'swin', device)
    examples['swin'] = select_best_example(data_loader, swin_model, device)
    print(f"  Selected example: Slice {examples['swin']['slice_idx']}, MSE: {examples['swin']['mse']:.4f}")
    
    # Load Haar if provided
    if args.haar_checkpoint:
        print(f"Loading Haar model from {args.haar_checkpoint}...")
        # TODO: Load haar model
        # haar_model = load_model(args.haar_checkpoint, 'wavelet_haar', device)
        # examples['haar'] = select_best_example(data_loader, haar_model, device)
        print("  Haar model loading not yet implemented - skipping")
    
    # Load db2 if provided
    if args.db2_checkpoint:
        print(f"Loading db2 model from {args.db2_checkpoint}...")
        # TODO: Load db2 model
        # db2_model = load_model(args.db2_checkpoint, 'wavelet_db2', device)
        # examples['db2'] = select_best_example(data_loader, db2_model, device)
        print("  db2 model loading not yet implemented - skipping")
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # If we selected an example and it contains a patient_id, append it to the filename
    try:
        pid = examples.get('swin', {}).get('patient_id', None)
        if pid:
            # Only append when the filename doesn't already contain the id
            stem = output_path.stem
            if str(pid) not in stem:
                output_path = output_path.with_name(f"{stem}_{pid}" + output_path.suffix)
    except Exception:
        pass
    
    # Generate figure
    if args.simple or len(examples) == 1:
        print("Note: Only Swin model available, cannot generate full comparison")
        print("Run with --haar_checkpoint and --db2_checkpoint for complete figure")
    else:
        print("Generating comparison figure...")
        if args.simple:
            create_simple_comparison(
                examples['swin'],
                examples.get('haar', examples['swin']),
                examples.get('db2', examples['swin']),
                args.output
            )
        else:
            create_comparison_figure(examples, args.output)
    
    print("Done!")


if __name__ == '__main__':
    main()