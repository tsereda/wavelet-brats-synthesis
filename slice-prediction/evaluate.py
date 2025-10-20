#!/usr/bin/env python3
"""
Evaluation script for middleslice reconstruction
Calculates MSE and SSIM for model predictions
NOW WITH WAVELET DECOMPOSITION SAVING, VISUALIZATION, AND WANDB LOGGING!
"""

import torch
import numpy as np
import argparse
import os
from pathlib import Path
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import wandb

from train import BraTS2D5Dataset
from monai.networks.nets import SwinUNETR


def visualize_wavelet_decomposition(coeffs, title, output_path):
    """
    Visualize wavelet decomposition coefficients
    
    Args:
        coeffs: torch.Tensor [C*4, H/2, W/2] - wavelet coefficients (LL, LH, HL, HH for each channel)
        title: str - title for the plot
        output_path: Path - where to save the visualization
    """
    # Ensure coeffs is 3D [total_channels, H, W]
    if coeffs.dim() == 4:
        # If [B, C*4, H, W], take first sample
        coeffs = coeffs[0]
    
    # Calculate number of modalities
    total_channels = coeffs.shape[0]
    C = total_channels // 4
    
    # Debug print
    print(f"  Debug: coeffs.shape = {coeffs.shape}, C = {C}, total_channels = {total_channels}")
    
    # For input (8 modalities), only show first 4
    if C > 4:
        print(f"  Warning: Expected 4 modalities but got C={C} (total_channels={total_channels})")
        print(f"  Will only visualize first 4 modalities")
        C = 4
    
    modalities = ['T1', 'T1ce', 'T2', 'FLAIR']
    
    fig = plt.figure(figsize=(16, 4*C))
    gs = GridSpec(C, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    for mod_idx in range(C):
        start_idx = mod_idx * 4
        
        # Extract 4 subbands for this modality
        ll = coeffs[start_idx].cpu().numpy()
        lh = coeffs[start_idx + 1].cpu().numpy()
        hl = coeffs[start_idx + 2].cpu().numpy()
        hh = coeffs[start_idx + 3].cpu().numpy()
        
        # Calculate common vmin/vmax for consistent scaling
        vmin = min(ll.min(), lh.min(), hl.min(), hh.min())
        vmax = max(ll.max(), lh.max(), hl.max(), hh.max())
        
        mod_name = modalities[mod_idx] if mod_idx < len(modalities) else f"Mod{mod_idx}"
        
        # Plot LL (approximation)
        ax = fig.add_subplot(gs[mod_idx, 0])
        im = ax.imshow(ll, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f'{mod_name} - LL (Approx)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Plot LH (horizontal detail)
        ax = fig.add_subplot(gs[mod_idx, 1])
        im = ax.imshow(lh, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f'{mod_name} - LH (Horiz)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Plot HL (vertical detail)
        ax = fig.add_subplot(gs[mod_idx, 2])
        im = ax.imshow(hl, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f'{mod_name} - HL (Vert)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Plot HH (diagonal detail)
        ax = fig.add_subplot(gs[mod_idx, 3])
        im = ax.imshow(hh, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f'{mod_name} - HH (Diag)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved wavelet visualization: {output_path}")


def calculate_metrics(prediction, ground_truth):
    """
    Calculate MSE and SSIM between prediction and ground truth
    
    Args:
        prediction: torch.Tensor [4, H, W] - predicted middle slice
        ground_truth: torch.Tensor [4, H, W] - real middle slice
    
    Returns:
        dict with 'mse' and 'ssim' keys
    """
    # Convert to numpy
    pred_np = prediction.cpu().numpy()
    gt_np = ground_truth.cpu().numpy()
    
    # Calculate MSE (per-modality, then average)
    mse_per_modality = np.mean((pred_np - gt_np) ** 2, axis=(1, 2))
    mse_avg = np.mean(mse_per_modality)
    
    # Calculate SSIM (per-modality, then average)
    ssim_scores = []
    for i in range(4):  # 4 modalities
        # SSIM requires data_range parameter
        data_range = max(pred_np[i].max(), gt_np[i].max()) - min(pred_np[i].min(), gt_np[i].min())
        if data_range == 0:
            data_range = 1.0
        
        score = ssim(
            gt_np[i], 
            pred_np[i], 
            data_range=data_range
        )
        ssim_scores.append(score)
    
    ssim_avg = np.mean(ssim_scores)
    
    return {
        'mse': mse_avg,
        'ssim': ssim_avg,
        'mse_t1': mse_per_modality[0],
        'mse_t1ce': mse_per_modality[1],
        'mse_t2': mse_per_modality[2],
        'mse_flair': mse_per_modality[3],
        'ssim_t1': ssim_scores[0],
        'ssim_t1ce': ssim_scores[1],
        'ssim_t2': ssim_scores[2],
        'ssim_flair': ssim_scores[3],
    }


def evaluate_model(model, data_loader, device, output_dir, save_wavelets=False):
    """
    Evaluate model on entire dataset
    
    Args:
        model: trained model
        data_loader: DataLoader for validation set
        device: cuda or cpu
        output_dir: where to save results
        save_wavelets: whether to save wavelet coefficients (only for wavelet models)
    
    Returns:
        dict with aggregated metrics
    """
    model.eval()
    
    all_metrics = []
    predictions_dir = Path(output_dir) / 'predictions'
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    # Create wavelet directories if needed
    if save_wavelets:
        wavelet_dir = Path(output_dir) / 'wavelets'
        wavelet_viz_dir = Path(output_dir) / 'wavelet_visualizations'
        wavelet_dir.mkdir(parents=True, exist_ok=True)
        wavelet_viz_dir.mkdir(parents=True, exist_ok=True)
        print("Will save wavelet coefficients and visualizations")
    
    print(f"Evaluating on {len(data_loader)} batches...")
    
    # For logging sample predictions to wandb
    sample_logged = False
    
    with torch.no_grad():
        for batch_idx, (inputs, targets, slice_indices) in enumerate(tqdm(data_loader)):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Save wavelet coefficients if this is a wavelet model
            if save_wavelets and hasattr(model, 'dwt2d_batch'):
                # Only save for first 10 batches to avoid too much storage
                if batch_idx < 10:
                    # Get wavelet decomposition of input
                    input_wavelets = model.dwt2d_batch(inputs)
                    
                    # Get wavelet decomposition of output
                    output_wavelets = model.dwt2d_batch(outputs)
                    
                    # Get wavelet decomposition of ground truth
                    target_wavelets = model.dwt2d_batch(targets)
                    
                    # Save first sample in batch
                    sample_idx = 0
                    
                    # Save coefficients as .npy files
                    np.save(
                        wavelet_dir / f'batch{batch_idx}_input_wavelets.npy',
                        input_wavelets[sample_idx].cpu().numpy()
                    )
                    np.save(
                        wavelet_dir / f'batch{batch_idx}_output_wavelets.npy',
                        output_wavelets[sample_idx].cpu().numpy()
                    )
                    np.save(
                        wavelet_dir / f'batch{batch_idx}_target_wavelets.npy',
                        target_wavelets[sample_idx].cpu().numpy()
                    )
                    
                    # Create visualizations
                    visualize_wavelet_decomposition(
                        input_wavelets[sample_idx],
                        f'Input Wavelet Decomposition (Batch {batch_idx})',
                        wavelet_viz_dir / f'batch{batch_idx}_input_wavelets.png'
                    )
                    visualize_wavelet_decomposition(
                        output_wavelets[sample_idx],
                        f'Output Wavelet Decomposition (Batch {batch_idx})',
                        wavelet_viz_dir / f'batch{batch_idx}_output_wavelets.png'
                    )
                    visualize_wavelet_decomposition(
                        target_wavelets[sample_idx],
                        f'Target Wavelet Decomposition (Batch {batch_idx})',
                        wavelet_viz_dir / f'batch{batch_idx}_target_wavelets.png'
                    )
            
            # Calculate metrics for each sample in batch
            batch_size = inputs.shape[0]
            for i in range(batch_size):
                metrics = calculate_metrics(outputs[i], targets[i])
                metrics['slice_idx'] = slice_indices[i].item()
                metrics['batch_idx'] = batch_idx
                all_metrics.append(metrics)
                
                # Optionally save predictions
                if batch_idx < 10:  # Save first 10 batches for inspection
                    pred_path = predictions_dir / f'batch{batch_idx}_sample{i}.npy'
                    np.save(pred_path, outputs[i].cpu().numpy())
                
                # Log first sample to wandb
                if batch_idx == 0 and i == 0 and not sample_logged:
                    # Create comparison visualization
                    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
                    modalities = ['T1', 'T1ce', 'T2', 'FLAIR']
                    
                    for mod_idx, mod_name in enumerate(modalities):
                        # Input Z-1
                        axes[mod_idx, 0].imshow(inputs[i, mod_idx].cpu().numpy(), cmap='gray')
                        axes[mod_idx, 0].set_title(f'{mod_name} Input (Z-1)' if mod_idx == 0 else '')
                        axes[mod_idx, 0].axis('off')
                        
                        # Input Z+1
                        axes[mod_idx, 1].imshow(inputs[i, mod_idx+4].cpu().numpy(), cmap='gray')
                        axes[mod_idx, 1].set_title(f'Input (Z+1)' if mod_idx == 0 else '')
                        axes[mod_idx, 1].axis('off')
                        
                        # Prediction
                        axes[mod_idx, 2].imshow(outputs[i, mod_idx].cpu().numpy(), cmap='gray')
                        axes[mod_idx, 2].set_title(f'Prediction (Z)' if mod_idx == 0 else '')
                        axes[mod_idx, 2].axis('off')
                        
                        # Ground truth
                        axes[mod_idx, 3].imshow(targets[i, mod_idx].cpu().numpy(), cmap='gray')
                        axes[mod_idx, 3].set_title(f'Ground Truth (Z)' if mod_idx == 0 else '')
                        axes[mod_idx, 3].axis('off')
                        
                        # Error map
                        error = np.abs(outputs[i, mod_idx].cpu().numpy() - targets[i, mod_idx].cpu().numpy())
                        im = axes[mod_idx, 4].imshow(error, cmap='hot', vmin=0, vmax=0.3)
                        axes[mod_idx, 4].set_title(f'|Error|' if mod_idx == 0 else '')
                        axes[mod_idx, 4].axis('off')
                    
                    plt.suptitle(f'Sample Prediction (Slice {slice_indices[i].item()})')
                    plt.tight_layout()
                    
                    # Save and log to wandb
                    sample_path = predictions_dir / 'sample_prediction.png'
                    plt.savefig(sample_path, dpi=150, bbox_inches='tight')
                    wandb.log({"eval/sample_prediction": wandb.Image(str(sample_path))})
                    plt.close()
                    
                    sample_logged = True
    
    # Aggregate metrics
    mse_values = [m['mse'] for m in all_metrics]
    ssim_values = [m['ssim'] for m in all_metrics]
    
    results = {
        'mse_mean': np.mean(mse_values),
        'mse_std': np.std(mse_values),
        'ssim_mean': np.mean(ssim_values),
        'ssim_std': np.std(ssim_values),
        'num_samples': len(all_metrics)
    }
    
    # Per-modality stats
    for mod in ['t1', 't1ce', 't2', 'flair']:
        mse_mod = [m[f'mse_{mod}'] for m in all_metrics]
        ssim_mod = [m[f'ssim_{mod}'] for m in all_metrics]
        results[f'mse_{mod}_mean'] = np.mean(mse_mod)
        results[f'ssim_{mod}_mean'] = np.mean(ssim_mod)
    
    return results, all_metrics


def save_results(results, all_metrics, output_dir):
    """Save results to CSV files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary statistics
    summary_path = output_dir / 'metrics_summary.csv'
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for key, value in results.items():
            writer.writerow([key, f'{value:.6f}'])
    
    print(f"Summary saved to {summary_path}")
    
    # Save per-sample metrics
    details_path = output_dir / 'metrics_detailed.csv'
    with open(details_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
        writer.writeheader()
        writer.writerows(all_metrics)
    
    print(f"Detailed metrics saved to {details_path}")


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate middleslice reconstruction model")
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to BraTS dataset')
    parser.add_argument('--output', type=str, default='./results/swin',
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--num_patients', type=int, default=None,
                       help='Number of patients to evaluate (default: all)')
    parser.add_argument('--img_size', type=int, default=256,
                       help='Image size')
    parser.add_argument('--model_type', type=str, default='swin',
                       choices=['swin', 'wavelet_haar', 'wavelet_db2', 'wavelet'],
                       help='Model architecture')
    parser.add_argument('--wavelet', type=str, default='haar',
                       help='Wavelet type (for wavelet model type)')
    parser.add_argument('--save_wavelets', action='store_true',
                       help='Save wavelet coefficients and visualizations (wavelet models only)')
    return parser.parse_args()


def load_model(checkpoint_path, model_type, wavelet_name, img_size, device):
    """Load trained model"""
    if model_type == 'swin':
        model = SwinUNETR(
            in_channels=8,
            out_channels=4,
            feature_size=24,
            spatial_dims=2
        ).to(device)
        print("Loaded Swin-UNETR model")
    
    elif model_type in ['wavelet', 'wavelet_haar', 'wavelet_db2']:
        from models.wavelet_diffusion import WaveletDiffusion
        
        # Determine wavelet type from model_type if specified
        if model_type == 'wavelet_haar':
            wavelet_name = 'haar'
        elif model_type == 'wavelet_db2':
            wavelet_name = 'db2'
        # else use the wavelet_name parameter
        
        model = WaveletDiffusion(
            wavelet_name=wavelet_name,
            in_channels=8,
            out_channels=4,
            timesteps=100
        ).to(device)
        print(f"Loaded Wavelet Diffusion model ({wavelet_name})")
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model


def main():
    args = get_args()
    
    # Initialize wandb
    run_name = f"eval_{args.model_type}_{args.wavelet if args.model_type in ['wavelet', 'wavelet_haar', 'wavelet_db2'] else 'baseline'}"
    wandb.init(
        project="brats-middleslice-wavelet-sweep",
        name=run_name,
        config=vars(args),
        tags=["evaluation"]
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = BraTS2D5Dataset(
        data_dir=args.data_dir,
        image_size=(args.img_size, args.img_size),
        spacing=(1.0, 1.0, 1.0),
        num_patients=args.num_patients
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Don't shuffle for reproducible evaluation
        num_workers=4
    )
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.model_type, args.wavelet, args.img_size, device)
    
    # Check if we should save wavelets
    is_wavelet_model = args.model_type in ['wavelet', 'wavelet_haar', 'wavelet_db2']
    save_wavelets = args.save_wavelets and is_wavelet_model
    if args.save_wavelets and not is_wavelet_model:
        print("Warning: --save_wavelets only works with wavelet models. Ignoring.")
    
    # Evaluate
    print("Running evaluation...")
    results, all_metrics = evaluate_model(model, data_loader, device, args.output, save_wavelets)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"MSE:  {results['mse_mean']:.6f} ± {results['mse_std']:.6f}")
    print(f"SSIM: {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")
    print(f"Samples evaluated: {results['num_samples']}")
    print("\nPer-modality MSE:")
    print(f"  T1:    {results['mse_t1_mean']:.6f}")
    print(f"  T1ce:  {results['mse_t1ce_mean']:.6f}")
    print(f"  T2:    {results['mse_t2_mean']:.6f}")
    print(f"  FLAIR: {results['mse_flair_mean']:.6f}")
    print("="*50)
    
    # Log results to wandb
    wandb.log({
        "eval/mse_mean": results['mse_mean'],
        "eval/mse_std": results['mse_std'],
        "eval/ssim_mean": results['ssim_mean'],
        "eval/ssim_std": results['ssim_std'],
        "eval/num_samples": results['num_samples'],
        "eval/mse_t1": results['mse_t1_mean'],
        "eval/mse_t1ce": results['mse_t1ce_mean'],
        "eval/mse_t2": results['mse_t2_mean'],
        "eval/mse_flair": results['mse_flair_mean'],
        "eval/ssim_t1": results['ssim_t1_mean'],
        "eval/ssim_t1ce": results['ssim_t1ce_mean'],
        "eval/ssim_t2": results['ssim_t2_mean'],
        "eval/ssim_flair": results['ssim_flair_mean'],
    })
    
    # Log example wavelet visualizations to wandb
    if save_wavelets:
        wavelet_viz_dir = Path(args.output) / 'wavelet_visualizations'
        for img_path in sorted(wavelet_viz_dir.glob('batch0_*.png'))[:3]:  # Log first batch
            wandb.log({f"wavelets/{img_path.stem}": wandb.Image(str(img_path))})
    
    # Save results
    save_results(results, all_metrics, args.output)
    
    print(f"\nResults saved to {args.output}/")
    if save_wavelets:
        print(f"Wavelet coefficients saved to {args.output}/wavelets/")
        print(f"Wavelet visualizations saved to {args.output}/wavelet_visualizations/")
    
    wandb.finish()


if __name__ == '__main__':
    main()