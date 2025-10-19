#!/usr/bin/env python3
"""
Evaluation script for middleslice reconstruction
Calculates MSE and SSIM for model predictions
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

from train import BraTS2D5Dataset
from monai.networks.nets import SwinUNETR


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


def evaluate_model(model, data_loader, device, output_dir):
    """
    Evaluate model on entire dataset
    
    Args:
        model: trained model
        data_loader: DataLoader for validation set
        device: cuda or cpu
        output_dir: where to save results
    
    Returns:
        dict with aggregated metrics
    """
    model.eval()
    
    all_metrics = []
    predictions_dir = Path(output_dir) / 'predictions'
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Evaluating on {len(data_loader)} batches...")
    
    with torch.no_grad():
        for batch_idx, (inputs, targets, slice_indices) in enumerate(tqdm(data_loader)):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
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
                       choices=['swin', 'wavelet_haar', 'wavelet_db2'],
                       help='Model architecture')
    return parser.parse_args()


def load_model(checkpoint_path, model_type, img_size, device):
    """Load trained model"""
    if model_type == 'swin':
        model = SwinUNETR(
            img_size=(img_size, img_size),  # Add img_size parameter
            in_channels=8,
            out_channels=4,
            feature_size=24,
            spatial_dims=2
        ).to(device)
    else:
        # TODO: Load wavelet diffusion models
        raise NotImplementedError(f"Model type {model_type} not yet implemented")
    
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
    model = load_model(args.checkpoint, args.model_type, args.img_size, device)
    
    # Evaluate
    print("Running evaluation...")
    results, all_metrics = evaluate_model(model, data_loader, device, args.output)
    
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
    
    # Save results
    save_results(results, all_metrics, args.output)
    
    print(f"\nResults saved to {args.output}/")


if __name__ == '__main__':
    main()