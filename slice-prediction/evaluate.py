#!/usr/bin/env python3
"""
Evaluation script for middleslice reconstruction
Calculates MSE and SSIM for model predictions
WITH FIXED SSIM CALCULATION, COMPLETE WAVELET VISUALIZATION, AND TIMING STATS!
FIXED: Updated to BraTS2023 GLI naming convention (t1n, t1c, t2w, t2f)
"""

import torch
import numpy as np
import argparse
import os
import cv2
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import wandb
from time import perf_counter
import pandas as pd

from train import BraTS2D5Dataset
from preprocessed_dataset import FastTensorSliceDataset
from monai.networks.nets import SwinUNETR
from utils import extract_patient_info, get_patient_output_dir, save_slice_outputs
from logging_utils import create_reconstruction_log_panel
from typing import Optional


def download_checkpoint_from_wandb(sweep_id: Optional[str] = None, run_id: Optional[str] = None, download_dir: str = './wandb_checkpoints', wandb_entity: str = 'timgsereda', wandb_project: str = 'brats-middleslice-wavelet-sweep'):
    """
    Download checkpoint from W&B sweep or run
    
    Args:
        sweep_id: W&B sweep ID (e.g., "5mfl25i8")
        run_id: W&B run ID (if you want specific run instead of sweep)
        download_dir: Where to save checkpoints
        wandb_entity: W&B entity/username
        wandb_project: W&B project name
    
    Returns:
        Path to downloaded checkpoint OR list of checkpoint dicts when sweep_id provided
    """
    api = wandb.Api()
    Path(download_dir).mkdir(parents=True, exist_ok=True)
    
    if run_id:
        # Single run mode
        run = api.run(f"{wandb_entity}/{wandb_project}/{run_id}")
        artifacts = run.logged_artifacts()
        for artifact in artifacts:
            if artifact.type == 'model' and any(alias in ['best', 'latest'] for alias in artifact.aliases):
                print(f"Downloading checkpoint from run {run_id}: {artifact.name}")
                artifact_dir = artifact.download(root=download_dir)
                # Find .pth file in downloaded directory
                ckpt_files = list(Path(artifact_dir).glob('*.pth'))
                if ckpt_files:
                    return str(ckpt_files[0])
        raise RuntimeError(f"No best checkpoint found in run {run_id}")
    
    elif sweep_id:
        # Sweep mode - download all checkpoints
        sweep = api.sweep(f"{wandb_entity}/{wandb_project}/{sweep_id}")
        runs = sweep.runs
        
        checkpoints = []
        for run in runs:
            try:
                artifacts = run.logged_artifacts()
                # Prefer 'best' over 'latest' to avoid downloading all checkpoints
                best_artifact = None
                latest_artifact = None
                for artifact in artifacts:
                    if artifact.type == 'model':
                        if 'best' in artifact.aliases:
                            best_artifact = artifact
                            break  # Found best, no need to continue
                        elif 'latest' in artifact.aliases:
                            latest_artifact = artifact
                
                # Download best if available, otherwise latest
                artifact_to_download = best_artifact or latest_artifact
                if artifact_to_download:
                    print(f"Downloading checkpoint from run {run.name}: {artifact_to_download.name}")
                    artifact_dir = artifact_to_download.download(root=f"{download_dir}/{run.id}")
                    ckpt_files = list(Path(artifact_dir).glob('*.pth'))
                    if ckpt_files:
                        checkpoints.append({
                            'path': str(ckpt_files[0]),
                            'run_id': run.id,
                            'run_name': run.name,
                            'config': run.config
                        })
            except Exception as e:
                # Safely derive a human-readable run identifier for logging
                try:
                    run_name = run.name
                except Exception:
                    try:
                        run_name = str(run)
                    except Exception:
                        run_name = "<unknown run>"
                print(f"Warning: Failed to download from run {run_name}: {e}")
                continue
        
        if not checkpoints:
            raise RuntimeError(f"No checkpoints found in sweep {sweep_id}")
        
        return checkpoints
    
    else:
        raise ValueError("Must provide either sweep_id or run_id")


# Timing stats tracker for evaluation
class EvaluationTimingStats:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.forward_times = []
        self.data_load_times = []
        self.metric_calculation_times = []
        self.wavelet_transform_times = []
        self.total_samples = 0
        
    def add_forward_time(self, elapsed):
        self.forward_times.append(elapsed)
        
    def add_data_load_time(self, elapsed):
        self.data_load_times.append(elapsed)
        
    def add_metric_time(self, elapsed):
        self.metric_calculation_times.append(elapsed)
        
    def add_wavelet_time(self, elapsed):
        self.wavelet_transform_times.append(elapsed)
        
    def add_samples(self, count):
        self.total_samples += count
    
    def get_stats(self):
        total_forward_time = sum(self.forward_times)
        total_data_time = sum(self.data_load_times)
        total_metric_time = sum(self.metric_calculation_times)
        total_wavelet_time = sum(self.wavelet_transform_times)
        
        return {
            'avg_forward_time_ms': np.mean(self.forward_times) * 1000 if self.forward_times else 0,
            'avg_data_load_time_ms': np.mean(self.data_load_times) * 1000 if self.data_load_times else 0,
            'avg_metric_time_ms': np.mean(self.metric_calculation_times) * 1000 if self.metric_calculation_times else 0,
            'avg_wavelet_time_ms': np.mean(self.wavelet_transform_times) * 1000 if self.wavelet_transform_times else 0,
            'total_forward_time_s': total_forward_time,
            'total_data_time_s': total_data_time,
            'total_metric_time_s': total_metric_time,
            'total_wavelet_time_s': total_wavelet_time,
            'samples_per_second': self.total_samples / total_forward_time if total_forward_time > 0 else 0,
            'total_samples': self.total_samples
        }


def visualize_wavelet_decomposition(coeffs, title, output_path):
    """
    Visualize wavelet decomposition coefficients - FIXED FOR ALL INPUT COMPONENTS
    
    Args:
        coeffs: torch.Tensor [C*4, H/2, W/2] - wavelet coefficients (LL, LH, HL, HH for each channel)
        title: str - title for the plot
        output_path: Path - where to save the visualization
    """
    import io
    from PIL import Image

    # Ensure coeffs is 3D [total_channels, H, W]
    if coeffs.dim() == 4:
        coeffs = coeffs[0]

    total_channels = coeffs.shape[0]
    C = total_channels // 4

    if C == 8:  # Input
        modalities = ['T1n(Z-1)', 'T1c(Z-1)', 'T2w(Z-1)', 'T2f(Z-1)',
                      'T1n(Z+1)', 'T1c(Z+1)', 'T2w(Z+1)', 'T2f(Z+1)']
        show_C = 8
    else:
        modalities = ['T1n', 'T1c', 'T2w', 'T2f']
        show_C = min(C, 4)

    fig = plt.figure(figsize=(16, 4 * show_C))
    gs = GridSpec(show_C, 4, figure=fig, hspace=0.3, wspace=0.3)

    for mod_idx in range(show_C):
        start_idx = mod_idx * 4

        ll = coeffs[start_idx].cpu().numpy()
        lh = coeffs[start_idx + 1].cpu().numpy()
        hl = coeffs[start_idx + 2].cpu().numpy()
        hh = coeffs[start_idx + 3].cpu().numpy()

        vmin = min(ll.min(), lh.min(), hl.min(), hh.min())
        vmax = max(ll.max(), lh.max(), hl.max(), hh.max())
        if vmin == vmax:
            vmax = vmin + 1e-6

        mod_name = modalities[mod_idx] if mod_idx < len(modalities) else f"Mod{mod_idx}"

        for i, (subband, name) in enumerate([(ll, 'LL'), (lh, 'LH'), (hl, 'HL'), (hh, 'HH')]):
            ax = fig.add_subplot(gs[mod_idx, i])
            im = ax.imshow(subband, cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_title(f'{mod_name} - {name}', fontsize=10)
            ax.axis('off')
            if i == 3:
                plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    #     print(f"  Saved: {output_path}")

    plt.close()


def calculate_ssim_pytorch(img1, img2, window_size=11, data_range=1.0):
    """
    Fast GPU-accelerated SSIM calculation using PyTorch
    ~200x faster than scikit-image implementation
    
    Args:
        img1: torch.Tensor [C, H, W] - first image
        img2: torch.Tensor [C, H, W] - second image  
        window_size: int - size of Gaussian window (default: 11)
        data_range: float - dynamic range of pixel values (default: 1.0)
    
    Returns:
        torch.Tensor [C] - SSIM score per channel
    """
    C, H, W = img1.shape
    device = img1.device
    
    # SSIM constants
    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    
    # Create Gaussian window (1D then outer product for 2D)
    sigma = 1.5
    coords = torch.arange(window_size, device=device, dtype=torch.float32) - window_size // 2
    gauss_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    window_2d = gauss_1d[:, None] * gauss_1d[None, :]
    window = window_2d.expand(C, 1, window_size, window_size).contiguous()
    
    # Compute local means using convolution
    pad = window_size // 2
    mu1 = torch.nn.functional.conv2d(img1.unsqueeze(0), window, padding=pad, groups=C).squeeze(0)
    mu2 = torch.nn.functional.conv2d(img2.unsqueeze(0), window, padding=pad, groups=C).squeeze(0)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Compute local variances and covariance
    sigma1_sq = torch.nn.functional.conv2d(img1.unsqueeze(0) ** 2, window, padding=pad, groups=C).squeeze(0) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(img2.unsqueeze(0) ** 2, window, padding=pad, groups=C).squeeze(0) - mu2_sq
    sigma12 = torch.nn.functional.conv2d((img1 * img2).unsqueeze(0), window, padding=pad, groups=C).squeeze(0) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    # Return mean SSIM per channel
    return ssim_map.mean(dim=(1, 2))


def calculate_metrics(prediction, ground_truth):
    """
    Calculate MSE and SSIM between prediction and ground truth
    OPTIMIZED: Uses fast GPU-accelerated PyTorch SSIM (200x faster than scikit-image)
    FIXED: Updated to BraTS2023 GLI naming (t1n, t1c, t2w, t2f)
    
    Args:
        prediction: torch.Tensor [4, H, W] - predicted middle slice
        ground_truth: torch.Tensor [4, H, W] - real middle slice
    
    Returns:
        dict with 'mse' and 'ssim' keys
    """
    # Calculate MSE (keep on GPU, no CPU transfer needed)
    mse_per_modality = torch.mean((prediction - ground_truth) ** 2, dim=(1, 2))
    mse_avg = torch.mean(mse_per_modality)
    
    # Calculate SSIM using fast PyTorch implementation (all 4 modalities vectorized)
    ssim_scores = calculate_ssim_pytorch(ground_truth, prediction, window_size=11, data_range=1.0)
    ssim_avg = torch.mean(ssim_scores)
    
    # Convert to Python scalars for logging
    return {
        'mse': float(mse_avg),
        'ssim': float(ssim_avg),
        'mse_t1n': float(mse_per_modality[0]),
        'mse_t1c': float(mse_per_modality[1]),
        'mse_t2w': float(mse_per_modality[2]),
        'mse_t2f': float(mse_per_modality[3]),
        'ssim_t1n': float(ssim_scores[0]),
        'ssim_t1c': float(ssim_scores[1]),
        'ssim_t2w': float(ssim_scores[2]),
        'ssim_t2f': float(ssim_scores[3]),
    }


def evaluate_model(model, data_loader, device, output_dir, save_wavelets=True):
    """
    Evaluate model on entire dataset WITH COMPLETE TIMING STATS
    
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
    
    # Initialize timing stats
    timing_stats = EvaluationTimingStats()
    
    # Create wavelet directories if needed
    if save_wavelets:
        wavelet_dir = Path(output_dir) / 'wavelets'
        wavelet_viz_dir = Path(output_dir) / 'wavelet_visualizations'
        wavelet_dir.mkdir(parents=True, exist_ok=True)
        wavelet_viz_dir.mkdir(parents=True, exist_ok=True)
        print("Will save wavelet coefficients and visualizations")
    
    print(f"Evaluating on {len(data_loader)} batches...")
    print(f"Measuring detailed timing statistics...")
    
    # For logging sample predictions to wandb
    sample_logged = False
    
    # Track running per-modality metrics for WandB - FIXED naming
    running_mse_per_mod = {'t1n': [], 't1c': [], 't2w': [], 't2f': []}
    running_ssim_per_mod = {'t1n': [], 't1c': [], 't2w': [], 't2f': []}
    
    # Collect data for W&B table
    table_data = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(data_loader)):
            # Time data loading (approximate)
            data_start = perf_counter()
            
            inputs, targets, slice_indices = batch_data
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            data_time = perf_counter() - data_start
            timing_stats.add_data_load_time(data_time)
            
            # Time forward pass
            torch.cuda.synchronize() if device.type == 'cuda' else None
            forward_start = perf_counter()
            
            # Forward pass
            outputs = model(inputs)
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            forward_time = perf_counter() - forward_start
            timing_stats.add_forward_time(forward_time)
            timing_stats.add_samples(inputs.shape[0])
            
            # Time wavelet transforms if this is a wavelet model
            # Separate timing from computation to ensure wavelets are available for ALL samples in table
            has_wavelets = save_wavelets and hasattr(model, 'dwt2d_batch')
            
            # Initialize wavelet variables (will be None if not computed)
            input_wavelets = None
            output_wavelets = None
            target_wavelets = None
            
            if has_wavelets:
                # Debug: Log on first batch
                if batch_idx == 0:
                    print(f"✓ Wavelet model detected - will compute wavelets for all batches")
                
                # Time wavelet computation only for first 10 batches (for performance metrics)
                measure_timing = batch_idx < 10
                if measure_timing:
                    wavelet_start = perf_counter()
                
                # But ALWAYS compute wavelets for all batches (needed for W&B table)
                # Get wavelet decomposition of input (ALL 8 CHANNELS)
                input_wavelets = model.dwt2d_batch(inputs)
                
                # Get wavelet decomposition of output
                output_wavelets = model.dwt2d_batch(outputs)
                
                # Get wavelet decomposition of ground truth
                target_wavelets = model.dwt2d_batch(targets)
                
                # Complete timing for first 10 batches
                if measure_timing:
                    torch.cuda.synchronize() if device.type == 'cuda' else None
                    wavelet_time = perf_counter() - wavelet_start
                    timing_stats.add_wavelet_time(wavelet_time)
                
                # Debug: Confirm wavelet computation on first batch
                if batch_idx == 0:
                    print(f"✓ Computed wavelets for batch 0: input_shape={input_wavelets.shape}, output_shape={output_wavelets.shape}, target_shape={target_wavelets.shape}")
            else:
                # Debug: Log on first batch
                if batch_idx == 0:
                    print(f"ℹ No wavelets will be computed (save_wavelets={save_wavelets}, has_dwt2d_batch={hasattr(model, 'dwt2d_batch')})")
            
            # Time metric calculation
            metric_start = perf_counter()
            
            # Calculate metrics for each sample in batch
            batch_size = inputs.shape[0]
            for i in range(batch_size):
                metrics = calculate_metrics(outputs[i], targets[i])
                
                # Extract slice index and patient ID using centralized utility
                slice_idx, patient_id = extract_patient_info(slice_indices, batch_idx, i)
                
                metrics['slice_idx'] = slice_idx
                metrics['patient_id'] = patient_id
                metrics['batch_idx'] = batch_idx
                all_metrics.append(metrics)
                
                # Track per-modality metrics for running average - FIXED naming
                for mod in ['t1n', 't1c', 't2w', 't2f']:
                    running_mse_per_mod[mod].append(metrics[f'mse_{mod}'])
                    running_ssim_per_mod[mod].append(metrics[f'ssim_{mod}'])
                
                # Save all predictions using patient-specific directories and add to table
                patient_dir = get_patient_output_dir(predictions_dir, patient_id, slice_idx)
                
                # Save slice outputs
                saved_files = save_slice_outputs(
                    patient_dir=patient_dir,
                    inputs=inputs[i],
                    target=targets[i],
                    output=outputs[i],
                    slice_idx=slice_idx,
                    patient_id=patient_id,
                    batch_idx=batch_idx
                )
                
                # Create reconstruction panel for this sample
                panel = create_reconstruction_log_panel(
                    inputs[i], targets[i], outputs[i],
                    slice_idx, batch_idx, patient_id=patient_id
                )
                panel_path = patient_dir / 'reconstruction_panel.png'
                cv2.imwrite(str(panel_path), panel)
                
                # Prepare table row
                table_row = {
                    "patient_id": patient_id,
                    "slice_idx": slice_idx,
                    "batch_idx": batch_idx,
                    "mse": float(metrics['mse']),
                    "ssim": float(metrics['ssim']),
                    "mse_t1n": float(metrics['mse_t1n']),
                    "mse_t1c": float(metrics['mse_t1c']),
                    "mse_t2w": float(metrics['mse_t2w']),
                    "mse_t2f": float(metrics['mse_t2f']),
                    "ssim_t1n": float(metrics['ssim_t1n']),
                    "ssim_t1c": float(metrics['ssim_t1c']),
                    "ssim_t2w": float(metrics['ssim_t2w']),
                    "ssim_t2f": float(metrics['ssim_t2f']),
                    "reconstruction": wandb.Image(str(panel_path)),
                }
                
                # Add wavelet visualizations if available (use pre-computed wavelets from batch)
                if has_wavelets and input_wavelets is not None:
                    # Use the wavelets already computed for this batch (no recomputation needed)
                    # Save wavelets using our utility
                    save_slice_outputs(
                        patient_dir=patient_dir,
                        inputs=inputs[i],
                        target=targets[i],
                        output=outputs[i],
                        slice_idx=slice_idx,
                        patient_id=patient_id,
                        batch_idx=batch_idx,
                        input_wavelets=input_wavelets[i],
                        output_wavelets=output_wavelets[i],
                        target_wavelets=target_wavelets[i]
                    )
                    
                    # Create wavelet visualizations
                    # visualize_wavelet_decomposition(
                    #     input_wavelets[i],
                    #     f'Input Wavelets - {patient_id}',
                    #     patient_dir / 'wavelet_input_viz.png'
                    # )
                    # visualize_wavelet_decomposition(
                    #     output_wavelets[i],
                    #     f'Output Wavelets - {patient_id}',
                    #     patient_dir / 'wavelet_output_viz.png'
                    # )
                    # visualize_wavelet_decomposition(
                    #     target_wavelets[i],
                    #     f'Target Wavelets - {patient_id}',
                    #     patient_dir / 'wavelet_target_viz.png'
                    # )
                    
                    # Add to table row
                    #table_row['wavelet_input'] = wandb.Image(str(patient_dir / 'wavelet_input_viz.png'))
                    #table_row['wavelet_output'] = wandb.Image(str(patient_dir / 'wavelet_output_viz.png'))
                    #table_row['wavelet_target'] = wandb.Image(str(patient_dir / 'wavelet_target_viz.png'))
                    
                    # Debug: Log on first few samples to confirm wavelets are being processed
                    #if batch_idx < 3 and i == 0:
                    #    print(f"✓ Batch {batch_idx}: Added wavelet images to table for {patient_id} (slice {slice_idx})")
                else:
                    # Debug: Log why wavelets are missing (only for first batch to avoid spam)
                    if batch_idx == 0 and i == 0:
                        print(f"⚠ Batch {batch_idx}: No wavelets added (has_wavelets={has_wavelets}, input_wavelets is None={input_wavelets is None})")
                    
                    table_row['wavelet_input'] = None
                    table_row['wavelet_output'] = None
                    table_row['wavelet_target'] = None
                
                # Add row to table data
                table_data.append(table_row)
            
            metric_time = perf_counter() - metric_start
            timing_stats.add_metric_time(metric_time)
            
            # Log running metrics every 10 batches - FIXED naming
            if batch_idx % 10 == 0 and batch_idx > 0:
                current_timing = timing_stats.get_stats()
                wandb.log({
                    "eval/running_mse_t1n": np.mean(running_mse_per_mod['t1n']),
                    "eval/running_mse_t1c": np.mean(running_mse_per_mod['t1c']),
                    "eval/running_mse_t2w": np.mean(running_mse_per_mod['t2w']),
                    "eval/running_mse_t2f": np.mean(running_mse_per_mod['t2f']),
                    "eval/running_ssim_t1n": np.mean(running_ssim_per_mod['t1n']),
                    "eval/running_ssim_t1c": np.mean(running_ssim_per_mod['t1c']),
                    "eval/running_ssim_t2w": np.mean(running_ssim_per_mod['t2w']),
                    "eval/running_ssim_t2f": np.mean(running_ssim_per_mod['t2f']),
                    "eval/progress": batch_idx / len(data_loader),
                    "eval/timing/avg_forward_time_ms": current_timing['avg_forward_time_ms'],
                    "eval/timing/samples_per_second": current_timing['samples_per_second']
                })
    
    # Get final timing statistics
    final_timing = timing_stats.get_stats()
    
    print(f"\n" + "="*60)
    print("EVALUATION TIMING STATISTICS")
    print("="*60)
    print(f"Total samples evaluated: {final_timing['total_samples']}")
    print(f"Average forward time: {final_timing['avg_forward_time_ms']:.2f}ms per batch")
    print(f"Average data load time: {final_timing['avg_data_load_time_ms']:.2f}ms per batch")
    print(f"Average metric calc time: {final_timing['avg_metric_time_ms']:.2f}ms per batch")
    if final_timing['avg_wavelet_time_ms'] > 0:
        print(f"Average wavelet time: {final_timing['avg_wavelet_time_ms']:.2f}ms per batch")
    print(f"Evaluation throughput: {final_timing['samples_per_second']:.1f} samples/second")
    print(f"Total evaluation time: {final_timing['total_forward_time_s']:.1f} seconds")
    print("="*60)
    
    # Log final timing to wandb
    wandb.log({
        "eval/final_timing/avg_forward_time_ms": final_timing['avg_forward_time_ms'],
        "eval/final_timing/avg_data_load_time_ms": final_timing['avg_data_load_time_ms'],
        "eval/final_timing/avg_metric_time_ms": final_timing['avg_metric_time_ms'],
        "eval/final_timing/samples_per_second": final_timing['samples_per_second'],
        "eval/final_timing/total_evaluation_time_s": final_timing['total_forward_time_s'],
        "eval/final_timing/total_samples": final_timing['total_samples'],
    })
    
    if final_timing['avg_wavelet_time_ms'] > 0:
        wandb.log({
            "eval/final_timing/avg_wavelet_time_ms": final_timing['avg_wavelet_time_ms'],
            "eval/final_timing/total_wavelet_time_s": final_timing['total_wavelet_time_s'],
        })
    
    # Create and log W&B table with all predictions
    print(f"\n" + "="*60)
    print(f"Creating W&B table with {len(table_data)} samples...")
    
    # Debug: Check wavelet data availability
    samples_with_wavelets = sum(1 for row in table_data if row.get('wavelet_input') is not None)
    print(f"Samples with wavelet data: {samples_with_wavelets}/{len(table_data)}")
    print("="*60)
    
    if wandb.run is not None and len(table_data) > 0:
        # Define table columns
        columns = [
            "patient_id", "slice_idx", "batch_idx",
            "mse", "ssim",
            "mse_t1n", "mse_t1c", "mse_t2w", "mse_t2f",
            "ssim_t1n", "ssim_t1c", "ssim_t2w", "ssim_t2f",
            "reconstruction"
        ]
        
        # Add wavelet columns if we have wavelet data
        has_wavelet_data = any(row.get('wavelet_input') is not None for row in table_data)
        if has_wavelet_data:
            columns.extend(["wavelet_input", "wavelet_output", "wavelet_target"])
            print(f"✓ Adding wavelet columns to table (found {samples_with_wavelets} samples with wavelets)")
        else:
            print(f"⚠ No wavelet columns will be added (no samples have wavelet data)")
        
        # Build table data rows
        table_rows = []
        for row in table_data:
            table_row = [
                row["patient_id"],
                row["slice_idx"],
                row["batch_idx"],
                row["mse"],
                row["ssim"],
                row["mse_t1n"],
                row["mse_t1c"],
                row["mse_t2w"],
                row["mse_t2f"],
                row["ssim_t1n"],
                row["ssim_t1c"],
                row["ssim_t2w"],
                row["ssim_t2f"],
                row["reconstruction"]
            ]
            
            # Add wavelet images if columns exist
            if has_wavelet_data:
                table_row.extend([
                    row.get("wavelet_input"),
                    row.get("wavelet_output"),
                    row.get("wavelet_target")
                ])
            
            table_rows.append(table_row)
        
        # Create and log table
        predictions_table = wandb.Table(columns=columns, data=table_rows)
        wandb.log({"eval/predictions_table": predictions_table})
        print(f"✓ Logged predictions table with {len(table_data)} samples to W&B")
    
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
    
    # Per-modality stats - FIXED naming
    for mod in ['t1n', 't1c', 't2w', 't2f']:
        mse_mod = [m[f'mse_{mod}'] for m in all_metrics]
        ssim_mod = [m[f'ssim_{mod}'] for m in all_metrics]
        results[f'mse_{mod}_mean'] = np.mean(mse_mod)
        results[f'mse_{mod}_std'] = np.std(mse_mod)
        results[f'ssim_{mod}_mean'] = np.mean(ssim_mod)
        results[f'ssim_{mod}_std'] = np.std(ssim_mod)
    
    # Add timing results to the evaluation results
    results.update({
        'eval_time_seconds': final_timing['total_forward_time_s'],
        'eval_throughput_samples_per_sec': final_timing['samples_per_second'],
        'avg_forward_time_ms': final_timing['avg_forward_time_ms'],
    })
    
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


def print_results(results):
    """Print evaluation results to console - FIXED naming"""
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"MSE:  {results['mse_mean']:.6f} ± {results['mse_std']:.6f}")
    print(f"SSIM: {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")
    print(f"Samples evaluated: {results['num_samples']}")
    print(f"Evaluation time: {results.get('eval_time_seconds', 0):.1f} seconds")
    print(f"Throughput: {results.get('eval_throughput_samples_per_sec', 0):.1f} samples/sec")
    print("\nPer-modality MSE:")
    print(f"  T1n:   {results['mse_t1n_mean']:.6f} ± {results['mse_t1n_std']:.6f}")
    print(f"  T1c:   {results['mse_t1c_mean']:.6f} ± {results['mse_t1c_std']:.6f}")
    print(f"  T2w:   {results['mse_t2w_mean']:.6f} ± {results['mse_t2w_std']:.6f}")
    print(f"  T2f:   {results['mse_t2f_mean']:.6f} ± {results['mse_t2f_std']:.6f}")
    print("\nPer-modality SSIM:")
    print(f"  T1n:   {results['ssim_t1n_mean']:.4f} ± {results['ssim_t1n_std']:.4f}")
    print(f"  T1c:   {results['ssim_t1c_mean']:.4f} ± {results['ssim_t1c_std']:.4f}")
    print(f"  T2w:   {results['ssim_t2w_mean']:.4f} ± {results['ssim_t2w_std']:.4f}")
    print(f"  T2f:   {results['ssim_t2f_mean']:.4f} ± {results['ssim_t2f_std']:.4f}")
    print("="*50)


def run_evaluation(model, data_loader, device, output_dir, model_type, wavelet_name, save_wavelets=True):
    """
    Main evaluation function that can be called from other scripts
    WITH COMPLETE TIMING STATS AND FIXED WAVELET VISUALIZATION
    
    Args:
        model: trained model
        data_loader: DataLoader for validation set
        device: cuda or cpu
        output_dir: where to save results
        model_type: type of model ('swin', 'wavelet', etc.)
        wavelet_name: wavelet type (for wavelet models)
        save_wavelets: whether to save wavelet coefficients
    
    Returns:
        results: dict with aggregated metrics
        all_metrics: list of per-sample metrics
    """
    # Run evaluation
    results, all_metrics = evaluate_model(model, data_loader, device, output_dir, save_wavelets)
    
    # Log comprehensive results to wandb - FIXED naming
    wandb.log({
        # Overall metrics
        "eval/mse_mean": results['mse_mean'],
        "eval/mse_std": results['mse_std'],
        "eval/ssim_mean": results['ssim_mean'],
        "eval/ssim_std": results['ssim_std'],
        "eval/num_samples": results['num_samples'],
        
        # Per-modality MSE
        "eval/mse_t1n_mean": results['mse_t1n_mean'],
        "eval/mse_t1n_std": results['mse_t1n_std'],
        "eval/mse_t1c_mean": results['mse_t1c_mean'],
        "eval/mse_t1c_std": results['mse_t1c_std'],
        "eval/mse_t2w_mean": results['mse_t2w_mean'],
        "eval/mse_t2w_std": results['mse_t2w_std'],
        "eval/mse_t2f_mean": results['mse_t2f_mean'],
        "eval/mse_t2f_std": results['mse_t2f_std'],
        
        # Per-modality SSIM
        "eval/ssim_t1n_mean": results['ssim_t1n_mean'],
        "eval/ssim_t1n_std": results['ssim_t1n_std'],
        "eval/ssim_t1c_mean": results['ssim_t1c_mean'],
        "eval/ssim_t1c_std": results['ssim_t1c_std'],
        "eval/ssim_t2w_mean": results['ssim_t2w_mean'],
        "eval/ssim_t2w_std": results['ssim_t2w_std'],
        "eval/ssim_t2f_mean": results['ssim_t2f_mean'],
        "eval/ssim_t2f_std": results['ssim_t2f_std'],
    })
    
    # Create per-modality comparison chart - FIXED naming
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    modalities = ['T1n', 'T1c', 'T2w', 'T2f']
    
    # MSE comparison
    mse_values = [results[f'mse_{m.lower()}_mean'] for m in modalities]
    mse_stds = [results[f'mse_{m.lower()}_std'] for m in modalities]
    axes[0].bar(modalities, mse_values, yerr=mse_stds, capsize=5, alpha=0.7)
    axes[0].set_ylabel('MSE')
    axes[0].set_title('MSE per Modality')
    axes[0].grid(True, alpha=0.3)
    
    # SSIM comparison
    ssim_values = [results[f'ssim_{m.lower()}_mean'] for m in modalities]
    ssim_stds = [results[f'ssim_{m.lower()}_std'] for m in modalities]
    axes[1].bar(modalities, ssim_values, yerr=ssim_stds, capsize=5, alpha=0.7, color='green')
    axes[1].set_ylabel('SSIM')
    axes[1].set_title('SSIM per Modality')
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    chart_path = Path(output_dir) / 'per_modality_comparison.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    wandb.log({"eval/per_modality_comparison": wandb.Image(str(chart_path))})
    plt.close()
    
    # Log example wavelet visualizations to wandb
    if save_wavelets:
        wavelet_viz_dir = Path(output_dir) / 'wavelet_visualizations'
        # Log input wavelet visualizations (match any patient-specific file)
        for img_path in sorted(wavelet_viz_dir.glob('*input_wavelets_ALL8*.png')):
            wandb.log({f"wavelets/{img_path.stem}": wandb.Image(str(img_path))})
        # Log output and target wavelets (patient-specific or batch-named)
        for img_path in sorted(wavelet_viz_dir.glob('*output_wavelets*.png'))[:6]:
            wandb.log({f"wavelets/{img_path.stem}": wandb.Image(str(img_path))})
        for img_path in sorted(wavelet_viz_dir.glob('*target_wavelets*.png'))[:6]:
            wandb.log({f"wavelets/{img_path.stem}": wandb.Image(str(img_path))})


def load_dataset_and_dataloader(args, device):
    """Helper function to load dataset and create dataloader.
    
    Args:
        args: Argument namespace with data configuration
        device: Device to use for pin_memory optimization
        
    Returns:
        tuple: (dataset, data_loader)
    """
    print("Loading dataset...")
    if args.preprocessed_dir:
        print(f"Using preprocessed dataset: {args.preprocessed_dir}")
        dataset = FastTensorSliceDataset(preprocessed_dir=args.preprocessed_dir)
        print(f"Loaded {len(dataset)} preprocessed samples")
    else:
        print(f"Using raw BraTS dataset: {args.data_dir}")
        dataset = BraTS2D5Dataset(
            data_dir=args.data_dir,
            image_size=(args.img_size, args.img_size),
            spacing=(1.0, 1.0, 1.0),
            num_patients=args.num_patients,
            cache_size=50
        )
        print(f"Loaded {len(dataset)} raw BraTS slices")
    
    pin_memory = device.type != 'cpu'
    num_workers = 0 if device.type == 'cpu' else 4
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return dataset, data_loader


def load_model(checkpoint_path, model_type, wavelet_name, img_size, device):
    """Load trained model - now supports all model types from training"""
    from monai.networks.nets import UNETR, BasicUNet
    
    # Determine if we need wavelet wrapper (use training logic)
    use_wavelet = wavelet_name != 'none'
    
    # Create base model based on architecture type
    if model_type == 'swin':
        base_model = SwinUNETR(
            in_channels=8,
            out_channels=4,
            feature_size=24,
            spatial_dims=2
        )
        print("Loaded Swin-UNETR model")
    
    elif model_type == 'unet':
        base_model = BasicUNet(
            spatial_dims=2,
            in_channels=8,
            out_channels=4,
            features=(32, 32, 64, 128, 256, 32),
            act='ReLU',
            norm='batch',
            dropout=0.0
        )
        print("Loaded BasicUNet model")
    
    elif model_type == 'unetr':
        base_model = UNETR(
            in_channels=8,
            out_channels=4,
            img_size=(img_size, img_size),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            proj_type='conv',
            norm_name='instance',
            res_block=True,
            dropout_rate=0.0,
            spatial_dims=2
        )
        print("Loaded UNETR model")
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Supported: 'swin', 'unet', 'unetr'")
    
    # Apply wavelet wrapper if used during training
    if use_wavelet:
        from models.wavelet_wrapper import WaveletWrapper
        model = WaveletWrapper(
            base_model=base_model,
            wavelet_name=wavelet_name,
            in_channels=8,
            out_channels=4
        ).to(device)
        print(f"Applied wavelet wrapper: {wavelet_name}")
    else:
        model = base_model.to(device)
        print("Using standard spatial domain processing (no wavelet)")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate middleslice reconstruction model")
    parser.add_argument('--checkpoint', type=str, required=False, default=None,
                       help='Path to model checkpoint (local). If not provided, use --wandb_run_id or --wandb_sweep_id')
    # NEW: W&B integration arguments
    parser.add_argument('--wandb_sweep_id', type=str, default=None,
                       help='W&B sweep ID to evaluate all checkpoints (e.g., "5mfl25i8")')
    parser.add_argument('--wandb_run_id', type=str, default=None,
                       help='W&B run ID to evaluate single checkpoint')
    parser.add_argument('--download_dir', type=str, default='./wandb_checkpoints',
                       help='Directory to cache downloaded checkpoints')
    parser.add_argument('--wandb_entity', type=str, default='timgsereda',
                       help='W&B entity/username')
    parser.add_argument('--wandb_project', type=str, default='brats-middleslice-wavelet-sweep',
                       help='W&B project name')
    parser.add_argument('--data_dir', type=str, required=False, default=None,
                       help='Path to BraTS dataset (for raw data)')
    parser.add_argument('--preprocessed_dir', type=str, required=False, default=None,
                       help='Path to preprocessed .pt files (faster, recommended)')
    parser.add_argument('--output', type=str, default='./results/swin',
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--num_patients', type=int, default=None,
                       help='Number of patients to evaluate (default: all)')
    parser.add_argument('--img_size', type=int, default=256,
                       help='Image size')
    parser.add_argument('--model_type', type=str, default='swin',
                       choices=['swin', 'unet', 'unetr'],
                       help='Model architecture')
    parser.add_argument('--wavelet', type=str, default='none',
                       choices=['none', 'haar', 'db2'],
                       help='Wavelet type (use "none" for standard spatial domain)')
    parser.add_argument('--no_save_wavelets', action='store_true',
                       help='Disable saving wavelet coefficients and visualizations')
    parser.add_argument('--timing_only', action='store_true',
                       help='Only measure timing, skip full evaluation')
    parser.add_argument('--test_mode', action='store_true',
                       help='Quick validation: 2 patients, reduced settings, auto-optimized')
    return parser.parse_args()


def setup_device_and_optimizations(args):
    """Setup device (always auto) and apply optimizations"""
    
    # Always auto-detect device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name()
        print(f"[DEVICE] Auto-detected: GPU ({gpu_name})")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"[DEVICE] Auto-detected: Apple Silicon MPS")
    else:
        device = torch.device('cpu')
        print(f"[DEVICE] Auto-detected: CPU ({torch.get_num_threads()} threads)")
    
    # Apply optimizations
    if device.type == 'cpu':
        apply_cpu_optimizations(args)
    
    if args.test_mode:
        apply_test_mode_settings(args, device)
        
    return device

def apply_cpu_optimizations(args):
    """Apply CPU optimizations only when needed"""
    total_cores = torch.get_num_threads()
    
    # Only optimize threads if PyTorch is using too many
    if total_cores > 8:  # Only intervene for high-core systems
        optimal_threads = max(4, total_cores // 2)
        torch.set_num_threads(optimal_threads)
        print(f"[CPU] Optimized threads: {total_cores} -> {optimal_threads}")
    
    # Conservative batch sizes for CPU
    original_batch = args.batch_size
    if args.batch_size > 4:
        args.batch_size = 4
        print(f"[CPU] Reduced batch_size: {original_batch} -> {args.batch_size}")
    
    # Warn about wavelets if being used
    if hasattr(args, 'wavelet') and args.wavelet != 'none':
        print(f"[WARNING] {args.wavelet} wavelets will be significantly slower on CPU")
        print(f"[TIP] Consider: --wavelet none for CPU evaluation")

def apply_test_mode_settings(args, device):
    """Apply test mode settings for evaluation"""
    print(f"[TEST] Test mode activated")
    
    # Force small dataset
    original_patients = getattr(args, 'num_patients', 'all')
    args.num_patients = 2
    print(f"[TEST] Patients: {original_patients} -> 2")
    
    # Conservative batch sizes for test mode
    original_batch = args.batch_size
    if device.type == 'cpu':
        args.batch_size = min(2, args.batch_size)
    else:
        args.batch_size = min(4, args.batch_size)
    
    if original_batch != args.batch_size:
        print(f"[TEST] Batch size: {original_batch} -> {args.batch_size}")
    
    # Time estimates
    if device.type == 'cpu':
        print(f"[TEST] Estimated time: 5-15 minutes (CPU)")
    else:
        print(f"[TEST] Estimated time: 1-3 minutes (GPU)")

def measure_timing_only(model, data_loader, device, num_batches=100):
    """
    Quick timing measurement without full evaluation
    """
    print(f"Running timing-only evaluation over {num_batches} batches...")
    
    model.eval()
    timing_stats = EvaluationTimingStats()
    
    with torch.no_grad():
        for i, (inputs, targets, _) in enumerate(data_loader):
            if i >= num_batches:
                break
            
            # Data loading time
            data_start = perf_counter()
            inputs = inputs.to(device)
            targets = targets.to(device)
            data_time = perf_counter() - data_start
            timing_stats.add_data_load_time(data_time)
            
            # Forward pass timing
            torch.cuda.synchronize() if device.type == 'cuda' else None
            forward_start = perf_counter()
            
            # Time wavelet transform if applicable
            if hasattr(model, 'dwt2d_batch'):
                wavelet_start = perf_counter()
                _ = model.dwt2d_batch(inputs)
                torch.cuda.synchronize() if device.type == 'cuda' else None
                wavelet_time = perf_counter() - wavelet_start
                timing_stats.add_wavelet_time(wavelet_time)
            
            # Full forward pass
            outputs = model(inputs)
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            forward_time = perf_counter() - forward_start
            timing_stats.add_forward_time(forward_time)
            timing_stats.add_samples(inputs.shape[0])
            
            if i % 20 == 0:
                current_stats = timing_stats.get_stats()
                print(f"Batch {i}/{num_batches}: {current_stats['avg_forward_time_ms']:.2f}ms/batch, "
                      f"{current_stats['samples_per_second']:.1f} samples/sec")
    
    return timing_stats.get_stats()


def main():
    """CLI entry point"""
    args = get_args()
    
    # Validate dataset arguments
    if not args.preprocessed_dir and not args.data_dir:
        raise ValueError("Must provide either --preprocessed_dir or --data_dir")
    if args.preprocessed_dir and args.data_dir:
        print("Warning: Both --preprocessed_dir and --data_dir provided. Using --preprocessed_dir (faster)")

    # If user asked to evaluate an entire training sweep, download and evaluate all "best" artifacts
    if args.wandb_sweep_id:
        print(f"Evaluating all checkpoints from sweep: {args.wandb_sweep_id}")
        checkpoints = download_checkpoint_from_wandb(
            sweep_id=args.wandb_sweep_id,
            download_dir=args.download_dir,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project
        )

        # Setup device once
        device = setup_device_and_optimizations(args)
        
        # Load dataset once before the loop
        dataset, data_loader = load_dataset_and_dataloader(args, device)

        all_results = []
        for ckpt_info in checkpoints:
            print(f"\n{'='*60}")
            print(f"Evaluating: {ckpt_info['run_name']}")
            print(f"Config: {ckpt_info['config']}")
            print(f"{'='*60}")

            # set checkpoint path and derive model config
            args.checkpoint = ckpt_info['path']
            model_type = ckpt_info['config'].get('model_type', 'swin')
            wavelet = ckpt_info['config'].get('wavelet', 'none')

            # Initialize a distinct W&B run per evaluation
            run_name = f"eval_{ckpt_info['run_name']}"
            # Prefer project name from checkpoint config, fall back to CLI argument
            project_name = ckpt_info['config'].get('wandb_project', args.wandb_project)
            wandb.init(
                project=project_name,
                name=run_name,
                config=ckpt_info['config'],
                tags=['evaluation', 'sweep_eval', f"sweep_{args.wandb_sweep_id}"],
                group=f"sweep_{args.wandb_sweep_id}"
            )

            # Load model and run evaluation
            print(f"Loading model from {args.checkpoint}...")
            model = load_model(args.checkpoint, model_type, wavelet, args.img_size, device)

            output_dir = f"{args.output}/{ckpt_info['run_id']}"
            results, _ = run_evaluation(
                model=model,
                data_loader=data_loader,
                device=device,
                output_dir=output_dir,
                model_type=model_type,
                wavelet_name=wavelet,
                save_wavelets=(wavelet != 'none')
            )

            # record metadata
            results['run_id'] = ckpt_info['run_id']
            results['run_name'] = ckpt_info['run_name']
            results['model_type'] = model_type
            results['wavelet'] = wavelet
            all_results.append(results)

            print_results(results)
            wandb.finish()

        # Summarize sweep results and write CSV
        print(f"\n{'='*60}")
        print("SWEEP EVALUATION SUMMARY")
        print(f"{'='*60}")
        
        # Validate all results have consistent keys before creating DataFrame
        if all_results:
            required_keys = {'run_name', 'model_type', 'wavelet', 'mse_mean', 'ssim_mean'}
            for i, result in enumerate(all_results):
                missing_keys = required_keys - set(result.keys())
                if missing_keys:
                    print(f"Warning: Result {i} missing keys: {missing_keys}")
        
        df = pd.DataFrame(all_results)
        print(df[['run_name', 'model_type', 'wavelet', 'mse_mean', 'ssim_mean']])

        comparison_path = Path(args.output) / f"sweep_{args.wandb_sweep_id}_comparison.csv"
        Path(args.output).mkdir(parents=True, exist_ok=True)
        df.to_csv(comparison_path, index=False)
        print(f"\nComparison saved to: {comparison_path}")
        return

    # Single-run download via --wandb_run_id
    if args.wandb_run_id:
        print(f"Downloading checkpoint from run: {args.wandb_run_id}")
        args.checkpoint = download_checkpoint_from_wandb(
            run_id=args.wandb_run_id,
            download_dir=args.download_dir,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project
        )
        # Try extract training run config
        try:
            api = wandb.Api()
            run = api.run(f"{args.wandb_entity}/{args.wandb_project}/{args.wandb_run_id}")
            # prefer run config where available, but do not override explicit CLI args
            if not hasattr(args, 'model_type') or getattr(args, 'model_type') is None:
                args.model_type = run.config.get('model_type', 'swin')
            if not hasattr(args, 'wavelet') or getattr(args, 'wavelet') is None:
                args.wavelet = run.config.get('wavelet', 'none')
            if not hasattr(args, 'img_size') or getattr(args, 'img_size') is None:
                args.img_size = run.config.get('img_size', 256)
            print(f"Model config: {args.model_type}, wavelet={args.wavelet}")
        except Exception as e:
            print(f"Warning: couldn't fetch run metadata: {e}")

    # Continue normal single-checkpoint evaluation (local checkpoint or downloaded run checkpoint)
    # Device + optimizations
    device = setup_device_and_optimizations(args)

    # Initialize W&B (if not already initialized above for per-run loop)
    run_name = f"eval_{args.model_type}_{args.wavelet if args.wavelet != 'none' else 'baseline'}"
    wandb_config = vars(args)
    wandb_config['device_type'] = device.type
    wandb_tags = ['evaluation']
    if args.test_mode:
        wandb_tags.append('test_mode')
    wandb.init(project=args.wandb_project, name=run_name, config=wandb_config, tags=wandb_tags)

    print(f"Using device: {device}")
    # Load dataset and dataloader using helper function
    dataset, data_loader = load_dataset_and_dataloader(args, device)

    # Load model
    if not args.checkpoint:
        raise ValueError("No checkpoint provided. Use --checkpoint, --wandb_run_id or --wandb_sweep_id.")
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.model_type, args.wavelet, args.img_size, device)

    # Decide about wavelet saving (wavelets are used when wavelet != 'none')
    is_wavelet_model = args.wavelet != 'none'
    save_wavelets = (not args.no_save_wavelets) and is_wavelet_model
    if (not args.no_save_wavelets) and not is_wavelet_model:
        print("Info: No wavelet processing enabled (wavelet='none'). Wavelet visualizations will be skipped.")


    # Timing-only mode
    if args.timing_only:
        timing_results = measure_timing_only(model, data_loader, device)
        print(f"\n" + "="*60)
        print("TIMING-ONLY RESULTS")
        print("="*60)
        print(f"Average forward time: {timing_results['avg_forward_time_ms']:.2f}ms per batch")
        print(f"Average data load time: {timing_results['avg_data_load_time_ms']:.2f}ms per batch")
        if timing_results['avg_wavelet_time_ms'] > 0:
            print(f"Average wavelet time: {timing_results['avg_wavelet_time_ms']:.2f}ms per batch")
        print(f"Throughput: {timing_results['samples_per_second']:.1f} samples/second")
        print(f"Total samples: {timing_results['total_samples']}")
        print("="*60)
        wandb.log({
            "timing_only/avg_forward_time_ms": timing_results['avg_forward_time_ms'],
            "timing_only/avg_data_load_time_ms": timing_results['avg_data_load_time_ms'],
            "timing_only/samples_per_second": timing_results['samples_per_second'],
            "timing_only/total_samples": timing_results['total_samples'],
        })
        if timing_results['avg_wavelet_time_ms'] > 0:
            wandb.log({"timing_only/avg_wavelet_time_ms": timing_results['avg_wavelet_time_ms']})
        wandb.finish()
        return

    # Run full evaluation
    print("Running full evaluation with timing statistics...")
    results, all_metrics = run_evaluation(
        model=model,
        data_loader=data_loader,
        device=device,
        output_dir=args.output,
        model_type=args.model_type,
        wavelet_name=args.wavelet,
        save_wavelets=save_wavelets
    )
    
    # Save results to CSV
    save_results(results, all_metrics, args.output)
    
    print(f"\nResults saved to {args.output}/")
    if save_wavelets:
        print(f"Wavelet coefficients saved to {args.output}/wavelets/")
        print(f"Wavelet visualizations saved to {args.output}/wavelet_visualizations/")
        print(f"  -> Now showing ALL 8 input channels (Z-1 and Z+1 slices)")
    
    print_results(results)
    wandb.finish()


if __name__ == '__main__':
    main()