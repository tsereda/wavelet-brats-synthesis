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
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import wandb
from time import perf_counter

from train import BraTS2D5Dataset
from monai.networks.nets import SwinUNETR
from utils import extract_patient_info, get_patient_output_dir, save_slice_outputs
from logging_utils import create_reconstruction_log_panel


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
            im = ax.imshow(subband, cmap='Reds', vmin=vmin, vmax=vmax)
            ax.set_title(f'{mod_name} - {name}', fontsize=10)
            ax.axis('off')
            if i == 3:
                plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    plt.close()


def calculate_metrics(prediction, ground_truth):
    """
    Calculate MSE and SSIM between prediction and ground truth
    FIXED: Uses stable data_range for SSIM calculation
    FIXED: Updated to BraTS2023 GLI naming (t1n, t1c, t2w, t2f)
    
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
    # FIXED: Use fixed data_range since inputs are normalized to [0, 1]
    ssim_scores = []
    for i in range(4):  # 4 modalities
        try:
            score = ssim(
                gt_np[i], 
                pred_np[i], 
                data_range=1.0  # Fixed range for normalized data
            )
            ssim_scores.append(score)
        except Exception as e:
            print(f"Warning: SSIM calculation failed for modality {i}: {e}")
            ssim_scores.append(0.0)  # Fallback value
    
    ssim_avg = np.mean(ssim_scores)
    
    return {
        'mse': mse_avg,
        'ssim': ssim_avg,
        'mse_t1n': mse_per_modality[0],   # FIXED: Changed from mse_t1
        'mse_t1c': mse_per_modality[1],   # FIXED: Changed from mse_t1ce
        'mse_t2w': mse_per_modality[2],   # FIXED: Changed from mse_t2
        'mse_t2f': mse_per_modality[3],   # FIXED: Changed from mse_flair
        'ssim_t1n': ssim_scores[0],       # FIXED: Changed from ssim_t1
        'ssim_t1c': ssim_scores[1],       # FIXED: Changed from ssim_t1ce
        'ssim_t2w': ssim_scores[2],       # FIXED: Changed from ssim_t2
        'ssim_t2f': ssim_scores[3],       # FIXED: Changed from ssim_flair
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
            if save_wavelets and hasattr(model, 'dwt2d_batch'):
                # Time wavelet computation only for first 10 batches (for performance metrics)
                if batch_idx < 10:
                    wavelet_start = perf_counter()
                
                # But ALWAYS compute wavelets for all batches (needed for W&B table)
                # Get wavelet decomposition of input (ALL 8 CHANNELS)
                input_wavelets = model.dwt2d_batch(inputs)
                
                # Get wavelet decomposition of output
                output_wavelets = model.dwt2d_batch(outputs)
                
                # Get wavelet decomposition of ground truth
                target_wavelets = model.dwt2d_batch(targets)
                
                # Complete timing for first 10 batches
                if batch_idx < 10:
                    torch.cuda.synchronize() if device.type == 'cuda' else None
                    wavelet_time = perf_counter() - wavelet_start
                    timing_stats.add_wavelet_time(wavelet_time)
                    
                    # Save first sample in batch using patient-specific directory (for detailed inspection)
                    sample_idx = 0
                    sample_slice_idx, sample_patient_id = extract_patient_info(slice_indices, batch_idx, sample_idx)
                    
                    # Create patient-specific subdirectory for wavelets
                    patient_wavelet_dir = get_patient_output_dir(predictions_dir, sample_patient_id, sample_slice_idx)
                    
                    # Save wavelet coefficients with clean filenames
                    np.save(
                        patient_wavelet_dir / 'wavelet_input.npy',
                        input_wavelets[sample_idx].cpu().numpy()
                    )
                    np.save(
                        patient_wavelet_dir / 'wavelet_output.npy',
                        output_wavelets[sample_idx].cpu().numpy()
                    )
                    np.save(
                        patient_wavelet_dir / 'wavelet_target.npy',
                        target_wavelets[sample_idx].cpu().numpy()
                    )

                    # Create wavelet visualizations with clean filenames
                    visualize_wavelet_decomposition(
                        input_wavelets[sample_idx],
                        f'Input Wavelet Decomposition - {sample_patient_id} - ALL 8 Channels',
                        patient_wavelet_dir / 'wavelet_input_viz.png'
                    )
                    visualize_wavelet_decomposition(
                        output_wavelets[sample_idx],
                        f'Output Wavelet Decomposition - {sample_patient_id}',
                        patient_wavelet_dir / 'wavelet_output_viz.png'
                    )
                    visualize_wavelet_decomposition(
                        target_wavelets[sample_idx],
                        f'Target Wavelet Decomposition - {sample_patient_id}',
                        patient_wavelet_dir / 'wavelet_target_viz.png'
                    )
            
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
                if save_wavelets and hasattr(model, 'dwt2d_batch'):
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
                    visualize_wavelet_decomposition(
                        input_wavelets[i],
                        f'Input Wavelets - {patient_id}',
                        patient_dir / 'wavelet_input_viz.png'
                    )
                    visualize_wavelet_decomposition(
                        output_wavelets[i],
                        f'Output Wavelets - {patient_id}',
                        patient_dir / 'wavelet_output_viz.png'
                    )
                    visualize_wavelet_decomposition(
                        target_wavelets[i],
                        f'Target Wavelets - {patient_id}',
                        patient_dir / 'wavelet_target_viz.png'
                    )
                    
                    # Add to table row
                    table_row['wavelet_input'] = wandb.Image(str(patient_dir / 'wavelet_input_viz.png'))
                    table_row['wavelet_output'] = wandb.Image(str(patient_dir / 'wavelet_output_viz.png'))
                    table_row['wavelet_target'] = wandb.Image(str(patient_dir / 'wavelet_target_viz.png'))
                else:
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
        if save_wavelets and any(row.get('wavelet_input') is not None for row in table_data):
            columns.extend(["wavelet_input", "wavelet_output", "wavelet_target"])
        
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
            
            # Add wavelet images if available
            if save_wavelets and any(r.get('wavelet_input') is not None for r in table_data):
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
    
    # Save results to CSV
    save_results(results, all_metrics, output_dir)
    
    print(f"\nResults saved to {output_dir}/")
    if save_wavelets:
        print(f"Wavelet coefficients saved to {output_dir}/wavelets/")
        print(f"Wavelet visualizations saved to {output_dir}/wavelet_visualizations/")
        print(f"  -> Now showing ALL 8 input channels (Z-1 and Z+1 slices)")
    
    return results, all_metrics


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
    parser.add_argument('--no_save_wavelets', action='store_true',
                       help='Disable saving wavelet coefficients and visualizations')
    parser.add_argument('--timing_only', action='store_true',
                       help='Only measure timing, skip full evaluation')
    return parser.parse_args()


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
        num_patients=args.num_patients,
        cache_size=50  # prevent OOM by limiting volume cache
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
    save_wavelets = (not args.no_save_wavelets) and is_wavelet_model
    if (not args.no_save_wavelets) and not is_wavelet_model:
        print("Warning: wavelet saving only works with wavelet models. Ignoring.")
    
    # Quick timing-only evaluation if requested
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
            wandb.log({
                "timing_only/avg_wavelet_time_ms": timing_results['avg_wavelet_time_ms']
            })
        
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
    
    # Print results
    print_results(results)
    
    wandb.finish()


if __name__ == '__main__':
    main()