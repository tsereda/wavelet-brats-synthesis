#!/usr/bin/env python3
"""
Complete evaluation pipeline for synthesis models.
Runs nnUNet segmentation and calculates Dice scores.

(MODIFIED TO INCLUDE WEIGHTS & BIASES LOGGING + VISUALIZATION)
"""

import os
import subprocess
import argparse
import nibabel as nib
import numpy as np
from pathlib import Path
import shutil
import wandb  # <-- ADDED
import matplotlib.pyplot as plt  # <-- ADDED

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Calculate Dice coefficient between two boolean masks."""
    # Ensure boolean
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    
    # Handle empty cases
    if not np.any(y_true) and not np.any(y_pred):
        return 1.0  # Both are empty, perfect match
    
    intersection = np.sum(y_true & y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

def calculate_brats_metrics(gt_data, pred_data):
    """
    Calculate BraTS region-specific Dice scores (ET, TC, WT).
    
    CORRECT (Standard nnUNet/BraTS) Labels:
    - 1: Necrotic/Non-Enhancing Core (NCR/NET)
    - 2: Peritumoral Edema (ED)
    - 3: Enhancing Tumor (ET)
    
    Evaluation Regions:
    - Enhancing Tumor (ET) = Label 3
    - Tumor Core (TC)      = Label 1 + Label 3
    - Whole Tumor (WT)     = Label 1 + Label 2 + Label 3
    """
    
    # Enhancing Tumor (ET) - Label 3
    gt_et = (gt_data == 3)
    pred_et = (pred_data == 3)
    dice_et = dice_coefficient(gt_et, pred_et)
    
    # Tumor Core (TC) - Labels 1 + 3  <-- THIS IS THE FIX
    gt_tc = np.logical_or(gt_data == 1, gt_data == 3)
    pred_tc = np.logical_or(pred_data == 1, pred_data == 3)
    dice_tc = dice_coefficient(gt_tc, pred_tc)
    
    # Whole Tumor (WT) - Labels 1 + 2 + 3
    # (gt_data > 0) is a safe shortcut for (1 | 2 | 3)
    gt_wt = (gt_data > 0)
    pred_wt = (pred_data > 0)
    dice_wt = dice_coefficient(gt_wt, pred_wt)
    
    return {
        "dice_et": float(dice_et),
        "dice_tc": float(dice_tc),
        "dice_wt": float(dice_wt)
    }

def fix_floating_point_labels(segmentation):
    """Fix floating point precision issues by rounding to nearest integers"""
    # Round to nearest integer and clip to valid range [0, 4]
    # BraTS labels are 0, 1, 2, 3. Clipping to 4 is safe.
    fixed_seg = np.round(segmentation).astype(np.int16)
    fixed_seg = np.clip(fixed_seg, 0, 4)
    return fixed_seg

# --- NEW FUNCTION ---
def find_best_slice(seg1, seg2):
    """Find slice with most non-background content in either segmentation"""
    # Combine masks to find the most representative slice
    combined_seg = np.logical_or(seg1 > 0, seg2 > 0)
    # Sum across x and y axes to get score per z-slice
    slice_scores = np.sum(combined_seg, axis=(0, 1))
    if np.sum(slice_scores) == 0:
        return seg1.shape[2] // 2 # Return middle slice if empty
    return np.argmax(slice_scores)

# --- NEW FUNCTION ---
def log_wandb_visualization(gt_data, pred_data, file_name):
    """Create and log a side-by-side visualization of GT and Pred slices."""
    try:
        best_slice = find_best_slice(gt_data, pred_data)
        
        gt_slice = gt_data[:, :, best_slice]
        pred_slice = pred_data[:, :, best_slice]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.imshow(gt_slice, cmap='jet', vmin=0, vmax=3)
        ax1.set_title(f"Ground Truth\n{file_name}")
        ax1.axis('off')
        
        ax2.imshow(pred_slice, cmap='jet', vmin=0, vmax=3)
        ax2.set_title(f"Prediction\n{file_name}")
        ax2.axis('off')
        
        plt.tight_layout()
        
        # Log the plot to wandb under "samples"
        wandb.log({f"samples/{Path(file_name).stem}": wandb.Image(fig)})
        
        plt.close(fig) # Close the figure to save memory
    except Exception as e:
        print(f"  [W&B] Error creating visualization for {file_name}: {e}")

# --- MODIFIED FUNCTION ---
def calculate_dice_scores(results_folder, ground_truth_folder, num_viz_samples=10):
    """Calculate Dice scores and log visualizations to W&B."""
    all_scores = {"dice_et": [], "dice_tc": [], "dice_wt": []}
    case_results = {}
    viz_counter = 0 # Counter for logging samples

    prediction_files = [f for f in os.listdir(results_folder) if f.endswith('.nii') or f.endswith('.nii.gz')]
    
    if not prediction_files:
        print(f"No segmentation files found in {results_folder}")
        return None, {}

    print(f"Found {len(prediction_files)} prediction files. Comparing with {ground_truth_folder}...")

    for file_name in prediction_files:
        result_path = os.path.join(results_folder, file_name)
        gt_path = os.path.join(ground_truth_folder, file_name)

        if os.path.exists(gt_path):
            try:
                result_img = nib.load(result_path)
                gt_img = nib.load(gt_path)

                result_data_raw = result_img.get_fdata()
                gt_data_raw = gt_img.get_fdata()

                result_data = fix_floating_point_labels(result_data_raw)
                gt_data = fix_floating_point_labels(gt_data_raw)

                metrics = calculate_brats_metrics(gt_data, result_data)
                
                case_results[file_name] = metrics
                
                all_scores["dice_et"].append(metrics["dice_et"])
                all_scores["dice_tc"].append(metrics["dice_tc"])
                all_scores["dice_wt"].append(metrics["dice_wt"])
                
                print(f"  {file_name}: ET={metrics['dice_et']:.4f}, TC={metrics['dice_tc']:.4f}, WT={metrics['dice_wt']:.4f}")

                # --- W&B VISUALIZATION LOGIC ---
                if wandb.run and viz_counter < num_viz_samples:
                    print(f"  [W&B] Logging visualization for {file_name}...")
                    log_wandb_visualization(gt_data, result_data, file_name)
                    viz_counter += 1
                # --- END W&B VISUALIZATION ---

            except Exception as e:
                print(f"Error processing {file_name}: {e}")
        else:
            print(f"Ground truth for {file_name} not found! (Looked for {gt_path})")

    if all_scores["dice_wt"]: 
        summary_stats = {}
        for region, scores in all_scores.items():
            if scores:
                summary_stats[region] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores))
                }
            else:
                summary_stats[region] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        print(f"\n📊 EVALUATION RESULTS:")
        for region, stats in summary_stats.items():
            print(f"  {region.upper()} Avg: {stats['mean']:.4f} ± {stats['std']:.4f} (Min: {stats['min']:.4f}, Max: {stats['max']:.4f})")
        print(f"Number of cases: {len(all_scores['dice_wt'])}")
        
        return summary_stats, case_results
    else:
        print("No valid segmentation files found for comparison.")
        return None, {}

# --- MODIFIED FUNCTION (v3) ---
def setup_nnunet_environment(local_model_path="3d_fullres"):
    """Setup nnUNet environment variables and move local model."""
    local_root = "./nnunet_data" 
    
    os.environ["nnUNet_raw_data_base"] = os.path.abspath(f"{local_root}/raw")
    os.environ["nnUNet_preprocessed"] = os.path.abspath(f"{local_root}/preprocessed") 
    os.environ["RESULTS_FOLDER"] = os.path.abspath(f"{local_root}/results")
    
    for path_key in ["nnUNet_raw_data_base", "nnUNet_preprocessed", "RESULTS_FOLDER"]: 
        path = os.environ[path_key]
        os.makedirs(path, exist_ok=True)
    
    nnunet_models_dir = Path(os.environ["RESULTS_FOLDER"]) / "nnUNet"
    os.makedirs(nnunet_models_dir, exist_ok=True)
    
    local_model_src = Path(local_model_path)
    local_model_dest = nnunet_models_dir / local_model_src.name # -> .../results/nnUNet/3d_fullres

    if local_model_src.exists() and local_model_src.is_dir():
        try:
            if local_model_dest.exists():
                print(f"Found existing model at {local_model_dest}, removing first.")
                shutil.rmtree(local_model_dest)
            
            print(f"Moving local model from {local_model_src} to {local_model_dest}...")
            shutil.move(str(local_model_src), str(local_model_dest))
            print(f"✅ Moved local model to {local_model_dest}")
        except Exception as e:
            print(f"⚠️  Could not move local model: {e}")
            print(f"    Ensure '{local_model_src}' exists and '{local_model_dest}' is writable.")
    elif local_model_dest.exists():
        print(f"✅ Local model already in place at {local_model_dest}")
    else:
        print(f"❌ Local model path '{local_model_src}' not found.")
        print("   Make sure you unzipped 'bratscp.zip' in the same directory as this script.")

    print("nnUNet environment variables set:")
    print(f"nnUNet_raw_data_base: {os.environ['nnUNet_raw_data_base']}")
    print(f"nnUNet_preprocessed: {os.environ['nnUNet_preprocessed']}")
    print(f"RESULTS_FOLDER: {os.environ['RESULTS_FOLDER']}")


# --- UNCHANGED FUNCTION ---
def download_nnunet_weights():
    """Bypassed: Using local nnUNet v1 model."""
    print("✅ Using local nnUNet v1 model. Skipping download.")
    return True

# --- UNCHANGED FUNCTION ---
def run_nnunet_prediction(dataset_dir, output_dir):
    """Run nnUNet (v1) prediction on the dataset."""
    
    print(f"Running nnUNet (v1) prediction...")
    
    input_dir_abs = os.path.abspath(f"{dataset_dir}/imagesTr")
    output_dir_abs = os.path.abspath(output_dir)
    
    print(f"Input: {input_dir_abs}")
    print(f"Output: {output_dir_abs}")
    
    cmd = [
        "nnUNet_predict",
        "-i", input_dir_abs,
        "-o", output_dir_abs,
        "-t", "Task082_BraTS2020",
        "-m", "3d_fullres",
        "-f", "0",
        "-tr", "nnUNetTrainerV2"
    ]
    
    try:
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=os.environ.copy()) 
        
        print("✅ nnUNet (v1) prediction completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ nnUNet (v1) prediction failed:")
        print(f"Return code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"❌ 'nnUNet_predict' command not found.")
        print("   Make sure nnUNet v1 (pip install nnunet) is installed and in your PATH.")
        return False

# --- MODIFIED FUNCTION ---
def main():
    parser = argparse.ArgumentParser(description="Evaluate synthesis models using segmentation performance")
    parser.add_argument("--dataset_dir", default="./Dataset137_BraTS21_Completed",
                       help="nnUNet format dataset directory")
    parser.add_argument("--output_dir", default="./segmentation_outputs",
                       help="Output directory for segmentation results")
    parser.add_argument("--skip_segmentation", action="store_true",
                       help="Skip segmentation and only calculate Dice scores")
    
    # --- NEW W&B ARGUMENTS ---
    parser.add_argument("--wandb_project", type=str, default="brats-synthesis-eval",
                        help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name (defaults to auto-generated)")
    parser.add_argument("--num_viz_samples", type=int, default=10,
                        help="Number of segmentation samples to log to W&B")
    # --- END W&B ARGUMENTS ---

    args = parser.parse_args()

    # --- W&B INIT ---
    try:
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args) # Log all command-line arguments
        )
        print(f"✅ W&B Run initialized: {run.url}")
    except Exception as e:
        print(f"❌ Could not initialize W&B: {e}")
        print("   Skipping W&B logging. Did you run 'wandb login'?")
        run = None # Set run to None to skip logging
    # --- END W&B INIT ---

    setup_nnunet_environment()
    
    if not os.path.exists(args.dataset_dir):
        print(f"❌ Dataset directory not found: {args.dataset_dir}")
        if run: run.finish(exit_code=1) # <-- ADDED
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not args.skip_segmentation:
        if not download_nnunet_weights():
            print("❌ Failed to setup nnUNet weights")
            if run: run.finish(exit_code=1) # <-- ADDED
            return
        
        if not run_nnunet_prediction(args.dataset_dir, args.output_dir):
            print("❌ nnUNet prediction failed")
            if run: run.finish(exit_code=1) # <-- ADDED
            return
    
    ground_truth_dir = f"{args.dataset_dir}/labelsTr"
    
    if not os.path.exists(ground_truth_dir):
        print(f"❌ Ground truth directory not found: {ground_truth_dir}")
        if run: run.finish(exit_code=1) # <-- ADDED
        return
    
    if not os.listdir(args.output_dir):
        print(f"❌ No segmentation results found in {args.output_dir}")
        if run: run.finish(exit_code=1) # <-- ADDED
        return
    
    print(f"\n🔍 Calculating Dice scores...")
    summary_stats, case_results = calculate_dice_scores(
        args.output_dir, 
        ground_truth_dir,
        args.num_viz_samples # Pass the new arg
    )
    
    # Save results
    if summary_stats: # Check if we have results
        results_file = "synthesis_evaluation_results.txt"
        with open(results_file, "w") as f:
            f.write("Synthesis Model Evaluation Results (Per-Region)\n")
            f.write("==================================================\n\n")
            f.write("Summary Statistics:\n")
            for region, stats in summary_stats.items():
                f.write(f"  {region.upper()} Avg: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                f.write(f"  {region.upper()} Min: {stats['min']:.4f}, Max: {stats['max']:.4f}\n")
            
            f.write(f"\nNumber of cases: {len(case_results)}\n")
            f.write("==================================================\n")
            f.write("\nPer-case results (Filename: ET, TC, WT):\n")
            for case, metrics in sorted(case_results.items()):
                f.write(f"  {case}: {metrics['dice_et']:.4f}, {metrics['dice_tc']:.4f}, {metrics['dice_wt']:.4f}\n")
        
        print(f"\n📄 Results saved to: {results_file}")
        
        # Summary
        print(f"\n🎯 FINAL RESULTS:")
        print(f"Your CWDM synthesis models achieved:")
        for region, stats in summary_stats.items():
            print(f"  Average {region.upper()} Dice: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"\nThis measures how well synthesized modalities preserve")
        print(f"segmentation-relevant information!")
        
        # --- W&B LOGGING ---
        if run:
            print("\n[W&B] Logging results to Weights & Biases...")
            
            # 1. Log summary statistics (flattened for easier plotting)
            summary_log = {}
            for region, stats in summary_stats.items():
                summary_log[f"avg_{region}_dice"] = stats['mean']
                summary_log[f"std_{region}_dice"] = stats['std']
                summary_log[f"min_{region}_dice"] = stats['min']
                summary_log[f"max_{region}_dice"] = stats['max']
            
            wandb.summary.update(summary_log)
            print("  [W&B] Logged summary statistics.")

            # 2. Log per-case results as a W&B Table
            table_columns = ["Filename", "DICE_ET", "DICE_TC", "DICE_WT"]
            table = wandb.Table(columns=table_columns)
            for case, metrics in sorted(case_results.items()):
                table.add_data(
                    case, 
                    metrics['dice_et'], 
                    metrics['dice_tc'], 
                    metrics['dice_wt']
                )
            run.log({"dice_results_per_case": table})
            print("  [W&B] Logged per-case results table.")
            
            # 3. Finish the run
            print(f"✅ W&B Run finished: {run.url}")
            run.finish()
        # --- END W&B LOGGING ---

if __name__ == "__main__":
    main()