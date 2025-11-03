"""
Complete evaluation pipeline for synthesis models.
Runs nnUNet segmentation and calculates Dice scores.
(MODIFIED TO USE LOCAL nnUNet v1 MODEL AND CORRECT ENV VARS + PATHS)
"""

import os
import subprocess
import argparse
import nibabel as nib
import numpy as np
from pathlib import Path
import shutil

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
    Assumes labels are: 1: Edema, 2: Necrosis, 3: Enhancing Tumor
    """
    # Enhancing Tumor (ET) - Label 3
    gt_et = (gt_data == 3)
    pred_et = (pred_data == 3)
    dice_et = dice_coefficient(gt_et, pred_et)
    
    # Tumor Core (TC) - Labels 2 + 3
    gt_tc = np.logical_or(gt_data == 2, gt_data == 3)
    pred_tc = np.logical_or(pred_data == 2, pred_data == 3)
    dice_tc = dice_coefficient(gt_tc, pred_tc)
    
    # Whole Tumor (WT) - Labels 1 + 2 + 3
    gt_wt = (gt_data > 0)
    pred_wt = (pred_data > 0)
    dice_wt = dice_coefficient(gt_wt, pred_wt)
    
    return {
        "dice_et": float(dice_et),
        "dice_tc": float(dice_tc),
        "dice_wt": float(dice_wt)
    }

def calculate_dice_scores(results_folder, ground_truth_folder):
    """Calculate Dice scores between predicted and ground truth segmentations."""
    # Use a dictionary to store lists of scores for each region
    all_scores = {"dice_et": [], "dice_tc": [], "dice_wt": []}
    case_results = {}

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

                result_data = result_img.get_fdata()
                gt_data = gt_img.get_fdata()

                # NEW: Calculate region-specific metrics
                metrics = calculate_brats_metrics(gt_data, result_data)
                
                # Store metrics for this case
                case_results[file_name] = metrics
                
                # Append to lists for overall average
                all_scores["dice_et"].append(metrics["dice_et"])
                all_scores["dice_tc"].append(metrics["dice_tc"])
                all_scores["dice_wt"].append(metrics["dice_wt"])
                
                # NEW: Updated print statement
                print(f"  {file_name}: ET={metrics['dice_et']:.4f}, TC={metrics['dice_tc']:.4f}, WT={metrics['dice_wt']:.4f}")
                
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
        else:
            print(f"Ground truth for {file_name} not found! (Looked for {gt_path})")

    if all_scores["dice_wt"]: # Check if we processed any files
        # NEW: Calculate summary stats for each region
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
    
    # --- CHANGED: Create the .../results/nnUNet/ directory ---
    # nnUNet v1 expects models to be in $RESULTS_FOLDER/nnUNet/
    nnunet_models_dir = Path(os.environ["RESULTS_FOLDER"]) / "nnUNet"
    os.makedirs(nnunet_models_dir, exist_ok=True)
    
    local_model_src = Path(local_model_path)
    # --- CHANGED: Destination is now INSIDE the nnUNet_models_dir ---
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

# --- UNCHANGED FUNCTION ---
def main():
    parser = argparse.ArgumentParser(description="Evaluate synthesis models using segmentation performance")
    parser.add_argument("--dataset_dir", default="./Dataset137_BraTS21_Completed",
                       help="nnUNet format dataset directory")
    parser.add_argument("--output_dir", default="./segmentation_outputs",
                       help="Output directory for segmentation results")
    parser.add_argument("--skip_segmentation", action="store_true",
                       help="Skip segmentation and only calculate Dice scores")
    
    args = parser.parse_args()
    
    setup_nnunet_environment()
    
    if not os.path.exists(args.dataset_dir):
        print(f"❌ Dataset directory not found: {args.dataset_dir}")
        print("Run prepare_nnunet_dataset.py first!")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not args.skip_segmentation:
        if not download_nnunet_weights():
            print("❌ Failed to setup nnUNet weights")
            return
        
        if not run_nnunet_prediction(args.dataset_dir, args.output_dir):
            print("❌ nnUNet prediction failed")
            return
    
    ground_truth_dir = f"{args.dataset_dir}/labelsTr"
    
    if not os.path.exists(ground_truth_dir):
        print(f"❌ Ground truth directory not found: {ground_truth_dir}")
        return
    
    if not os.listdir(args.output_dir):
        print(f"❌ No segmentation results found in {args.output_dir}")
        return
    
    print(f"\n🔍 Calculating Dice scores...")
    # NEW: Updated return values
    summary_stats, case_results = calculate_dice_scores(args.output_dir, ground_truth_dir)
    
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

if __name__ == "__main__":
    main()