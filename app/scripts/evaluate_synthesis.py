#!/usr/bin/env python3
import os
import subprocess
import argparse
import nibabel as nib
import numpy as np
from pathlib import Path
import shutil
import wandb
import matplotlib.pyplot as plt
import tempfile
import glob

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    
    # Handle empty ground truth and empty prediction
    if not np.any(y_true) and not np.any(y_pred):
        return 1.0
    
    intersection = np.sum(y_true & y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

def calculate_brats_metrics(gt_data, pred_data):
    # ET = Enhancing Tumor (Label 3)
    gt_et = (gt_data == 3)
    pred_et = (pred_data == 3)
    dice_et = dice_coefficient(gt_et, pred_et)
    
    # TC = Tumor Core (Labels 1 + 3)
    gt_tc = np.logical_or(gt_data == 2, gt_data == 3)
    pred_tc = np.logical_or(pred_data == 2, pred_data == 3)
    dice_tc = dice_coefficient(gt_tc, pred_tc)
    
    # WT = Whole Tumor (All labels > 0, i.e., 1 + 2 + 3)
    gt_wt = (gt_data > 0)
    pred_wt = (pred_data > 0)
    dice_wt = dice_coefficient(gt_wt, pred_wt)
    
    # --- DIAGNOSTIC METRIC ---
    # NCR = Necrotic/Non-Enhancing Core (Label 1)
    gt_ncr = (gt_data == 1)
    pred_ncr = (pred_data == 1)
    dice_ncr = dice_coefficient(gt_ncr, pred_ncr)
    
    return {
        "dice_et": float(dice_et),
        "dice_tc": float(dice_tc),
        "dice_wt": float(dice_wt),
        "dice_ncr": float(dice_ncr)  # <-- Diagnostic metric
    }

def fix_floating_point_labels(segmentation):
    fixed_seg = np.round(segmentation).astype(np.int16)
    fixed_seg = np.clip(fixed_seg, 0, 4) # Clip to 4 just in case
    return fixed_seg

def find_best_slice(seg1, seg2):
    combined_seg = np.logical_or(seg1 > 0, seg2 > 0)
    slice_scores = np.sum(combined_seg, axis=(0, 1))
    if np.sum(slice_scores) == 0:
        return seg1.shape[2] // 2
    return np.argmax(slice_scores)

def log_wandb_visualization(gt_data, pred_data, file_name):
    """
    Logs BOTH visualization formats to W&B:
    1. Original full segmentation with all labels (jet colormap)
    2. Tumor Core (TC) binary masks for detailed analysis
    """
    try:
        best_slice = find_best_slice(gt_data, pred_data)
        
        # === FORMAT 1: Original full segmentation visualization ===
        gt_slice_full = gt_data[:, :, best_slice]
        pred_slice_full = pred_data[:, :, best_slice]
        
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.imshow(gt_slice_full, cmap='jet', vmin=0, vmax=3)
        ax1.set_title(f"Ground Truth\n{file_name}")
        ax1.axis('off')
        
        ax2.imshow(pred_slice_full, cmap='jet', vmin=0, vmax=3)
        ax2.set_title(f"Prediction\n{file_name}")
        ax2.axis('off')
        
        plt.tight_layout()
        
        # Log original format
        wandb.log({f"samples/{Path(file_name).stem}": wandb.Image(fig1)})
        plt.close(fig1)
        
        # === FORMAT 2: Tumor Core (TC) binary masks ===
        # Create TC masks (Label 1 + Label 3)
        gt_tc_mask = np.logical_or(gt_data == 1, gt_data == 3)
        pred_tc_mask = np.logical_or(pred_data == 1, pred_data == 3)
        
        gt_slice_tc = gt_tc_mask[:, :, best_slice]
        pred_slice_tc = pred_tc_mask[:, :, best_slice]
        
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Use grayscale cmap for binary masks
        ax1.imshow(gt_slice_tc, cmap='gray', vmin=0, vmax=1)
        ax1.set_title(f"Ground Truth (Tumor Core)\n{file_name}")
        ax1.axis('off')
        
        ax2.imshow(pred_slice_tc, cmap='gray', vmin=0, vmax=1)
        ax2.set_title(f"Prediction (Tumor Core)\n{file_name}")
        ax2.axis('off')
        
        plt.tight_layout()
        
        # Log TC mask format
        wandb.log({f"samples_TC_Mask/{Path(file_name).stem}": wandb.Image(fig2)})
        plt.close(fig2)
        
    except Exception as e:
        print(f"  [W&B] Error creating visualization for {file_name}: {e}")

def check_label_proportions(results_folder, ground_truth_folder):
    """
    Loads the first sample from GT and Pred folders to check
    label proportions for debugging.
    """
    print("\n--- 🔬 LABEL PROPORTION PRE-CHECK ---")
    try:
        prediction_files = sorted([f for f in os.listdir(results_folder) if f.endswith('.nii') or f.endswith('.nii.gz')])
        if not prediction_files:
            print("  [Warning] No prediction files found to check.")
            return

        first_file = prediction_files[0]
        pred_path = os.path.join(results_folder, first_file)
        gt_path = os.path.join(ground_truth_folder, first_file)

        if not os.path.exists(gt_path):
            print(f"  [Warning] Ground truth not found for first file: {gt_path}")
            return

        print(f"  Checking first file: {first_file}")

        # Load images
        pred_img = nib.load(pred_path)
        gt_img = nib.load(gt_path)

        # Get and fix data using the same logic as the main script
        pred_data = fix_floating_point_labels(pred_img.get_fdata())
        gt_data = fix_floating_point_labels(gt_img.get_fdata())

        total_voxels = gt_data.size
        if total_voxels == 0:
            print("  [Warning] Loaded empty image.")
            return

        # Calculate proportions
        gt_counts = {label: 0 for label in range(5)} # Check 0,1,2,3,4
        pred_counts = {label: 0 for label in range(5)}

        gt_labels, gt_cts = np.unique(gt_data, return_counts=True)
        for label, count in zip(gt_labels, gt_cts):
            if label in gt_counts:
                gt_counts[label] = count

        pred_labels, pred_cts = np.unique(pred_data, return_counts=True)
        for label, count in zip(pred_labels, pred_cts):
            if label in pred_counts:
                pred_counts[label] = count

        # Print report
        print("  Label Proportions (% of total voxels):")
        print("  Label | Ground Truth | Prediction   | Description (BraTS 2021/23)")
        print("  ------------------------------------------------------------------")
        
        labels_desc = {
            0: "Background",
            1: "Label 1 (NCR/NET)",
            2: "Label 2 (Edema)",
            3: "Label 3 (ET)",
            4: "Label 4 (Legacy ET - SHOULD BE 0%)" 
        }

        for i in range(5): # Check 0, 1, 2, 3, 4
            gt_prop = (gt_counts[i] / total_voxels) * 100
            pred_prop = (pred_counts[i] / total_voxels) * 100
            print(f"    {i}   | {gt_prop:10.6f}% | {pred_prop:10.6f}% | {labels_desc.get(i, 'Unknown')}")
        
        print("  ------------------------------------------------------------------")
        
        if gt_counts[4] > 0 or pred_counts[4] > 0:
             print("  [ALERT] Label '4' detected! This script expects ET=3.")
        else:
             print("  [INFO] No Label '4' detected. Labeling seems correct (ET=3).")
        print("--- END PRE-CHECK ---\n")

    except Exception as e:
        print(f"  [Error] Failed to run label proportion check: {e}")

def calculate_dice_scores(results_folder, ground_truth_folder, num_viz_samples=10):
    # Add 'dice_ncr' to the metrics to track
    all_scores = {"dice_et": [], "dice_tc": [], "dice_wt": [], "dice_ncr": []}
    case_results = {}
    viz_counter = 0

    prediction_files = sorted([f for f in os.listdir(results_folder) if f.endswith('.nii') or f.endswith('.nii.gz')])
    
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
                
                # Append all metrics
                all_scores["dice_et"].append(metrics["dice_et"])
                all_scores["dice_tc"].append(metrics["dice_tc"])
                all_scores["dice_wt"].append(metrics["dice_wt"])
                all_scores["dice_ncr"].append(metrics["dice_ncr"])
                
                print(f"  {file_name}: ET={metrics['dice_et']:.4f}, TC={metrics['dice_tc']:.4f}, WT={metrics['dice_wt']:.4f}, NCR={metrics['dice_ncr']:.4f}", flush=True)

                if wandb.run and viz_counter < num_viz_samples:
                    print(f"  [W&B] Logging both visualization formats for {file_name}...", flush=True)
                    # This will log both formats to W&B
                    log_wandb_visualization(gt_data, result_data, file_name)
                    viz_counter += 1

            except Exception as e:
                print(f"Error processing {file_name}: {e}", flush=True)
        else:
            print(f"Ground truth for {file_name} not found! (Looked for {gt_path})", flush=True)

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

def setup_nnunet_environment(local_model_path="3d_fullres"):
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
    local_model_dest = nnunet_models_dir / local_model_src.name

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

def download_nnunet_weights():
    print("✅ Using local nnUNet v1 model. Skipping download.")
    return True

def run_nnunet_prediction(dataset_dir, output_dir):
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
        result = subprocess.run(cmd, check=True, text=True, env=os.environ.copy()) 
        
        print("✅ nnUNet (v1) prediction completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ nnUNet (v1) prediction failed:")
        print(f"Return code: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ 'nnUNet_predict' command not found.")
        print("   Make sure nnUNet v1 (pip install nnunet) is installed and in your PATH.")
        return False

def setup_subset_directory(original_dataset_dir, num_cases):
    try:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="nnunet_subset_")
        temp_dir_path = temp_dir_obj.name
        
        print(f"Creating temporary subset directory for {num_cases} cases at: {temp_dir_path}")
        
        temp_images_tr = os.path.join(temp_dir_path, "imagesTr")
        temp_labels_tr = os.path.join(temp_dir_path, "labelsTr")
        os.makedirs(temp_images_tr, exist_ok=True)
        os.makedirs(temp_labels_tr, exist_ok=True)
        
        base_image_files = sorted(glob.glob(f"{original_dataset_dir}/imagesTr/*_0000.nii.gz"))
        
        if not base_image_files:
            print("[Warning] No '*_0000.nii.gz' files found. Trying to glob all '*.nii*'...")
            base_image_files = sorted(glob.glob(f"{original_dataset_dir}/imagesTr/*.nii*"))
            case_ids = sorted(list(set([f.split(os.sep)[-1].split('_')[0] for f in base_image_files])))
            base_image_files = [os.path.join(original_dataset_dir, "imagesTr", f"{cid}_0000.nii.gz") for cid in case_ids]
            
            print("Trying robust fallback: globbing labels and finding matching images.")
            all_labels = sorted(glob.glob(f"{original_dataset_dir}/labelsTr/*.nii*"))
            base_image_files = []
            for label_path in all_labels:
                label_name = os.path.basename(label_path)
                img_name = label_name.replace(".nii.gz", "_0000.nii.gz")
                img_path = os.path.join(original_dataset_dir, "imagesTr", img_name)
                if os.path.exists(img_path):
                    base_image_files.append(img_path)
            
            if not base_image_files:
                raise FileNotFoundError(f"Could not determine base image files in {original_dataset_dir}/imagesTr")
            print(f"Found {len(base_image_files)} base files via label matching.")

        cases_to_link = base_image_files[:num_cases]
        print(f"Linking {len(cases_to_link)} cases...")
        
        count = 0
        for base_img_path in cases_to_link:
            base_file_name = os.path.basename(base_img_path)
            
            # --- TYPO FIX 1 ---
            case_id = base_file_name.replace("_0000.nii.gz", "")
            
            label_name = f"{case_id}.nii.gz"
            label_path = os.path.join(original_dataset_dir, "labelsTr", label_name)

            if not os.path.exists(label_path):
                print(f"  [Warning] No label found for case {case_id}. (Looked for {label_path}). Skipping case.")
                continue
            
            os.symlink(os.path.abspath(label_path), os.path.join(temp_labels_tr, label_name))
            
            modalities_found = 0
            for i in range(4): # 0000, 0001, 0002, 0003
                modality_file = f"{case_id}_000{i}.nii.gz"
                modality_path = os.path.join(original_dataset_dir, "imagesTr", modality_file)
                
                if os.path.exists(modality_path):
                    os.symlink(os.path.abspath(modality_path), os.path.join(temp_images_tr, modality_file))
                    modalities_found += 1
                else:
                    print(f"  [Warning] Missing modality {modality_file} for case {case_id}")
            
            if modalities_found == 4:
                count += 1
            else:
                print(f"  [Warning] Incomplete case {case_id}. Only found {modalities_found}/4 modalities. nnU-Net may fail.")
                count += 1 

        print(f"✅ Successfully linked {count} cases.")
        
        if count == 0:
            raise RuntimeError("Failed to link any files. Check paths and file names.")
            
        return temp_dir_path, temp_dir_obj
        
    except Exception as e:
        print(f"❌ Error creating subset directory: {e}")
        if 'temp_dir_obj' in locals():
            temp_dir_obj.cleanup()
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Evaluate synthesis models using segmentation performance")
    parser.add_argument("--dataset_dir", default="./Dataset137_BraTS21_Completed",
                       help="nnUNet format dataset directory")
    parser.add_argument("--output_dir", default="./segmentation_outputs",
                       help="Output directory for segmentation results")
    parser.add_argument("--skip_segmentation", action="store_true",
                       help="Skip segmentation and only calculate Dice scores")
    
    parser.add_argument("--wandb_project", type=str, default="brats-synthesis-eval",
                        help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name (defaults to auto-generated)")
    parser.add_argument("--num_viz_samples", type=int, default=10,
                        help="Number of segmentation samples to log to W&B")
    
    parser.add_argument("--num_test_files", type=int, default=None,
                        help="Run evaluation on a subset of N files (default: all)")

    args = parser.parse_args()

    try:
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
        print(f"✅ W&B Run initialized: {run.url}")
    except Exception as e:
        print(f"❌ Could not initialize W&B: {e}")
        print("   Skipping W&B logging. Did you run 'wandb login'?")
        run = None

    # --- TYPO FIX 2 ---
    dataset_dir_to_use = args.dataset_dir
    temp_dir_obj = None
    
    try:
        if args.num_test_files:
            dataset_dir_to_use, temp_dir_obj = setup_subset_directory(
                args.dataset_dir, 
                args.num_test_files
            )
            if dataset_dir_to_use is None:
                raise RuntimeError("Failed to create temporary subset directory.")
        
        setup_nnunet_environment()
        
        if not os.path.exists(dataset_dir_to_use):
            print(f"❌ Dataset directory not found: {dataset_dir_to_use}")
            if run: run.finish(exit_code=1)
            return
        
        if not args.skip_segmentation:
            print(f"Cleaning output directory: {args.output_dir}")
            shutil.rmtree(args.output_dir, ignore_errors=True)
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        if not args.skip_segmentation:
            if not download_nnunet_weights():
                print("❌ Failed to setup nnUNet weights")
                if run: run.finish(exit_code=1)
                return
            
            if not run_nnunet_prediction(dataset_dir_to_use, args.output_dir):
                print("❌ nnU-Net prediction failed")
                if run: run.finish(exit_code=1)
                return
        
        ground_truth_dir = f"{dataset_dir_to_use}/labelsTr"
        
        if not os.path.exists(ground_truth_dir):
            print(f"❌ Ground truth directory not found: {ground_truth_dir}")
            if run: run.finish(exit_code=1)
            return
        
        if not os.listdir(args.output_dir):
            print(f"❌ No segmentation results found in {args.output_dir}")
            if run: run.finish(exit_code=1)
            return
        
        # --- NEW FUNCTION CALL ADDED HERE ---
        check_label_proportions(args.output_dir, ground_truth_dir)
        
        print(f"\n🔍 Calculating Dice scores...")
        summary_stats, case_results = calculate_dice_scores(
            args.output_dir, 
            ground_truth_dir,
            args.num_viz_samples
        )
        
        if summary_stats:
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
                f.write("\nPer-case results (Filename: ET, TC, WT, NCR):\n")
                for case, metrics in sorted(case_results.items()):
                    f.write(f"  {case}: {metrics['dice_et']:.4f}, {metrics['dice_tc']:.4f}, {metrics['dice_wt']:.4f}, {metrics['dice_ncr']:.4f}\n")
            
            print(f"\n📄 Results saved to: {results_file}")
            
            print(f"\n🎯 FINAL RESULTS:")
            print(f"Your CWDM synthesis models achieved:")
            for region, stats in summary_stats.items():
                print(f"  Average {region.upper()} Dice: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"\nThis measures how well synthesized modalities preserve")
            print(f"segmentation-relevant information!")
            
            if run:
                print("\n[W&B] Logging results to Weights & Biases...")
                
                summary_log = {}
                for region, stats in summary_stats.items():
                    summary_log[f"avg_{region}_dice"] = stats['mean']
                    summary_log[f"std_{region}_dice"] = stats['std']
                    summary_log[f"min_{region}_dice"] = stats['min']
                    summary_log[f"max_{region}_dice"] = stats['max']
                
                wandb.summary.update(summary_log)
                print("  [WB] Logged summary statistics.")

                # Include NCR in the table columns
                table_columns = ["Filename", "DICE_ET", "DICE_TC", "DICE_WT", "DICE_NCR"]
                table = wandb.Table(columns=table_columns)
                for case, metrics in sorted(case_results.items()):
                    table.add_data(
                        case, 
                        metrics['dice_et'], 
                        metrics['dice_tc'], 
                        metrics['dice_wt'],
                        metrics['dice_ncr'] # Add new metric
                    )
                run.log({"dice_results_per_case": table})
                print("  [W&B] Logged per-case results table.")
                print("  [W&B] Logged both visualization formats: 'samples/' and 'samples_TC_Mask/'")
                
                print(f"✅ W&B Run finished: {run.url}")
                run.finish()
            
    except Exception as e:
        print(f"\n--- SCRIPT FAILED ---")
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        if run:
            print("Finishing W&B run with exit code 1...")
            run.finish(exit_code=1)
    finally:
        if temp_dir_obj:
            print(f"\nCleaning up temporary subset directory: {temp_dir_obj.name}")
            temp_dir_obj.cleanup()
            print("✅ Cleanup complete.")

if __name__ == "__main__":
    main()