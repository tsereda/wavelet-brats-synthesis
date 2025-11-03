"""
Complete evaluation pipeline for synthesis models.
Runs nnUNet segmentation and calculates Dice scores.
(MODIFIED TO USE LOCAL nnUNet v1 MODEL)
"""

import os
import subprocess
import argparse
import nibabel as nib
import numpy as np
from pathlib import Path
import shutil  # <-- ADDED IMPORT

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Calculate Dice coefficient between two binary masks."""
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

def dice_coef_multilabel(y_true, y_pred, numLabels):
    """Calculate average Dice coefficient across multiple labels."""
    dice = 0
    for index in range(numLabels):
        dice += dice_coefficient(y_true == index, y_pred == index)
    return dice / numLabels

def calculate_dice_scores(results_folder, ground_truth_folder):
    """Calculate Dice scores between predicted and ground truth segmentations."""
    dice_scores = []
    case_results = {}

    for file_name in os.listdir(results_folder):
        if file_name.endswith('.nii') or file_name.endswith('.nii.gz'):
            result_path = os.path.join(results_folder, file_name)
            gt_path = os.path.join(ground_truth_folder, file_name)

            if os.path.exists(gt_path):
                try:
                    result_img = nib.load(result_path)
                    gt_img = nib.load(gt_path)

                    result_data = result_img.get_fdata()
                    gt_data = gt_img.get_fdata()

                    dice = dice_coef_multilabel(gt_data, result_data, 4) # 4 labels for BraTS
                    dice_scores.append(dice)
                    case_results[file_name] = dice
                    
                    print(f"{file_name}: Dice = {dice:.4f}")
                    
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")
            else:
                print(f"Ground truth for {file_name} not found!")

    if dice_scores:
        avg_dice = np.mean(dice_scores)
        std_dice = np.std(dice_scores)
        print(f"\n📊 EVALUATION RESULTS:")
        print(f"Average Dice Coefficient: {avg_dice:.4f} ± {std_dice:.4f}")
        print(f"Number of cases: {len(dice_scores)}")
        print(f"Min Dice: {np.min(dice_scores):.4f}")
        print(f"Max Dice: {np.max(dice_scores):.4f}")
        
        return avg_dice, std_dice, case_results
    else:
        print("No valid segmentation files found for comparison.")
        return None, None, {}

# --- MODIFIED FUNCTION ---
def setup_nnunet_environment(local_model_path="3d_fullres"):
    """Setup nnUNet environment variables and move local model."""
    local_root = "./nnunet_data" 
    
    os.environ["nnUNet_raw"] = os.path.abspath(f"{local_root}/raw")
    os.environ["nnUNet_preprocessed"] = os.path.abspath(f"{local_root}/preprocessed") 
    os.environ["nnUNet_results"] = os.path.abspath(f"{local_root}/results")
    
    # Create directories
    for path_key in ["nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"]:
        path = os.environ[path_key]
        os.makedirs(path, exist_ok=True)
    
    # --- NEW PART ---
    # Move the unzipped model (3d_fullres) into the expected nnUNet_results directory
    local_model_src = Path(local_model_path)
    local_model_dest = Path(os.environ["nnUNet_results"]) / local_model_src.name

    if local_model_src.exists() and local_model_src.is_dir():
        try:
            # If it already exists in the destination, remove it first
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
    # --- END NEW PART ---

    print("nnUNet environment variables set:")
    print(f"nnUNet_raw: {os.environ['nnUNet_raw']}")
    print(f"nnUNet_preprocessed: {os.environ['nnUNet_preprocessed']}")
    print(f"nnUNet_results: {os.environ['nnUNet_results']}")

# --- MODIFIED FUNCTION ---
def download_nnunet_weights():
    """Bypassed: Using local nnUNet v1 model."""
    print("✅ Using local nnUNet v1 model. Skipping download.")
    return True

# --- MODIFIED FUNCTION ---
def run_nnunet_prediction(dataset_dir, output_dir):
    """Run nnUNet (v1) prediction on the dataset."""
    
    print(f"Running nnUNet (v1) prediction...")
    
    # Input directory (needs to be absolute)
    input_dir_abs = os.path.abspath(f"{dataset_dir}/imagesTr")
    # Output directory (needs to be absolute)
    output_dir_abs = os.path.abspath(output_dir)
    
    print(f"Input: {input_dir_abs}")
    print(f"Output: {output_dir_abs}")
    
    # --- NOTE ---
    # The v1 `nnUNet_predict` command takes -i and -o directly.
    # No need to symlink into nnUNet_raw for prediction.

    # Run nnUNet (v1) prediction
    cmd = [
        "nnUNet_predict",
        "-i", input_dir_abs,
        "-o", output_dir_abs,
        "-t", "Task082_BraTS2020",   # From your unzipped model
        "-m", "3d_fullres",
        "-f", "0",                    # From your unzipped model (fold_0)
        "-tr", "nnUNetTrainerV2"     # From your unzipped model (nnUNetTrainerV2__...)
        # "--disable_tta",            # Add this for faster prediction (optional)
    ]
    
    try:
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
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

def main():
    parser = argparse.ArgumentParser(description="Evaluate synthesis models using segmentation performance")
    parser.add_argument("--dataset_dir", default="./Dataset137_BraTS21_Completed",
                       help="nnUNet format dataset directory")
    parser.add_argument("--output_dir", default="./segmentation_outputs",
                       help="Output directory for segmentation results")
    parser.add_argument("--skip_segmentation", action="store_true",
                       help="Skip segmentation and only calculate Dice scores")
    
    args = parser.parse_args()
    
    # Setup nnUNet environment (this will now also move your local model)
    setup_nnunet_environment()
    
    # Check if dataset exists
    if not os.path.exists(args.dataset_dir):
        print(f"❌ Dataset directory not found: {args.dataset_dir}")
        print("Run prepare_nnunet_dataset.py first!")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not args.skip_segmentation:
        # Download nnUNet weights if needed (this is now bypassed)
        if not download_nnunet_weights():
            print("❌ Failed to setup nnUNet weights")
            return
        
        # Run nnUNet prediction (this now runs the v1 command)
        if not run_nnunet_prediction(args.dataset_dir, args.output_dir):
            print("❌ nnUNet prediction failed")
            return
    
    # Calculate Dice scores
    ground_truth_dir = f"{args.dataset_dir}/labelsTr"
    
    if not os.path.exists(ground_truth_dir):
        print(f"❌ Ground truth directory not found: {ground_truth_dir}")
        return
    
    if not os.listdir(args.output_dir):
        print(f"❌ No segmentation results found in {args.output_dir}")
        return
    
    print(f"\n🔍 Calculating Dice scores...")
    avg_dice, std_dice, case_results = calculate_dice_scores(args.output_dir, ground_truth_dir)
    
    # Save results
    if avg_dice is not None:
        results_file = "synthesis_evaluation_results.txt"
        with open(results_file, "w") as f:
            f.write("Synthesis Model Evaluation Results\n")
            f.write("=====================================\n\n")
            f.write(f"Average Dice Coefficient: {avg_dice:.4f} ± {std_dice:.4f}\n")
            f.write(f"Number of cases: {len(case_results)}\n")
            f.write(f"Min Dice: {min(case_results.values()):.4f}\n")
            f.write(f"Max Dice: {max(case_results.values()):.4f}\n\n")
            f.write("Per-case results:\n")
            for case, dice in sorted(case_results.items()):
                f.write(f"  {case}: {dice:.4f}\n")
        
        print(f"\n📄 Results saved to: {results_file}")
        
        # Summary
        print(f"\n🎯 FINAL RESULTS:")
        print(f"Your CWDM synthesis models achieved:")
        print(f"Average Dice Score: {avg_dice:.4f} ± {std_dice:.4f}")
        print(f"This measures how well synthesized modalities preserve")
        print(f"segmentation-relevant information!")

if __name__ == "__main__":
    main()