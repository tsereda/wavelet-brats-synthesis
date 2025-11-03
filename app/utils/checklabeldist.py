#!/usr/bin/env python3
"""
Inspects and compares unique labels in ground truth and prediction NIfTI files.
Helps diagnose missing labels in ground truth data.
"""

import argparse
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import warnings

def fix_floating_point_labels(segmentation):
    """Fix floating point precision issues by rounding to nearest integers"""
    fixed_seg = np.round(segmentation).astype(np.int16)
    fixed_seg = np.clip(fixed_seg, 0, 4)
    return fixed_seg

def find_best_slice(seg1, seg2):
    """Find slice with most non-background content in either segmentation"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try:
            combined_seg = np.logical_or(seg1 > 0, seg2 > 0)
            slice_scores = np.sum(combined_seg, axis=(0, 1))
            if np.sum(slice_scores) == 0:
                return seg1.shape[2] // 2
            return np.argmax(slice_scores)
        except Exception:
            return seg1.shape[2] // 2

def save_comparison_image(gt_data, pred_data, file_name, output_dir):
    """Create and save a side-by-side visualization of GT and Pred slices."""
    try:
        best_slice = find_best_slice(gt_data, pred_data)
        
        gt_slice = gt_data[:, :, best_slice]
        pred_slice = pred_data[:, :, best_slice]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.imshow(gt_slice, cmap='jet', vmin=0, vmax=3)
        ax1.set_title(f"Ground Truth (Slice {best_slice})")
        ax1.axis('off')
        
        ax2.imshow(pred_slice, cmap='jet', vmin=0, vmax=3)
        ax2.set_title(f"Prediction (Slice {best_slice})")
        ax2.axis('off')

        fig.suptitle(file_name, fontsize=16)
        plt.tight_layout()
        
        output_path = output_dir / f"mismatch_{Path(file_name).stem}.png"
        plt.savefig(output_path, dpi=100)
        plt.close(fig)
        return output_path
        
    except Exception as e:
        print(f"    [Error creating visualization for {file_name}: {e}]")
        plt.close(fig)
        return None

def main():
    parser = argparse.ArgumentParser(description="Compare unique labels in GT and Pred segmentations.")
    parser.add_argument("gt_dir", type=Path, help="Path to the ground truth labels directory (e.g., .../labelsTr)")
    parser.add_argument("pred_dir", type=Path, help="Path to the prediction directory (e.g., .../segmentation_outputs)")
    parser.add_argument("--viz_dir", type=Path, default="label_inspection_viz", help="Directory to save mismatch visualizations")
    parser.add_argument("--num_viz", type=int, default=10, help="Number of mismatch images to save")
    
    args = parser.parse_args()

    print(f"🔬 Starting Label Inspection...")
    print(f"  Ground Truth: {args.gt_dir}")
    print(f"  Prediction:   {args.pred_dir}")
    
    args.viz_dir.mkdir(exist_ok=True)
    
    all_gt_labels = set()
    all_pred_labels = set()
    problem_cases = []
    viz_counter = 0

    pred_files = list(args.pred_dir.glob('*.nii.gz'))
    if not pred_files:
        pred_files = list(args.pred_dir.glob('*.nii'))
    
    if not pred_files:
        print(f"❌ No .nii.gz or .nii files found in {args.pred_dir}")
        return

    print(f"Found {len(pred_files)} prediction files. Comparing...")

    for pred_file in tqdm(pred_files, desc="Inspecting Files"):
        gt_file = args.gt_dir / pred_file.name
        
        if not gt_file.exists():
            print(f"  [Warning] Skipping {pred_file.name}: No matching GT file found.")
            continue
            
        try:
            gt_nii = nib.load(gt_file)
            pred_nii = nib.load(pred_file)
            
            gt_data = fix_floating_point_labels(gt_nii.get_fdata())
            pred_data = fix_floating_point_labels(pred_nii.get_fdata())
            
            gt_unique = np.unique(gt_data)
            pred_unique = np.unique(pred_data)
            
            all_gt_labels.update(gt_unique)
            all_pred_labels.update(pred_unique)
            
            # This is the hypothesis check:
            if 3 not in gt_unique and 3 in pred_unique:
                problem_cases.append(pred_file.name)
                
                if viz_counter < args.num_viz:
                    img_path = save_comparison_image(gt_data, pred_data, pred_file.name, args.viz_dir)
                    if img_path:
                        tqdm.write(f"  🚨 Mismatch found in {pred_file.name} (GT: {gt_unique}, Pred: {pred_unique}). Saving viz to {img_path}")
                        viz_counter += 1

        except Exception as e:
            print(f"  [Error] Failed to process {pred_file.name}: {e}")

    print("\n" + "="*50)
    print("📊 INSPECTION COMPLETE")
    print("="*50)
    
    print("\n--- Label Summary ---")
    print(f"All unique labels found in Ground Truth: {sorted(list(all_gt_labels))}")
    print(f"All unique labels found in Predictions:  {sorted(list(all_pred_labels))}")
    
    print("\n--- Hypothesis Test: Missing Enhancing Tumor (Label 3) ---")
    print(f"Total files processed: {len(pred_files)}")
    print(f"🚨🚨 Total cases where Pred has Label 3 but GT does NOT: {len(problem_cases)} 🚨🚨")

    if problem_cases:
        print("\nThis confirms the hypothesis: Your model is predicting Enhancing Tumor (Label 3),")
        print("but the ground truth files you are using have those regions labeled as Background (Label 0).")
        print("This is the direct cause of the `DICE_ET = 0.0000` scores.")
        print(f"\nSaved {viz_counter} comparison images to: {args.viz_dir}/")
        print("First 10 problem cases:")
        for f in problem_cases[:10]:
            print(f"  - {f}")
    else:
        print("\n✅ No mismatches found. If ET scores are 0, it means your model is not predicting Label 3.")


if __name__ == "__main__":
    main()