#!/usr/bin/env python3
"""
BraTS 2023 Inference with BraTS 2021 Trained Weights (SwinUNETR)

This script adapts the inference for BraTS 2023 data format using weights
trained on BraTS 2021 and calculates Dice scores if labels are available.
"""

import os
import torch
import numpy as np
import glob
from pathlib import Path
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
    MapTransform,
)
from monai.inferers import sliding_window_inference
from monai.data import Dataset, DataLoader
from monai.metrics import DiceMetric  # <--- ADDED
import nibabel as nib


# <--- ADDED: Class definition from the training notebook to convert labels ---
class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key not in d:  # <--- ADDED: Skip if key (e.g., 'label') is missing
                continue
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            # label 2 is ET (we use 4 in brats 2023, but 2021 model used 2)
            # Assuming Enhancing tumor is label 2 from source, or 4 from new data
            # Let's check BraTS format. ET is 4, NCR/NET is 1, ED is 2.
            # Training script used: 1=ED, 2=ET, 3=NCR/NET
            # This class seems to match the notebook:
            # result.append(torch.logical_or(d[key] == 2, d[key] == 3))  # TC (ET + NCR/NET)
            # result.append(
            #     torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1)
            # )  # WT (ET + NCR/NET + ED)
            # result.append(d[key] == 2)  # ET
            
            # Let's use the standard BraTS labels 1, 2, 4
            # TC (Tumor Core) = 1 (NCR/NET) + 4 (ET)
            result.append(torch.logical_or(d[key] == 1, d[key] == 4))
            # WT (Whole Tumor) = 1 (NCR/NET) + 4 (ET) + 2 (ED)
            result.append(torch.logical_or(torch.logical_or(d[key] == 1, d[key] == 4), d[key] == 2))
            # ET (Enhancing Tumor) = 4
            result.append(d[key] == 4)
            
            d[key] = torch.stack(result, axis=0).float()
        return d
# -------------------------------------------------------------------------


class BraTSInference2023:
    """BraTS 2023 inference using BraTS 2021 trained weights"""

    def __init__(self, model_path, device=None):
        """
        Initialize the inference class
        
        Args:
            model_path: Path to the trained model (.pth file)
            device: torch device (cuda/cpu), auto-detected if None
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=4,
            out_channels=3,
            feature_size=48,
            use_checkpoint=False,
        ).to(self.device)

        # <--- ADDED: Metrics for Dice score ---
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch")
        # -------------------------------------
        
        # Load weights
        self.load_model(model_path)

        # Setup transforms
        self.setup_transforms()

    def load_model(self, model_path):
        """Load pre-trained model weights"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        print(f"Loading BraTS 2021 trained model from: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Handle checkpoints saved with "state_dict" key or as raw state_dict
        if "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint) # <--- MODIFIED: Handle raw state_dict too
        
        self.model.eval()
        print("Model loaded successfully!")

    def setup_transforms(self):
        """Setup preprocessing and postprocessing transforms"""
        # <--- MODIFIED: To handle both image and label ---
        self.preprocess = Compose([
            LoadImaged(keys=["image", "label"], allow_missing_keys=True),
            EnsureChannelFirstd(keys=["image", "label"], allow_missing_keys=True),
            EnsureTyped(keys=["image", "label"], allow_missing_keys=True),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label", allow_missing_keys=True),
            Orientationd(keys=["image", "label"], axcodes="RAS", allow_missing_keys=True),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
                allow_missing_keys=True,
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ])
        # -------------------------------------------------
        
        self.postprocess = Compose([
            Activations(sigmoid=True),
            AsDiscrete(threshold=0.5)
        ])

    def find_brats_2023_files(self, case_dir):
        """
        Auto-detect BraTS 2023 file naming convention
        
        Args:
            case_dir: Directory containing the case files
            
        Returns:
            Tuple: (image_paths, label_path)
        """
        case_dir = Path(case_dir)
        case_name = case_dir.name
        
        files = {
            "t1n": case_dir / f"{case_name}-t1n.nii.gz",
            "t1c": case_dir / f"{case_name}-t1c.nii.gz",
            "t2f": case_dir / f"{case_name}-t2f.nii.gz",
            "t2w": case_dir / f"{case_name}-t2w.nii.gz",
        }
        
        missing_files = []
        for modality, filepath in files.items():
            if not filepath.exists():
                missing_files.append(str(filepath))
        
        if missing_files:
            raise FileNotFoundError(f"Missing files: {missing_files}")
        
        image_paths = [
            str(files["t2f"]),  # FLAIR
            str(files["t1n"]),  # T1 native  
            str(files["t1c"]),  # T1 contrast enhanced
            str(files["t2w"]),  # T2 weighted
        ]
        
        # <--- ADDED: Find label file ---
        label_path = case_dir / f"{case_name}-seg.nii.gz"
        if not label_path.exists():
            print(f"Warning: No label file found for {case_name}")
            label_path = None
        # ----------------------------------

        return image_paths, str(label_path) if label_path else None # <--- MODIFIED

    def predict_case(self, case_dir, use_amp=True):
        """
        Predict on a BraTS 2023 case.
        
        Args:
            case_dir: Directory containing case files
            use_amp: Use automatic mixed precision
            
        Returns:
            Tuple: (prediction_tensor, label_tensor)
        """
        # <--- MODIFIED: Get both image and label paths ---
        image_paths, label_path = self.find_brats_2023_files(case_dir)
        
        print(f"Processing case: {Path(case_dir).name}")
        print(f"Found modalities: {[Path(p).name for p in image_paths]}")
        if label_path:
            print(f"Found label: {Path(label_path).name}")
            
        data = {"image": image_paths, "label": label_path}
        if label_path is None:
            data.pop("label")
        # -------------------------------------------------

        # Apply transforms
        data = self.preprocess(data)
        image = data["image"].unsqueeze(0).to(self.device)
        label = data.get("label")  # Will be None if not found
        
        print(f"Input shape: {image.shape}")
        
        # Run inference
        with torch.no_grad():
            if use_amp and self.device.type == "cuda":
                with torch.autocast("cuda"):
                    prediction = sliding_window_inference(
                        inputs=image,
                        roi_size=(96, 96, 96),
                        sw_batch_size=4,
                        predictor=self.model,
                        overlap=0.5,
                    )
            else:
                prediction = sliding_window_inference(
                    inputs=image,
                    roi_size=(96, 96, 96),
                    sw_batch_size=4,
                    predictor=self.model,
                    overlap=0.5,
                )
        
        # Postprocess
        prediction = self.postprocess(prediction[0])
        
        # <--- MODIFIED: Return tensors for metric calculation ---
        return prediction, label

    def save_prediction(self, prediction_tensor, reference_image_path, output_path): # <--- MODIFIED
        """
        Save prediction as NIfTI file
        
        Args:
            prediction_tensor: Prediction tensor [3, H, W, D]
            reference_image_path: Path to reference NIfTI file for header
            output_path: Output path for segmentation
        """
        # <--- ADDED: Convert tensor to numpy ---
        prediction = prediction_tensor.cpu().numpy()
        
        # Load reference to get header info
        ref_img = nib.load(reference_image_path)
        
        # Convert multi-channel to single label image
        # Following BraTS convention: 1=NCR/NET, 2=ED, 4=ET
        label_img = np.zeros(prediction.shape[1:], dtype=np.uint8)
        
        # ET (Enhancing Tumor) = label 4
        # This is channel 2 (TC, WT, ET)
        label_img[prediction[2] > 0] = 4
        
        # TC (Tumor Core: NCR/NET) = label 1  
        # This is channel 0 (TC, WT, ET)
        # We only want label 1 (NCR/NET), so we exclude ET (4)
        ncr_net_mask = (prediction[0] > 0) & (label_img == 0)
        label_img[ncr_net_mask] = 1
        
        # ED (Edema) = label 2
        # This is channel 1 (TC, WT, ET)
        # We exclude areas already labeled as TC or ET
        edema_mask = (prediction[1] > 0) & (label_img == 0)
        label_img[edema_mask] = 2
        
        # Create new NIfTI image
        pred_img = nib.Nifti1Image(label_img, ref_img.affine, ref_img.header)
        
        # Save
        nib.save(pred_img, output_path)
        print(f"Prediction saved to: {output_path}")
        
        # Print statistics
        unique_labels, counts = np.unique(label_img, return_counts=True)
        total_voxels = label_img.size
        
        print(f"Segmentation statistics:")
        print(f"  Background (0): {counts[0] if 0 in unique_labels else 0} voxels")
        for label, count in zip(unique_labels, counts):
            if label > 0:
                percentage = (count / total_voxels) * 100
                label_name = {1: "NCR/NET", 2: "Edema", 4: "Enhancing"}[label]
                print(f"  {label_name} ({label}): {count} voxels ({percentage:.2f}%)")

    def process_dataset(self, dataset_dir, output_dir):
        """
        Process entire BraTS 2023 dataset
        
        Args:
            dataset_dir: Directory containing all cases
            output_dir: Directory to save predictions
        """
        dataset_dir = Path(dataset_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        case_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
        print(f"Found {len(case_dirs)} cases to process")
        
        for i, case_dir in enumerate(case_dirs):
            print(f"\n--- Processing {i+1}/{len(case_dirs)}: {case_dir.name} ---")
            
            try:
                # <--- MODIFIED: Get prediction and label tensors ---
                prediction_tensor, label_tensor = self.predict_case(case_dir)
                
                # Get reference image (use FLAIR)
                image_paths, label_path = self.find_brats_2023_files(case_dir)
                reference_path = image_paths[0]  # FLAIR
                
                # Save prediction
                output_path = output_dir / f"{case_dir.name}-seg.nii.gz"
                self.save_prediction(prediction_tensor, reference_path, output_path)
                
                # <--- ADDED: Calculate Dice score for this case ---
                if label_tensor is not None:
                    # Add batch dimension for metric
                    self.dice_metric(
                        y_pred=prediction_tensor.unsqueeze(0).to(self.device),
                        y=label_tensor.unsqueeze(0).to(self.device)
                    )
                    self.dice_metric_batch(
                        y_pred=prediction_tensor.unsqueeze(0).to(self.device),
                        y=label_tensor.unsqueeze(0).to(self.device)
                    )
                # ------------------------------------------------
                
            except Exception as e:
                print(f"Error processing {case_dir.name}: {e}")
                continue
        
        # <--- ADDED: Aggregate and print Dice scores ---
        try:
            metric = self.dice_metric.aggregate().item()
            metric_batch = self.dice_metric_batch.aggregate()
            
            metric_tc = metric_batch[0].item()
            metric_wt = metric_batch[1].item()
            metric_et = metric_batch[2].item()

            print(f"\n\n--- Dataset Dice Score ---")
            print(f"Overall Mean Dice: {metric:.4f}")
            print(f"TC Dice (Channel 0): {metric_tc:.4f}")
            print(f"WT Dice (Channel 1): {metric_wt:.4f}")
            print(f"ET Dice (Channel 2): {metric_et:.4f}")

            self.dice_metric.reset()
            self.dice_metric_batch.reset()
        
        except Exception as e:
            print(f"\nCould not calculate Dice scores: {e}. No labels found or all labels were empty.")
        # -----------------------------------------------


def main():
    """Example usage"""
    
    model_path = "best_metric_model_2021.pth"
    
    # Example paths - update these for your setup
    case_dir = "ASNR-MICCAI-BraTS2023-GLI-MET-TrainingData/BraTS-GLI-00732-001"
    dataset_dir = "ASNR-MICCAI-BraTS2023-GLI-MET-TrainingData"
    output_dir = "predictions"
    
    try:
        print("Initializing BraTS 2023 predictor with 2021 weights...")
        predictor = BraTSInference2023(model_path)
        
        # Example 1: Process single case
        if os.path.exists(case_dir):
            print(f"\n=== Processing single case ===")
            # <--- MODIFIED: Get both tensors ---
            prediction_tensor, label_tensor = predictor.predict_case(case_dir)
            
            # Save prediction
            image_paths, label_path = predictor.find_brats_2023_files(case_dir)
            output_path = f"{Path(case_dir).name}_prediction.nii.gz"
            predictor.save_prediction(prediction_tensor, image_paths[0], output_path) # <--- MODIFIED
            
            # <--- ADDED: Dice score for single case ---
            if label_tensor is not None:
                predictor.dice_metric(
                    y_pred=prediction_tensor.unsqueeze(0).to(predictor.device),
                    y=label_tensor.unsqueeze(0).to(predictor.device)
                )
                metric = predictor.dice_metric.aggregate().item()
                print(f"Dice for single case: {metric:.4f}")
                predictor.dice_metric.reset()
            # ------------------------------------------

        # Example 2: Process entire dataset
        if os.path.exists(dataset_dir):
            print(f"\n=== Processing entire dataset ===")
            predictor.process_dataset(dataset_dir, output_dir)
        
        if not os.path.exists(case_dir) and not os.path.exists(dataset_dir):
            print("Please update the paths in the script to point to your data.")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. BraTS 2021 trained model weights")
        print("2. BraTS 2023 data in the correct format")
        print("3. Updated paths in this script")


if __name__ == "__main__":
    main()