#!/usr/bin/env python3
"""
BraTS 2023 Inference with BraTS 2021 Trained Weights (SwinUNETR)

This script adapts the inference for BraTS 2023 data format using weights
trained on BraTS 2021 and calculates Dice scores if labels are available.

CHANGELOG:
- Added CropForegroundd before normalization to match training preprocessing.
- Correctly configured Invertd to use "t2f" as the reference key.
- Applied inverse transform to both prediction and label for correct Dice/saving.
- Removed buggy padding logic from save_prediction.
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
    ConcatItemsd,
    CropForegroundd, # <--- ADDED
    Invertd          # <--- ADDED
)
from monai.inferers import sliding_window_inference
from monai.data import Dataset, DataLoader
from monai.metrics import DiceMetric
import nibabel as nib


# <--- NEW TRANSFORM: Maps BraTS 2023 labels to model output channels ---
class MapBraTS2023LabelsToModelOutputd(MapTransform):
    """
    Convert BraTS 2023 ground truth labels to 3-channel format 
    to match the BraTS 2021 model's output.

    BraTS 2023 Labels:
    - Label 1: Enhancing Tumor (ET)
    - Label 2: Tumor Core (TC)
    - Label 3: Whole Tumor (WT)

    Model (BraTS 2021) Output Channels:
    - Channel 0: TC (Tumor Core)
    - Channel 1: WT (Whole Tumor)
    - Channel 2: ET (Enhancing Tumor)
    """
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key not in d or d[key] is None:
                continue
            
            # Input d[key] has shape (1, H, W, D) with values 0, 1, 2, 3
            label_vol = d[key]
            
            result = []
            # Channel 0: TC (Tumor Core) -> BraTS 2023 Label 2
            result.append(label_vol == 2)
            # Channel 1: WT (Whole Tumor) -> BraTS 2023 Label 3
            result.append(label_vol == 3)
            # Channel 2: ET (Enhancing Tumor) -> BraTS 2023 Label 1
            result.append(label_vol == 1)
            
            # Concat along channel dim (axis=0) to get (3, H, W, D)
            d[key] = torch.cat(result, axis=0).float()
        return d
# ---------------------------------------------------------------------


class BraTSInference2023:
    """BraTS 2023 inference using BraTS 2021 trained weights"""

    def __init__(self, model_path, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=4,
            out_channels=3,
            feature_size=48,
            use_checkpoint=False,
        ).to(self.device)

        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch")
        
        self.load_model(model_path)
        self.setup_transforms()

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        print(f"Loading BraTS 2021 trained model from: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        print("Model loaded successfully!")

    # <--- MODIFIED: Updated transform pipeline ---
    def setup_transforms(self):
        """Setup preprocessing and postprocessing transforms"""
        
        image_keys = ["t2f", "t1n", "t1c", "t2w"]
        all_keys = image_keys + ["label"]

        self.preprocess = Compose([
            LoadImaged(keys=all_keys, allow_missing_keys=True),
            EnsureChannelFirstd(keys=all_keys, allow_missing_keys=True),
            EnsureTyped(keys=all_keys, allow_missing_keys=True),
            Orientationd(keys=all_keys, axcodes="RAS", allow_missing_keys=True),
            Spacingd(
                keys=all_keys,
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "bilinear", "bilinear", "bilinear", "nearest"),
                allow_missing_keys=True,
            ),
            
            # --- ADDED CROPFOREGROUNDD ---
            CropForegroundd(
                keys=all_keys, 
                source_key="t2f", # Use FLAIR as the source
                allow_missing_keys=True,
                k_divisible=[96, 96, 96] # Pad to be divisible by patch size
            ),
            # -----------------------------
            
            NormalizeIntensityd(keys=image_keys, nonzero=True, channel_wise=True, allow_missing_keys=True),
            MapBraTS2023LabelsToModelOutputd(keys="label", allow_missing_keys=True),
            ConcatItemsd(keys=["t2f", "t1n", "t1c", "t2w"], name="image", dim=0), # Stack channels
        ])
        
        self.postprocess = Compose([
            Activations(sigmoid=True),
            AsDiscrete(threshold=0.5)
        ])
        
        # --- ADDED INVERSE TRANSFORM ---
        # This will be used to revert the crop/pad after prediction
        self.inverse_transform = Compose([
            Invertd(
                keys=["pred", "label"], # Invert both pred and label
                transform=self.preprocess,
                orig_keys="t2f", # Use t2f's metadata as the reference
                meta_keys=["pred_meta_dict", "label_meta_dict"],
                orig_meta_keys="t2f_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=True, # Use nearest neighbor for seg masks
                allow_missing_keys=True # Don't fail if label is missing
            )
        ])
    # -------------------------------------------------

    def find_brats_2023_files(self, case_dir):
        """
        Auto-detect BraTS 2023 file naming convention
        """
        case_dir = Path(case_dir)
        case_name = case_dir.name
        
        files = {
            "t1n": case_dir / f"{case_name}-t1n.nii.gz",
            "t1c": case_dir / f"{case_name}-t1c.nii.gz",
            "t2f": case_dir / f"{case_name}-t2f.nii.gz",
            "t2w": case_dir / f"{case_name}-t2w.nii.gz",
            "label": case_dir / f"{case_name}-seg.nii.gz",
        }
        
        existing_files = {}
        missing_images = []
        for key, path in files.items():
            if path.exists():
                existing_files[key] = str(path)
            elif key != "label":
                missing_images.append(str(path))
        
        if missing_images:
            raise FileNotFoundError(f"Missing image files: {missing_images}")
        
        if "label" not in existing_files:
             print(f"Warning: No label file found for {case_name}")

        return existing_files
    

    # <--- MODIFIED: Updated data loading & inverse transform ---
    def predict_case(self, case_dir, use_amp=True):
        """
        Predict on a BraTS 2023 case.
        """
        data = self.find_brats_2023_files(case_dir)
        
        print(f"Processing case: {Path(case_dir).name}")
        print(f"Found modalities: {[Path(p).name for k, p in data.items() if k != 'label']}")
        if "label" in data:
            print(f"Found label: {Path(data['label']).name}")
            
        # Apply preprocessing (data is now cropped)
        data = self.preprocess(data)
        
        image = data["image"].unsqueeze(0).to(self.device) 
        
        print(f"Input shape after crop/preprocess: {image.shape}")
        
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
        
        # Postprocess the (cropped) prediction
        prediction = self.postprocess(prediction[0])
        
        # --- APPLY INVERSE TRANSFORM ---
        # Add the cropped prediction back to the dictionary
        data["pred"] = prediction 
        
        # Apply the inverse transform to "pred" and "label" keys
        data = self.inverse_transform(data) 
        
        # Now get the *inverted* (full-size) tensors
        prediction_full = data["pred"]
        label_full = data.get("label") # This will be the full-size label, or None
        
        print(f"Output shape after inverse crop: {prediction_full.shape}")
        if label_full is not None:
            print(f"Label shape after inverse crop: {label_full.shape}")
        
        return prediction_full, label_full
    # -----------------------------------------------------------

    # <--- MODIFIED: Removed padding logic ---
    def save_prediction(self, prediction_tensor, reference_image_path, output_path):
        """
        Save prediction as NIfTI file in BraTS 2023 format (Labels 1, 2, 3)
        """
        prediction = prediction_tensor.cpu().numpy()
        
        ref_img = nib.load(reference_image_path)
        
        # Check if inverse transform worked. If not, save cropped as fallback.
        if prediction.shape[1:] != ref_img.shape:
            print(f"Error: Prediction shape {prediction.shape[1:]} does not match reference {ref_img.shape}.")
            print("Saving cropped image as fallback.")
            # This is not ideal but better than crashing
            ref_img.header.set_data_shape(prediction.shape[1:])
        else:
            print("Prediction shape matches reference shape.")

        
        label_img = np.zeros(prediction.shape[1:], dtype=np.uint8)
        
        # Apply labels in order (WT -> TC -> ET)
        label_img[prediction[1] > 0] = 3
        label_img[prediction[0] > 0] = 2
        label_img[prediction[2] > 0] = 1
        
        pred_img = nib.Nifti1Image(label_img, ref_img.affine, ref_img.header)
        
        nib.save(pred_img, output_path)
        print(f"Prediction saved to: {output_path}")
        
        unique_labels, counts = np.unique(label_img, return_counts=True)
        total_voxels = label_img.size
        
        print(f"Segmentation statistics (BraTS 2023 format):")
        label_map = {0: "Background", 1: "Enhancing (ET)", 2: "Tumor Core (TC)", 3: "Whole Tumor (WT)"}
        stats = {label: 0 for label in label_map}
        for label, count in zip(unique_labels, counts):
            if label in stats:
                stats[label] = count

        for label, name in label_map.items():
            count = stats[label]
            if label == 0:
                print(f"  {name} (0): {count} voxels")
            else:
                percentage = (count / total_voxels) * 100
                print(f"  {name} ({label}): {count} voxels ({percentage:.2f}%)")
    # ---------------------------------------------------------------------

    def process_dataset(self, dataset_dir, output_dir):
        """
        Process entire BraTS 2023 dataset
        """
        dataset_dir = Path(dataset_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        case_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
        print(f"Found {len(case_dirs)} cases to process")
        
        # --- Limit to 20 cases for testing ---
        case_dirs = case_dirs[:20]
        # ------------------------------------
        
        for i, case_dir in enumerate(case_dirs):
            print(f"\n--- Processing {i+1}/{len(case_dirs)}: {case_dir.name} ---")
            
            try:
                # Both tensors should now be full-size
                prediction_tensor, label_tensor = self.predict_case(case_dir)
                
                data_paths = self.find_brats_2023_files(case_dir)
                reference_path = data_paths["t2f"]  # Use FLAIR as reference
                
                output_path = output_dir / f"{case_dir.name}-seg.nii.gz"
                self.save_prediction(prediction_tensor, reference_path, output_path)
                
                if label_tensor is not None:
                    # Add batch dimension for metric calculation
                    y_pred_batch = prediction_tensor.unsqueeze(0).to(self.device)
                    y_label_batch = label_tensor.unsqueeze(0).to(self.device)
                    
                    self.dice_metric(y_pred=y_pred_batch, y=y_label_batch)
                    self.dice_metric_batch(y_pred=y_pred_batch, y=y_label_batch)
                
            except Exception as e:
                print(f"Error processing {case_dir.name}: {e}")
                continue
        
        # <--- MODIFIED: Robust metric aggregation ---
        try:
            metric = self.dice_metric.aggregate().item()
            metric_batch = self.dice_metric_batch.aggregate()
            
            print(f"\n\n--- Dataset Dice Score (First {len(case_dirs)} cases) ---")
            print(f"Overall Mean Dice: {metric:.4f}")

            num_channels_aggregated = metric_batch.numel()

            if num_channels_aggregated == 3:
                metric_tc = metric_batch[0].item()
                metric_wt = metric_batch[1].item()
                metric_et = metric_batch[2].item()
                print(f"TC Dice (Channel 0): {metric_tc:.4f}")
                print(f"WT Dice (Channel 1): {metric_wt:.4f}")
                print(f"ET Dice (Channel 2): {metric_et:.4f}")
            else:
                print(f"Warning: Aggregated metrics only found {num_channels_aggregated} channels, expected 3.")
                if num_channels_aggregated > 0:
                    print(f"Channel 0 Dice: {metric_batch[0].item():.4f}")
                if num_channels_aggregated > 1:
                    print(f"Channel 1 Dice: {metric_batch[1].item():.4f}")

            self.dice_metric.reset()
            self.dice_metric_batch.reset()
        
        except Exception as e:
            print(f"\nCould not calculate Dice scores: {e}. Check if labels were loaded correctly.")
    # -----------------------------------------------------------------


def main():
    """Example usage"""
    
    # !! UPDATE THIS PATH !!
    model_path = "best_metric_model_2021.pth"
    
    # !! UPDATE THESE PATHS !!
    case_dir = "ASNR-MICCAI-BraTS2023-GLI-MET-TrainingData/BraTS-GLI-00732-001"
    dataset_dir = "ASNR-MICCAI-BraTS2023-GLI-MET-TrainingData"
    output_dir = "predictions_2023_format"
    
    try:
        print("Initializing BraTS 2023 predictor with 2021 weights...")
        predictor = BraTSInference2023(model_path)
        
        if os.path.exists(case_dir) and os.path.isdir(case_dir):
            print(f"\n=== Processing single case ===")
            prediction_tensor, label_tensor = predictor.predict_case(case_dir)
            
            data_paths = predictor.find_brats_2023_files(case_dir)
            output_path = f"{Path(case_dir).name}_prediction.nii.gz"
            predictor.save_prediction(prediction_tensor, data_paths["t2f"], output_path)
            
            if label_tensor is not None:
                predictor.dice_metric(
                    y_pred=prediction_tensor.unsqueeze(0).to(predictor.device),
                    y=label_tensor.unsqueeze(0).to(predictor.device)
                )
                metric = predictor.dice_metric.aggregate().item()
                print(f"Dice for single case: {metric:.4f}")
                predictor.dice_metric.reset()

        elif os.path.exists(dataset_dir) and os.path.isdir(dataset_dir):
            print(f"\n=== Processing entire dataset ===")
            predictor.process_dataset(dataset_dir, output_dir)
        
        else:
            print("Please update the paths in the script to point to your data.")
            print(f"Checked 'case_dir': {case_dir}")
            print(f"Checked 'dataset_dir': {dataset_dir}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. BraTS 2021 trained model weights (best_metric_model_2021.pth)")
        print("2. BraTS 2023 data in the correct format")
        print("3. Updated paths in this script")


if __name__ == "__main__":
    main()