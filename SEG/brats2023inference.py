#!/usr/bin/env python3
"""
BraTS 2023 Inference with BraTS 2021 Trained Weights

This script adapts the inference for BraTS 2023 data format using weights
trained on BraTS 2021. The tumor definitions are consistent across years.
"""

import os
import torch
import numpy as np
import glob
from pathlib import Path
from monai.networks.nets import SegResNet
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
import nibabel as nib


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
        
        # Create model (same architecture as 2021)
        self.model = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=4,
            out_channels=3,
            dropout_prob=0.2,
        ).to(self.device)
        
        # Load weights
        self.load_model(model_path)
        
        # Setup transforms
        self.setup_transforms()
        
    def load_model(self, model_path):
        """Load pre-trained model weights"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading BraTS 2021 trained model from: {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        print("Model loaded successfully!")
        
    def setup_transforms(self):
        """Setup preprocessing and postprocessing transforms"""
        self.preprocess = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ])
        
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
            Dictionary with modality paths
        """
        case_dir = Path(case_dir)
        case_name = case_dir.name
        
        # BraTS 2023 naming convention
        files = {
            "t1n": case_dir / f"{case_name}-t1n.nii.gz",      # T1 native
            "t1c": case_dir / f"{case_name}-t1c.nii.gz",      # T1 contrast enhanced  
            "t2f": case_dir / f"{case_name}-t2f.nii.gz",      # T2 FLAIR
            "t2w": case_dir / f"{case_name}-t2w.nii.gz",      # T2 weighted
        }
        
        # Check if files exist
        missing_files = []
        for modality, filepath in files.items():
            if not filepath.exists():
                missing_files.append(str(filepath))
        
        if missing_files:
            raise FileNotFoundError(f"Missing files: {missing_files}")
        
        # Map to expected order: [FLAIR, T1, T1ce, T2]
        image_paths = [
            str(files["t2f"]),  # FLAIR
            str(files["t1n"]),  # T1 native  
            str(files["t1c"]),  # T1 contrast enhanced
            str(files["t2w"]),  # T2 weighted
        ]
        
        return image_paths
    
    def predict_case(self, case_dir, use_amp=True):
        """
        Predict on a BraTS 2023 case
        
        Args:
            case_dir: Directory containing case files
            use_amp: Use automatic mixed precision
            
        Returns:
            Segmentation prediction as numpy array [3, H, W, D]
        """
        # Auto-detect files
        image_paths = self.find_brats_2023_files(case_dir)
        
        print(f"Processing case: {Path(case_dir).name}")
        print(f"Found modalities: {[Path(p).name for p in image_paths]}")
        
        # Prepare data dictionary
        data = {"image": image_paths}
        
        # Apply transforms
        data = self.preprocess(data)
        image = data["image"].unsqueeze(0).to(self.device)
        
        print(f"Input shape: {image.shape}")
        
        # Run inference
        with torch.no_grad():
            if use_amp and self.device.type == "cuda":
                with torch.autocast("cuda"):
                    prediction = sliding_window_inference(
                        inputs=image,
                        roi_size=(240, 240, 160),
                        sw_batch_size=1,
                        predictor=self.model,
                        overlap=0.5,
                    )
            else:
                prediction = sliding_window_inference(
                    inputs=image,
                    roi_size=(240, 240, 160),
                    sw_batch_size=1,
                    predictor=self.model,
                    overlap=0.5,
                )
        
        # Postprocess
        prediction = self.postprocess(prediction[0])
        
        return prediction.cpu().numpy()
    
    def save_prediction(self, prediction, reference_image_path, output_path):
        """
        Save prediction as NIfTI file
        
        Args:
            prediction: Prediction array [3, H, W, D]
            reference_image_path: Path to reference NIfTI file for header
            output_path: Output path for segmentation
        """
        # Load reference to get header info
        ref_img = nib.load(reference_image_path)
        
        # Convert multi-channel to single label image
        # Following BraTS convention: 1=NCR/NET, 2=ED, 4=ET
        label_img = np.zeros(prediction.shape[1:], dtype=np.uint8)
        
        # ET (Enhancing Tumor) = label 4
        label_img[prediction[2] > 0] = 4
        
        # TC (Tumor Core: NCR/NET) = label 1  
        label_img[prediction[0] > 0] = 1
        
        # ED (Edema) = label 2 (but exclude areas already labeled as TC or ET)
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
        
        # Find all case directories
        case_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
        print(f"Found {len(case_dirs)} cases to process")
        
        for i, case_dir in enumerate(case_dirs):
            print(f"\n--- Processing {i+1}/{len(case_dirs)}: {case_dir.name} ---")
            
            try:
                # Run prediction
                prediction = self.predict_case(case_dir)
                
                # Get reference image (use FLAIR)
                image_paths = self.find_brats_2023_files(case_dir)
                reference_path = image_paths[0]  # FLAIR
                
                # Save prediction
                output_path = output_dir / f"{case_dir.name}-seg.nii.gz"
                self.save_prediction(prediction, reference_path, output_path)
                
            except Exception as e:
                print(f"Error processing {case_dir.name}: {e}")
                continue


def main():
    """Example usage"""
    
    # Configuration
    model_path = "best_metric_model_2021.pth"  # Your BraTS 2021 trained weights
    
    # Example paths - update these for your setup
    case_dir = "path/to/BraTS2023_case"  # Single case directory
    dataset_dir = "path/to/BraTS2023_dataset"  # Full dataset directory
    output_dir = "predictions"
    
    try:
        # Initialize predictor
        print("Initializing BraTS 2023 predictor with 2021 weights...")
        predictor = BraTSInference2023(model_path)
        
        # Example 1: Process single case
        if os.path.exists(case_dir):
            print(f"\n=== Processing single case ===")
            prediction = predictor.predict_case(case_dir)
            
            # Save prediction
            image_paths = predictor.find_brats_2023_files(case_dir)
            output_path = f"{Path(case_dir).name}_prediction.nii.gz"
            predictor.save_prediction(prediction, image_paths[0], output_path)
        
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