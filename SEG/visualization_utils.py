#!/usr/bin/env python3
"""
Visualization and Analysis Utilities for Brain Tumor Segmentation

This script contains utility functions for visualizing and analyzing
the brain tumor segmentation dataset and results.
"""

import os
import matplotlib.pyplot as plt
import torch
from monai.apps import DecathlonDataset
from monai.data import DataLoader
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
)


class ConvertToMultiChannelBasedOnBratsClassesd:
    """
    Convert labels to multi channels based on brats classes
    """
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(
                torch.logical_or(
                    torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                )
            )
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d


def setup_dataset(root_dir, section="training"):
    """Setup the dataset for visualization"""
    transform = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])
    
    dataset = DecathlonDataset(
        root_dir=root_dir,
        task="Task01_BrainTumour",
        transform=transform,
        section=section,
        download=True,
        cache_rate=0.0,
        num_workers=4,
    )
    
    return dataset


def visualize_sample(dataset, index=0, slice_idx=70):
    """
    Visualize a single sample from the dataset
    
    Args:
        dataset: MONAI dataset
        index: Sample index to visualize
        slice_idx: Axial slice index to display
    """
    sample = dataset[index]
    image = sample["image"]
    label = sample["label"]
    
    # Display the 4 input modalities
    plt.figure("Input Modalities", figsize=(20, 5))
    modality_names = ["FLAIR", "T1w", "T1gd", "T2w"]
    
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.title(f"{modality_names[i]} (channel {i})")
        plt.imshow(image[i, :, :, slice_idx].detach().cpu(), cmap="gray")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Display the 3 label channels
    plt.figure("Ground Truth Labels", figsize=(15, 5))
    label_names = ["Tumor Core (TC)", "Whole Tumor (WT)", "Enhancing Tumor (ET)"]
    
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(f"{label_names[i]} (channel {i})")
        plt.imshow(label[i, :, :, slice_idx].detach().cpu())
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def compare_multiple_slices(dataset, index=0, num_slices=5):
    """
    Compare multiple axial slices of a single sample
    
    Args:
        dataset: MONAI dataset
        index: Sample index to visualize
        num_slices: Number of slices to display
    """
    sample = dataset[index]
    image = sample["image"]
    label = sample["label"]
    
    # Get slice indices
    depth = image.shape[-1]
    slice_indices = [int(depth * (i + 1) / (num_slices + 1)) for i in range(num_slices)]
    
    # Display FLAIR modality across slices
    plt.figure("FLAIR across slices", figsize=(20, 4))
    for i, slice_idx in enumerate(slice_indices):
        plt.subplot(1, num_slices, i + 1)
        plt.title(f"Slice {slice_idx}")
        plt.imshow(image[0, :, :, slice_idx].detach().cpu(), cmap="gray")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Display tumor core labels across slices
    plt.figure("Tumor Core across slices", figsize=(20, 4))
    for i, slice_idx in enumerate(slice_indices):
        plt.subplot(1, num_slices, i + 1)
        plt.title(f"TC Slice {slice_idx}")
        plt.imshow(label[0, :, :, slice_idx].detach().cpu())
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def analyze_dataset_statistics(dataset):
    """
    Analyze and print dataset statistics
    
    Args:
        dataset: MONAI dataset
    """
    print(f"Dataset size: {len(dataset)}")
    
    # Analyze first sample
    sample = dataset[0]
    image = sample["image"]
    label = sample["label"]
    
    print(f"Image shape: {image.shape}")
    print(f"Label shape: {label.shape}")
    print(f"Image data type: {image.dtype}")
    print(f"Label data type: {label.dtype}")
    
    # Calculate statistics for each modality
    print("\nImage intensity statistics per modality:")
    modality_names = ["FLAIR", "T1w", "T1gd", "T2w"]
    for i in range(4):
        modality_data = image[i]
        print(f"{modality_names[i]}:")
        print(f"  Min: {modality_data.min():.4f}")
        print(f"  Max: {modality_data.max():.4f}")
        print(f"  Mean: {modality_data.mean():.4f}")
        print(f"  Std: {modality_data.std():.4f}")
    
    # Calculate label statistics
    print("\nLabel statistics:")
    label_names = ["Tumor Core (TC)", "Whole Tumor (WT)", "Enhancing Tumor (ET)"]
    for i in range(3):
        label_data = label[i]
        voxel_count = label_data.sum().item()
        total_voxels = label_data.numel()
        percentage = (voxel_count / total_voxels) * 100
        print(f"{label_names[i]}:")
        print(f"  Positive voxels: {voxel_count}")
        print(f"  Percentage: {percentage:.2f}%")


def visualize_multiple_samples(dataset, num_samples=4, slice_idx=70):
    """
    Visualize multiple samples from the dataset
    
    Args:
        dataset: MONAI dataset
        num_samples: Number of samples to display
        slice_idx: Axial slice index to display
    """
    fig, axes = plt.subplots(num_samples, 7, figsize=(28, 4 * num_samples))
    
    modality_names = ["FLAIR", "T1w", "T1gd", "T2w"]
    label_names = ["TC", "WT", "ET"]
    
    for sample_idx in range(num_samples):
        sample = dataset[sample_idx]
        image = sample["image"]
        label = sample["label"]
        
        # Display modalities
        for mod_idx in range(4):
            ax = axes[sample_idx, mod_idx] if num_samples > 1 else axes[mod_idx]
            ax.imshow(image[mod_idx, :, :, slice_idx].detach().cpu(), cmap="gray")
            ax.set_title(f"Sample {sample_idx} - {modality_names[mod_idx]}")
            ax.axis('off')
        
        # Display labels
        for label_idx in range(3):
            ax = axes[sample_idx, 4 + label_idx] if num_samples > 1 else axes[4 + label_idx]
            ax.imshow(label[label_idx, :, :, slice_idx].detach().cpu())
            ax.set_title(f"Sample {sample_idx} - {label_names[label_idx]}")
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function for visualization utilities"""
    # Setup
    root_dir = os.environ.get("MONAI_DATA_DIRECTORY", "/tmp/monai_data")
    print(f"Data directory: {root_dir}")
    
    # Load dataset
    print("Loading training dataset...")
    train_dataset = setup_dataset(root_dir, section="training")
    
    print("Loading validation dataset...")
    val_dataset = setup_dataset(root_dir, section="validation")
    
    # Analyze dataset
    print("\n=== Training Dataset Analysis ===")
    analyze_dataset_statistics(train_dataset)
    
    print("\n=== Validation Dataset Analysis ===")
    analyze_dataset_statistics(val_dataset)
    
    # Visualize samples (only if matplotlib can display)
    try:
        print("\n=== Visualizing Training Samples ===")
        visualize_sample(train_dataset, index=0)
        compare_multiple_slices(train_dataset, index=0)
        visualize_multiple_samples(train_dataset, num_samples=3)
        
        print("\n=== Visualizing Validation Samples ===")
        visualize_sample(val_dataset, index=0)
        
    except Exception as e:
        print(f"Could not display visualization plots: {e}")
        print("This is normal if running in a headless environment.")


if __name__ == "__main__":
    main()
