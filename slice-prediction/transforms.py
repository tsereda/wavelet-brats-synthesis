from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    Resized,
    EnsureTyped,
)

def get_train_transforms(image_size, spacing):
    """
    Get MONAI transforms for loading and preprocessing BraTS volumes
    Args:
        image_size: tuple (H, W) - target 2D slice size
        spacing: tuple (x, y, z) - target voxel spacing
    Returns:
        MONAI Compose transform
    Note: Updated for BraTS2023 GLI format with keys:
        t1n (native T1), t1c (contrast-enhanced T1), t2w (T2-weighted), t2f (T2-FLAIR)
    """
    # BraTS2023 GLI format keys
    keys = ["t1n", "t1c", "t2w", "t2f", "label"]
    
    return Compose([
        # Load NIfTI files - allow missing keys for validation data
        LoadImaged(keys=keys, allow_missing_keys=True),
        
        # Ensure channel-first format [C, H, W, D]
        EnsureChannelFirstd(keys=keys, allow_missing_keys=True),
        
        # Reorient to standard orientation
        Orientationd(keys=keys, axcodes="RAS", allow_missing_keys=True),
        
        # Resample to target spacing
        Spacingd(
            keys=keys,
            pixdim=spacing,
            mode=("bilinear", "bilinear", "bilinear", "bilinear", "nearest"),
            allow_missing_keys=True,
        ),
        
        # Normalize intensity for imaging modalities (only modalities, not label)
        ScaleIntensityRanged(
            keys=["t1n", "t1c", "t2w", "t2f"],
            a_min=0.0,
            a_max=1000.0,  # Adjust based on your data
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        
        # Crop to foreground (remove empty space)
        CropForegroundd(
            keys=keys,
            source_key="t1c",  # Use T1c (contrast-enhanced) to detect brain
            margin=10,
            allow_missing_keys=True,
        ),
        
        # Resize spatial dimensions (H, W) - depth stays same
        Resized(
            keys=keys,
            spatial_size=(image_size[0], image_size[1], -1),  # -1 keeps original depth
            mode=("trilinear", "trilinear", "trilinear", "trilinear", "nearest"),
            allow_missing_keys=True,
        ),
        
        # Convert to PyTorch tensors
        EnsureTyped(keys=keys, allow_missing_keys=True),
    ])