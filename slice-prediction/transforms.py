# transforms.py
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
    """
    keys = ["t1", "t1ce", "t2", "flair", "label"]
    
    return Compose([
        # Load NIfTI files
        LoadImaged(keys=keys),
        
        # Ensure channel-first format [C, H, W, D]
        EnsureChannelFirstd(keys=keys),
        
        # Reorient to standard orientation
        Orientationd(keys=keys, axcodes="RAS"),
        
        # Resample to target spacing
        Spacingd(
            keys=keys,
            pixdim=spacing,
            mode=("bilinear", "bilinear", "bilinear", "bilinear", "nearest"),
        ),
        
        # Normalize intensity for imaging modalities
        ScaleIntensityRanged(
            keys=["t1", "t1ce", "t2", "flair"],
            a_min=0.0,
            a_max=1000.0,  # Adjust based on your data
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        
        # Crop to foreground (remove empty space)
        CropForegroundd(
            keys=keys,
            source_key="t1ce",  # Use T1ce to detect brain
            margin=10,
        ),
        
        # Resize spatial dimensions (H, W) - depth stays same
        Resized(
            keys=keys,
            spatial_size=(image_size[0], image_size[1], -1),  # -1 keeps original depth
            mode=("trilinear", "trilinear", "trilinear", "trilinear", "nearest"),
        ),
        
        # Convert to PyTorch tensors
        EnsureTyped(keys=keys),
    ])