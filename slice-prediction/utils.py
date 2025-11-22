"""
Utility functions for patient ID extraction and organized output file management.
"""
import numpy as np
from pathlib import Path
from typing import Tuple, Union, Optional
import torch


def extract_patient_info(slice_indices, batch_idx: int, sample_idx: int) -> Tuple[int, str]:
    """
    Centralized logic for extracting slice index and patient ID from dataset outputs.
    
    Args:
        slice_indices: Output from dataset __getitem__ (can be int, tensor, or (slice_idx, patient_id) tuple)
        batch_idx: Current batch index (for fallback naming)
        sample_idx: Sample index within batch (for fallback naming)
    
    Returns:
        tuple: (slice_idx: int, patient_id: str)
    
    Examples:
        >>> # From a batch of tuples
        >>> slice_indices = [(78, 'BraTS-GLI-00002-000'), (79, 'BraTS-GLI-00002-000')]
        >>> extract_patient_info(slice_indices, 0, 0)
        (78, 'BraTS-GLI-00002-000')
        
        >>> # From a single tensor
        >>> slice_indices = torch.tensor([78, 79])
        >>> extract_patient_info(slice_indices, 0, 0)
        (78, 'sample0_slice78')
    """
    try:
        slice_info = slice_indices[sample_idx] if hasattr(slice_indices, '__getitem__') else slice_indices
        
        if isinstance(slice_info, (list, tuple)) and len(slice_info) >= 2:
            slice_idx = int(slice_info[0])
            patient_id = str(slice_info[1])
        else:
            # Handle tensor or simple int
            try:
                slice_idx = int(slice_info.item()) if hasattr(slice_info, 'item') else int(slice_info)
            except Exception:
                slice_idx = int(slice_info)
            patient_id = f"sample{sample_idx}_slice{slice_idx}"
        
        return slice_idx, patient_id
        
    except Exception:
        # Ultimate fallback
        return -1, f"sample{sample_idx}"


def get_patient_output_dir(base_dir: Path, patient_id: str, slice_idx: int) -> Path:
    """
    Create and return patient-specific output directory.
    
    Args:
        base_dir: Base predictions directory
        patient_id: Patient identifier
        slice_idx: Slice index
    
    Returns:
        Path to patient-specific directory (created if doesn't exist)
    
    Example:
        >>> base_dir = Path('./predictions')
        >>> patient_dir = get_patient_output_dir(base_dir, 'BraTS-GLI-00002-000', 78)
        >>> # Returns: ./predictions/BraTS-GLI-00002-000_slice078/
    """
    # Create directory name with patient_id and slice number
    dir_name = f"{patient_id}_slice{slice_idx:03d}"
    patient_dir = base_dir / dir_name
    patient_dir.mkdir(parents=True, exist_ok=True)
    return patient_dir


def save_slice_outputs(
    patient_dir: Path,
    inputs: torch.Tensor,
    target: torch.Tensor,
    output: torch.Tensor,
    slice_idx: int,
    patient_id: str,
    batch_idx: int,
    input_wavelets: Optional[torch.Tensor] = None,
    output_wavelets: Optional[torch.Tensor] = None,
    target_wavelets: Optional[torch.Tensor] = None
) -> dict:
    """
    Save all outputs for a single slice in organized patient-specific directory.
    
    Args:
        patient_dir: Patient-specific output directory
        inputs: Input tensor [8, H, W] (prev + next slices, 4 modalities each)
        target: Target tensor [4, H, W] (ground truth middle slice)
        output: Output tensor [4, H, W] (predicted middle slice)
        slice_idx: Slice index
        patient_id: Patient identifier
        batch_idx: Batch index
        input_wavelets: Optional wavelet coefficients for input
        output_wavelets: Optional wavelet coefficients for output
        target_wavelets: Optional wavelet coefficients for target
    
    Returns:
        dict: Paths to all saved files
    
    File structure created:
        {patient_id}_slice{slice_idx:03d}/
        ├── input_prev_slice.npy
        ├── input_next_slice.npy
        ├── target_middle_slice.npy
        ├── prediction_middle_slice.npy
        ├── wavelet_input.npy (if wavelets provided)
        ├── wavelet_output.npy (if wavelets provided)
        └── wavelet_target.npy (if wavelets provided)
    """
    saved_files = {}
    
    # Save input slices (split into prev and next)
    inputs_np = inputs.cpu().numpy()
    prev_slice = inputs_np[:4]  # First 4 channels (t1n, t1c, t2w, t2f at Z-1)
    next_slice = inputs_np[4:]  # Last 4 channels (t1n, t1c, t2w, t2f at Z+1)
    
    prev_path = patient_dir / 'input_prev_slice.npy'
    next_path = patient_dir / 'input_next_slice.npy'
    np.save(prev_path, prev_slice)
    np.save(next_path, next_slice)
    saved_files['input_prev'] = prev_path
    saved_files['input_next'] = next_path
    
    # Save target (ground truth middle slice)
    target_path = patient_dir / 'target_middle_slice.npy'
    np.save(target_path, target.cpu().numpy())
    saved_files['target'] = target_path
    
    # Save prediction (model output)
    prediction_path = patient_dir / 'prediction_middle_slice.npy'
    np.save(prediction_path, output.cpu().numpy())
    saved_files['prediction'] = prediction_path
    
    # Save wavelets if provided
    if input_wavelets is not None:
        wavelet_input_path = patient_dir / 'wavelet_input.npy'
        np.save(wavelet_input_path, input_wavelets.cpu().numpy())
        saved_files['wavelet_input'] = wavelet_input_path
    
    if output_wavelets is not None:
        wavelet_output_path = patient_dir / 'wavelet_output.npy'
        np.save(wavelet_output_path, output_wavelets.cpu().numpy())
        saved_files['wavelet_output'] = wavelet_output_path
    
    if target_wavelets is not None:
        wavelet_target_path = patient_dir / 'wavelet_target.npy'
        np.save(wavelet_target_path, target_wavelets.cpu().numpy())
        saved_files['wavelet_target'] = wavelet_target_path
    
    # Save metadata
    metadata_path = patient_dir / 'metadata.npz'
    np.savez(
        metadata_path,
        slice_idx=slice_idx,
        batch_idx=batch_idx,
        patient_id=patient_id
    )
    saved_files['metadata'] = metadata_path
    
    return saved_files
