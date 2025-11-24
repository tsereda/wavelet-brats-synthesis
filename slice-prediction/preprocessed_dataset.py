"""Fast dataset that loads pre-saved .pt slice triplets.

Each .pt file should contain a dict with keys: 'input', 'target', 'patient', 'slice_idx'
as produced by `preprocess_slices_to_tensors.py`.

This version uses a non-recursive loop in __getitem__ to prevent RecursionError 
when encountering corrupted files.
"""
import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path


class FastTensorSliceDataset(Dataset):
    """Loads individual sample files saved as .pt for maximal speed.

    Args:
        preprocessed_dir: directory containing .pt files and optional preprocessed_slices.csv
    """
    def __init__(self, preprocessed_dir):
        self.dir = Path(preprocessed_dir)
        if not self.dir.exists():
            raise FileNotFoundError(f"Preprocessed directory not found: {preprocessed_dir}")

        csv_index = self.dir / 'preprocessed_slices.csv'
        if csv_index.exists():
            df = pd.read_csv(csv_index)
            # Expect first column to be filepath
            if 'filepath' in df.columns:
                self.files = df['filepath'].tolist()
            else:
                # fallback: try to build from rows
                self.files = df.iloc[:, 0].astype(str).tolist()
        else:
            # find all .pt files
            self.files = sorted([str(p) for p in self.dir.glob('*.pt')])

        if not self.files:
            raise RuntimeError(f"No .pt files found in {preprocessed_dir}")
        
        # Store the original starting index to check for full wrap-around
        self.initial_len = len(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # We use a non-recursive loop to safely skip corrupted files.
        start_idx = idx
        current_idx = idx
        
        while True:
            path = self.files[current_idx]
            try:
                # FIX: Set weights_only=False to allow loading of files saved 
                # with older PyTorch versions that included NumPy globals.
                sample = torch.load(path, map_location='cpu', weights_only=False)
                
                # --- Successful Load, now validate keys ---
                input_t = sample.get('input')
                target_t = sample.get('target')

                if input_t is None or target_t is None:
                    # Treat missing keys as a file error and skip it
                    raise KeyError(f"Sample {path} missing 'input' or 'target' keys")

                # Return (input, target, (slice_idx, patient_id)) tuple for compatibility
                slice_idx = sample.get('slice_idx', -1)
                patient_id = sample.get('patient', f'unknown_{current_idx}')
                
                # Ensure float32 tensors
                return input_t.float(), target_t.float(), (slice_idx, patient_id)

            except Exception as e:
                # Log the error and move to the next index
                print(f"File Load Error: Skipping corrupted file {path}. Error: {e}")
                
                # Calculate the next index (wrap-around)
                current_idx = (current_idx + 1) % self.initial_len
                
                # If we have wrapped all the way back to the starting index, 
                # we've checked every file and nothing worked.
                if current_idx == start_idx:
                    raise RuntimeError("Failed to load any valid samples after checking the entire dataset list.")
                
                # Continue the loop to try the new index (current_idx)

# Note: The DataLoader will autom
# atically handle the exception raised by this 
# loop if no files can be loaded.