"""Fast dataset that loads pre-saved .pt slice triplets.

Each .pt file should contain a dict with keys: 'input', 'target', 'patient', 'slice_idx'
as produced by `preprocess_slices_to_tensors.py`.
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

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            sample = torch.load(path, map_location='cpu')
        except Exception as e:
            # If a file fails to load, skip to the next one (wrap-around)
            new_idx = (idx + 1) % len(self.files)
            if new_idx == idx:
                raise RuntimeError(f"Failed to load any samples; error loading {path}: {e}")
            return self.__getitem__(new_idx)

        # Expect tensors under keys 'input' and 'target'
        input_t = sample.get('input')
        target_t = sample.get('target')

        if input_t is None or target_t is None:
            raise KeyError(f"Sample {path} missing 'input' or 'target' keys")

        # Return (input, target, (slice_idx, patient_id)) tuple for compatibility
        slice_idx = sample.get('slice_idx', -1)
        patient_id = sample.get('patient', f'unknown_{idx}')
        
        # Ensure float32 tensors
        return input_t.float(), target_t.float(), (slice_idx, patient_id)
