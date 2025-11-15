"""
Patient-level dataset splitting utilities.

Functions:
 - extract_patient_ids(dataset) -> list of patient_id strings (one per sample index)
 - split_dataset_by_patients(dataset, train_ratio, val_ratio, test_ratio, seed) -> (train_indices, val_indices, test_indices, patient_splits)
 - get_split_loaders(dataset, batch_size, eval_batch_size, num_workers, train_ratio, val_ratio, test_ratio, seed)

Provides conservative heuristics to derive patient IDs from three dataset classes used in this repo:
 - BraTS2D5Dataset (has .slice_map and .files)
 - FastTensorSliceDataset (has .files list of .pt paths and .dir)
 - SimpleCSVTripletDataset (has .samples DataFrame with t1c or filepaths)

If it cannot infer patient ids, it raises a clear error.

The module also includes a small CLI for quick inspection.
"""

from pathlib import Path
import os
import random
from collections import defaultdict
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader


def extract_patient_ids(dataset):
    """Return a list patient_id for each sample index in the dataset.

    The function tries multiple heuristics depending on dataset attributes.
    """
    # BraTS2D5Dataset (slice_map + files)
    try:
        if hasattr(dataset, 'slice_map') and hasattr(dataset, 'files'):
            patient_ids = []
            for (pidx, _) in dataset.slice_map:
                # patient_files is a dict with modality paths
                patient_files = dataset.files[pidx]
                first_path = None
                for v in patient_files.values():
                    if isinstance(v, str) and len(v) > 0:
                        first_path = v
                        break
                if first_path:
                    pid = Path(first_path).parent.name
                else:
                    pid = f"patient_{pidx}"
                patient_ids.append(pid)
            return patient_ids
    except Exception:
        pass

    # FastTensorSliceDataset: dataset.files is a list of .pt paths (strings)
    try:
        if hasattr(dataset, 'files') and all(isinstance(p, str) for p in dataset.files):
            # Try to read preprocessed_slices.csv if present
            if hasattr(dataset, 'dir'):
                csvp = Path(getattr(dataset, 'dir')) / 'preprocessed_slices.csv'
                if csvp.exists():
                    try:
                        df = pd.read_csv(csvp)
                        if 'patient' in df.columns:
                            return df['patient'].astype(str).tolist()
                    except Exception:
                        pass
            # Fallback to parent directory name of each file
            patient_ids = [Path(p).parent.name for p in dataset.files]
            return patient_ids
    except Exception:
        pass

    # SimpleCSVTripletDataset: dataset.samples is a pandas.DataFrame
    try:
        if hasattr(dataset, 'samples') and isinstance(dataset.samples, pd.DataFrame):
            df = dataset.samples
            # Prefer t1c column
            if 't1c' in df.columns:
                return [Path(str(p)).parent.name for p in df['t1c'].astype(str).tolist()]
            # Fallback: inspect first filepath-like column
            for col in df.columns:
                if df[col].dtype == object:
                    return [Path(str(p)).parent.name for p in df[col].astype(str).tolist()]
    except Exception:
        pass

    raise NotImplementedError("Could not extract patient ids from dataset; unsupported dataset type or missing attributes")


def split_dataset_by_patients(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Split a dataset by patient id, returning sample index lists for each split.

    Returns: train_idx, val_idx, test_idx, patient_splits
    where patient_splits is a dict with keys 'train','val','test' mapping to lists of patient ids.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    patient_ids = extract_patient_ids(dataset)
    n = len(patient_ids)
    if n == 0:
        raise RuntimeError("Dataset appears empty")

    # Map patient -> list of sample indices
    p2idx = defaultdict(list)
    for i, pid in enumerate(patient_ids):
        p2idx[pid].append(i)

    unique_patients = list(p2idx.keys())
    rng = random.Random(seed)
    rng.shuffle(unique_patients)

    n_pat = len(unique_patients)
    n_train = int(round(n_pat * train_ratio))
    n_val = int(round(n_pat * val_ratio))
    # ensure at least 1 patient in each split if possible
    n_test = n_pat - n_train - n_val
    if n_test <= 0:
        # adjust conservatively
        n_test = max(1, n_pat - n_train - n_val)
        if n_train + n_val + n_test > n_pat:
            # reduce train first
            while n_train + n_val + n_test > n_pat and n_train > 1:
                n_train -= 1
    train_patients = unique_patients[:n_train]
    val_patients = unique_patients[n_train:n_train + n_val]
    test_patients = unique_patients[n_train + n_val:]

    # Ensure non-empty
    if len(train_patients) == 0:
        train_patients = unique_patients[:max(1, min(len(unique_patients) - 2, 1))]
    if len(val_patients) == 0 and len(unique_patients) > 1:
        val_patients = unique_patients[n_train:n_train + 1]
    if len(test_patients) == 0 and len(unique_patients) > (len(train_patients) + len(val_patients)):
        test_patients = unique_patients[n_train + n_val:n_train + n_val + 1]

    # Now build sample index lists
    train_idx = []
    val_idx = []
    test_idx = []
    for pid in train_patients:
        train_idx.extend(p2idx[pid])
    for pid in val_patients:
        val_idx.extend(p2idx[pid])
    for pid in test_patients:
        test_idx.extend(p2idx[pid])

    # Sort indices for reproducibility
    train_idx = sorted(train_idx)
    val_idx = sorted(val_idx)
    test_idx = sorted(test_idx)

    patient_splits = {'train': train_patients, 'val': val_patients, 'test': test_patients}

    # Sanity check: no patient appears in more than one split
    overlap = set(train_patients) & set(val_patients) | set(train_patients) & set(test_patients) | set(val_patients) & set(test_patients)
    if overlap:
        raise RuntimeError(f"Patient overlap detected between splits: {overlap}")

    return train_idx, val_idx, test_idx, patient_splits


def get_split_loaders(dataset, batch_size=8, eval_batch_size=16, num_workers=0,
                      train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42,
                      pin_memory=True, persistent_workers=False, shuffle_train=True):
    """Return (train_loader, val_loader, test_loader, train_subset, val_subset, test_subset, patient_splits).

    Uses Subset wrappers around the provided dataset and constructs DataLoaders.
    """
    train_idx, val_idx, test_idx, patient_splits = split_dataset_by_patients(
        dataset, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
    )

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    test_subset = Subset(dataset, test_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

    return train_loader, val_loader, test_loader, train_subset, val_subset, test_subset, patient_splits


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Inspect patient-level splitting for a dataset')
    parser.add_argument('--preprocessed_dir', type=str, default=None, help='Path to preprocessed .pt slices directory')
    parser.add_argument('--csv_index', type=str, default=None, help='CSV index of triplets (for SimpleCSVTripletDataset)')
    parser.add_argument('--data_dir', type=str, default=None, help='BraTS dataset directory (for BraTS2D5Dataset)')
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Lazily load the dataset types to avoid circular imports
    from importlib import import_module
    dataset = None
    if args.preprocessed_dir:
        mod = import_module('preprocessed_dataset')
        dataset = mod.FastTensorSliceDataset(args.preprocessed_dir)
    elif args.csv_index:
        mod = import_module('train')
        dataset = mod.SimpleCSVTripletDataset(args.csv_index, image_size=(256,256), spacing=(1,1,1))
    elif args.data_dir:
        mod = import_module('train')
        dataset = mod.BraTS2D5Dataset(args.data_dir, image_size=(256,256), spacing=(1,1,1))
    else:
        raise SystemExit('Provide --preprocessed_dir or --csv_index or --data_dir')

    train_loader, val_loader, test_loader, train_subset, val_subset, test_subset, patient_splits = get_split_loaders(
        dataset, batch_size=8, eval_batch_size=16, num_workers=0,
        train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed,
        pin_memory=False, persistent_workers=False
    )

    print('Dataset total samples:', len(dataset))
    print('Train samples:', len(train_subset), 'Val samples:', len(val_subset), 'Test samples:', len(test_subset))
    print('Unique patients:', len(patient_splits['train']) + len(patient_splits['val']) + len(patient_splits['test']))
    print('Patients by split sizes:', {k: len(v) for k, v in patient_splits.items()})
    # Quick overlap check
    overlap = set(patient_splits['train']) & set(patient_splits['val'])
    overlap |= set(patient_splits['train']) & set(patient_splits['test'])
    overlap |= set(patient_splits['val']) & set(patient_splits['test'])
    if overlap:
        print('ERROR: Overlap detected:', overlap)
    else:
        print('No patient overlap between splits (OK)')
