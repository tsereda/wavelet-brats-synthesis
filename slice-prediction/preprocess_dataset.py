#!/usr/bin/env python3
"""
Create a CSV index of non-overlapping triplet slice centers per patient.

Outputs a CSV with columns:
  t1n,t1c,t2w,t2f,label,slice_prev,slice_mid,slice_next,patient_dir

Each row corresponds to one triplet (center-1, center, center+1).
"""
import os
import glob
import csv
import random
import argparse
from typing import List, Tuple

try:
    import nibabel as nib
except Exception:
    nib = None


def find_patient_files(patient_dir: str):
    brats2023_modalities = {
        't1n': 't1n',
        't1c': 't1c',
        't2w': 't2w',
        't2f': 't2f'
    }
    files = {}
    for key, suf in brats2023_modalities.items():
        matches = glob.glob(os.path.join(patient_dir, f"*-{suf}.nii*"))
        if not matches:
            return None
        files[key] = matches[0]

    # segmentation - accept *seg.nii* or *label.nii*
    seg_matches = glob.glob(os.path.join(patient_dir, "*seg.nii*"))
    if not seg_matches:
        seg_matches = glob.glob(os.path.join(patient_dir, "*label.nii*"))
    if not seg_matches:
        return None
    files['label'] = seg_matches[0]
    return files


def get_depth_from_nifti(path: str) -> int:
    if nib is None:
        # fallback: try reading header with simple parsing (slower but avoids hard failure)
        # As a last resort return 0
        raise RuntimeError("nibabel is required for reading NIfTI shapes. Install nibabel or run with it available.")
    img = nib.load(path)
    data = img.header.get_data_shape()
    # NIfTI may be (H, W, D) or (X, Y, Z). We assume depth is last dimension
    return int(data[-1])


def sample_triplets(max_depth: int, num_triplets: int = 5, seed: int = None) -> List[Tuple[int, int, int]]:
    if seed is not None:
        random.seed(seed)

    valid_start = 25
    valid_end = max_depth - 32

    if valid_end - valid_start < 5:
        # Not enough depth to sample; return empty
        return []

    available_centers = list(range(valid_start + 1, valid_end - 1))

    centers = []
    attempts = 0
    while len(centers) < num_triplets and attempts < 5000:
        candidate = random.choice(available_centers)
        if all(abs(candidate - c) >= 3 for c in centers):
            centers.append(candidate)
        attempts += 1

    centers = sorted(centers)
    triplets = [(c - 1, c, c + 1) for c in centers]
    return triplets


def build_index(data_dir: str, output_csv: str, num_triplets: int = 5, num_patients: int = None, seed: int = None):
    patient_dirs = sorted(glob.glob(os.path.join(data_dir, "BraTS*")))
    if num_patients is not None:
        patient_dirs = patient_dirs[:num_patients]

    print(f"Found {len(patient_dirs)} patient dirs, scanning...")

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['t1n', 't1c', 't2w', 't2f', 'label', 'slice_prev', 'slice_mid', 'slice_next', 'patient_dir'])

        for p in patient_dirs:
            patient_name = os.path.basename(p)
            files = find_patient_files(p)
            if files is None:
                print(f"Skipping {patient_name}: missing modality/seg files")
                continue
            try:
                depth = get_depth_from_nifti(files['t1c'])
            except Exception as e:
                print(f"Skipping {patient_name}: failed to read depth: {e}")
                continue

            triplets = sample_triplets(depth, num_triplets=num_triplets, seed=seed)
            if not triplets:
                print(f"Warning: no triplets for {patient_name} (depth={depth})")
                continue

            for (prev, mid, nxt) in triplets:
                writer.writerow([
                    files['t1n'], files['t1c'], files['t2w'], files['t2f'], files['label'],
                    prev, mid, nxt, p
                ])

    print(f"Wrote index to {output_csv}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True, help='Path to BraTS root containing BraTS* patient dirs')
    p.add_argument('--output', required=True, help='Output CSV path')
    p.add_argument('--num_triplets', type=int, default=5)
    p.add_argument('--num_patients', type=int, default=None)
    p.add_argument('--seed', type=int, default=None)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    build_index(args.data_dir, args.output, num_triplets=args.num_triplets, num_patients=args.num_patients, seed=args.seed)
