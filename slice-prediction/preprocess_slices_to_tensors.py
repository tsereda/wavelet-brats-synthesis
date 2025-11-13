#!/usr/bin/env python3
"""
Preprocess BraTS volumes once and save 2.5D triplet slices as individual .pt files.

Usage:
    python preprocess_slices_to_tensors.py \
        --data_dir /path/to/BraTSFolder \
        --output_dir ./preprocessed_slices \
        --img_size 256

This script applies the same MONAI transforms used during training and
saves each valid triplet (prev, mid, next) as a small .pt file. It also
writes a CSV index `preprocessed_slices.csv` in the output directory for
fast loading.
"""
import os
import glob
import argparse
import gzip
import csv
from time import time
import torch
from pathlib import Path

from transforms import get_train_transforms


def validate_patient_files(patient_files):
    for key, filepath in patient_files.items():
        if filepath is None:
            return False, f"Missing file {key}"
        try:
            if filepath.endswith('.gz'):
                with gzip.open(filepath, 'rb') as f:
                    f.read(10240)
            else:
                if os.path.getsize(filepath) == 0:
                    return False, f"Empty file {key}"
        except Exception as e:
            return False, str(e)
    return True, None


def build_patient_list(data_dir):
    patient_dirs = sorted(glob.glob(os.path.join(data_dir, 'BraTS*')))
    return patient_dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--spacing', type=float, nargs=3, default=(1.0, 1.0, 1.0))
    parser.add_argument('--max_slices_per_patient', type=int, default=9999,
                        help='Optional cap on saved slices per patient (useful for debugging)')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / 'preprocessed_slices.csv'

    transforms = get_train_transforms((args.img_size, args.img_size), tuple(args.spacing))

    patients = build_patient_list(args.data_dir)
    if not patients:
        raise RuntimeError(f"No patient directories found under {args.data_dir}")

    start = time()
    total_saved = 0

    # CSV header
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filepath', 'patient', 'slice_idx'])

        for p_idx, p in enumerate(patients):
            patient_name = os.path.basename(p)
            try:
                # find modalities
                modalities = {}
                for suffix in ['t1n', 't1c', 't2w', 't2f']:
                    matches = glob.glob(os.path.join(p, f'*-{suffix}.nii*'))
                    if not matches:
                        raise FileNotFoundError(f"Missing *-{suffix} in {patient_name}")
                    modalities[suffix] = matches[0]
                seg = None
                seg_matches = glob.glob(os.path.join(p, '*seg.nii*'))
                if not seg_matches:
                    seg_matches = glob.glob(os.path.join(p, '*label.nii*'))
                if seg_matches:
                    seg = seg_matches[0]

                patient_files = {**modalities, 'label': seg}

                ok, err = validate_patient_files(patient_files)
                if not ok:
                    print(f"Skipping {patient_name}: {err}")
                    continue

                # apply full transforms (this yields tensors with shape [C,H,W,D])
                processed = transforms(patient_files)

                # concatenate modalities into a single tensor [C_total, H, W, D]
                import torch as _th
                img_modalities = _th.cat([processed['t1n'], processed['t1c'], processed['t2w'], processed['t2f']], dim=0)

                depth = img_modalities.shape[3]
                saved_for_patient = 0

                # iterate slices, avoid edges (need prev and next)
                for z in range(1, depth - 1):
                    if saved_for_patient >= args.max_slices_per_patient:
                        break

                    mid_slice = img_modalities[:, :, :, z]
                    prev_slice = img_modalities[:, :, :, z - 1]
                    next_slice = img_modalities[:, :, :, z + 1]

                    # input: concat(prev, next) -> channels doubled
                    input_tensor = _th.cat([prev_slice, next_slice], dim=0).contiguous()
                    target_tensor = mid_slice.contiguous()

                    fname = f"{patient_name}_slice_{z:04d}.pt"
                    out_path = out_dir / fname
                    _th.save({'input': input_tensor, 'target': target_tensor, 'patient': patient_name, 'slice_idx': int(z)}, str(out_path))

                    writer.writerow([str(out_path), patient_name, int(z)])
                    saved_for_patient += 1
                    total_saved += 1

                print(f"Patient {p_idx+1}/{len(patients)}: {patient_name} -> saved {saved_for_patient} slices")

            except Exception as e:
                print(f"Error processing {patient_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

    elapsed = time() - start
    print(f"Preprocessing complete: saved {total_saved} slices to {out_dir} in {elapsed:.1f}s")
    print(f"CSV index saved to: {csv_path}")


if __name__ == '__main__':
    main()