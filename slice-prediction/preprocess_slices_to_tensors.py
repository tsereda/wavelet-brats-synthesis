#!/usr/bin/env python3
"""
Preprocess BraTS volumes once and save 2.5D triplet slices as individual .pt files.
MODIFIED: Saves only 5 random non-overlapping triplets per patient with MULTIPROCESSING.

Usage:
    python preprocess_slices_to_tensors.py \
        --data_dir /path/to/BraTSFolder \
        --output_dir ./preprocessed_slices \
        --img_size 256 \
        --num_workers 8
"""
import os
import glob
import argparse
import gzip
import csv
from time import time
import torch
import random
from pathlib import Path
from multiprocessing import Pool, cpu_count

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


def select_non_overlapping_triplets(depth, num_triplets=5, margin=3, seed=None):
    """
    Select num_triplets random non-overlapping triplet centers.
    
    Args:
        depth: total number of slices
        num_triplets: number of triplets to select (default 5)
        margin: minimum spacing between triplet centers to avoid overlap
        seed: random seed (optional, for reproducibility per patient)
    
    Returns:
        List of slice indices for triplet centers
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    # Valid range: need prev and next slice, so [1, depth-2]
    # Also apply conservative bounds similar to your training code
    safe_start = max(1, int(0.1 * depth))
    safe_end = min(depth - 2, int(0.8 * depth))
    
    if safe_end - safe_start < num_triplets * margin:
        # Not enough room for non-overlapping triplets, just sample what we can
        valid_indices = list(range(safe_start, safe_end + 1))
        return rng.sample(valid_indices, min(num_triplets, len(valid_indices)))
    
    # Greedy selection of non-overlapping centers
    valid_indices = list(range(safe_start, safe_end + 1))
    rng.shuffle(valid_indices)
    
    selected = []
    for idx in valid_indices:
        # Check if this index is far enough from all previously selected
        if all(abs(idx - s) >= margin for s in selected):
            selected.append(idx)
            if len(selected) >= num_triplets:
                break
    
    return sorted(selected)


def process_patient(patient_info):
    """
    Process a single patient and save triplet slices.
    Returns list of (filepath, patient_name, slice_idx) tuples for CSV writing.
    
    patient_info is a tuple: (patient_path, patient_idx, output_dir, img_size, 
                               spacing, triplets_per_patient, triplet_margin, base_seed)
    """
    patient_path, patient_idx, output_dir, img_size, spacing, triplets_per_patient, triplet_margin, base_seed = patient_info
    
    patient_name = os.path.basename(patient_path)
    results = []
    
    # Use patient-specific seed for reproducibility
    patient_seed = base_seed + patient_idx
    
    try:
        # Find modalities
        modalities = {}
        for suffix in ['t1n', 't1c', 't2w', 't2f']:
            matches = glob.glob(os.path.join(patient_path, f'*-{suffix}.nii*'))
            if not matches:
                raise FileNotFoundError(f"Missing *-{suffix} in {patient_name}")
            modalities[suffix] = matches[0]
        
        seg = None
        seg_matches = glob.glob(os.path.join(patient_path, '*seg.nii*'))
        if not seg_matches:
            seg_matches = glob.glob(os.path.join(patient_path, '*label.nii*'))
        if seg_matches:
            seg = seg_matches[0]

        patient_files = {**modalities, 'label': seg}

        ok, err = validate_patient_files(patient_files)
        if not ok:
            print(f"✗ Skipping {patient_name}: {err}")
            return results

        # Get transforms (need to create fresh ones per process)
        transforms = get_train_transforms((img_size, img_size), spacing)
        
        # Apply full transforms
        processed = transforms(patient_files)

        # Concatenate modalities into a single tensor [C_total, H, W, D]
        img_modalities = torch.cat([
            processed['t1n'], 
            processed['t1c'], 
            processed['t2w'], 
            processed['t2f']
        ], dim=0)

        depth = img_modalities.shape[3]
        
        # Select random non-overlapping triplet centers
        selected_slices = select_non_overlapping_triplets(
            depth, 
            num_triplets=triplets_per_patient,
            margin=triplet_margin,
            seed=patient_seed
        )

        # Save only the selected triplets
        for z in selected_slices:
            mid_slice = img_modalities[:, :, :, z]
            prev_slice = img_modalities[:, :, :, z - 1]
            next_slice = img_modalities[:, :, :, z + 1]

            # input: concat(prev, next) -> channels doubled
            input_tensor = torch.cat([prev_slice, next_slice], dim=0).contiguous()
            target_tensor = mid_slice.contiguous()

            fname = f"{patient_name}_slice_{z:04d}.pt"
            out_path = output_dir / fname
            torch.save({
                'input': input_tensor, 
                'target': target_tensor, 
                'patient': patient_name, 
                'slice_idx': int(z)
            }, str(out_path))

            results.append((str(out_path), patient_name, int(z)))

        print(f"✓ {patient_name} -> saved {len(selected_slices)} slices at indices {selected_slices}")
        
    except Exception as e:
        print(f"✗ Error processing {patient_name}: {e}")
        import traceback
        traceback.print_exc()
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--spacing', type=float, nargs=3, default=(1.0, 1.0, 1.0))
    parser.add_argument('--triplets_per_patient', type=int, default=5,
                        help='Number of random non-overlapping triplets to save per patient')
    parser.add_argument('--triplet_margin', type=int, default=3,
                        help='Minimum spacing between triplet centers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of worker processes (default: CPU count)')
    args = parser.parse_args()

    # Determine number of workers
    if args.num_workers is None:
        args.num_workers = cpu_count()
    
    print(f"Using {args.num_workers} worker processes")

    # Set random seed for main process
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / 'preprocessed_slices.csv'

    patients = build_patient_list(args.data_dir)
    if not patients:
        raise RuntimeError(f"No patient directories found under {args.data_dir}")

    print(f"Found {len(patients)} patients")
    print(f"Preprocessing with {args.triplets_per_patient} random triplets per patient")
    print(f"Using base seed: {args.seed}")

    start = time()

    # Prepare arguments for each worker
    worker_args = [
        (p, idx, out_dir, args.img_size, tuple(args.spacing), 
         args.triplets_per_patient, args.triplet_margin, args.seed)
        for idx, p in enumerate(patients)
    ]

    # Process patients in parallel
    with Pool(processes=args.num_workers) as pool:
        all_results = pool.map(process_patient, worker_args)

    # Flatten results and write CSV
    total_saved = 0
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filepath', 'patient', 'slice_idx'])
        
        for patient_results in all_results:
            for row in patient_results:
                writer.writerow(row)
                total_saved += 1

    elapsed = time() - start
    print(f"\n{'='*60}")
    print(f"Preprocessing complete!")
    print(f"Total slices saved: {total_saved}")
    print(f"Average per patient: {total_saved/len(patients):.1f}")
    print(f"Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"Throughput: {len(patients)/elapsed:.2f} patients/second")
    print(f"Output directory: {out_dir}")
    print(f"CSV index: {csv_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()