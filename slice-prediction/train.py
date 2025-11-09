# BraTS2023 GLI format only (t1n, t1c, t2w, t2f)

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import gzip
import logging
from collections import OrderedDict
import argparse
from time import time, perf_counter
import torch.multiprocessing
import cv2
import wandb
from monai.networks.nets import SwinUNETR, UNETR, BasicUNet
from torch.nn import L1Loss, MSELoss
from transforms import get_train_transforms
from logging_utils import create_reconstruction_log_panel
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# Create timing stats tracker
class TimingStats:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.batch_times = []
        self.forward_times = []
        self.backward_times = []
        self.data_load_times = []
        self.wavelet_transform_times = []
        
    def add_batch_time(self, elapsed):
        self.batch_times.append(elapsed)
    
    def add_forward_time(self, elapsed):
        self.forward_times.append(elapsed)
        
    def add_backward_time(self, elapsed):
        self.backward_times.append(elapsed)
        
    def add_data_load_time(self, elapsed):
        self.data_load_times.append(elapsed)
        
    def add_wavelet_time(self, elapsed):
        self.wavelet_transform_times.append(elapsed)
    
    def get_stats(self):
        return {
            'avg_batch_time': np.mean(self.batch_times) if self.batch_times else 0,
            'avg_forward_time': np.mean(self.forward_times) if self.forward_times else 0,
            'avg_backward_time': np.mean(self.backward_times) if self.backward_times else 0,
            'avg_data_load_time': np.mean(self.data_load_times) if self.data_load_times else 0,
            'avg_wavelet_time': np.mean(self.wavelet_transform_times) if self.wavelet_transform_times else 0,
            'total_batch_time': sum(self.batch_times),
            'samples_per_second': len(self.batch_times) / sum(self.batch_times) if self.batch_times else 0
        }


class BraTS2D5Dataset(Dataset):
    """Memory-efficient BraTS 2.5D dataset with on-demand volume loading and LRU caching.
    
    Supports BraTS2023 GLI Challenge format:
    - BraTS2023 GLI: *-t1n.nii.gz, *-t1c.nii.gz, *-t2w.nii.gz, *-t2f.nii.gz

    Keeps a small LRU cache of processed volumes to avoid pre-loading the entire dataset
    into memory (prevents OOM for large datasets). Builds the slice index map by
    processing volumes one-by-one and then discarding them.
    """
    def __init__(self, data_dir, image_size, spacing, num_patients=None, cache_size=50):
        self.image_size = image_size
        self.cache_size = cache_size
        # Track corrupted patients (for reporting)
        self.corrupted_patients = []
        patient_dirs = sorted(glob.glob(os.path.join(data_dir, "BraTS*")))
        if num_patients is not None:
            print(f"--- Using a subset of {num_patients} patients for testing. ---")
            patient_dirs = patient_dirs[:num_patients]
        if not patient_dirs:
            raise FileNotFoundError(f"No patient data found in '{data_dir}'. Check your --data_dir path.")

        print(f"Found {len(patient_dirs)} patient directories")

        # Build file list for BraTS2023 GLI format
        self.files = []
        brats2023_modalities = {
            't1n': 't1n',
            't1c': 't1c', 
            't2w': 't2w',
            't2f': 't2f'
        }
        
        for p in patient_dirs:
            patient_name = os.path.basename(p)
            try:
                patient_files = {}
                
                # Find each modality file
                for key, modality_suffix in brats2023_modalities.items():
                    pattern = os.path.join(p, f"*-{modality_suffix}.nii*")
                    matches = glob.glob(pattern)
                    if not matches:
                        raise FileNotFoundError(f"No {modality_suffix} file found in {patient_name}")
                    patient_files[key] = matches[0]

                # Find segmentation file
                seg_matches = glob.glob(os.path.join(p, "*seg.nii*"))
                if not seg_matches:
                    seg_matches = glob.glob(os.path.join(p, "*label.nii*"))
                if not seg_matches:
                    raise FileNotFoundError(f"No segmentation file found in {patient_name}")
                patient_files['label'] = seg_matches[0]

                # Validate files for obvious corruption (gzip integrity and non-zero size)
                is_valid, error_msg = self._validate_patient_files(patient_files, patient_name)
                if is_valid:
                    self.files.append(patient_files)
                else:
                    # Keep a short log of corrupted entries for later reporting
                    print(f"  ⚠️  Skipping {patient_name}: {error_msg}")
                    continue
            except FileNotFoundError as e:
                print(f"Warning: Skipping patient {patient_name}: {e}")
                continue

        if not self.files:
            raise RuntimeError("No valid patients found! Check your dataset structure.")

        print(f"✓ Detected BraTS2023 GLI format (t1n, t1c, t2w, t2f)")
        print(f"Successfully found {len(self.files)} patient entries")

        # Report corruption/skipped stats
        try:
            total_candidates = len(patient_dirs)
            valid_count = len(self.files)
            skipped = total_candidates - valid_count
            if skipped > 0:
                pct = (skipped / float(total_candidates)) * 100.0 if total_candidates > 0 else 0.0
                print(f"⚠️  Corrupted/skipped: {skipped} ({pct:.2f}% of candidates)")
                if getattr(self, 'corrupted_patients', None):
                    print("  Examples of corrupted files (up to 5):")
                    for p, key, err in self.corrupted_patients[:5]:
                        print(f"    {p}: {key} -> {err}")
        except Exception:
            pass

        # Prepare transforms and LRU cache
        self.transforms = get_train_transforms(image_size, spacing)
        self.volume_cache = OrderedDict()

        # Build slice map using FAST lightweight method to avoid expensive full transforms
        # This only loads T1c for slice counting and defers full processing until data is requested
        self.slice_map = []
        self._build_slice_map_fast()

    def _get_lightweight_transforms(self):
        """
        Create minimal transforms for fast slice counting.
        Only loads T1c, and does minimal typing/channel insertion.
        """
        from monai.transforms import (
            Compose,
            LoadImaged,
            EnsureChannelFirstd,
            EnsureTyped,
        )
        return Compose([
            LoadImaged(keys=["t1c"]),
            EnsureChannelFirstd(keys=["t1c"]),
            EnsureTyped(keys=["t1c"]),
        ])

    def _build_slice_map_fast(self):
        """
        FAST slice mapping - much quicker than running full training transforms on every
        patient. Loads only T1c and performs a quick intensity check per slice.
        """
        # Safer FAST slice mapping: estimate preprocessing reduction, use conservative
        # bounds and avoid indexing slices that may be cropped/resized out by full
        # preprocessing pipeline.
        print("🚀 SAFE FAST slice mapping mode - with conservative bounds!")
        print("(Sampling a few patients to estimate preprocessing effects, then building a safe slice map)")
        start_time = time()

        lightweight_transforms = self._get_lightweight_transforms()

        # Step 1: sample a few patients to estimate how preprocessing changes slice counts
        reduction_factor = 1.0
        sample_size = min(5, len(self.files))
        for i in range(sample_size):
            try:
                patient_files = self.files[i]
                proc_light = lightweight_transforms({'t1c': patient_files['t1c']})
                raw_slices = proc_light['t1c'].shape[3]

                # Try a full transform on the same patient to estimate final size
                try:
                    proc_full = self.transforms(patient_files)
                    final_slices = proc_full['t1c'].shape[3]
                    current_factor = final_slices / float(raw_slices) if raw_slices > 0 else 1.0
                    reduction_factor = min(reduction_factor, current_factor)
                    del proc_full
                except Exception:
                    # If full transform fails for sampling, skip but keep default factor
                    pass

                del proc_light
            except Exception as e:
                print(f"  Warning: Could not sample patient {i}: {e}")
                continue

        print(f"Using conservative reduction factor: {reduction_factor:.3f}")

        # Step 2: build slice map using conservative estimated final slice counts
        for i, patient_files in enumerate(self.files):
            try:
                proc = lightweight_transforms({'t1c': patient_files['t1c']})
                raw_num_slices = proc['t1c'].shape[3]

                # Estimate final number of slices after preprocessing and add safety margin
                estimated_final = max(3, int(raw_num_slices * reduction_factor * 0.9))

                # Avoid edges that are often cropped; choose conservative start/end
                safe_start = max(1, int(0.1 * estimated_final))
                safe_end = min(estimated_final - 1, int(0.8 * estimated_final))

                # Also ensure not to exceed raw slices
                safe_end = min(safe_end, raw_num_slices - 2)

                if safe_start >= safe_end:
                    print(f"Warning: Patient {i} has insufficient slices after safety margins")
                    del proc
                    continue

                for slice_idx in range(safe_start, safe_end):
                    # Extra guard: don't read out-of-range on the lightweight proc
                    if slice_idx >= proc['t1c'].shape[3] or slice_idx <= 0:
                        continue
                    brain_slice = proc['t1c'][0, :, :, slice_idx]
                    if torch.mean(brain_slice) > 50.0:
                        self.slice_map.append((i, slice_idx))

                del proc

            except Exception as e:
                print(f"Warning: Failed to process patient {i}: {e}")
                continue

            # Progress reporting with ETA
            if (i + 1) % 25 == 0 or (i + 1) == len(self.files):
                elapsed = time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta_seconds = (len(self.files) - i - 1) / rate if rate > 0 else 0

                print(f"   📊 Indexed {i + 1:4d}/{len(self.files)} patients "
                      f"({rate:5.1f} patients/sec, "
                      f"ETA: {eta_seconds/60:4.1f}min, "
                      f"Found: {len(self.slice_map):5d} slices)")

        total_time = time() - start_time
        print(f"\n🚀 SAFE FAST slice mapping completed!")
        print(f"   ⏱️  Total time: {total_time:.1f} seconds")
        print(f"   📈 Processing rate: {len(self.files)/total_time:.1f} patients/sec")
        print(f"   🧠 Found {len(self.slice_map)} brain slices")
        if len(self.files) > 0:
            print(f"   💾 Average {len(self.slice_map)/len(self.files):.1f} slices per patient")
        print(f"   🛡️  Using safety factor: {reduction_factor:.3f}")

    def __len__(self):
        return len(self.slice_map)

    def _validate_patient_files(self, patient_files, patient_name):
        """Validate files are not corrupted.

        Checks gzip files by reading the first chunk and non-zero size for uncompressed files.
        Returns (True, None) on success or (False, error_message) on failure.
        """
        for key, filepath in patient_files.items():
            try:
                if filepath.endswith('.gz'):
                    # Try to read a small chunk to verify gzip integrity
                    with gzip.open(filepath, 'rb') as f:
                        f.read(10240)
                else:
                    # For uncompressed files, a quick size check
                    if os.path.getsize(filepath) == 0:
                        return False, f"Empty file for {key}"
            except Exception as e:
                # Record corrupted patient info for later reporting
                try:
                    self.corrupted_patients.append((patient_name, key, str(e)))
                except Exception:
                    pass
                return False, f"Error reading {key}: {e}"
        return True, None

    def _get_volume(self, vol_idx):
        """Return processed volume for vol_idx using LRU cache."""
        if vol_idx in self.volume_cache:
            # mark as recently used
            self.volume_cache.move_to_end(vol_idx)
            return self.volume_cache[vol_idx]

        # load and process
        processed = self.transforms(self.files[vol_idx])
        self.volume_cache[vol_idx] = processed
        # enforce cache size
        if len(self.volume_cache) > self.cache_size:
            self.volume_cache.popitem(last=False)
        return processed

    def __getitem__(self, index):
        volume_idx, slice_idx = self.slice_map[index]
        patient_data = self._get_volume(volume_idx)

        img_modalities = torch.cat([patient_data['t1n'], patient_data['t1c'],
                                    patient_data['t2w'], patient_data['t2f']], dim=0)

        # CRITICAL FIX: Bounds checking to prevent IndexError when preprocessing
        # (cropping/resizing) has changed the final number of slices relative to
        # the lightweight T1c count used when building the slice map.
        max_slice = img_modalities.shape[3] - 1

        # Ensure slice_idx isn't on the extreme boundaries
        if slice_idx <= 0 or slice_idx >= max_slice:
            slice_idx = max(1, min(slice_idx, max_slice - 1))

        prev_idx = max(0, slice_idx - 1)
        next_idx = min(max_slice, slice_idx + 1)

        try:
            prev_slice = img_modalities[:, :, :, prev_idx]
            next_slice = img_modalities[:, :, :, next_idx]
            target_tensor = img_modalities[:, :, :, slice_idx]

            input_tensor = torch.cat([prev_slice, next_slice], dim=0)
            return input_tensor, target_tensor, slice_idx

        except IndexError as e:
            # Fallback: log and use middle slice of volume
            print(f"IndexError caught in __getitem__: volume_idx={volume_idx}, slice_idx={slice_idx}, "
                  f"volume_shape={img_modalities.shape}, max_slice={max_slice}")
            print(f"Attempted indices: prev={prev_idx}, curr={slice_idx}, next={next_idx}")

            middle_slice = max_slice // 2
            middle_prev = max(0, middle_slice - 1)
            middle_next = min(max_slice, middle_slice + 1)

            prev_slice = img_modalities[:, :, :, middle_prev]
            next_slice = img_modalities[:, :, :, middle_next]
            target_tensor = img_modalities[:, :, :, middle_slice]

            input_tensor = torch.cat([prev_slice, next_slice], dim=0)
            print(f"Using fallback middle slice: {middle_slice}")
            return input_tensor, target_tensor, middle_slice


def get_args():
    parser = argparse.ArgumentParser(description="2.5D Middleslice Reconstruction - BraTS2023 GLI")
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to BraTS2023 GLI dataset directory')
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--model_type', type=str, default='swin', 
                       choices=['swin', 'unet', 'unetr'],
                       help='Model architecture to use')
    parser.add_argument('--wavelet', type=str, default='none',
                       help='Wavelet type (none, haar, db2, db4, sym3, coif2, etc.)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--num_patients', type=int, default=None,
                       help='Number of patients to use (default: all)')
    parser.add_argument('--eval_batch_size', type=int, default=16,
                       help='Batch size for evaluation (default: 16)')
    parser.add_argument('--skip_eval', action='store_true',
                       help='Skip evaluation after training')
    parser.add_argument('--timing_frequency', type=int, default=50,
                       help='How often to report detailed timing stats (batches)')
    return parser.parse_args()


def get_model(model_type, wavelet_name, img_size, device):
    """Load model based on type, optionally with wavelet wrapper"""
    
    use_wavelet = wavelet_name != 'none'
    
    print("\n" + "="*60)
    print(f"INITIALIZING MODEL: {model_type.upper()}")
    if use_wavelet:
        print(f"WITH WAVELET PROCESSING: {wavelet_name}")
    print("="*60)
    
    # Create base model based on architecture type
    if model_type == 'swin':
        base_model = SwinUNETR(
            in_channels=8, 
            out_channels=4, 
            feature_size=24, 
            spatial_dims=2
        )
        print("Base Model: Swin-UNETR")
        print(f"Input channels: 8 (4 modalities × 2 slices)")
        print(f"Output channels: 4 (4 modalities)")
        print(f"Feature size: 24")
        print(f"Spatial dims: 2D")
    
    elif model_type == 'unet':
        base_model = BasicUNet(
            spatial_dims=2,
            in_channels=8,
            out_channels=4,
            features=(32, 32, 64, 128, 256, 32),
            act='ReLU',
            norm='batch',
            dropout=0.0
        )
        print("Base Model: Basic U-Net")
        print(f"Input channels: 8 (4 modalities × 2 slices)")
        print(f"Output channels: 4 (4 modalities)")
        print(f"Feature progression: [32, 64, 128, 256, 512]")
        print(f"Activation: ReLU")
        print(f"Normalization: BatchNorm")
    
    elif model_type == 'unetr':
        base_model = UNETR(
            in_channels=8,
            out_channels=4,
            img_size=(img_size, img_size),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            proj_type='conv',
            norm_name='instance',
            res_block=True,
            dropout_rate=0.0,
            spatial_dims=2
        )
        print("Base Model: UNETR (Vision Transformer + CNN decoder)")
        print(f"Input channels: 8 (4 modalities × 2 slices)")
        print(f"Output channels: 4 (4 modalities)")
        print(f"Input size: {img_size}×{img_size}")
        print(f"Hidden size: 768")
        print(f"MLP dimension: 3072")
        print(f"Number of heads: 12")
        print(f"Projection type: conv")
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Apply wavelet wrapper if requested
    if use_wavelet:
        from models.wavelet_wrapper import WaveletWrapper
        model = WaveletWrapper(
            base_model=base_model,
            wavelet_name=wavelet_name,
            in_channels=8,
            out_channels=4
        ).to(device)
        print(f"\n✓ Wavelet wrapper applied: Processing in {wavelet_name} wavelet domain")
    else:
        model = base_model.to(device)
        print("\n✓ Standard spatial domain processing (no wavelet)")
    
    # Calculate and print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print("="*60 + "\n")
    
    return model


def visualize_wavelet_filters(wavelet_name):
    """
    Visualize the wavelet filter itself
    
    Args:
        wavelet_name: name of the wavelet (e.g., 'haar', 'db2')
    
    Returns:
        matplotlib figure
    """
    import pywt
    
    wavelet = pywt.Wavelet(wavelet_name)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f'Wavelet Filters: {wavelet_name}', fontsize=14, fontweight='bold')
    
    # Decomposition filters
    axes[0, 0].stem(wavelet.dec_lo, basefmt=' ')
    axes[0, 0].set_title('Decomposition Low-pass (LL)')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].stem(wavelet.dec_hi, basefmt=' ')
    axes[0, 1].set_title('Decomposition High-pass (LH/HL/HH)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Reconstruction filters
    axes[1, 0].stem(wavelet.rec_lo, basefmt=' ')
    axes[1, 0].set_title('Reconstruction Low-pass')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].stem(wavelet.rec_hi, basefmt=' ')
    axes[1, 1].set_title('Reconstruction High-pass')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def visualize_wavelet_decomposition(coeffs, title, num_modalities=4):
    """
    Visualize wavelet decomposition coefficients
    
    Args:
        coeffs: torch.Tensor [C*4, H/2, W/2] - wavelet coefficients (LL, LH, HL, HH for each channel)
        title: str - title for the plot
        num_modalities: int - number of modalities to visualize (4 for output, 8 for input)
    
    Returns:
        matplotlib figure
    """
    # Ensure coeffs is 3D [total_channels, H, W]
    if coeffs.dim() == 4:
        # If [B, C*4, H, W], take first sample
        coeffs = coeffs[0]
    
    # Calculate number of modalities
    total_channels = coeffs.shape[0]
    C = total_channels // 4
    
    # Only show requested number of modalities
    C = min(C, num_modalities)
    
    # Labels for modalities (including Z-1 and Z+1 for input)
    modalities = [
        'T1n(Z-1)', 'T1c(Z-1)', 'T2w(Z-1)', 'T2f(Z-1)',  # First 4: Z-1 slices
        'T1n(Z+1)', 'T1c(Z+1)', 'T2w(Z+1)', 'T2f(Z+1)'   # Next 4: Z+1 slices
    ]
    
    # For output, use simple labels
    if num_modalities == 4:
        modalities = ['T1n', 'T1c', 'T2w', 'T2f']
    
    fig = plt.figure(figsize=(16, 4*C))
    gs = GridSpec(C, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    for mod_idx in range(C):
        start_idx = mod_idx * 4
        
        # Extract 4 subbands for this modality
        ll = coeffs[start_idx].cpu().numpy()
        lh = coeffs[start_idx + 1].cpu().numpy()
        hl = coeffs[start_idx + 2].cpu().numpy()
        hh = coeffs[start_idx + 3].cpu().numpy()
        
        # Calculate common vmin/vmax for consistent scaling
        vmin = min(ll.min(), lh.min(), hl.min(), hh.min())
        vmax = max(ll.max(), lh.max(), hl.max(), hh.max())
        
        mod_name = modalities[mod_idx] if mod_idx < len(modalities) else f"Mod{mod_idx}"
        
        # Plot LL (approximation)
        ax = fig.add_subplot(gs[mod_idx, 0])
        im = ax.imshow(ll, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f'{mod_name} - LL (Approx)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Plot LH (horizontal detail)
        ax = fig.add_subplot(gs[mod_idx, 1])
        im = ax.imshow(lh, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f'{mod_name} - LH (Horiz)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Plot HL (vertical detail)
        ax = fig.add_subplot(gs[mod_idx, 2])
        im = ax.imshow(hl, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f'{mod_name} - HL (Vert)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Plot HH (diagonal detail)
        ax = fig.add_subplot(gs[mod_idx, 3])
        im = ax.imshow(hh, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f'{mod_name} - HH (Diag)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def log_wavelet_visualizations(model, inputs, outputs, targets, wavelet_name, epoch):
    """
    Create and log wavelet decomposition visualizations to WandB
    Only works for wavelet models - FIXED TO SHOW ALL INPUT COMPONENTS
    """
    """FIXED: Create and log wavelet decomposition visualizations to WandB"""
    if not hasattr(model, 'dwt2d_batch'):
        return

    import io
    from PIL import Image

    with torch.no_grad():
        try:
            # Get decompositions
            input_wavelets = model.dwt2d_batch(inputs[:1])
            output_wavelets = model.dwt2d_batch(outputs[:1])
            target_wavelets = model.dwt2d_batch(targets[:1])

            print(f"\n>>> Wavelet shapes: Input {input_wavelets.shape}, Output {output_wavelets.shape}")

            # Helper: convert matplotlib fig to PIL Image
            def fig_to_pil(fig):
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                pil_image = Image.open(buf).convert('RGB')
                plt.close(fig)
                return pil_image

            # Log filter visualization (only once)
            if epoch == 0:
                filter_fig = visualize_wavelet_filters(wavelet_name)
                filter_image = fig_to_pil(filter_fig)
                wandb.log({f"wavelet_filters/{wavelet_name}": wandb.Image(filter_image)})

            # Log decompositions
            input_fig = visualize_wavelet_decomposition(input_wavelets[0], f'Input Wavelets (Epoch {epoch})', num_modalities=8)
            input_image = fig_to_pil(input_fig)
            wandb.log({f"wavelets/input_epoch_{epoch}": wandb.Image(input_image)})

            output_fig = visualize_wavelet_decomposition(output_wavelets[0], f'Output Wavelets (Epoch {epoch})', num_modalities=4)
            output_image = fig_to_pil(output_fig)
            wandb.log({f"wavelets/output_epoch_{epoch}": wandb.Image(output_image)})

            target_fig = visualize_wavelet_decomposition(target_wavelets[0], f'Target Wavelets (Epoch {epoch})', num_modalities=4)
            target_image = fig_to_pil(target_fig)
            wandb.log({f"wavelets/target_epoch_{epoch}": wandb.Image(target_image)})

            print(f"✓ Wavelet visualizations logged to WandB")

        except Exception as e:
            print(f"✗ Error in wavelet logging: {e}")


def measure_inference_timing(model, data_loader, device, num_batches=100):
    """
    Measure detailed inference timing statistics
    
    Args:
        model: trained model
        data_loader: DataLoader
        device: cuda/cpu
        num_batches: number of batches to time
    
    Returns:
        dict with timing statistics
    """
    print(f"\n>>> Measuring inference timing over {num_batches} batches...")
    
    model.eval()
    timing_stats = TimingStats()
    
    with torch.no_grad():
        for i, (inputs, targets, _) in enumerate(data_loader):
            if i >= num_batches:
                break
            
            # Data loading time (approximation)
            data_start = perf_counter()
            inputs = inputs.to(device)
            targets = targets.to(device)
            data_time = perf_counter() - data_start
            timing_stats.add_data_load_time(data_time)
            
            # Forward pass timing
            torch.cuda.synchronize() if device.type == 'cuda' else None
            forward_start = perf_counter()
            
            # Time wavelet transform if applicable
            if hasattr(model, 'dwt2d_batch'):
                wavelet_start = perf_counter()
                _ = model.dwt2d_batch(inputs)
                torch.cuda.synchronize() if device.type == 'cuda' else None
                wavelet_time = perf_counter() - wavelet_start
                timing_stats.add_wavelet_time(wavelet_time)
            
            # Full forward pass
            outputs = model(inputs)
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            forward_time = perf_counter() - forward_start
            timing_stats.add_forward_time(forward_time)
    
    return timing_stats.get_stats()


def main(args):
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    # Determine if using wavelet
    use_wavelet = args.wavelet != 'none'
    
    # Create run name with model type and wavelet info
    if use_wavelet:
        run_name = f"{args.model_type}_wavelet_{args.wavelet}_{int(time())}"
    else:
        run_name = f"{args.model_type}_nowavelet_{int(time())}"
    
    wandb.init(project="brats-middleslice-wavelet-sweep", config=vars(args), name=run_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    print("\n" + "="*60)
    print("LOADING DATASET")
    print("="*60)
    dataset_start = time()
    dataset = BraTS2D5Dataset(
        data_dir=args.data_dir, 
        image_size=(args.img_size, args.img_size),
        spacing=(1.0, 1.0, 1.0), 
        num_patients=args.num_patients,
        cache_size=25
    )
    dataset_time = time() - dataset_start
    print(f"Dataset loading took {dataset_time:.2f} seconds")
    print("="*60 + "\n")
    
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Load model
    model = get_model(args.model_type, args.wavelet, args.img_size, device)
    wandb.watch(model, log="all", log_freq=100)
    
    # Loss function
    if use_wavelet:
        loss_function = MSELoss()
        print("Using MSE loss (for wavelet domain processing)")
    else:
        loss_function = L1Loss()
        print("Using L1 loss (MAE)")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"\nStarting training for {args.model_type.upper()}{' with ' + args.wavelet + ' wavelet' if use_wavelet else ' (no wavelet)'}...")
    print(f"Dataset: {len(dataset)} slices")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Timing frequency: every {args.timing_frequency} batches\n")
    
    best_loss = float('inf')
    timing_stats = TimingStats()
    
    # Log wavelet filter visualization once at the start (for wavelet models)
    if use_wavelet:
        filter_fig = visualize_wavelet_filters(args.wavelet)
        wandb.log({"wavelet_filter_kernels": wandb.Image(filter_fig)})
        plt.close(filter_fig)
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        num_batches = len(data_loader)
        epoch_start = time()
        
        for i, (inputs, targets, slice_indices) in enumerate(data_loader):
            batch_start = perf_counter()
            
            # Data loading timing (approximate)
            data_start = perf_counter()
            inputs, targets = inputs.to(device), targets.to(device)
            data_time = perf_counter() - data_start
            timing_stats.add_data_load_time(data_time)
            
            # Print dimensions on first batch of first epoch
            if epoch == 0 and i == 0:
                print(f"\n>>> Data Dimensions:")
                print(f"    Input: {inputs.shape} (batch, channels, height, width)")
                print(f"    Target: {targets.shape}")
                print(f"    Batch contains slices: {slice_indices.tolist()[:5]}...\n")
            
            optimizer.zero_grad()
            
            # Forward pass with timing
            torch.cuda.synchronize() if device.type == 'cuda' else None
            forward_start = perf_counter()
            
            # Time wavelet transform if applicable
            if use_wavelet and hasattr(model, 'dwt2d_batch') and i == 0:
                wavelet_start = perf_counter()
                _ = model.dwt2d_batch(inputs)
                torch.cuda.synchronize() if device.type == 'cuda' else None
                wavelet_time = perf_counter() - wavelet_start
                timing_stats.add_wavelet_time(wavelet_time)
            
            outputs = model(inputs)
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            forward_time = perf_counter() - forward_start
            timing_stats.add_forward_time(forward_time)
                
            # Print wavelet dimensions on first batch if using wavelet
            if epoch == 0 and i == 0 and use_wavelet and hasattr(model, 'dwt2d_batch'):
                with torch.no_grad():
                    test_wavelets = model.dwt2d_batch(inputs)
                    print(f">>> Wavelet Transform Dimensions:")
                    print(f"    Input: {inputs.shape} -> Wavelets: {test_wavelets.shape}")
                    print(f"    (Each of 8 input channels becomes 4 subbands)")
                    print(f"    Total wavelet channels: {test_wavelets.shape[1]} = 8 × 4\n")
            
            loss = loss_function(outputs, targets)
            
            # Backward pass with timing
            torch.cuda.synchronize() if device.type == 'cuda' else None
            backward_start = perf_counter()
            
            loss.backward()
            optimizer.step()
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            backward_time = perf_counter() - backward_start
            timing_stats.add_backward_time(backward_time)
            
            # Total batch time
            batch_time = perf_counter() - batch_start
            timing_stats.add_batch_time(batch_time)
            
            epoch_loss += loss.item()
            
            # Log to W&B
            wandb.log({
                "batch_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "batch_time_ms": batch_time * 1000,
                "forward_time_ms": forward_time * 1000,
                "backward_time_ms": backward_time * 1000,
                "data_load_time_ms": data_time * 1000,
            })
            
            # Add wavelet timing if applicable
            if use_wavelet and len(timing_stats.wavelet_transform_times) > 0:
                wandb.log({
                    "wavelet_transform_time_ms": timing_stats.wavelet_transform_times[-1] * 1000
                })
            
            # Log qualitative visualization every 100 batches
            if i % 100 == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, Batch {i}/{num_batches}, Loss: {loss.item():.6f}")
                
                # Create reconstruction visualization
                with torch.no_grad():
                    panel = create_reconstruction_log_panel(
                        inputs[0], targets[0], outputs[0], 
                        slice_indices[0].item(), i
                    )
                    wandb.log({"reconstruction_preview": wandb.Image(panel)})
                
                # Log wavelet decompositions (only for wavelet models, only first batch of epoch)
                if i == 0 and use_wavelet:
                    log_wavelet_visualizations(
                        model, inputs, outputs, targets, args.wavelet, epoch
                    )
            
            # Report timing stats periodically
            if i > 0 and i % args.timing_frequency == 0:
                timing = timing_stats.get_stats()
                print(f"\n>>> Timing Stats (Batch {i}):")
                print(f"    Avg batch time: {timing['avg_batch_time']*1000:.2f}ms")
                print(f"    Avg forward time: {timing['avg_forward_time']*1000:.2f}ms")
                print(f"    Avg backward time: {timing['avg_backward_time']*1000:.2f}ms")
                print(f"    Avg data load time: {timing['avg_data_load_time']*1000:.2f}ms")
                if use_wavelet and timing['avg_wavelet_time'] > 0:
                    print(f"    Avg wavelet time: {timing['avg_wavelet_time']*1000:.2f}ms")
                print(f"    Samples/sec: {timing['samples_per_second']*args.batch_size:.1f}")
                
                # Log timing stats to wandb
                wandb.log({
                    "timing/avg_batch_time_ms": timing['avg_batch_time'] * 1000,
                    "timing/avg_forward_time_ms": timing['avg_forward_time'] * 1000,
                    "timing/avg_backward_time_ms": timing['avg_backward_time'] * 1000,
                    "timing/avg_data_load_time_ms": timing['avg_data_load_time'] * 1000,
                    "timing/samples_per_second": timing['samples_per_second'] * args.batch_size,
                })
                
                if use_wavelet and timing['avg_wavelet_time'] > 0:
                    wandb.log({
                        "timing/avg_wavelet_time_ms": timing['avg_wavelet_time'] * 1000
                    })
        
        # Calculate average epoch loss and epoch time
        avg_epoch_loss = epoch_loss / num_batches
        epoch_time = time() - epoch_start
        
        print(f"Epoch {epoch+1}/{args.epochs} - Average Loss: {avg_epoch_loss:.6f}, Time: {epoch_time:.1f}s")
        
        wandb.log({
            "epoch": epoch + 1,
            "avg_epoch_loss": avg_epoch_loss,
            "epoch_time_seconds": epoch_time,
        })
        
        # Save best checkpoint
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            if use_wavelet:
                checkpoint_name = f"{args.model_type}_wavelet_{args.wavelet}_best.pth"
            else:
                checkpoint_name = f"{args.model_type}_baseline_best.pth"
            checkpoint_path = os.path.join(args.output_dir, checkpoint_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': vars(args)
            }, checkpoint_path)
            print(f"Saved best checkpoint to {checkpoint_path}")
            wandb.save(checkpoint_path)
    
    # Final training timing summary
    final_timing = timing_stats.get_stats()
    print(f"\n" + "="*60)
    print("TRAINING TIMING SUMMARY")
    print("="*60)
    print(f"Average batch time: {final_timing['avg_batch_time']*1000:.2f}ms")
    print(f"Average forward time: {final_timing['avg_forward_time']*1000:.2f}ms") 
    print(f"Average backward time: {final_timing['avg_backward_time']*1000:.2f}ms")
    print(f"Average data load time: {final_timing['avg_data_load_time']*1000:.2f}ms")
    if use_wavelet and final_timing['avg_wavelet_time'] > 0:
        print(f"Average wavelet time: {final_timing['avg_wavelet_time']*1000:.2f}ms")
    print(f"Training throughput: {final_timing['samples_per_second']*args.batch_size:.1f} samples/sec")
    print(f"Total training time: {final_timing['total_batch_time']/60:.1f} minutes")
    print("="*60)
    
    # Log final timing stats
    wandb.log({
        "final_timing/avg_batch_time_ms": final_timing['avg_batch_time'] * 1000,
        "final_timing/avg_forward_time_ms": final_timing['avg_forward_time'] * 1000,
        "final_timing/avg_backward_time_ms": final_timing['avg_backward_time'] * 1000,
        "final_timing/samples_per_second": final_timing['samples_per_second'] * args.batch_size,
        "final_timing/total_training_minutes": final_timing['total_batch_time'] / 60,
    })
    
    if use_wavelet and final_timing['avg_wavelet_time'] > 0:
        wandb.log({
            "final_timing/avg_wavelet_time_ms": final_timing['avg_wavelet_time'] * 1000
        })
    
    # ===== INFERENCE TIMING MEASUREMENT =====
    print(f"\n" + "="*60)
    print("MEASURING INFERENCE TIMING")
    print("="*60)
    
    # Reload best checkpoint for inference timing
    if use_wavelet:
        checkpoint_name = f"{args.model_type}_wavelet_{args.wavelet}_best.pth"
    else:
        checkpoint_name = f"{args.model_type}_baseline_best.pth"
    checkpoint_path = os.path.join(args.output_dir, checkpoint_name)
    
    print(f"Loading best checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create inference dataloader (no shuffle for reproducibility)
    inference_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Measure inference timing
    inference_timing = measure_inference_timing(model, inference_loader, device, num_batches=100)
    
    print(f"\nINFERENCE TIMING RESULTS:")
    print(f"Average inference time per batch: {inference_timing['avg_forward_time']*1000:.2f}ms")
    print(f"Average data loading time: {inference_timing['avg_data_load_time']*1000:.2f}ms") 
    if use_wavelet and inference_timing['avg_wavelet_time'] > 0:
        print(f"Average wavelet transform time: {inference_timing['avg_wavelet_time']*1000:.2f}ms")
    print(f"Inference throughput: {(1/inference_timing['avg_forward_time'])*args.batch_size:.1f} samples/sec")
    print("="*60)
    
    # Log inference timing
    wandb.log({
        "inference_timing/avg_forward_time_ms": inference_timing['avg_forward_time'] * 1000,
        "inference_timing/avg_data_load_time_ms": inference_timing['avg_data_load_time'] * 1000,
        "inference_timing/samples_per_second": (1/inference_timing['avg_forward_time']) * args.batch_size,
    })
    
    if use_wavelet and inference_timing['avg_wavelet_time'] > 0:
        wandb.log({
            "inference_timing/avg_wavelet_time_ms": inference_timing['avg_wavelet_time'] * 1000
        })
    
    # Final summary
    print(f"\nTraining completed! Best loss: {best_loss:.6f}")
    wandb.log({"best_loss": best_loss})
    
    # ===== POST-TRAINING EVALUATION =====
    if not args.skip_eval:
        print("\n" + "="*60)
        print("Running evaluation on best checkpoint...")
        print("="*60)
        
        # Create evaluation dataloader (no shuffle for reproducibility)
        eval_loader = DataLoader(
            dataset, 
            batch_size=args.eval_batch_size,  # Larger batch for faster eval
            shuffle=False,  # Don't shuffle for reproducible evaluation
            num_workers=4
        )
        
        # Import and run evaluation
        try:
            from evaluate import run_evaluation
            
            # Determine output directory for results
            if use_wavelet:
                results_dir = f"./results/{args.model_type}_wavelet_{args.wavelet}"
            else:
                results_dir = f"./results/{args.model_type}_baseline"
            
            # Run evaluation (this logs to the same wandb run)
            print(f"Evaluating model...")
            results, _ = run_evaluation(
                model=model,
                data_loader=eval_loader,
                device=device,
                output_dir=results_dir,
                model_type=args.model_type,
                wavelet_name=args.wavelet if args.wavelet != 'none' else 'N/A',
                save_wavelets=(args.wavelet != 'none')  # Only save wavelets for wavelet models
            )
            
            # Print final evaluation results
            print("\n" + "="*60)
            print("FINAL EVALUATION RESULTS")
            print("="*60)
            print(f"MSE:  {results['mse_mean']:.6f} ± {results['mse_std']:.6f}")
            print(f"SSIM: {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")
            print(f"Samples evaluated: {results['num_samples']}")
            print("\nPer-modality MSE:")
            print(f"  T1n:   {results['mse_t1n_mean']:.6f} ± {results['mse_t1n_std']:.6f}")
            print(f"  T1c:   {results['mse_t1c_mean']:.6f} ± {results['mse_t1c_std']:.6f}")
            print(f"  T2w:   {results['mse_t2w_mean']:.6f} ± {results['mse_t2w_std']:.6f}")
            print(f"  T2f:   {results['mse_t2f_mean']:.6f} ± {results['mse_t2f_std']:.6f}")
            print("\nPer-modality SSIM:")
            print(f"  T1n:   {results['ssim_t1n_mean']:.4f} ± {results['ssim_t1n_std']:.4f}")
            print(f"  T1c:   {results['ssim_t1c_mean']:.4f} ± {results['ssim_t1c_std']:.4f}")
            print(f"  T2w:   {results['ssim_t2w_mean']:.4f} ± {results['ssim_t2w_std']:.4f}")
            print(f"  T2f:   {results['ssim_t2f_mean']:.4f} ± {results['ssim_t2f_std']:.4f}")
            print("="*60)
            
            print(f"\nEvaluation results saved to {results_dir}/")
            
        except ImportError:
            print("Warning: evaluate.py not found, skipping post-training evaluation")
            
    else:
        print("\nSkipping evaluation (--skip_eval flag set)")
    
    wandb.finish()


if __name__ == '__main__':
    try:
        args = get_args()
        main(args)
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        wandb.finish(exit_code=1)
        raise