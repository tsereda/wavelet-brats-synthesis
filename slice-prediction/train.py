#!/usr/bin/env python3
# train.py - Complete training script with wavelet: none option

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import argparse
from time import time
import torch.multiprocessing
import cv2
import wandb
from monai.networks.nets import SwinUNETR, UNETR, BasicUNet
from torch.nn import L1Loss, MSELoss
from transforms import get_train_transforms
from logging_utils import create_reconstruction_log_panel
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class BraTS2D5Dataset(Dataset):
    def __init__(self, data_dir, image_size, spacing, num_patients=None):
        self.image_size = image_size
        patient_dirs = sorted(glob.glob(os.path.join(data_dir, "BraTS*")))
        if num_patients is not None:
            print(f"--- Using a subset of {num_patients} patients for testing. ---")
            patient_dirs = patient_dirs[:num_patients]
        if not patient_dirs:
            raise FileNotFoundError(f"No patient data found in '{data_dir}'. Check your --data_dir path.")
        
        print(f"Found {len(patient_dirs)} patient directories")
        
        # Build file list with better error handling
        self.files = []
        for p in patient_dirs:
            patient_name = os.path.basename(p)
            
            try:
                patient_files = {}
                
                # Find each modality with flexible pattern matching
                for modality in ['t1', 't1ce', 't2', 'flair']:
                    pattern = os.path.join(p, f"*{modality}.nii*")
                    matches = glob.glob(pattern)
                    if not matches:
                        raise FileNotFoundError(f"No {modality} file found in {patient_name}")
                    patient_files[modality] = matches[0]
                
                # Find segmentation (might be 'seg' or 'label')
                seg_matches = glob.glob(os.path.join(p, "*seg.nii*"))
                if not seg_matches:
                    # Try alternative pattern
                    seg_matches = glob.glob(os.path.join(p, "*label.nii*"))
                if not seg_matches:
                    raise FileNotFoundError(f"No segmentation file found in {patient_name}")
                patient_files['label'] = seg_matches[0]
                
                self.files.append(patient_files)
                
            except FileNotFoundError as e:
                print(f"Warning: Skipping patient {patient_name}: {e}")
                continue
        
        if not self.files:
            raise RuntimeError("No valid patients found! Check your dataset structure.")
        
        print(f"Successfully loaded {len(self.files)} patients")
        
        # Process volumes
        transforms = get_train_transforms(image_size, spacing)
        print("--- Pre-loading and processing volumes... ---")
        start_time = time()
        self.processed_volumes = []
        for i, patient_files in enumerate(self.files):
            self.processed_volumes.append(transforms(patient_files))
            if (i + 1) % 10 == 0 or (i + 1) == len(self.files):
                print(f"   Processed {i + 1}/{len(self.files)} patients...")
        print(f"--- Volume processing took {time() - start_time:.2f} seconds. ---")
        
        # Create slice map
        self.slice_map = []
        print("Mapping and filtering slices to create dataset...")
        for vol_idx, p_data in enumerate(self.processed_volumes):
            num_slices = p_data["label"].shape[3]
            for slice_idx in range(1, num_slices - 1): 
                brain_slice = p_data["t1ce"][0, :, :, slice_idx]
                if torch.mean(brain_slice) > 0.1:
                    self.slice_map.append((vol_idx, slice_idx))
        print(f"Dataset ready. Found {len(self.slice_map)} valid slices from {len(self.files)} volumes.")

    def __len__(self):
        return len(self.slice_map)
    
    def __getitem__(self, index):
        volume_idx, slice_idx = self.slice_map[index]
        patient_data = self.processed_volumes[volume_idx]
        
        img_modalities = torch.cat([patient_data['t1'], patient_data['t1ce'], 
                                    patient_data['t2'], patient_data['flair']], dim=0)
        
        prev_slice = img_modalities[:, :, :, slice_idx - 1]
        next_slice = img_modalities[:, :, :, slice_idx + 1]
        input_tensor = torch.cat([prev_slice, next_slice], dim=0)
        target_tensor = img_modalities[:, :, :, slice_idx]

        return input_tensor, target_tensor, slice_idx


def get_args():
    parser = argparse.ArgumentParser(description="2.5D Middleslice Reconstruction")
    parser.add_argument('--data_dir', type=str, required=True)
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
    parser.add_argument('--num_patients', type=int, default=None)
    parser.add_argument('--eval_batch_size', type=int, default=16,
                       help='Batch size for evaluation (default: 16)')
    parser.add_argument('--skip_eval', action='store_true',
                       help='Skip evaluation after training')
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
    
    modalities = ['T1', 'T1ce', 'T2', 'FLAIR', 'T1(Z+1)', 'T1ce(Z+1)', 'T2(Z+1)', 'FLAIR(Z+1)']
    
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
    Only works for wavelet models
    """
    if not hasattr(model, 'dwt2d_batch'):
        return  # Not a wavelet model, skip
    
    with torch.no_grad():
        # Get wavelet decomposition of input (8 channels)
        input_wavelets = model.dwt2d_batch(inputs[:1])  # Just first sample
        
        # Get wavelet decomposition of output (4 channels)
        output_wavelets = model.dwt2d_batch(outputs[:1])
        
        # Get wavelet decomposition of ground truth (4 channels)
        target_wavelets = model.dwt2d_batch(targets[:1])
        
        # Print dimensions for debugging
        print(f"\n>>> Wavelet Decomposition Dimensions (Epoch {epoch}):")
        print(f"    Input shape: {inputs.shape} -> Wavelet shape: {input_wavelets.shape}")
        print(f"    Output shape: {outputs.shape} -> Wavelet shape: {output_wavelets.shape}")
        print(f"    Target shape: {targets.shape} -> Wavelet shape: {target_wavelets.shape}")
        print(f"    (Each modality becomes 4 subbands: LL, LH, HL, HH)\n")
        
        # Visualize the wavelet filter itself (only once at epoch 0)
        if epoch == 0:
            filter_fig = visualize_wavelet_filters(wavelet_name)
            wandb.log({f"wavelet_filters/{wavelet_name}": wandb.Image(filter_fig)})
            plt.close(filter_fig)
        
        # ... (rest of visualization code remains the same)


def main(args):
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    # Determine if using wavelet
    use_wavelet = args.wavelet != 'none'
    
    # Create run name with model type and wavelet info
    if use_wavelet:
        run_name = f"{args.model_type}_wavelet_{args.wavelet}_{int(time())}"
    else:
        run_name = f"{args.model_type}_baseline_{int(time())}"
    
    wandb.init(project="brats-middleslice-wavelet-sweep", config=vars(args), name=run_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    print("Loading dataset...")
    dataset = BraTS2D5Dataset(
        data_dir=args.data_dir, 
        image_size=(args.img_size, args.img_size),
        spacing=(1.0, 1.0, 1.0), 
        num_patients=args.num_patients
    )
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
    print(f"Epochs: {args.epochs}\n")
    
    best_loss = float('inf')
    
    # Log wavelet filter visualization once at the start (for wavelet models)
    if use_wavelet:
        filter_fig = visualize_wavelet_filters(args.wavelet)
        wandb.log({"wavelet_filter_kernels": wandb.Image(filter_fig)})
        plt.close(filter_fig)
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        num_batches = len(data_loader)
        
        for i, (inputs, targets, slice_indices) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Print dimensions on first batch of first epoch
            if epoch == 0 and i == 0:
                print(f"\n>>> Data Dimensions:")
                print(f"    Input: {inputs.shape} (batch, channels, height, width)")
                print(f"    Target: {targets.shape}")
                print(f"    Batch contains slices: {slice_indices.tolist()[:5]}...\n")
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
                
            # Print wavelet dimensions on first batch if using wavelet
            if epoch == 0 and i == 0 and use_wavelet and hasattr(model, 'dwt2d_batch'):
                with torch.no_grad():
                    test_wavelets = model.dwt2d_batch(inputs)
                    print(f">>> Wavelet Transform Dimensions:")
                    print(f"    Input: {inputs.shape} -> Wavelets: {test_wavelets.shape}")
                    print(f"    (Each of 8 input channels becomes 4 subbands)")
                    print(f"    Total wavelet channels: {test_wavelets.shape[1]} = 8 × 4\n")
            
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Log to W&B
            wandb.log({
                "batch_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
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
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{args.epochs} - Average Loss: {avg_epoch_loss:.6f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "avg_epoch_loss": avg_epoch_loss,
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
    
    # Final summary
    print(f"\nTraining completed! Best loss: {best_loss:.6f}")
    wandb.log({"best_loss": best_loss})
    
    # ===== POST-TRAINING EVALUATION =====
    if not args.skip_eval:
        print("\n" + "="*60)
        print("Running evaluation on best checkpoint...")
        print("="*60)
        
        # Reload best checkpoint
        if use_wavelet:
            checkpoint_name = f"{args.model_type}_wavelet_{args.wavelet}_best.pth"
        else:
            checkpoint_name = f"{args.model_type}_baseline_best.pth"
        checkpoint_path = os.path.join(args.output_dir, checkpoint_name)
        
        print(f"Loading best checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create evaluation dataloader (no shuffle for reproducibility)
        eval_loader = DataLoader(
            dataset, 
            batch_size=args.eval_batch_size,  # Larger batch for faster eval
            shuffle=False,  # Don't shuffle for reproducible evaluation
            num_workers=4
        )
        
        # Import and run evaluation
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
        print(f"  T1:    {results['mse_t1_mean']:.6f} ± {results['mse_t1_std']:.6f}")
        print(f"  T1ce:  {results['mse_t1ce_mean']:.6f} ± {results['mse_t1ce_std']:.6f}")
        print(f"  T2:    {results['mse_t2_mean']:.6f} ± {results['mse_t2_std']:.6f}")
        print(f"  FLAIR: {results['mse_flair_mean']:.6f} ± {results['mse_flair_std']:.6f}")
        print("\nPer-modality SSIM:")
        print(f"  T1:    {results['ssim_t1_mean']:.4f} ± {results['ssim_t1_std']:.4f}")
        print(f"  T1ce:  {results['ssim_t1ce_mean']:.4f} ± {results['ssim_t1ce_std']:.4f}")
        print(f"  T2:    {results['ssim_t2_mean']:.4f} ± {results['ssim_t2_std']:.4f}")
        print(f"  FLAIR: {results['ssim_flair_mean']:.4f} ± {results['ssim_flair_std']:.4f}")
        print("="*60)
        
        print(f"\nEvaluation results saved to {results_dir}/")
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