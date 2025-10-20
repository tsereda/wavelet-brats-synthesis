#!/usr/bin/env python3
# train.py - Complete training script with FIXED dataset loader for BraTS2020 + AUTO-EVALUATION

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
from monai.networks.nets import SwinUNETR
from torch.nn import L1Loss, MSELoss
from transforms import get_train_transforms
from logging_utils import create_reconstruction_log_panel
from skimage.metrics import structural_similarity as ssim


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
            
            # Try different naming patterns for BraTS2020
            # Pattern 1: BraTS20_Training_001_t1.nii (no .gz)
            # Pattern 2: BraTS20_Training_001_t1.nii.gz
            
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
                       choices=['swin', 'wavelet'],
                       help='Model architecture to use')
    parser.add_argument('--wavelet', type=str, default='haar',
                       help='Wavelet type (only for wavelet model)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--num_patients', type=int, default=None)
    parser.add_argument('--skip_eval', action='store_true',
                       help='Skip automatic evaluation after training')
    return parser.parse_args()


def get_model(model_type, wavelet_name, img_size, device):
    """Load model based on type"""
    if model_type == 'swin':
        model = SwinUNETR(
            in_channels=8, 
            out_channels=4, 
            feature_size=24, 
            spatial_dims=2
        ).to(device)
        print("Loaded Swin-UNETR model")
    
    elif model_type == 'wavelet':
        from models.wavelet_diffusion import WaveletDiffusion
        model = WaveletDiffusion(
            wavelet_name=wavelet_name,
            in_channels=8,
            out_channels=4,
            timesteps=100
        ).to(device)
        print(f"Loaded Wavelet Diffusion model ({wavelet_name})")
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def evaluate_model_quick(model, data_loader, device, max_batches=50):
    """
    Quick evaluation to get MSE and SSIM metrics
    
    Args:
        model: trained model
        data_loader: DataLoader for evaluation
        device: cuda or cpu
        max_batches: maximum number of batches to evaluate (to save time)
    
    Returns:
        dict with evaluation metrics
    """
    model.eval()
    
    all_mse = []
    all_ssim = []
    all_mse_per_mod = {mod: [] for mod in ['t1', 't1ce', 't2', 'flair']}
    all_ssim_per_mod = {mod: [] for mod in ['t1', 't1ce', 't2', 'flair']}
    
    print(f"Evaluating on up to {max_batches} batches...")
    
    with torch.no_grad():
        for batch_idx, (inputs, targets, slice_indices) in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
                
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate metrics for each sample in batch
            batch_size = outputs.shape[0]
            for i in range(batch_size):
                pred = outputs[i].cpu().numpy()
                gt = targets[i].cpu().numpy()
                
                # Calculate MSE per modality
                mse_per_modality = np.mean((pred - gt) ** 2, axis=(1, 2))
                all_mse.append(np.mean(mse_per_modality))
                
                for mod_idx, mod_name in enumerate(['t1', 't1ce', 't2', 'flair']):
                    all_mse_per_mod[mod_name].append(mse_per_modality[mod_idx])
                
                # Calculate SSIM per modality
                ssim_scores = []
                for mod_idx, mod_name in enumerate(['t1', 't1ce', 't2', 'flair']):
                    data_range = max(pred[mod_idx].max(), gt[mod_idx].max()) - min(pred[mod_idx].min(), gt[mod_idx].min())
                    if data_range == 0:
                        data_range = 1.0
                    
                    score = ssim(
                        gt[mod_idx], 
                        pred[mod_idx], 
                        data_range=data_range
                    )
                    ssim_scores.append(score)
                    all_ssim_per_mod[mod_name].append(score)
                
                all_ssim.append(np.mean(ssim_scores))
    
    # Aggregate results
    results = {
        'eval/mse_mean': np.mean(all_mse),
        'eval/mse_std': np.std(all_mse),
        'eval/ssim_mean': np.mean(all_ssim),
        'eval/ssim_std': np.std(all_ssim),
        'eval/num_samples': len(all_mse)
    }
    
    # Add per-modality metrics
    for mod_name in ['t1', 't1ce', 't2', 'flair']:
        results[f'eval/mse_{mod_name}'] = np.mean(all_mse_per_mod[mod_name])
        results[f'eval/ssim_{mod_name}'] = np.mean(all_ssim_per_mod[mod_name])
    
    return results


def main(args):
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    # Create run name with wavelet type
    if args.model_type == 'wavelet':
        run_name = f"wavelet_{args.wavelet}_{int(time())}"
    else:
        run_name = f"{args.model_type}_reconstruction_{int(time())}"
    
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
    if args.model_type == 'swin':
        loss_function = L1Loss()
        print("Using L1 loss (MAE)")
    else:
        loss_function = MSELoss()
        print("Using MSE loss (for diffusion)")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"Starting training for {args.model_type} ({args.wavelet if args.model_type == 'wavelet' else 'N/A'})...")
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        num_batches = len(data_loader)
        
        for i, (inputs, targets, slice_indices) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            if args.model_type == 'swin':
                outputs = model(inputs)
            else:
                outputs = model(inputs, t=None)
            
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
                
                # Create visualization
                with torch.no_grad():
                    panel = create_reconstruction_log_panel(
                        inputs[0], targets[0], outputs[0], 
                        slice_indices[0].item(), i
                    )
                    wandb.log({"reconstruction_preview": wandb.Image(panel)})
        
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
            checkpoint_name = f"{args.model_type}_{args.wavelet if args.model_type == 'wavelet' else 'baseline'}_best.pth"
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
    
    # ===== AUTOMATIC EVALUATION =====
    if not args.skip_eval:
        print("\n" + "="*60)
        print("STARTING AUTOMATIC EVALUATION")
        print("="*60)
        
        # Reload best model
        checkpoint_name = f"{args.model_type}_{args.wavelet if args.model_type == 'wavelet' else 'baseline'}_best.pth"
        checkpoint_path = os.path.join(args.output_dir, checkpoint_name)
        
        print(f"Loading best checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create evaluation dataloader (no shuffle, larger batch size)
        eval_loader = DataLoader(
            dataset, 
            batch_size=args.batch_size * 2,
            shuffle=False, 
            num_workers=4
        )
        
        # Run evaluation
        eval_results = evaluate_model_quick(model, eval_loader, device, max_batches=50)
        
        # Log to WandB
        wandb.log(eval_results)
        
        # Print results
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"MSE:  {eval_results['eval/mse_mean']:.6f} ± {eval_results['eval/mse_std']:.6f}")
        print(f"SSIM: {eval_results['eval/ssim_mean']:.4f} ± {eval_results['eval/ssim_std']:.4f}")
        print(f"Samples evaluated: {eval_results['eval/num_samples']}")
        print("\nPer-modality MSE:")
        print(f"  T1:    {eval_results['eval/mse_t1']:.6f}")
        print(f"  T1ce:  {eval_results['eval/mse_t1ce']:.6f}")
        print(f"  T2:    {eval_results['eval/mse_t2']:.6f}")
        print(f"  FLAIR: {eval_results['eval/mse_flair']:.6f}")
        print("\nPer-modality SSIM:")
        print(f"  T1:    {eval_results['eval/ssim_t1']:.4f}")
        print(f"  T1ce:  {eval_results['eval/ssim_t1ce']:.4f}")
        print(f"  T2:    {eval_results['eval/ssim_t2']:.4f}")
        print(f"  FLAIR: {eval_results['eval/ssim_flair']:.4f}")
        print("="*60)
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