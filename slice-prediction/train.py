# train.py - Updated with wavelet diffusion models

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


class BraTS2D5Dataset(Dataset):
    def __init__(self, data_dir, image_size, spacing, num_patients=None):
        self.image_size = image_size
        patient_dirs = sorted(glob.glob(os.path.join(data_dir, "BraTS*")))
        if num_patients is not None:
            print(f"--- Using a subset of {num_patients} patients for testing. ---")
            patient_dirs = patient_dirs[:num_patients]
        if not patient_dirs:
            raise FileNotFoundError(f"No patient data found in '{data_dir}'. Check your --data_dir path.")
        self.files = [{"t1": glob.glob(os.path.join(p, "*_t1.nii"))[0],"t1ce": glob.glob(os.path.join(p, "*_t1ce.nii"))[0],"t2": glob.glob(os.path.join(p, "*_t2.nii"))[0],"flair": glob.glob(os.path.join(p, "*_flair.nii"))[0],"label": glob.glob(os.path.join(p, "*_seg.nii"))[0]} for p in patient_dirs]
        transforms = get_train_transforms(image_size, spacing)
        print("--- Pre-loading and processing volumes... ---")
        start_time = time()
        self.processed_volumes = []
        for i, patient_files in enumerate(self.files):
            self.processed_volumes.append(transforms(patient_files))
            if (i + 1) % 10 == 0 or (i + 1) == len(self.files):
                print(f"   Processed {i + 1}/{len(self.files)} patients...")
        print(f"--- Volume processing took {time() - start_time:.2f} seconds. ---")
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
        
        img_modalities = torch.cat([patient_data['t1'], patient_data['t1ce'], patient_data['t2'], patient_data['flair']], dim=0)
        
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
                       help='Wavelet type (only for wavelet model). Options: haar, db1-20, sym2-20, coif1-17, bior1.1-6.8, rbio1.1-6.8')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--num_patients', type=int, default=None)
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

    print(f"Starting training for {args.model_type}...")
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