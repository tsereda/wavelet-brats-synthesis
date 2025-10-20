import torch
import torch.nn as nn
import pywt
import numpy as np


class WaveletDiffusion(nn.Module):
    """
    Universal Wavelet Diffusion Model supporting all PyWavelets
    Can be used with any discrete wavelet: haar, db, sym, coif, bior, rbio, dmey
    """
    def __init__(self, wavelet_name='haar', in_channels=8, out_channels=4, 
                 timesteps=100, base_channels=64):
        super().__init__()
        self.timesteps = timesteps
        self.wavelet_name = wavelet_name
        
        # Validate wavelet
        try:
            self.wavelet = pywt.Wavelet(wavelet_name)
            if not self.wavelet.orthogonal and not self.wavelet.biorthogonal:
                raise ValueError(f"Wavelet {wavelet_name} is not suitable for DWT")
        except:
            raise ValueError(f"Invalid wavelet name: {wavelet_name}. "
                           f"Use one from: {pywt.wavelist(kind='discrete')}")
        
        print(f"Using wavelet: {self.wavelet.name} (family: {self.wavelet.family_name})")
        
        # Encoder for context (adjacent slices)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1),
            nn.GroupNorm(16, base_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1),
            nn.GroupNorm(32, base_channels*4),
            nn.ReLU(inplace=True),
        )
        
        # Wavelet coefficient predictor
        # After DWT, we get 4 subbands (LL, LH, HL, HH) for each of 4 modalities = 16 channels
        self.wavelet_predictor = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*2, 3, padding=1),
            nn.GroupNorm(16, base_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 16, 3, padding=1),  # 4 modalities * 4 subbands
        )
        
        # Upsample back to original resolution
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
    def dwt2d_batch(self, x):
        """
        Apply 2D DWT to entire batch
        x: [B, C, H, W]
        returns: [B, C*4, H/2, W/2] with LL, LH, HL, HH coefficients
        """
        batch_size, channels, h, w = x.shape
        device = x.device
        
        # Process entire batch at once
        x_np = x.cpu().numpy()
        coeffs_batch = []
        
        for b in range(batch_size):
            coeffs_sample = []
            for c in range(channels):
                # Apply DWT
                coeffs = pywt.dwt2(x_np[b, c], self.wavelet_name)
                cA, (cH, cV, cD) = coeffs
                
                # Stack: LL, LH, HL, HH
                stacked = np.stack([cA, cH, cV, cD], axis=0)
                coeffs_sample.append(stacked)
            
            # Stack all channels: [C*4, H/2, W/2]
            coeffs_batch.append(np.concatenate(coeffs_sample, axis=0))
        
        # Convert back to tensor: [B, C*4, H/2, W/2]
        return torch.from_numpy(np.stack(coeffs_batch, axis=0)).float().to(device)
    
    def idwt2d_batch(self, coeffs, target_shape):
        """
        Apply inverse 2D DWT to entire batch
        coeffs: [B, C*4, H/2, W/2]
        returns: [B, C, H, W]
        """
        batch_size, total_channels, h, w = coeffs.shape
        num_modalities = total_channels // 4
        device = coeffs.device
        
        coeffs_np = coeffs.cpu().numpy()
        reconstructed_batch = []
        
        for b in range(batch_size):
            reconstructed_sample = []
            
            for i in range(num_modalities):
                # Extract 4 subbands for this modality
                start_idx = i * 4
                cA = coeffs_np[b, start_idx]
                cH = coeffs_np[b, start_idx + 1]
                cV = coeffs_np[b, start_idx + 2]
                cD = coeffs_np[b, start_idx + 3]
                
                # Reconstruct
                reconstructed = pywt.idwt2((cA, (cH, cV, cD)), self.wavelet_name)
                reconstructed_sample.append(reconstructed)
            
            reconstructed_batch.append(np.stack(reconstructed_sample, axis=0))
        
        result = torch.from_numpy(np.stack(reconstructed_batch, axis=0)).float().to(device)
        
        # Ensure correct shape
        if result.shape[2:] != target_shape:
            result = torch.nn.functional.interpolate(
                result, size=target_shape, mode='bilinear', align_corners=False
            )
        
        return result
    
    def forward(self, x, t=None):
        """
        Forward pass
        x: [B, 8, H, W] - concatenated prev and next slices
        t: timestep (not used in simplified version)
        Returns: [B, 4, H, W] - predicted middle slice
        """
        target_shape = (x.shape[2], x.shape[3])
        
        # Encode context
        context = self.encoder(x)  # [B, base_channels*4, H/2, W/2]
        
        # Predict wavelet coefficients
        wavelet_coeffs = self.wavelet_predictor(context)  # [B, 16, H/2, W/2]
        
        # Upsample coefficients to match DWT output size
        wavelet_coeffs = self.upsample(wavelet_coeffs)  # [B, 16, H, W]
        
        # Reconstruct from wavelets
        output = self.idwt2d_batch(wavelet_coeffs, target_shape)
        
        return output