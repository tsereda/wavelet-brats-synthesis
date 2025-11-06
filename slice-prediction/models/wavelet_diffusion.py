import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np


class WaveletDiffusion(nn.Module):
    """
    Fast-cWDM for 2D slice prediction with differentiable wavelet transforms
    """
    def __init__(self, wavelet_name='haar', in_channels=8, out_channels=4, 
                 timesteps=100, base_channels=64):
        super().__init__()
        self.timesteps = timesteps
        self.wavelet_name = wavelet_name
        
        # Validate wavelet
        try:
            self.wavelet = pywt.Wavelet(wavelet_name)
        except:
            raise ValueError(f"Invalid wavelet: {wavelet_name}")
        
        print(f"Using wavelet: {self.wavelet.name} (family: {self.wavelet.family_name})")
        print(f"Filter lengths - Low: {len(self.wavelet.dec_lo)}, High: {len(self.wavelet.dec_hi)}")
        
        # Pre-compute and register wavelet filter coefficients as buffers
        self._init_wavelet_filters()
        
        # Calculate wavelet space dimensions
        wavelet_channels = in_channels * 4  # Each input channel becomes 4 subbands
        print(f"\nWavelet space configuration:")
        print(f"  Input: {in_channels} channels -> {wavelet_channels} wavelet channels")
        print(f"  Output: {out_channels} channels -> {out_channels * 4} wavelet channels")
        print(f"  Spatial reduction: H×W -> (H/2)×(W/2)")
        
        # U-Net for processing in wavelet space
        self.encoder = nn.Sequential(
            nn.Conv2d(wavelet_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1),
            nn.GroupNorm(16, base_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1),
            nn.GroupNorm(32, base_channels*4),
            nn.ReLU(inplace=True),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*2, 3, padding=1),
            nn.GroupNorm(16, base_channels*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels*2, base_channels, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, out_channels * 4, 3, padding=1),
        )
        
        print(f"  Encoder: {wavelet_channels} -> {base_channels} -> {base_channels*2} -> {base_channels*4}")
        print(f"  Decoder: {base_channels*4} -> {base_channels*2} -> {base_channels} -> {out_channels*4}")
    
    def _init_wavelet_filters(self):
        """Initialize wavelet filters as conv kernels for differentiable operations"""
        # Get wavelet filter coefficients
        dec_lo = torch.tensor(self.wavelet.dec_lo[::-1], dtype=torch.float32)
        dec_hi = torch.tensor(self.wavelet.dec_hi[::-1], dtype=torch.float32)
        rec_lo = torch.tensor(self.wavelet.rec_lo, dtype=torch.float32)
        rec_hi = torch.tensor(self.wavelet.rec_hi, dtype=torch.float32)
        
        # Create 2D separable filters (outer product)
        # Decomposition filters (for DWT)
        self.register_buffer('dec_ll', (dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0))
        self.register_buffer('dec_lh', (dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0))
        self.register_buffer('dec_hl', (dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0))
        self.register_buffer('dec_hh', (dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0))
        
        # Reconstruction filters (for IDWT)
        self.register_buffer('rec_ll', (rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0))
        self.register_buffer('rec_lh', (rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0))
        self.register_buffer('rec_hl', (rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0))
        self.register_buffer('rec_hh', (rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0))
        
        # Calculate padding for even filter length
        self.padding = (self.dec_ll.shape[-1] - 1) // 2
    
    def dwt2d_batch(self, x):
        """
        Differentiable 2D DWT using conv2d
        x: [B, C, H, W]
        returns: [B, C*4, H/2, W/2]
        """
        B, C, H, W = x.shape
        
        # Apply filters to each channel separately
        coeffs = []
        for i in range(C):
            x_ch = x[:, i:i+1, :, :]  # [B, 1, H, W]
            
            # Apply 4 separable filters with stride 2 (downsampling)
            ll = F.conv2d(x_ch, self.dec_ll, stride=2, padding=self.padding)
            lh = F.conv2d(x_ch, self.dec_lh, stride=2, padding=self.padding)
            hl = F.conv2d(x_ch, self.dec_hl, stride=2, padding=self.padding)
            hh = F.conv2d(x_ch, self.dec_hh, stride=2, padding=self.padding)
            
            # Stack subbands: [B, 4, H/2, W/2]
            coeffs.append(torch.cat([ll, lh, hl, hh], dim=1))
        
        # Concatenate all channels: [B, C*4, H/2, W/2]
        result = torch.cat(coeffs, dim=1)
        
        # Print dimensions on first call (training mode only)
        if self.training and not hasattr(self, '_dims_printed'):
            print(f"\nDWT Transform: Input {x.shape} -> Wavelet coeffs {result.shape}")
            print(f"  Each channel split into 4 subbands (LL, LH, HL, HH)")
            self._dims_printed = True
        
        return result
    
    def idwt2d_batch(self, coeffs, target_shape):
        """
        Differentiable inverse 2D DWT using conv_transpose2d
        coeffs: [B, C*4, H/2, W/2]
        returns: [B, C, H, W]
        """
        B, total_ch, H_half, W_half = coeffs.shape
        C = total_ch // 4
        
        reconstructed = []
        for i in range(C):
            # Extract 4 subbands for this channel
            start = i * 4
            ll = coeffs[:, start:start+1, :, :]
            lh = coeffs[:, start+1:start+2, :, :]
            hl = coeffs[:, start+2:start+3, :, :]
            hh = coeffs[:, start+3:start+4, :, :]
            
            # Upsample and apply reconstruction filters
            ll_up = F.conv_transpose2d(ll, self.rec_ll * 4, stride=2, padding=self.padding)
            lh_up = F.conv_transpose2d(lh, self.rec_lh * 4, stride=2, padding=self.padding)
            hl_up = F.conv_transpose2d(hl, self.rec_hl * 4, stride=2, padding=self.padding)
            hh_up = F.conv_transpose2d(hh, self.rec_hh * 4, stride=2, padding=self.padding)
            
            # Sum all components
            x_rec = ll_up + lh_up + hl_up + hh_up
            reconstructed.append(x_rec)
        
        result = torch.cat(reconstructed, dim=1)  # [B, C, H, W]
        
        # Ensure correct output size (handle odd dimensions)
        if result.shape[2:] != target_shape:
            result = F.interpolate(result, size=target_shape, mode='bilinear', align_corners=False)
        
        # Print dimensions on first call (training mode only)
        if self.training and not hasattr(self, '_idwt_dims_printed'):
            print(f"IDWT Transform: Wavelet coeffs {coeffs.shape} -> Output {result.shape}")
            self._idwt_dims_printed = True
        
        return result
    
    def forward(self, x, t=None):
        """
        Forward pass with differentiable wavelet transforms
        x: [B, 8, H, W] - concatenated prev and next slices
        Returns: [B, 4, H, W] - predicted middle slice
        """
        target_shape = (x.shape[2], x.shape[3])
        
        # Transform to wavelet space (differentiable!)
        wavelet_input = self.dwt2d_batch(x)  # [B, 32, H/2, W/2]
        
        # Process in wavelet space
        features = self.encoder(wavelet_input)
        wavelet_output = self.decoder(features)
        
        # Transform back to image space (differentiable!)
        output = self.idwt2d_batch(wavelet_output, target_shape)
        
        return output