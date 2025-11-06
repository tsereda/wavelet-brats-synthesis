import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np


class WaveletWrapper(nn.Module):
    """
    Wrapper that adds wavelet transform processing to any base model.
    The base model processes data in wavelet domain instead of spatial domain.
    """
    
    def __init__(self, base_model, wavelet_name='haar', in_channels=8, out_channels=4):
        super().__init__()
        self.wavelet_name = wavelet_name
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Validate wavelet
        try:
            self.wavelet = pywt.Wavelet(wavelet_name)
        except:
            raise ValueError(f"Invalid wavelet: {wavelet_name}")
        
        print(f"\n>>> Wavelet Wrapper Configuration:")
        print(f"    Wavelet: {self.wavelet.name} (family: {self.wavelet.family_name})")
        print(f"    Filter lengths - Low: {len(self.wavelet.dec_lo)}, High: {len(self.wavelet.dec_hi)}")
        
        # Pre-compute and register wavelet filter coefficients as buffers
        self._init_wavelet_filters()
        
        # Modify base model's input/output channels for wavelet space
        # Input: each channel becomes 4 subbands (LL, LH, HL, HH)
        # Output: we need 4 subbands per output channel
        wavelet_in_channels = in_channels * 4
        wavelet_out_channels = out_channels * 4
        
        print(f"    Input transform: {in_channels} channels -> {wavelet_in_channels} wavelet channels")
        print(f"    Output transform: {wavelet_out_channels} wavelet channels -> {out_channels} channels")
        print(f"    Spatial reduction: H×W -> (H/2)×(W/2) in wavelet domain")
        
        # Store the base model with modified channels
        self.base_model = self._modify_model_channels(base_model, wavelet_in_channels, wavelet_out_channels)
        
        print(f"    Base model adapted for wavelet processing")
    
    def _modify_model_channels(self, model, new_in_channels, new_out_channels):
        """
        Modify the first and last layers of the model to handle wavelet channels
        """
        # Find and replace first conv layer (input layer)
        for name, module in model.named_children():
            if isinstance(module, nn.Conv2d):
                # Found first conv layer, replace it
                old_conv = module
                new_conv = nn.Conv2d(
                    new_in_channels, 
                    old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    dilation=old_conv.dilation,
                    groups=1,  # Reset groups for new channel count
                    bias=old_conv.bias is not None
                )
                setattr(model, name, new_conv)
                print(f"    Modified input layer: {name} ({old_conv.in_channels} -> {new_in_channels} channels)")
                break
            elif len(list(module.children())) > 0:
                # Recursively search in submodules
                self._modify_first_conv(module, new_in_channels)
                break
        
        # Find and replace last conv layer (output layer)
        self._modify_last_conv(model, new_out_channels)
        
        return model
    
    def _modify_first_conv(self, module, new_in_channels):
        """Recursively find and modify the first conv layer"""
        for name, submodule in module.named_children():
            if isinstance(submodule, nn.Conv2d):
                old_conv = submodule
                new_conv = nn.Conv2d(
                    new_in_channels,
                    old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    dilation=old_conv.dilation,
                    groups=1,
                    bias=old_conv.bias is not None
                )
                setattr(module, name, new_conv)
                return True
            elif len(list(submodule.children())) > 0:
                if self._modify_first_conv(submodule, new_in_channels):
                    return True
        return False
    
    def _modify_last_conv(self, module, new_out_channels):
        """Find and modify the last conv layer (output layer)"""
        last_conv_name = None
        last_conv_parent = None
        
        def find_last_conv(module, parent=None, name=None):
            nonlocal last_conv_name, last_conv_parent
            for child_name, child in module.named_children():
                if isinstance(child, nn.Conv2d):
                    last_conv_name = child_name
                    last_conv_parent = module
                elif len(list(child.children())) > 0:
                    find_last_conv(child, module, child_name)
        
        find_last_conv(module)
        
        if last_conv_parent and last_conv_name:
            old_conv = getattr(last_conv_parent, last_conv_name)
            new_conv = nn.Conv2d(
                old_conv.in_channels,
                new_out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                dilation=old_conv.dilation,
                groups=1,
                bias=old_conv.bias is not None
            )
            setattr(last_conv_parent, last_conv_name, new_conv)
            print(f"    Modified output layer: {last_conv_name} ({old_conv.out_channels} -> {new_out_channels} channels)")
    
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
            print(f"\n    DWT Transform: Input {x.shape} -> Wavelet coeffs {result.shape}")
            print(f"    Each channel split into 4 subbands (LL, LH, HL, HH)")
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
            print(f"    IDWT Transform: Wavelet coeffs {coeffs.shape} -> Output {result.shape}")
            self._idwt_dims_printed = True
        
        return result
    
    def forward(self, x):
        """
        Forward pass with wavelet transform
        1. Transform input to wavelet domain
        2. Process with base model in wavelet domain
        3. Transform output back to spatial domain
        """
        target_shape = (x.shape[2], x.shape[3])
        
        # Transform to wavelet space
        wavelet_input = self.dwt2d_batch(x)
        
        # Process in wavelet space with base model
        wavelet_output = self.base_model(wavelet_input)
        
        # Transform back to image space
        output = self.idwt2d_batch(wavelet_output, target_shape)
        
        return output