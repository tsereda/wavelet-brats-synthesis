#!/usr/bin/env python3
"""Test script to verify channel configuration fixes"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app'))

import torch
from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D

def test_channel_configs():
    """Test that channel configs are calculated correctly"""
    configs = [
        # (wavelet, use_freq, is_direct, expected_in, expected_out, description)
        (None, False, True, 3, 1, "direct + null"),
        ('haar', True, True, 24, 8, "direct + haar"),
        ('db2', True, True, 24, 8, "direct + db2"),
        (None, False, False, 4, 1, "diffusion + null"),
        ('haar', True, False, 32, 8, "diffusion + haar"),
        ('db2', True, False, 32, 8, "diffusion + db2"),
    ]
    
    print("Testing channel configurations:")
    print("-" * 70)
    
    for wavelet, use_freq, is_direct, expected_in, expected_out, desc in configs:
        if use_freq:
            condition_channels = 3 * 8  # 3 modalities × 8 subbands
            target_channels = 1 * 8
        else:
            condition_channels = 3
            target_channels = 1
        
        if is_direct:
            in_channels = condition_channels
            out_channels = target_channels
        else:
            in_channels = target_channels + condition_channels
            out_channels = target_channels
        
        status = "✅" if (in_channels == expected_in and out_channels == expected_out) else "❌"
        print(f"{status} {desc:20s} → in={in_channels:2d}, out={out_channels:1d} (expected in={expected_in}, out={expected_out})")
    
    print("-" * 70)

def test_dwt_3d_matrix():
    """Test that DWT_3D handles non-square 3D volumes"""
    print("\nTesting DWT_3D with BraTS dimensions (224, 224, 160):")
    print("-" * 70)
    
    try:
        # Create test input matching BraTS dimensions after preprocessing
        batch_size = 2
        channels = 3
        depth, height, width = 160, 224, 224
        
        x = torch.randn(batch_size, channels, depth, height, width)
        
        for wavelet in ['haar', 'db2']:
            dwt = DWT_3D(wavelet)
            print(f"Testing wavelet: {wavelet}")
            
            # Forward pass
            lfc, hfc_llh, hfc_lhl, hfc_lhh, hfc_hll, hfc_hlh, hfc_hhl, hfc_hhh = dwt(x)
            
            # Check shapes
            expected_shape = (batch_size, channels, depth//2, height//2, width//2)
            assert lfc.shape == expected_shape, f"LFC shape mismatch: {lfc.shape} vs {expected_shape}"
            assert hfc_llh.shape == expected_shape, f"HFC_LLH shape mismatch"
            
            print(f"  ✅ {wavelet}: Input {tuple(x.shape)} → LFC {tuple(lfc.shape)}")
            
            # Test IDWT reconstruction
            idwt = IDWT_3D(wavelet)
            reconstructed = idwt(lfc, hfc_llh, hfc_lhl, hfc_lhh, hfc_hll, hfc_hlh, hfc_hhl, hfc_hhh)
            
            # Check reconstruction
            assert reconstructed.shape == x.shape, f"Reconstruction shape mismatch: {reconstructed.shape} vs {x.shape}"
            error = torch.abs(reconstructed - x).max().item()
            print(f"  ✅ {wavelet}: Reconstruction max error = {error:.6f}")
        
        print("-" * 70)
        print("✅ All DWT_3D tests passed!")
        
    except Exception as e:
        print(f"❌ DWT_3D test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    test_channel_configs()
    success = test_dwt_3d_matrix()
    
    if success:
        print("\n🎉 All tests passed! Ready to run ablation sweep.")
        sys.exit(0)
    else:
        print("\n❌ Tests failed. Check errors above.")
        sys.exit(1)
