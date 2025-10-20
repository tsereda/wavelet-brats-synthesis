#!/usr/bin/env python3
"""
Debug script to test wavelet model before full training
Tests different wavelets to ensure they work correctly
"""

import torch
import pywt
import numpy as np
from models.wavelet_diffusion import WaveletDiffusion


def test_wavelet(wavelet_name, batch_size=2, img_size=256):
    """Test a specific wavelet"""
    print(f"\n{'='*60}")
    print(f"Testing wavelet: {wavelet_name}")
    print(f"{'='*60}")
    
    try:
        # Create model
        model = WaveletDiffusion(
            wavelet_name=wavelet_name,
            in_channels=8,
            out_channels=4,
            timesteps=100
        )
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, 8, img_size, img_size)
        
        print(f"Input shape: {dummy_input.shape}")
        
        # Forward pass
        output = model(dummy_input)
        
        print(f"Output shape: {output.shape}")
        print(f"✓ Forward pass successful!")
        
        # Check shapes match
        assert output.shape == (batch_size, 4, img_size, img_size), \
            f"Shape mismatch! Expected {(batch_size, 4, img_size, img_size)}, got {output.shape}"
        
        # Check for NaN/Inf
        if torch.isnan(output).any():
            print(f"✗ WARNING: Output contains NaN values!")
            return False
        if torch.isinf(output).any():
            print(f"✗ WARNING: Output contains Inf values!")
            return False
        
        print(f"✓ Output is valid (no NaN/Inf)")
        print(f"✓ Test PASSED for {wavelet_name}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test FAILED for {wavelet_name}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dwt_idwt_roundtrip(wavelet_name):
    """Test that DWT -> IDWT reconstruction works"""
    print(f"\nTesting DWT/IDWT roundtrip for {wavelet_name}...")
    
    try:
        model = WaveletDiffusion(wavelet_name=wavelet_name)
        
        # Create test image
        test_img = torch.randn(1, 4, 256, 256)
        
        # DWT
        coeffs = model.dwt2d_batch(test_img)
        print(f"  DWT output shape: {coeffs.shape}")
        
        # IDWT
        reconstructed = model.idwt2d_batch(coeffs, target_shape=(256, 256))
        print(f"  IDWT output shape: {reconstructed.shape}")
        
        # Check reconstruction error
        error = torch.mean((test_img - reconstructed) ** 2).item()
        print(f"  Reconstruction MSE: {error:.6f}")
        
        if error < 1e-4:
            print(f"  ✓ Roundtrip successful (low error)")
            return True
        else:
            print(f"  ✗ Roundtrip has high error!")
            return False
            
    except Exception as e:
        print(f"  ✗ Roundtrip test failed: {e}")
        return False


def main():
    print("="*60)
    print("WAVELET MODEL DEBUG SCRIPT")
    print("="*60)
    
    # Test wavelets from your sweep config
    test_wavelets = [
        'haar',
        'db1', 'db2', 'db3', 'db4',
        'sym2', 'sym3',
        'coif1', 'coif2',
        'bior1.3', 'bior2.2',
        'rbio1.3',
    ]
    
    results = {}
    
    for wavelet in test_wavelets:
        # Test forward pass
        forward_ok = test_wavelet(wavelet)
        
        # Test DWT/IDWT roundtrip
        roundtrip_ok = test_dwt_idwt_roundtrip(wavelet)
        
        results[wavelet] = forward_ok and roundtrip_ok
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = [w for w, ok in results.items() if ok]
    failed = [w for w, ok in results.items() if not ok]
    
    print(f"\n✓ PASSED ({len(passed)}/{len(test_wavelets)}):")
    for w in passed:
        print(f"  - {w}")
    
    if failed:
        print(f"\n✗ FAILED ({len(failed)}/{len(test_wavelets)}):")
        for w in failed:
            print(f"  - {w}")
    
    print("\n" + "="*60)
    
    if not failed:
        print("All wavelets passed! Ready to train.")
    else:
        print("Some wavelets failed. Check the errors above.")
        print("Consider removing failed wavelets from sweep.yml")


if __name__ == '__main__':
    main()