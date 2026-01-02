#!/usr/bin/env python3
"""
Test script to verify the fixed wavelet wrapper works with all model types
"""

import torch
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_model_configuration(model_type, wavelet='none', img_size=256, batch_size=2):
    """Test a specific model configuration"""
    use_wavelet = wavelet != 'none'
    config_name = f"{model_type.upper()}{' + ' + wavelet + ' wavelet' if use_wavelet else ' (baseline)'}"
    print(f"\n{'='*70}")
    print(f"Testing {config_name}")
    print(f"{'='*70}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    try:
        # Import get_model function from train.py
        from train import get_model
        
        # Create model
        model = get_model(model_type, wavelet, img_size, device)
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, 8, img_size, img_size).to(device)
        print(f"\nInput shape: {dummy_input.shape}")
        print(f"Input dimensions: [batch={batch_size}, channels=8, height={img_size}, width={img_size}]")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"Output shape: {output.shape}")
        print(f"Expected shape: [{batch_size}, 4, {img_size}, {img_size}]")
        
        # Verify output shape
        assert output.shape == (batch_size, 4, img_size, img_size), \
            f"Shape mismatch! Expected {(batch_size, 4, img_size, img_size)}, got {output.shape}"
        
        # Check for NaN/Inf
        if torch.isnan(output).any():
            print("✗ WARNING: Output contains NaN values!")
            return False
        if torch.isinf(output).any():
            print("✗ WARNING: Output contains Inf values!")
            return False
        
        # Calculate model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nModel Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")
        
        # Test backward pass
        print(f"\nTesting backward pass...")
        model.train()
        dummy_input.requires_grad_(True)
        output = model(dummy_input)
        loss = output.mean()
        loss.backward()
        print(f"  Backward pass successful!")
        
        # If using wavelet, test the transforms
        if use_wavelet and hasattr(model, 'dwt2d_batch'):
            print(f"\nTesting wavelet transforms...")
            test_img = torch.randn(1, 4, img_size, img_size).to(device)
            
            # Forward transform
            coeffs = model.dwt2d_batch(test_img)
            print(f"  DWT: {test_img.shape} -> {coeffs.shape}")
            
            # Inverse transform
            reconstructed = model.idwt2d_batch(coeffs, (img_size, img_size))
            print(f"  IDWT: {coeffs.shape} -> {reconstructed.shape}")
            
            # Check reconstruction error
            error = torch.mean((test_img - reconstructed) ** 2).item()
            print(f"  Reconstruction MSE: {error:.6f}")
            
            if error < 1e-4:
                print(f"  ✓ Wavelet roundtrip successful")
            else:
                print(f"  ⚠️ Wavelet roundtrip has higher error than expected")
        
        print(f"\n✓ {config_name} test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n✗ {config_name} test FAILED!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*70)
    print("FIXED WAVELET WRAPPER TEST SUITE")
    print("Testing all architecture + wavelet combinations")
    print("="*70)
    
    # Define test configurations
    test_configs = [
        # Baselines (no wavelet)
        ('swin', 'none'),
        ('unet', 'none'),
        ('unetr', 'none'),
        
        # With wavelets
        ('swin', 'haar'),
        ('swin', 'db2'),
        ('unet', 'haar'),
        ('unet', 'db2'),
        ('unetr', 'haar'),
        ('unetr', 'sym3'),
    ]
    
    results = []
    for model_type, wavelet in test_configs:
        passed = test_model_configuration(
            model_type=model_type,
            wavelet=wavelet
        )
        results.append(((model_type, wavelet), passed))
    
    # Summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    
    print("\nConfiguration                     Status")
    print("-" * 40)
    for (model_type, wavelet), passed in results:
        if wavelet != 'none':
            config_str = f"{model_type.upper():6s} + {wavelet:6s}"
        else:
            config_str = f"{model_type.upper():6s} baseline"
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{config_str:25s} {status}")
    
    all_passed = all(p for _, p in results)
    if all_passed:
        print("\nAll configurations passed! Ready to train.")
    else:
        failed_configs = [(m,w) for (m,w), p in results if not p]
        print(f"\n⚠️ {len(failed_configs)} configuration(s) failed. Please check the errors above.")
    
    # Print usage examples
    print(f"\n{'='*70}")
    print("USAGE EXAMPLES")
    print(f"{'='*70}")
    print("\n# Train baseline models (no wavelet):")
    print("python train.py --model_type swin --wavelet none --data_dir /path/to/BraTS")
    print("python train.py --model_type unet --wavelet none --data_dir /path/to/BraTS")
    print("\n# Train with wavelet processing:")
    print("python train.py --model_type swin --wavelet haar --data_dir /path/to/BraTS")
    print("python train.py --model_type unet --wavelet db2 --data_dir /path/to/BraTS")
    print("\n# Run hyperparameter sweep:")
    print("wandb sweep sweep.yml")
    print("wandb agent <sweep_id>")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())