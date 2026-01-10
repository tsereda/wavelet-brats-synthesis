#!/usr/bin/env python3
"""
Unified test suite for Fast-cWDM BraTS Synthesis project.

Usage:
    python test.py          # Show available tests
    python test.py shapes   # Channel configuration math (~5s)
    python test.py pipeline # Full GPU pipeline with synthetic data (~2min)
    python test.py all      # Run all tests
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app'))

import argparse
import torch as th
from guided_diffusion import gaussian_diffusion as gd
from guided_diffusion.unet import UNetModel
from guided_diffusion.wunet import WavUNetModel


def calculate_channels(model_mode, wavelet):
    """Calculate in/out channels based on model mode and wavelet."""
    use_freq = wavelet is not None and wavelet != 'null'
    
    if model_mode == 'direct':
        if use_freq:
            return 24, 8  # 3 conditions × 8 subbands → 8 subbands
        else:
            return 3, 1   # 3 conditions → 1 target
    else:
        if use_freq:
            return 32, 8  # (8 + 24) subbands → 8 subbands
        else:
            return 4, 1   # (1 + 3) channels → 1 channel


def test_shapes():
    """Channel configuration math verification."""
    print("\n" + "="*80)
    print("SHAPE TESTS - Channel Configuration Math")
    print("="*80)
    
    test_cases = [
        ('direct', 'null', False, 3, 1, 'Direct regression, image space'),
        ('direct', 'haar', True, 24, 8, 'Direct regression, Haar wavelet'),
        ('direct', 'db2', True, 24, 8, 'Direct regression, db2 wavelet'),
        ('diffusion_fast', 'null', False, 4, 1, 'Fast diffusion, image space'),
        ('diffusion_fast', 'haar', True, 32, 8, 'Fast diffusion, Haar [SUBMISSION]'),
        ('diffusion_fast', 'db2', True, 32, 8, 'Fast diffusion, db2'),
        ('diffusion_standard', 'null', False, 4, 1, 'Standard diffusion, image space'),
        ('diffusion_standard', 'haar', True, 32, 8, 'Standard diffusion, Haar'),
        ('diffusion_standard', 'db2', True, 32, 8, 'Standard diffusion, db2'),
    ]
    
    all_passed = True
    for model_mode, wavelet, use_freq, expected_in, expected_out, desc in test_cases:
        in_ch, out_ch = calculate_channels(model_mode, wavelet)
        status = "✅" if (in_ch == expected_in and out_ch == expected_out) else "❌"
        if status == "❌":
            all_passed = False
        
        print(f"\n[{desc}]")
        print(f"  {status} in_channels: {in_ch} (expected {expected_in})")
        print(f"  {status} out_channels: {out_ch} (expected {expected_out})")
    
    print("\n" + "="*80)
    print("✨ ALL SHAPE TESTS PASSED!" if all_passed else "⚠️  SOME TESTS FAILED!")
    print("="*80)
    return all_passed


def test_pipeline():
    """Full pipeline test with GPU."""
    print("\n" + "="*80)
    print("PIPELINE TESTS - Full Training Pipeline")
    print("="*80)
    
    configs = [
        ('direct', 'nowavelet'), ('direct', 'haar'), ('direct', 'db2'),
        ('diffusion_fast', 'nowavelet'), ('diffusion_fast', 'haar'), ('diffusion_fast', 'db2'),
        ('diffusion_standard', 'nowavelet'), ('diffusion_standard', 'haar'), ('diffusion_standard', 'db2'),
    ]
    
    results = []
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    
    for model_mode, wavelet in configs:
        print(f"\n{'='*80}\nTesting: {model_mode} + {wavelet}\n{'='*80}")
        
        use_freq = wavelet != 'nowavelet'
        is_direct = model_mode == 'direct'
        
        # Direct regression has different channel counts (no noisy input)
        if is_direct:
            in_channels, out_channels = (24, 8) if use_freq else (3, 1)
        else:
            in_channels, out_channels = (32, 8) if use_freq else (4, 1)
        
        image_size = 112 if use_freq else 224
        resample_2d = not use_freq  # FALSE for wavelets (3D DWT/IDWT), TRUE for image space
        # When diffusion.py transforms data to wavelet space, model operates on wavelet coefficients
        # but doesn't need WavUNet's internal wavelet layers - use regular UNet instead
        ModelClass = UNetModel
        
        try:
            model_kwargs = {
                'image_size': image_size, 'in_channels': in_channels, 'model_channels': 32,
                'out_channels': out_channels, 'num_res_blocks': 1, 'attention_resolutions': [image_size // 16],
                'dropout': 0.0, 'channel_mult': (1, 2, 4), 'num_classes': None, 'use_checkpoint': False,
                'num_heads': 1, 'num_head_channels': -1, 'num_heads_upsample': -1,
                'use_scale_shift_norm': False, 'resblock_updown': False, 'use_new_attention_order': False,
                'dims': 3, 'num_groups': 8, 'bottleneck_attention': False, 'resample_2d': resample_2d, 'additive_skips': False,
            }
            model = ModelClass(**model_kwargs)
            model.to(device)  # Move model to device (modifies in-place for modules)
            
            schedule = 'sampled' if model_mode == 'diffusion_fast' else 'direct'
            diffusion = gd.GaussianDiffusion(
                betas=gd.get_named_beta_schedule('linear', 100, schedule),
                model_mean_type=gd.ModelMeanType.EPSILON, model_var_type=gd.ModelVarType.FIXED_SMALL,
                loss_type=gd.LossType.MSE, rescale_timesteps=False, use_freq=use_freq,
                wavelet=wavelet if use_freq else None, mse_loss_weight=1.0
            )
            
            # Move DWT/IDWT to device if using wavelets
            if hasattr(diffusion, 'dwt') and diffusion.dwt:
                diffusion.dwt.to(device)
            if hasattr(diffusion, 'idwt') and diffusion.idwt:
                diffusion.idwt.to(device)
            
            batch = {k: th.randn(1, 1, 224, 224, 160).to(device) for k in ['t1n', 't1c', 't2w', 't2f']}
            t = th.randint(0, 100, (1,)).to(device)
            
            input_shape = [None]
            def capture(m, inp): 
                input_shape[0] = list(inp[0].shape)
            
            with th.no_grad():
                handle = model.register_forward_pre_hook(capture)
                if is_direct:
                    # Direct regression uses DirectRegressionLoop.forward_backward logic
                    _ = diffusion.direct_regression_loss(model, batch, contr='t2f')
                else:
                    _ = diffusion.training_losses(model, batch, t, mode='i2i', contr='t2f')
                handle.remove()
            
            # Expected shapes differ for direct (no noisy input concatenation)
            if is_direct:
                expected = [1, 24, 112, 112, 80] if use_freq else [1, 3, 224, 224, 160]
            else:
                expected = [1, 32, 112, 112, 80] if use_freq else [1, 4, 224, 224, 160]
            passed = input_shape[0] == expected
            
            print(f"✅ {'PASS' if passed else 'FAIL'}: shape {input_shape[0]} (expected {expected})")
            results.append((model_mode, wavelet, passed))
            
        except Exception as e:
            import traceback
            print(f"❌ FAIL: {e}")
            print(f"Traceback:")
            traceback.print_exc()
            results.append((model_mode, wavelet, False))
    
    print(f"\n{'='*80}\nSUMMARY\n{'='*80}")
    for mode, wav, passed in results:
        print(f"{'✅ PASS' if passed else '❌ FAIL'}: {mode} + {wav}")
    
    all_passed = all(p for _, _, p in results)
    print(f"\n{'✨ ALL TESTS PASSED!' if all_passed else '⚠️  SOME TESTS FAILED!'}\n{'='*80}\n")
    return all_passed


def show_help():
    print(__doc__)
    print("Available Tests:")
    print("  shapes   - Channel math (~5s)")
    print("  pipeline - Full GPU pipeline (~2min)")
    print("  all      - Run all tests\n")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('test', nargs='?', choices=['shapes', 'pipeline', 'all'], help='Test to run')
    args = parser.parse_args()
    
    if not args.test:
        show_help()
        return 0
    
    results = []
    if args.test in ['shapes', 'all']:
        results.append(test_shapes())
    if args.test in ['pipeline', 'all']:
        results.append(test_pipeline())
    
    return 0 if all(results) else 1


if __name__ == '__main__':
    exit(main())
