#!/usr/bin/env python3
"""Minimal unit test for i2i channel concatenation - no OOM."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app'))

import torch as th
from guided_diffusion.gaussian_diffusion import GaussianDiffusion, ModelMeanType, ModelVarType, LossType, get_named_beta_schedule
from guided_diffusion.unet import UNetModel

device = 'cpu'  # Use CPU to avoid OOM

print("="*60)
print("MINIMAL I2I CHANNEL CONCATENATION TEST (CPU)")
print("="*60)

# Test both wavelet and image space
for use_freq, wavelet, spatial_size, in_ch, desc in [
    (True, 'haar', (16, 16, 16), 32, "Wavelet space (8+24=32)"),
    (False, None, (32, 32, 32), 4, "Image space (1+3=4)")
]:
    print(f"\n--- Testing {desc} ---")
    
    # Create diffusion
    betas = get_named_beta_schedule('linear', 100, 'direct')  # 100 steps
    diffusion = GaussianDiffusion(
        betas=betas,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
        mode='i2i',
        use_freq=use_freq,
        wavelet=wavelet
    )
    
    # Create tiny model
    model = UNetModel(
        image_size=spatial_size[0],
        in_channels=in_ch,
        model_channels=16,  # Tiny!
        out_channels=8 if use_freq else 1,
        num_res_blocks=1,
        attention_resolutions=tuple(),  # No attention
        dropout=0.0,
        channel_mult=(1,),  # Single level
        dims=3,
        num_heads=1,
        num_groups=8,  # 16 channels / 8 groups = 2 ch/group
        use_checkpoint=False  # Faster
    ).to(device).eval()
    
    # Create test data
    noise_ch = 8 if use_freq else 1
    cond_ch = 24 if use_freq else 3
    
    x = th.randn(1, noise_ch, *spatial_size, device=device)
    cond = th.randn(1, cond_ch, *spatial_size, device=device)
    t = th.tensor([5], device=device)
    
    print(f"  Model in_channels: {model.in_channels}")
    print(f"  x (noise): {x.shape}")
    print(f"  cond:      {cond.shape}")
    print(f"  Expected concat: [1, {in_ch}, {spatial_size[0]}, {spatial_size[1]}, {spatial_size[2]}]")
    
    try:
        with th.no_grad():
            out = diffusion.p_mean_variance(model, x, t, cond=cond, model_kwargs={})
        print(f"  ✅ SUCCESS! Output: {out['mean'].shape}")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("\n" + "="*60)
print("✅ ALL TESTS PASSED - Channel concatenation working!")
print("="*60)
