#!/usr/bin/env python3
"""Unit test for i2i mode channel concatenation."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app'))

import torch as th
from guided_diffusion.gaussian_diffusion import GaussianDiffusion, ModelMeanType, ModelVarType, LossType, get_named_beta_schedule
from guided_diffusion.unet import UNetModel

device = 'cuda' if th.cuda.is_available() else 'cpu'
print(f"Device: {device}\n")

# Create diffusion with i2i mode
betas = get_named_beta_schedule('linear', 100, 'direct')
diffusion = GaussianDiffusion(
    betas=betas,
    model_mean_type=ModelMeanType.EPSILON,
    model_var_type=ModelVarType.FIXED_SMALL,
    loss_type=LossType.MSE,
    mode='i2i',
    use_freq=True,
    wavelet='haar'
)

print(f"✓ Diffusion mode: {diffusion.mode}")
print(f"\nCreating UNetModel...")

# Create model expecting 32 channels (8 noise + 24 conditioning)
# Use default num_channels=128 which is divisible by num_groups=32
try:
    model = UNetModel(
        image_size=112,
        in_channels=32,
        model_channels=128,  # Default value, divisible by num_groups=32
        out_channels=8,
        num_res_blocks=1,
        attention_resolutions=tuple(),
        dropout=0.0,
        channel_mult=(1, 2),
        dims=3,
        num_heads=4
    )
    print(f"Model created: {type(model)}")
    model = model.to(device).eval()
    print(f"✓ Model moved to {device}")
except Exception as e:
    print(f"❌ Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"✓ Model in_channels: {model.in_channels}")
print(f"✓ Model out_channels: {model.out_channels}")

# Simulate validation data
x = th.randn(1, 8, 112, 112, 80, device=device)      # 8 wavelet channels (noise)
cond = th.randn(1, 24, 112, 112, 80, device=device)  # 24 channels (3 modalities × 8)
t = th.tensor([50], device=device)

print(f"\nTest inputs:")
print(f"  x (noise): {x.shape}")
print(f"  cond:      {cond.shape}")
print(f"  Expected after concat: [1, 32, 112, 112, 80]\n")

try:
    with th.no_grad():
        out = diffusion.p_mean_variance(model, x, t, cond=cond, model_kwargs={})
    print(f"✅ SUCCESS!")
    print(f"   Output shape: {out['mean'].shape}")
    print(f"   Channel concatenation worked correctly!")
except ValueError as e:
    print(f"❌ FAILED with ValueError: {e}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
