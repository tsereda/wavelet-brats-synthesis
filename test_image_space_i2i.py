#!/usr/bin/env python3
"""Unit test for i2i mode in IMAGE SPACE (no wavelets)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app'))

import torch as th
from guided_diffusion.gaussian_diffusion import GaussianDiffusion, ModelMeanType, ModelVarType, LossType, get_named_beta_schedule
from guided_diffusion.unet import UNetModel

device = 'cuda' if th.cuda.is_available() else 'cpu'
print(f"Device: {device}\n")

# Create diffusion with i2i mode, NO WAVELETS (image space)
betas = get_named_beta_schedule('linear', 100, 'direct')
diffusion = GaussianDiffusion(
    betas=betas,
    model_mean_type=ModelMeanType.EPSILON,
    model_var_type=ModelVarType.FIXED_SMALL,
    loss_type=LossType.MSE,
    mode='i2i',
    use_freq=False,  # IMAGE SPACE
    wavelet=None
)

# Create model expecting 4 channels (1 noise + 3 conditioning)
model = UNetModel(
    image_size=224,
    in_channels=4,  # 1 target + 3 conditions
    model_channels=128,
    out_channels=1,  # 1 target channel
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

print(f"✓ Diffusion mode: {diffusion.mode}")
print(f"✓ Diffusion use_freq: {diffusion.use_freq}")
print(f"✓ Model in_channels: {model.in_channels}")
print(f"✓ Model out_channels: {model.out_channels}")

# Simulate validation data (IMAGE SPACE - no wavelets)
x = th.randn(1, 1, 224, 224, 160, device=device)  # 1 channel (noise)
cond = th.randn(1, 3, 224, 224, 160, device=device)  # 3 channels (conditions)
t = th.tensor([50], device=device)

print(f"\nTest inputs (IMAGE SPACE):")
print(f"  x (noise): {x.shape}")
print(f"  cond:      {cond.shape}")
print(f"  Expected after concat: [1, 4, 224, 224, 160]\n")

try:
    with th.no_grad():
        out = diffusion.p_mean_variance(model, x, t, cond=cond, model_kwargs={})
    print(f"✅ SUCCESS!")
    print(f"   Output shape: {out['mean'].shape}")
    print(f"   Channel concatenation worked correctly in IMAGE SPACE!")
except ValueError as e:
    print(f"❌ FAILED with ValueError: {e}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
