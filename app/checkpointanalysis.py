#!/usr/bin/env python3
"""
Raw Checkpoint Inspector - shows actual data structure
"""

import torch
import os
import sys
from collections import defaultdict

def inspect_raw(checkpoint_path):
    """Show raw checkpoint contents"""
    
    print(f"File: {checkpoint_path}")
    print(f"Size: {os.path.getsize(checkpoint_path) / (1024**3):.2f} GB")
    print()
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Show top-level structure
    if isinstance(checkpoint, dict):
        print("Top-level keys:")
        for key in checkpoint.keys():
            value = checkpoint[key]
            if isinstance(value, torch.Tensor):
                print(f"  {key}: tensor {list(value.shape)} {value.dtype}")
            elif isinstance(value, dict):
                print(f"  {key}: dict with {len(value)} keys")
            else:
                print(f"  {key}: {type(value).__name__} {value}")
        print()
        
        # Get the actual model weights
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("Using checkpoint['state_dict']")
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
            print("Using checkpoint['model']")
        else:
            state_dict = checkpoint
            print("Using checkpoint directly")
    else:
        state_dict = checkpoint
        print("Checkpoint is the state_dict directly")
    
    print(f"Total parameters: {len(state_dict)}")
    print()
    
    # Group parameters by prefix
    prefixes = defaultdict(list)
    for key in state_dict.keys():
        parts = key.split('.')
        if len(parts) >= 2:
            prefix = '.'.join(parts[:2])
        else:
            prefix = parts[0]
        prefixes[prefix].append(key)
    
    print("Parameter groups:")
    for prefix in sorted(prefixes.keys()):
        count = len(prefixes[prefix])
        first_key = prefixes[prefix][0]
        first_shape = list(state_dict[first_key].shape)
        print(f"  {prefix}: {count} params, first shape {first_shape}")
    print()
    
    # Show first few actual parameter names and shapes
    print("First 20 parameters:")
    for i, (key, tensor) in enumerate(state_dict.items()):
        if i >= 20:
            break
        print(f"  {key}: {list(tensor.shape)} {tensor.dtype}")
    
    print(f"... and {len(state_dict) - 20} more")
    print()
    
    # Find conv layers specifically
    conv_layers = []
    for key, tensor in state_dict.items():
        if key.endswith('.weight') and len(tensor.shape) in [4, 5]:  # Conv weights
            conv_layers.append((key, tensor.shape))
    
    print(f"Conv layers ({len(conv_layers)}):")
    for key, shape in conv_layers[:10]:  # First 10
        print(f"  {key}: {list(shape)}")
    if len(conv_layers) > 10:
        print(f"  ... and {len(conv_layers) - 10} more")
    print()
    
    # Channel analysis
    if conv_layers:
        channels = [shape[0] for key, shape in conv_layers]  # Output channels
        unique_channels = sorted(set(channels))
        print(f"All output channel counts: {unique_channels}")
        
        # Find first conv to get base
        first_conv = None
        for key, shape in conv_layers:
            if 'input_blocks.0' in key:
                first_conv = (key, shape)
                break
        
        if first_conv:
            base = first_conv[1][0]
            multipliers = [ch // base for ch in unique_channels if ch % base == 0]
            print(f"Base channels: {base}")
            print(f"Channel multipliers: {sorted(set(multipliers))}")
            print(f"First conv: {first_conv[0]} -> {list(first_conv[1])}")

def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        candidates = [
            "brats_t1c_060000_sampled_100.pt",
            "checkpoints/brats_t1c_060000_sampled_100.pt", 
            "../checkpoints/brats_t1c_060000_sampled_100.pt"
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if not path:
            print("Usage: python inspect.py <checkpoint.pt>")
            return
    
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
        
    try:
        inspect_raw(path)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()