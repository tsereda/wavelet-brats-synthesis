#!/usr/bin/env python3
"""
Direct Checkpoint Inspector - loads the actual checkpoint and tells us the exact architecture
"""

import torch
import os
import sys

def inspect_checkpoint(checkpoint_path):
    """Direct inspection of checkpoint file"""
    print(f"🔍 INSPECTING: {checkpoint_path}")
    print("="*60)
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Get state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("✅ Found 'state_dict' key")
        else:
            state_dict = checkpoint
            print("✅ Using checkpoint as state_dict")
            
        print(f"Total parameters: {len(state_dict)}")
        
        # Find the first conv layer to get base channels
        first_conv = None
        for key in state_dict.keys():
            if key.startswith('input_blocks.0') and 'weight' in key and len(state_dict[key].shape) == 5:
                first_conv = (key, state_dict[key].shape)
                break
                
        if first_conv:
            key, shape = first_conv
            in_channels = shape[1]
            out_channels = shape[0] 
            print(f"\n🎯 FIRST CONV LAYER: {key}")
            print(f"   Shape: {shape}")
            print(f"   Input channels: {in_channels}")
            print(f"   Output channels (base model channels): {out_channels}")
        
        # Find all conv layer output channels to determine channel progression
        conv_channels = []
        input_block_channels = []
        
        for key in state_dict.keys():
            if 'in_layers.2.weight' in key and len(state_dict[key].shape) == 5:
                out_ch = state_dict[key].shape[0]
                conv_channels.append(out_ch)
                if 'input_blocks' in key:
                    block_num = int(key.split('.')[1])
                    input_block_channels.append((block_num, out_ch))
        
        # Remove duplicates and sort
        unique_channels = sorted(set(conv_channels))
        input_block_channels.sort()
        
        print(f"\n📊 CHANNEL ANALYSIS:")
        print(f"All conv output channels: {unique_channels}")
        
        if first_conv:
            base = first_conv[1][0]  # Base channels from first conv
            multipliers = [ch // base for ch in unique_channels if ch % base == 0]
            multipliers = sorted(set(multipliers))
            
            print(f"\n🎯 DETERMINED ARCHITECTURE:")
            print(f"   Base channels: {base}")
            print(f"   Channel multipliers: {multipliers}")
            print(f"   Channel progression: {[base * m for m in multipliers]}")
            
            # Show input block progression
            print(f"\n📋 INPUT BLOCK PROGRESSION:")
            for block_num, channels in input_block_channels[:8]:  # First 8 blocks
                mult = channels // base if channels % base == 0 else "?"
                print(f"   Block {block_num}: {channels} channels (×{mult})")
                
            return base, ','.join(map(str, multipliers))
        else:
            print("❌ Could not find first conv layer")
            return None, None
            
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None, None

def main():
    # Look for checkpoint files in likely locations
    possible_paths = [
        "../checkpoints/brats_t1c_060000_sampled_100.pt",
        "./checkpoints/brats_t1c_060000_sampled_100.pt", 
        "checkpoints/brats_t1c_060000_sampled_100.pt",
        "brats_t1c_060000_sampled_100.pt"
    ]
    
    checkpoint_path = None
    for path in possible_paths:
        if os.path.exists(path):
            checkpoint_path = path
            break
            
    if not checkpoint_path:
        print("❌ Checkpoint not found. Please provide the path:")
        print("Usage: python inspect_checkpoint.py [path_to_checkpoint.pt]")
        if len(sys.argv) > 1:
            checkpoint_path = sys.argv[1]
        else:
            return
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ File not found: {checkpoint_path}")
        return
        
    base_channels, channel_mult = inspect_checkpoint(checkpoint_path)
    
    if base_channels:
        print(f"\n" + "="*60)
        print("🎯 COPY THIS CONFIGURATION:")
        print("="*60)
        print(f"args.num_channels = {base_channels}")
        print(f'args.channel_mult = "{channel_mult}"')
        print("="*60)

if __name__ == "__main__":
    main()