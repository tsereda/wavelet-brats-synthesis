#!/usr/bin/env python3
"""
Enhanced Checkpoint Inspector - Comprehensive model architecture analysis
"""

import torch
import os
import sys
from collections import defaultdict
import re

def analyze_architecture(state_dict):
    """Extract comprehensive architecture information from state dict"""
    
    # Architecture detection patterns
    patterns = {
        'input_conv': r'input_blocks\.0\.[^.]*\.weight',
        'input_blocks': r'input_blocks\.(\d+)\..*in_layers\.2\.weight',
        'middle_blocks': r'middle_block\..*in_layers\.2\.weight', 
        'output_blocks': r'output_blocks\.(\d+)\..*in_layers\.2\.weight',
        'attention': r'.*attn.*',
        'time_embed': r'time_embed\..*',
        'class_embed': r'label_emb\..*'
    }
    
    info = {
        'total_params': len(state_dict),
        'base_channels': None,
        'channel_mult': [],
        'input_channels': None,
        'attention_blocks': [],
        'has_class_conditioning': False,
        'time_embed_dim': None,
        'block_structure': defaultdict(list)
    }
    
    # Analyze each parameter
    for key, tensor in state_dict.items():
        # Skip non-conv weights
        if not key.endswith('.weight') or len(tensor.shape) not in [4, 5]:
            continue
            
        # Input conv layer (first layer)
        if re.match(patterns['input_conv'], key):
            info['input_channels'] = tensor.shape[1]
            info['base_channels'] = tensor.shape[0]
            
        # Input blocks
        elif match := re.match(patterns['input_blocks'], key):
            block_num = int(match.group(1))
            channels = tensor.shape[0]
            info['block_structure']['input'].append((block_num, channels))
            
        # Output blocks  
        elif match := re.match(patterns['output_blocks'], key):
            block_num = int(match.group(1))
            channels = tensor.shape[0]
            info['block_structure']['output'].append((block_num, channels))
            
        # Middle block
        elif re.match(patterns['middle_blocks'], key):
            channels = tensor.shape[0]
            info['block_structure']['middle'].append(channels)
    
    # Check for attention and conditioning
    for key in state_dict.keys():
        if 'attn' in key:
            block_match = re.search(r'(input_blocks|output_blocks)\.(\d+)', key)
            if block_match:
                block_type, block_num = block_match.groups()
                info['attention_blocks'].append(f"{block_type}_{block_num}")
                
        if 'label_emb' in key or 'class_embed' in key:
            info['has_class_conditioning'] = True
            
        if 'time_embed' in key and key.endswith('.weight'):
            info['time_embed_dim'] = state_dict[key].shape[0]
    
    # Calculate channel multipliers
    if info['base_channels']:
        all_channels = []
        for block_list in info['block_structure'].values():
            if isinstance(block_list[0], tuple):
                all_channels.extend([ch for _, ch in block_list])
            else:
                all_channels.extend(block_list)
                
        unique_channels = sorted(set(all_channels))
        base = info['base_channels']
        info['channel_mult'] = [ch // base for ch in unique_channels if ch % base == 0]
        info['channel_mult'] = sorted(set(info['channel_mult']))
    
    return info

def print_analysis(info, checkpoint_path):
    """Print comprehensive analysis results"""
    
    print(f"🔍 {os.path.basename(checkpoint_path)}")
    print("=" * 60)
    
    # Basic info
    print(f"📊 Total parameters: {info['total_params']:,}")
    print(f"🎯 Input channels: {info['input_channels']}")
    print(f"🏗️  Base channels: {info['base_channels']}")
    print(f"📈 Channel multipliers: {info['channel_mult']}")
    print(f"⏰ Time embed dim: {info['time_embed_dim']}")
    print(f"🏷️  Class conditioning: {'Yes' if info['has_class_conditioning'] else 'No'}")
    
    # Block structure
    if info['block_structure']['input']:
        input_blocks = sorted(info['block_structure']['input'])
        print(f"\n📥 Input blocks ({len(input_blocks)}):")
        for i, (block_num, channels) in enumerate(input_blocks[:6]):  # Show first 6
            mult = channels // info['base_channels'] if info['base_channels'] else '?'
            print(f"   Block {block_num}: {channels} (×{mult})")
        if len(input_blocks) > 6:
            print(f"   ... and {len(input_blocks) - 6} more")
    
    if info['block_structure']['middle']:
        mid_ch = info['block_structure']['middle'][0]
        mult = mid_ch // info['base_channels'] if info['base_channels'] else '?'
        print(f"\n🎯 Middle block: {mid_ch} (×{mult})")
    
    # Attention blocks
    if info['attention_blocks']:
        att_blocks = sorted(set(info['attention_blocks']))[:5]  # Show first 5
        print(f"\n🧠 Attention blocks: {', '.join(att_blocks)}")
        if len(info['attention_blocks']) > 5:
            print(f"   ... and {len(info['attention_blocks']) - 5} more")
    
    # Configuration output
    print("\n" + "=" * 60)
    print("🎯 CONFIGURATION:")
    print("=" * 60)
    print(f"num_channels = {info['base_channels']}")
    print(f"channel_mult = \"{','.join(map(str, info['channel_mult']))}\"")
    print(f"num_res_blocks = 2  # typical")
    print(f"attention_resolutions = \"16,8\"  # typical")
    if info['has_class_conditioning']:
        print(f"num_classes = 1000  # adjust as needed")
    print("=" * 60)

def inspect_checkpoint(checkpoint_path):
    """Main inspection function"""
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ File not found: {checkpoint_path}")
        return False
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract state dict
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        # Analyze and print results
        info = analyze_architecture(state_dict)
        print_analysis(info, checkpoint_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main entry point"""
    
    # Get checkpoint path
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        # Look for common checkpoint files
        candidates = [
            "brats_t1c_060000_sampled_100.pt",
            "checkpoints/brats_t1c_060000_sampled_100.pt", 
            "../checkpoints/brats_t1c_060000_sampled_100.pt"
        ]
        
        checkpoint_path = next((p for p in candidates if os.path.exists(p)), None)
        
        if not checkpoint_path:
            print("Usage: python inspect_checkpoint.py <checkpoint.pt>")
            print("\nOr place checkpoint in current directory as:")
            for candidate in candidates:
                print(f"  {candidate}")
            return
    
    inspect_checkpoint(checkpoint_path)

if __name__ == "__main__":
    main()