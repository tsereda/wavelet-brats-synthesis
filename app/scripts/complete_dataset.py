#!/usr/bin/env python3
"""
SMART: Auto-detecting medical image synthesis script
Automatically detects checkpoint architecture and configures model accordingly
"""

import argparse
import nibabel as nib
import numpy as np
import os
import torch as th
import glob
import sys
import random
import re
import time
import wandb

sys.path.append(".")

from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    model_and_diffusion_defaults, 
    create_model_and_diffusion,
    args_to_dict
)
from guided_diffusion.bratsloader import clip_and_normalize
from DWT_IDWT.DWT_IDWT_layer import IDWT_3D, DWT_3D
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
try:
    from PIL import Image
except ImportError:
    print("Warning: PIL not available. Visual sampling will be disabled.")
    Image = None

# Constants
MODALITIES = ['t1n', 't1c', 't2w', 't2f']

def detect_model_architecture(checkpoint_path):
    """Automatically detect model architecture from checkpoint"""
    print(f"🔍 Auto-detecting architecture from: {os.path.basename(checkpoint_path)}")
    
    try:
        checkpoint = th.load(checkpoint_path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # Look for first conv layer to determine base channels
        first_conv_key = None
        for key in state_dict.keys():
            if 'input_blocks.0' in key and 'weight' in key and len(state_dict[key].shape) == 5:
                first_conv_key = key
                break
                
        if not first_conv_key:
            raise ValueError("Could not find first conv layer in checkpoint")
            
        first_conv_shape = state_dict[first_conv_key].shape
        num_channels = first_conv_shape[0]  # Output channels of first conv
        
        print(f"✅ Detected num_channels: {num_channels}")
        
        # Try to infer channel multipliers by looking at layer patterns
        channel_sizes = []
        for key in state_dict.keys():
            if 'input_blocks' in key and 'in_layers.2.weight' in key:
                shape = state_dict[key].shape
                channel_sizes.append(shape[0])
                
        # Remove duplicates and sort
        unique_channels = sorted(set(channel_sizes))
        print(f"  Detected channel progression: {unique_channels}")
        
        # Infer channel multipliers
        if len(unique_channels) >= 1:
            base = unique_channels[0]
            multipliers = [ch // base for ch in unique_channels[:5]]  # Take first 5
            channel_mult = ",".join(map(str, multipliers))
        else:
            # Fallback
            channel_mult = "1,2,4,4"
            
        print(f"  Inferred channel_mult: {channel_mult}")
        
        return num_channels, channel_mult
        
    except Exception as e:
        print(f"⚠️ Auto-detection failed: {e}")
        print("  Falling back to default architecture")
        return 64, "1,2,4,4"


def create_brain_mask_from_target(target, threshold=0.01):
    """Create brain mask from target image"""
    if target.dim() > 3:
        target_for_mask = target.squeeze()
    else:
        target_for_mask = target
    
    brain_mask = (target_for_mask > threshold).float()
    
    while brain_mask.dim() < target.dim():
        brain_mask = brain_mask.unsqueeze(0)
    
    return brain_mask

class ComprehensiveMetrics:
    """Calculate metrics for synthesis evaluation with brain masking"""
    
    def __init__(self, device='cuda'):
        self.device = device
        
    def calculate_metrics(self, predicted, target, case_name=""):
        """Calculate L1 and MSE metrics with brain masking"""
        metrics = {}
        
        with th.no_grad():
            predicted = predicted.to(self.device)
            target = target.to(self.device)
            
            if predicted.dim() == 3:
                predicted = predicted.unsqueeze(0).unsqueeze(0)
            elif predicted.dim() == 4:
                predicted = predicted.unsqueeze(1)
                
            if target.dim() == 3:
                target = target.unsqueeze(0).unsqueeze(0)
            elif target.dim() == 4:
                target = target.unsqueeze(1)
            
            brain_mask = create_brain_mask_from_target(target, threshold=0.01)
            predicted_masked = predicted * brain_mask
            target_masked = target * brain_mask
            
            l1_loss = F.l1_loss(predicted_masked, target_masked).item()
            mse_loss = F.mse_loss(predicted_masked, target_masked).item()
            
            brain_volume = brain_mask.sum().item()
            total_volume = brain_mask.numel()
            brain_ratio = brain_volume / total_volume
            
            metrics = {
                'l1': l1_loss,
                'mse': mse_loss,
                'brain_volume_ratio': brain_ratio
            }
            
            if case_name:
                print(f"  {case_name}: MSE={mse_loss:.6f} (brain region = {brain_ratio:.1%})")
            
        return metrics

def load_image(file_path):
    """Load and preprocess image EXACTLY like training dataloader."""
    print(f"Loading: {file_path}")
    
    img = nib.load(file_path).get_fdata()
    print(f"  Original shape: {img.shape}")
    
    img_normalized = clip_and_normalize(img)
    
    img_tensor = th.zeros(1, 240, 240, 160)
    img_tensor[:, :, :, :155] = th.tensor(img_normalized)
    img_tensor = img_tensor[:, 8:-8, 8:-8, :]
    
    print(f"  Preprocessed shape: {img_tensor.shape}")
    return img_tensor.float()

def find_missing_modality(case_dir, evaluation_mode=False, target_modality=None):
    """Find which modality is missing (real) or select one to exclude (evaluation)."""
    case_name = os.path.basename(case_dir)
    
    if evaluation_mode:
        if target_modality:
            return target_modality
        else:
            return random.choice(MODALITIES)
    else:
        for modality in MODALITIES:
            file_path = os.path.join(case_dir, f"{case_name}-{modality}.nii.gz")
            if not os.path.exists(file_path):
                return modality
        return None

def check_complete_case(case_dir):
    """Check if case has all 4 modalities (for evaluation mode)."""
    case_name = os.path.basename(case_dir)
    
    for modality in MODALITIES:
        file_path = os.path.join(case_dir, f"{case_name}-{modality}.nii.gz")
        if not os.path.exists(file_path):
            return False
    return True

def load_available_modalities(case_dir, missing_modality, evaluation_mode=False):
    """Load all available modalities (excluding the missing/target one)."""
    case_name = os.path.basename(case_dir)
    available = [m for m in MODALITIES if m != missing_modality]
    
    modalities = {}
    for modality in available:
        file_path = os.path.join(case_dir, f"{case_name}-{modality}.nii.gz")
        if os.path.exists(file_path):
            modalities[modality] = load_image(file_path)
        elif not evaluation_mode:
            print(f"  Warning: Expected file missing: {file_path}")
    
    return modalities

def find_checkpoint(missing_modality, checkpoint_dir):
    """Find the best checkpoint for the missing modality."""
    pattern = f"brats_{missing_modality}_*.pt"
    best_files = glob.glob(os.path.join(checkpoint_dir, pattern))
    
    if best_files:
        checkpoint = best_files[0]
        print(f"Found checkpoint: {checkpoint}")
        return checkpoint
    
    pattern = f"brats_{missing_modality}_*.pt"
    regular_files = glob.glob(os.path.join(checkpoint_dir, pattern))
    
    if not regular_files:
        raise FileNotFoundError(f"No checkpoint found for {missing_modality}")
    
    def get_iteration(filename):
        parts = os.path.basename(filename).split('_')
        try:
            return int(parts[2])
        except (IndexError, ValueError):
            return 0
    
    regular_files.sort(key=get_iteration, reverse=True)
    checkpoint = regular_files[0]
    print(f"Found checkpoint: {checkpoint}")
    return checkpoint

def parse_checkpoint_info(checkpoint_path):
    """Parse checkpoint filename to get training parameters."""
    basename = os.path.basename(checkpoint_path)
    
    diffusion_steps = 1000
    sample_schedule = "direct"
    
    print(f"Parsing checkpoint: {basename}")
    
    if "_BEST100epoch_" in basename:
        parts = basename.split('_')
        if len(parts) >= 4:
            sample_schedule = parts[3]
        if len(parts) >= 5:
            try:
                diffusion_steps = int(parts[4].split('.')[0])
            except ValueError:
                pass
    elif "_sampled_" in basename:
        parts = basename.split('_')
        for i, part in enumerate(parts):
            if part == "sampled" and i + 1 < len(parts):
                try:
                    diffusion_steps = int(parts[i + 1].split('.')[0])
                    sample_schedule = "direct"
                    break
                except ValueError:
                    pass
    else:
        match = re.search(r'sampled[_-](\d+)', basename)
        if match:
            diffusion_steps = int(match.group(1))
            sample_schedule = "direct"
    
    print(f"✅ Checkpoint config: schedule={sample_schedule}, steps={diffusion_steps}")
    return sample_schedule, diffusion_steps

def create_model_args_auto(checkpoint_path, sample_schedule="direct", diffusion_steps=1000):
    """Create model arguments - AUTO-DETECTED from checkpoint."""
    class Args:
        pass
    
    args = Args()
    
    # AUTO-DETECT architecture from checkpoint
    num_channels, channel_mult = detect_model_architecture(checkpoint_path)
    
    # Model architecture - AUTO-CONFIGURED
    args.image_size = 224
    args.num_channels = num_channels
    args.num_res_blocks = 2
    args.channel_mult = channel_mult
    args.learn_sigma = False
    args.class_cond = False
    args.use_checkpoint = False
    args.attention_resolutions = ""
    args.num_heads = 1
    args.num_head_channels = -1
    args.num_heads_upsample = -1
    args.use_scale_shift_norm = False
    args.dropout = 0.0
    args.resblock_updown = True
    args.use_fp16 = False
    args.use_new_attention_order = False
    args.dims = 3
    args.num_groups = 32
    args.bottleneck_attention = False
    args.resample_2d = False
    args.additive_skips = False
    args.use_freq = False
    
    # Diffusion parameters
    args.predict_xstart = True
    args.noise_schedule = "linear"
    args.timestep_respacing = ""
    args.use_kl = False
    args.rescale_timesteps = False
    args.rescale_learned_sigmas = False
    
    # Model channels: 8 (target) + 24 (3 modalities * 8 DWT components each)
    args.in_channels = 32
    args.out_channels = 8
    
    # From checkpoint
    args.diffusion_steps = diffusion_steps
    args.sample_schedule = sample_schedule
    args.mode = 'i2i'
    
    return args

def synthesize_modality(modalities, missing_modality, checkpoint_path, device, 
                       diffusion_steps=100, wavelet=True):
    """Synthesize missing modality with AUTO-DETECTED model configuration."""
    start_time = time.time()
    
    # Parse checkpoint config
    sample_schedule, checkpoint_diffusion_steps = parse_checkpoint_info(checkpoint_path)
    
    # Create model args with AUTO-DETECTED configuration
    args = create_model_args_auto(checkpoint_path, sample_schedule, checkpoint_diffusion_steps)
    
    # Override diffusion steps from user
    args.diffusion_steps = diffusion_steps
    
    print(f"🔧 Model config: channels={args.num_channels}, mult={args.channel_mult}, steps={args.diffusion_steps}")
    
    # Create model and diffusion
    try:
        model_dict = args_to_dict(args, model_and_diffusion_defaults().keys())
        model, diffusion = create_model_and_diffusion(**model_dict)
        
        print(f"✅ Model created successfully")
        
        # Load checkpoint
        checkpoint = th.load(checkpoint_path, map_location=device)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict, strict=True)
        print(f"✅ Checkpoint loaded successfully")
        
    except Exception as e:
        print(f"❌ Model creation/loading error: {e}")
        return None, 0
    
    model.to(device)
    model.eval()
    
    # Continue with synthesis...
    try:
        # Order modalities consistently
        available_modalities = sorted(modalities.keys())
        print(f"Available modalities: {available_modalities}")
        
        # Stack modalities
        input_stack = []
        for mod in available_modalities:
            input_stack.append(modalities[mod])
        input_tensor = th.cat(input_stack, dim=0).to(device)
        
        print(f"Input tensor shape: {input_tensor.shape}")
        
        # Apply wavelet transform if enabled
        if wavelet:
            dwt = DWT_3D("haar").to(device)
            input_dwt = dwt(input_tensor)
            print(f"DWT input shape: {input_dwt.shape}")
        else:
            input_dwt = input_tensor
        
        # Prepare for diffusion
        input_batch = input_dwt.unsqueeze(0)  # Add batch dimension
        
        # Sample
        with th.no_grad():
            print(f"🎲 Sampling with {diffusion_steps} steps...")
            sample = diffusion.p_sample_loop(
                model,
                (1, 8, 224, 224, 160),  # batch_size, channels, H, W, D
                model_kwargs={"y": input_batch},
                clip_denoised=True,
                progress=True
            )
        
        # Apply inverse wavelet transform if enabled
        if wavelet:
            idwt = IDWT_3D("haar").to(device)
            output = idwt(sample.squeeze(0))
        else:
            output = sample.squeeze(0)
        
        sample_time = time.time() - start_time
        print(f"✅ Synthesis completed in {sample_time:.2f}s")
        
        return output.cpu(), sample_time
        
    except Exception as e:
        print(f"❌ Synthesis error: {e}")
        return None, 0

def process_case(case_dir, output_dir, checkpoint_dir, device, metrics_calculator, 
                evaluation_mode=False, target_modality=None, diffusion_steps=100,
                case_index=0, visual_args=None, wavelet=True):
    """Process a single case."""
    case_name = os.path.basename(case_dir)
    print(f"\n=== Processing {case_name} ===")
    
    # Check if case is complete (for evaluation mode)
    if evaluation_mode and not check_complete_case(case_dir):
        print(f"⚠️ Skipping {case_name}: incomplete case (missing modalities)")
        return False, {}
    
    # Find missing/target modality
    missing_modality = find_missing_modality(case_dir, evaluation_mode, target_modality)
    
    if not missing_modality:
        print(f"No missing modality in {case_name}")
        return False, {}
    
    print(f"Target modality: {missing_modality}")
    
    # Load available modalities
    modalities = load_available_modalities(case_dir, missing_modality, evaluation_mode)
    
    if len(modalities) < 3:
        print(f"⚠️ Not enough modalities for synthesis (need 3, got {len(modalities)})")
        return False, {}
    
    # Load ground truth for evaluation
    if evaluation_mode:
        gt_path = os.path.join(case_dir, f"{case_name}-{missing_modality}.nii.gz")
        print(f"Loading ground truth: {gt_path}")
        ground_truth = load_image(gt_path)
    
    # Find checkpoint
    try:
        checkpoint_path = find_checkpoint(missing_modality, checkpoint_dir)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return False, {}
    
    # Synthesize
    print(f"\n=== Synthesizing {missing_modality} ===")
    try:
        synthesized, sample_time = synthesize_modality(
            modalities, missing_modality, checkpoint_path, device, 
            diffusion_steps, wavelet
        )
        
        if synthesized is None:
            return False, {}
            
    except Exception as e:
        print(f"❌ Error processing {case_name}: {e}")
        return False, {}
    
    metrics = {'sample_time': sample_time}
    
    # Evaluate metrics if requested
    if evaluation_mode and metrics_calculator:
        # Calculate metrics
        case_metrics = metrics_calculator.calculate_metrics(
            synthesized, ground_truth, case_name
        )
        metrics.update(case_metrics)
    
    # Save synthesized image
    if not evaluation_mode:
        output_path = os.path.join(output_dir, f"{case_name}-{missing_modality}.nii.gz")
        
        # Convert back to original format
        synthesized_np = synthesized.squeeze().numpy()
        
        # Create NIfTI image
        nii_img = nib.Nifti1Image(synthesized_np, affine=np.eye(4))
        nib.save(nii_img, output_path)
        print(f"💾 Saved: {output_path}")
    
    return True, metrics

def main():
    parser = argparse.ArgumentParser(description="Medical Image Synthesis - AUTO-DETECTING VERSION")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="synthesis_output")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--evaluation_mode", action="store_true")
    parser.add_argument("--target_modality", type=str, choices=MODALITIES)
    parser.add_argument("--evaluate_metrics", action="store_true")
    parser.add_argument("--diffusion_steps", type=int, default=100)
    parser.add_argument("--max_cases", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wavelet", action="store_true", default=True)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_project", type=str, default="brats-synthesis")
    parser.add_argument("--log_visual_samples", action="store_true")
    parser.add_argument("--save_visual_samples", action="store_true")
    parser.add_argument("--visual_sample_frequency", type=int, default=10)
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    
    # Initialize device
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        mode=args.wandb_mode,
        name="inference_synth_auto",
        config=vars(args)
    )
    print(f"🔮 Wandb initialized: {wandb.run.name}")
    
    # Print mode
    if args.evaluation_mode:
        print("EVALUATION MODE: Testing synthesis quality")
    else:
        print("SYNTHESIS MODE: Using incomplete dataset")
    
    if args.evaluate_metrics:
        print("Synthesis metrics enabled")
    
    # Initialize metrics calculator
    metrics_calculator = None
    if args.evaluate_metrics:
        metrics_calculator = ComprehensiveMetrics(device)
    
    # Find cases
    case_dirs = [d for d in os.listdir(args.input_dir) 
                if os.path.isdir(os.path.join(args.input_dir, d))]
    case_dirs.sort()
    
    if args.max_cases:
        case_dirs = case_dirs[:args.max_cases]
    
    print(f"Found {len(case_dirs)} cases")
    
    # Log run configuration to wandb
    wandb.log({
        "config/total_cases": len(case_dirs),
        "config/max_cases": args.max_cases or len(case_dirs),
        "config/evaluation_mode": args.evaluation_mode,
        "config/target_modality": args.target_modality or "random",
        "config/seed": args.seed,
        "config/visual_sampling": args.log_visual_samples and args.evaluation_mode,
        "config/visual_frequency": args.visual_sample_frequency if args.log_visual_samples else 0
    })
    
    # Process cases
    if not args.evaluation_mode:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare visual arguments
    visual_args = {
        'log_visual_samples': args.log_visual_samples,
        'save_visual_samples': args.save_visual_samples,
        'visual_sample_frequency': args.visual_sample_frequency
    }
    
    successful = 0
    all_metrics = {modality: [] for modality in MODALITIES}
    sample_times = []
    total_start_time = time.time()
    
    for i, case_dir_name in enumerate(case_dirs):
        case_dir = os.path.join(args.input_dir, case_dir_name)
        
        # Show progress and ETA
        if i > 0 and sample_times:
            avg_time = np.mean(sample_times)
            remaining_cases = len(case_dirs) - i
            eta_seconds = avg_time * remaining_cases
            eta_minutes = eta_seconds / 60
            print(f"\nProgress: {i}/{len(case_dirs)} | Avg: {avg_time:.1f}s/case | ETA: {eta_minutes:.1f} min")
            
            # Log progress to wandb
            wandb.log({
                "progress/processed_cases": i,
                "progress/successful_cases": successful,
                "progress/success_rate": successful / i if i > 0 else 0,
                "progress/avg_sample_time": avg_time,
                "progress/eta_minutes": eta_minutes
            })
        
        success, metrics = process_case(
            case_dir, args.output_dir, args.checkpoint_dir, device,
            metrics_calculator, args.evaluation_mode, args.target_modality, 
            args.diffusion_steps, case_index=i, visual_args=visual_args,
            wavelet=args.wavelet
        )
        
        if success:
            successful += 1
            # Track sample times
            if 'sample_time' in metrics:
                sample_times.append(metrics['sample_time'])
                print(f"⏱Case sample time: {metrics['sample_time']:.2f}s")
            
            # Collect metrics if available
            if metrics:
                if args.evaluation_mode:
                    # In evaluation mode, we know which modality was synthesized
                    target_mod = find_missing_modality(case_dir, args.evaluation_mode, args.target_modality)
                    all_metrics[target_mod].append(metrics)
                else:
                    # In synthesis mode, find the actual missing modality
                    missing_modality = find_missing_modality(case_dir, False)
                    if missing_modality:
                        all_metrics[missing_modality].append(metrics)
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    print(f"\n=== Summary ===")
    print(f"Successful: {successful}/{len(case_dirs)}")
    print(f"Total time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    
    # Log final summary to wandb
    final_summary = {
        "summary/successful_cases": successful,
        "summary/total_cases": len(case_dirs),
        "summary/success_rate": successful / len(case_dirs) if len(case_dirs) > 0 else 0,
        "summary/total_time_seconds": total_duration,
        "summary/total_time_minutes": total_duration / 60
    }
    
    # Print comprehensive metrics if available
    if args.evaluate_metrics and any(all_metrics.values()):
        print(f"\n=== METRICS SUMMARY ===")
        for modality, metrics_list in all_metrics.items():
            if metrics_list:
                print(f"\n{modality.upper()} Synthesis:")
                
                if any('l1' in m for m in metrics_list):
                    avg_metrics = {
                        'l1': np.mean([m['l1'] for m in metrics_list if 'l1' in m]),
                        'mse': np.mean([m['mse'] for m in metrics_list if 'mse' in m]),
                    }
                    std_metrics = {
                        'l1': np.std([m['l1'] for m in metrics_list if 'l1' in m]),
                        'mse': np.std([m['mse'] for m in metrics_list if 'mse' in m]),
                    }
                    print(f"  L1:   {avg_metrics['l1']:.6f} ± {std_metrics['l1']:.6f}")
                    print(f"  MSE:  {avg_metrics['mse']:.6f} ± {std_metrics['mse']:.6f}")
                    
                    # Log modality-specific metrics to wandb
                    modality_summary = {
                        f"metrics/{modality}/l1_mean": avg_metrics['l1'],
                        f"metrics/{modality}/l1_std": std_metrics['l1'],
                        f"metrics/{modality}/mse_mean": avg_metrics['mse'],
                        f"metrics/{modality}/mse_std": std_metrics['mse'],
                    }
                    final_summary.update(modality_summary)
                
                print(f"  Cases: {len(metrics_list)}")
                final_summary[f"metrics/{modality}/cases"] = len(metrics_list)
    
    # Log final summary to wandb
    wandb.log(final_summary)
    wandb.finish()

if __name__ == "__main__":
    main()