"""
A script for training a diffusion model for paired image-to-image translation.
FIXED: Proper import handling for app/ directory structure
ENHANCED: Support for direct regression (no diffusion) mode
"""

import argparse
import numpy as np
import random
import os
import sys
import torch as th
import wandb
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ✅ CRITICAL FIX: Add the parent directory and app directory to Python path
# This allows importing from guided_diffusion when running from app/scripts/
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # This gets us to the 'app' directory
root_dir = os.path.dirname(parent_dir)     # This gets us to the project root

# Add both app directory and project root to path
sys.path.insert(0, parent_dir)
sys.path.insert(0, root_dir)

# Now the imports should work
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults, 
    create_model_and_diffusion,
    args_to_dict, 
    add_dict_to_argparser
)
from guided_diffusion.train_util import TrainLoop, DirectRegressionLoop
from guided_diffusion.bratsloader import BRATSVolumes


def setup_training(args):
    """Setup and run training."""
    # Validate
    if not args.data_dir or not os.path.exists(args.data_dir):
        raise ValueError(f"Invalid data_dir: {args.data_dir}")
    
    # Set seeds
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Initialize distributed training (REQUIRED - even for single GPU)
    dist_util.setup_dist()
    
    # Ensure checkpoint directory exists
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Configure logger with checkpoint directory
    logger.configure(dir=args.checkpoint_dir)
    
    # Validate checkpoint directory is writable
    test_file = os.path.join(args.checkpoint_dir, '.write_test')
    try:
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print(f"[CHECK] ✅ Checkpoint directory writable: {args.checkpoint_dir}")
    except Exception as e:
        print(f"[ERROR] ❌ Cannot write to {args.checkpoint_dir}: {e}")
        raise
    
    # Process special checkpoint steps
    if isinstance(args.special_checkpoint_steps, str):
        special_checkpoint_steps = [int(x.strip()) for x in args.special_checkpoint_steps.split(',') if x.strip()]
    else:
        special_checkpoint_steps = args.special_checkpoint_steps
    
    # 🆕 Parse model_mode to set sample_schedule
    model_mode = args.model_mode
    if model_mode == 'direct':
        # Direct regression mode - no diffusion
        use_direct_regression = True
        sample_schedule = 'direct'  # Unused but set for consistency
        print("🚀 MODE: Direct Regression (no diffusion)")
    elif model_mode == 'diffusion_fast':
        # Fast-cWDM: T=100, sampled schedule
        use_direct_regression = False
        sample_schedule = 'sampled'
        print("🚀 MODE: Fast Diffusion (T=100, sampled schedule)")
    elif model_mode == 'diffusion_standard':
        # Standard DDPM: T=100, direct schedule
        use_direct_regression = False
        sample_schedule = 'direct'
        print("🚀 MODE: Standard Diffusion (T=100, direct schedule)")
    else:
        raise ValueError(f"Invalid model_mode: {model_mode}. Must be 'direct', 'diffusion_fast', or 'diffusion_standard'")
    
    # Override sample_schedule from mode
    args.sample_schedule = sample_schedule
    
    # 🆕 Handle wavelet parameter (null → None for use_freq logic)
    if args.wavelet == 'nowavelet' or args.wavelet is None:
        args.wavelet = None
        args.use_freq = False
        print("🔧 Wavelet: nowavelet (baseline, image space)")
    else:
        args.use_freq = True
        print(f"🔧 Wavelet: {args.wavelet} (wavelet space)")
    
    # Set max iterations from max_iterations parameter
    if args.max_iterations > 0:
        args.lr_anneal_steps = args.max_iterations
    
    print(f"[CONFIG] model_mode={model_mode}")
    print(f"[CONFIG] lr={args.lr}, batch_size={args.batch_size}, contr={args.contr}")
    print(f"[CONFIG] max_iterations={args.max_iterations}, lr_anneal_steps={args.lr_anneal_steps}")
    print(f"[CONFIG] sample_schedule={sample_schedule}, diffusion_steps={args.diffusion_steps}")
    print(f"[CONFIG] wavelet={args.wavelet}, use_freq={args.use_freq}")
    print(f"[CONFIG] checkpoint_dir={args.checkpoint_dir}")
    print(f"[CONFIG] val_interval={args.val_interval}")
    print(f"[CONFIG] special_checkpoint_steps={special_checkpoint_steps}")
    print(f"[CONFIG] save_to_wandb={args.save_to_wandb}")
    print(f"[CONFIG] resume_checkpoint={args.resume_checkpoint}")
    print(f"[CONFIG] resume_step={args.resume_step}")
    
    # 🆕 AUTO-CONFIGURE CHANNELS based on mode and wavelet
    # Direct regression: input = 3 condition modalities
    # Diffusion: input = noisy target + 3 condition modalities
    # Wavelet: multiply by 8 subbands
    if args.use_freq:  # Wavelet space
        condition_channels = 3 * 8  # 3 modalities × 8 subbands = 24
        target_channels = 1 * 8      # 1 target modality × 8 subbands = 8
    else:  # Image space
        condition_channels = 3       # 3 modalities
        target_channels = 1          # 1 target modality
    
    if use_direct_regression:
        # Direct: input = condition only
        args.in_channels = condition_channels
        args.out_channels = target_channels
    else:
        # Diffusion: input = noisy target + condition
        args.in_channels = target_channels + condition_channels
        args.out_channels = target_channels
    
    print(f"[CONFIG] ✅ Auto-configured channels: in_channels={args.in_channels}, out_channels={args.out_channels}")
    
    # 🆕 CRITICAL: Set mode='i2i' for conditional image-to-image synthesis
    args.mode = 'i2i'
    print(f"[CONFIG] mode={args.mode} (conditional synthesis)")
    
    # Create model and diffusion
    model_args = args_to_dict(args, model_and_diffusion_defaults().keys())
    model, diffusion = create_model_and_diffusion(**model_args)
    model.to(dist_util.dev())
    
    # Load dataset and split into train/val (70/30)
    full_ds = BRATSVolumes(args.data_dir, mode='train', wavelet=args.wavelet)
    
    # Calculate split sizes
    train_size = int(0.7 * len(full_ds))
    val_size = len(full_ds) - train_size
    
    print(f"📊 Dataset split: {len(full_ds)} total → {train_size} train + {val_size} val")
    
    # Split with seeded generator for reproducibility
    train_ds, val_ds = th.utils.data.random_split(
        full_ds, [train_size, val_size],
        generator=th.Generator().manual_seed(args.seed)
    )
    
    # Create data loaders
    train_loader = th.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True
    )
    
    val_loader = th.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )
    
    # Select training loop based on mode
    if use_direct_regression:
        print("Starting Direct Regression training...")
        TrainLoopClass = DirectRegressionLoop
        # Direct regression doesn't need schedule_sampler
        loop_kwargs = dict(
            schedule_sampler=None
        )
    else:
        print("Starting Diffusion Model training...")
        TrainLoopClass = TrainLoop
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion, maxt=diffusion.num_timesteps
        )
        loop_kwargs = dict(
            schedule_sampler=schedule_sampler
        )
    
    TrainLoopClass(
        model=model,
        diffusion=diffusion,
        data=train_loader,
        val_data=val_loader,
        batch_size=args.batch_size,
        in_channels=args.in_channels,
        image_size=args.image_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        resume_step=args.resume_step,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        dataset=args.dataset,
        summary_writer=None,
        mode='i2i',
        contr=args.contr,
        sample_schedule=args.sample_schedule,
        diffusion_steps=args.diffusion_steps,
        wavelet=args.wavelet,
        special_checkpoint_steps=special_checkpoint_steps,
        save_to_wandb=args.save_to_wandb,
        val_interval=args.val_interval,
        checkpoint_dir=args.checkpoint_dir,  # <--- pass PVC path
        **loop_kwargs
    ).run_loop()


def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        max_iterations=200000,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=5000,
        resume_checkpoint='',
        resume_step=0,
        use_fp16=True,
        fp16_scale_growth=1e-3,
        use_checkpoint=True,
        dataset='brats',
        num_workers=0,
        contr='t1n',
        sample_schedule='direct',
        special_checkpoint_steps="75400,100000,200000",
        save_to_wandb=False,
        model_mode='diffusion_fast',
        checkpoint_dir='/checkpoints', 
        val_interval=10000,             
    )
    # Add model/diffusion defaults first
    defaults.update(model_and_diffusion_defaults())
    # Then set project-specific defaults (will override model defaults if they overlap)
    defaults.update({
        'wavelet': 'haar',  # Project default: haar wavelet
    })
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def train_with_wandb_sweep():
    """W&B sweep training mode."""
    
    # Check if we're resuming from an existing run
    resume_run_id = os.getenv('WANDB_RUN_ID', '')
    
    if resume_run_id:
        print(f"🔄 Resuming W&B run: {resume_run_id}")
        # Initialize with specific run ID to resume
        wandb.init(
            project=os.getenv('WANDB_PROJECT', 'test-brats-3dsynth'),
            entity=os.getenv('WANDB_ENTITY', 'timgsereda'),
        )
    else:
        # Normal sweep mode - let sweep agent handle init
        # DON'T call wandb.init() here - the sweep agent does it
        pass
    
    args = create_argparser().parse_args([])
    
    # Override with sweep config ONLY if wandb is initialized and has config
    try:
        if wandb.run is not None and hasattr(wandb, 'config') and wandb.config:
            for key, value in wandb.config.items():
                if hasattr(args, key):
                    setattr(args, key, value)
                    print(f"[SWEEP] {key}={value}")
        else:
            print("[RESUME] No sweep config available - using defaults")
    except Exception as e:
        print(f"[RESUME] Could not access wandb config: {e}")
        print("[RESUME] Proceeding with default arguments")
    
    # Auto-resume from local checkpoint if available
    latest_checkpoint = os.getenv('LATEST_CHECKPOINT_FILE', '')
    latest_step = os.getenv('LATEST_CHECKPOINT_STEP', '')
    resume_modality = os.getenv('RESUME_MODALITY', '')
    
    if latest_checkpoint and os.path.exists(latest_checkpoint):
        print(f"🔄 Auto-resuming from local checkpoint: {latest_checkpoint}")
        print(f"   Starting from step: {latest_step}")
        args.resume_checkpoint = latest_checkpoint
        if latest_step:
            args.resume_step = int(latest_step)
    
    # Use detected modality if available
    if resume_modality:
        print(f"🧠 Using detected modality: {resume_modality}")
        args.contr = resume_modality
    
    setup_training(args)


def main():
    """Normal training mode."""
    args = create_argparser().parse_args()
    wandb.init(project="fast-cwmd-brats", entity="timgsereda", config=vars(args))
    setup_training(args)


if __name__ == "__main__":
    try:
        # Check if we're in resume mode or sweep mode
        if os.getenv('RESUME_RUN') or os.getenv('WANDB_RUN_ID'):
            # Resume mode - handle direct training with W&B resume
            train_with_wandb_sweep()
        elif os.getenv('SWEEP_ID'):  # Fixed: was WANDB_SWEEP_ID, now SWEEP_ID
            # Sweep mode - let agent handle everything
            train_with_wandb_sweep()
        else:
            # Normal training mode
            main()
    except KeyboardInterrupt:
        print("\nTraining interrupted")
        wandb.finish()
    except Exception as e:
        print(f"\nError: {e}")
        wandb.finish()
        raise