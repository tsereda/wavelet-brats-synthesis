"""
A script for training a diffusion model for paired image-to-image translation.
FIXED: Proper import handling for app/ directory structure
"""

import argparse
import numpy as np
import random
import os
import sys
import torch as th
import wandb

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
from guided_diffusion.train_util import TrainLoop
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
    
    # Configure logger
    logger.configure()
    
    # Process special checkpoint steps
    if isinstance(args.special_checkpoint_steps, str):
        special_checkpoint_steps = [int(x.strip()) for x in args.special_checkpoint_steps.split(',') if x.strip()]
    else:
        special_checkpoint_steps = args.special_checkpoint_steps
    
    print(f"[CONFIG] lr={args.lr}, batch_size={args.batch_size}, contr={args.contr}")
    print(f"[CONFIG] sample_schedule={args.sample_schedule}, diffusion_steps={args.diffusion_steps}")
    print(f"[CONFIG] wavelet={args.wavelet}")
    print(f"[CONFIG] special_checkpoint_steps={special_checkpoint_steps}")
    print(f"[CONFIG] save_to_wandb={args.save_to_wandb}")
    print(f"[CONFIG] resume_checkpoint={args.resume_checkpoint}")
    print(f"[CONFIG] resume_step={args.resume_step}")
    
    # Create model and diffusion
    model_args = args_to_dict(args, model_and_diffusion_defaults().keys())
    model, diffusion = create_model_and_diffusion(**model_args)
    model.to(dist_util.dev())
    
    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler, diffusion, maxt=diffusion.num_timesteps
    )
    
    # Load dataset
    ds = BRATSVolumes(args.data_dir, mode='train', wavelet=args.wavelet)
    dataloader = th.utils.data.DataLoader(
        ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True
    )
    
    print("Starting training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=dataloader,
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
        schedule_sampler=schedule_sampler,
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
    ).run_loop()


def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=5000,
        resume_checkpoint='',
        resume_step=0,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        dataset='brats',
        num_workers=0,
        contr='t1n',
        sample_schedule='direct',
        wavelet='haar',
        special_checkpoint_steps="75400,100000,200000",
        save_to_wandb=True,
    )
    defaults.update(model_and_diffusion_defaults())
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
            project=os.getenv('WANDB_PROJECT', 'wavelet-brats-synthesis'),
            entity=os.getenv('WANDB_ENTITY', 'timgsereda'),
            id=resume_run_id,
            resume="must"
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