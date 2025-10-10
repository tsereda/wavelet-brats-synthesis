"""
A script for training a diffusion model for paired image-to-image translation.
Enhanced with dual checkpoint saving to wandb and PVC.
"""

import argparse
import numpy as np
import random
import os
import torch as th
import wandb
import shutil
from pathlib import Path

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


def parse_save_iterations(save_iters_str):
    """Parse save iterations from environment variable or argument"""
    if not save_iters_str:
        return None
    
    try:
        return [int(x.strip()) for x in save_iters_str.split(',')]
    except ValueError:
        print(f"⚠️ Warning: Invalid save_iterations format: {save_iters_str}")
        return None


def setup_dual_checkpoint_saving():
    """Setup dual checkpoint saving configuration"""
    dual_save_enabled = os.getenv('DUAL_CHECKPOINT_SAVE', 'false').lower() == 'true'
    pvc_checkpoint_dir = os.getenv('PVC_CHECKPOINT_DIR', '/data/checkpoints')
    save_iters_str = os.getenv('SAVE_ITERATIONS', '')
    
    config = {
        'enabled': dual_save_enabled,
        'pvc_dir': pvc_checkpoint_dir,
        'save_iterations': parse_save_iterations(save_iters_str)
    }
    
    if dual_save_enabled:
        # Create PVC checkpoint directory if it doesn't exist
        os.makedirs(pvc_checkpoint_dir, exist_ok=True)
        print(f"💾 Dual checkpoint saving enabled:")
        print(f"  - Wandb: Automatic cloud backup and versioning")
        print(f"  - PVC: {pvc_checkpoint_dir}")
        if config['save_iterations']:
            print(f"  - Save at iterations: {config['save_iterations']}")
    
    return config


def save_checkpoint_dual(checkpoint_path, pvc_dir, step, modality):
    """
    Save checkpoint to both local (for wandb) and PVC
    
    Args:
        checkpoint_path: Local checkpoint path (will be uploaded to wandb)
        pvc_dir: PVC directory for persistent storage
        step: Training step number
        modality: Training modality (t1n, t1c, t2w, t2f)
    """
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Warning: Checkpoint not found at {checkpoint_path}")
        return
    
    # Extract filename
    checkpoint_name = os.path.basename(checkpoint_path)
    
    # Save to PVC with informative naming
    pvc_checkpoint_path = os.path.join(pvc_dir, f"{modality}_{checkpoint_name}")
    
    try:
        shutil.copy2(checkpoint_path, pvc_checkpoint_path)
        print(f"💾 Saved to PVC: {pvc_checkpoint_path}")
        
        # Also save to wandb
        if wandb.run is not None:
            wandb.save(checkpoint_path, base_path=os.path.dirname(checkpoint_path))
            print(f"☁️ Saved to wandb: {checkpoint_name}")
            
            # Log checkpoint metadata
            wandb.log({
                f"checkpoint/{modality}/step": step,
                f"checkpoint/{modality}/saved": True
            })
            
    except Exception as e:
        print(f"❌ Error saving checkpoint: {e}")


def setup_training(args):
    """Setup and run training with dual checkpoint saving."""
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
    
    # Setup dual checkpoint saving
    dual_checkpoint_config = setup_dual_checkpoint_saving()
    
    print(f"[CONFIG] lr={args.lr}, batch_size={args.batch_size}, contr={args.contr}")
    print(f"[CONFIG] sample_schedule={args.sample_schedule}, diffusion_steps={args.diffusion_steps}")
    print(f"[CONFIG] wavelet={args.wavelet}")
    
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
        dual_checkpoint_config=dual_checkpoint_config,
        save_checkpoint_callback=save_checkpoint_dual,
    ).run_loop()


def main():
    """Normal training mode."""
    args = create_argparser().parse_args()
    
    # Initialize wandb with sweep support
    if os.getenv('WANDB_SWEEP_ID'):
        # Wandb sweep mode - wandb.init() called by sweep agent
        pass
    else:
        # Normal mode - manual wandb init
        wandb.init(
            project=os.getenv('WANDB_PROJECT', 'fast-cwmd-brats'),
            entity=os.getenv('WANDB_ENTITY', 'timgsereda'),
            config=vars(args)
        )
    
    setup_training(args)


def train_with_wandb_sweep():
    """W&B sweep training mode."""
    project = os.getenv('WANDB_PROJECT', 'wavelet-brats-synthesis')
    entity = os.getenv('WANDB_ENTITY', 'timgsereda')
    
    wandb.init(project=project, entity=entity)
    
    args = create_argparser().parse_args([])
    
    # Override with sweep config
    for key, value in wandb.config.items():
        if hasattr(args, key):
            setattr(args, key, value)
            print(f"[SWEEP] {key}={value}")
    
    setup_training(args)


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
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    
    # Add dual checkpoint saving arguments
    parser.add_argument('--save_iterations', type=str, default=None,
                       help='Comma-separated list of iterations to save checkpoints (e.g., "100,500,1000")')
    
    return parser


if __name__ == "__main__":
    try:
        if os.getenv('WANDB_SWEEP_ID'):
            train_with_wandb_sweep()
        else:
            main()
    except KeyboardInterrupt:
        print("\nTraining interrupted")
        wandb.finish()
    except Exception as e:
        print(f"\nError: {e}")
        wandb.finish()
        raise