"""
A script for training a diffusion model for paired image-to-image translation.
"""

import argparse
import numpy as np
import random
import os
import torch as th
import wandb

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
    dist_util.setup_dist()  # ← ADD THIS LINE
    
    # Configure logger
    logger.configure()
    
    print(f"[CONFIG] lr={args.lr}, batch_size={args.batch_size}, contr={args.contr}")
    print(f"[CONFIG] sample_schedule={args.sample_schedule}, diffusion_steps={args.diffusion_steps}")
    
    # Create model and diffusion
    model_args = args_to_dict(args, model_and_diffusion_defaults().keys())
    model, diffusion = create_model_and_diffusion(**model_args)
    model.to(dist_util.dev())
    
    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler, diffusion, maxt=diffusion.num_timesteps
    )
    
    # Load dataset
    ds = BRATSVolumes(args.data_dir, mode='train')
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
    ).run_loop()


def main():
    """Normal training mode."""
    args = create_argparser().parse_args()
    wandb.init(project="fast-cwmd-brats", entity="timgsereda", config=vars(args))
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
        resume_step=400000,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        dataset='brats',
        num_workers=0,
        contr='t1n',
        sample_schedule='direct',
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
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