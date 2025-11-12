"""
train_with_wandb_checkpoints.py

Minimal example training wrapper that shows how to save PyTorch checkpoints
locally and upload them to Weights & Biases as versioned artifacts. The
script intentionally focuses on checkpoint management and artifact metadata.

This file is meant to be adaptable into the project's real training loop.
Usage (example):
  python train_with_wandb_checkpoints.py --project my-proj --entity my-entity \
    --epochs 50 --save_freq 5 --save_dir ./checkpoints

The script will:
 - Save periodic checkpoints every `save_freq` epochs
 - Save a "best" checkpoint when validation metric improves
 - Save a final checkpoint at the end of training
 - Upload each saved checkpoint to W&B as an artifact with metadata and aliases

Note: this script does not implement a concrete model/train loop. Replace the
placeholder `train_one_epoch` and `validate` functions with the project's logic.
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import wandb


def save_local_checkpoint(state: dict, path: str):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def upload_checkpoint_artifact(run: wandb.sdk.wandb_run.Run, file_path: str,
                               checkpoint_type: str, epoch: int, metrics: dict, config: dict):
    """Upload a checkpoint file as a W&B artifact with metadata and aliases.

    - checkpoint_type: one of 'periodic', 'best', 'final'
    - metrics: snapshot of validation/train metrics
    - config: training configuration (args)
    """
    fname = os.path.basename(file_path)
    artifact_name = f"{run.name or run.id}-{fname}"

    metadata = {
        "checkpoint_type": checkpoint_type,
        "epoch": epoch,
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": metrics or {},
        "config": config or {},
        "pytorch_version": torch.__version__,
    }

    art = wandb.Artifact(name=artifact_name, type="model", metadata=metadata)
    art.add_file(file_path)

    # Aliases make it easy to retrieve 'best' or 'latest' directly
    aliases = [f"epoch-{epoch}"]
    if checkpoint_type == "best":
        aliases.append("best")
    if checkpoint_type == "final":
        aliases.append("final")

    run.log_artifact(art, aliases=aliases)
    # return artifact object (not strictly necessary)
    return art


def train_one_epoch(epoch: int):
    """Placeholder - replace with real training logic."""
    # Simulate some training time
    time.sleep(0.01)
    # Return mocked training loss
    return max(0.0, 1.0 - epoch * 0.01)


def validate(epoch: int):
    """Placeholder validation that returns a mocked validation loss.

    Replace this with your validation function that returns a scalar metric
    to monitor (e.g., validation loss or negative dice).
    """
    # Simulate validation time
    time.sleep(0.005)
    return max(0.0, 1.0 - epoch * 0.012)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=None,
                        help="W&B project name. If omitted, will use WANDB_PROJECT env var")
    parser.add_argument("--entity", default=None,
                        help="W&B entity name. If omitted, will use WANDB_ENTITY env var")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_freq", type=int, default=5,
                        help="Save periodic checkpoint every N epochs")
    parser.add_argument("--monitor", type=str, default="val_loss",
                        help="Metric name to monitor for best checkpoint")
    parser.add_argument("--monitor_mode", choices=["min", "max"], default="min")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume (local file)")
    args = parser.parse_args()

    # Resolve project/entity from args or environment variables
    args.project = args.project or os.getenv("WANDB_PROJECT")
    args.entity = args.entity or os.getenv("WANDB_ENTITY")
    if not args.project or not args.entity:
        parser.error("project and entity must be provided via --project/--entity or WANDB_PROJECT/WANDB_ENTITY env vars")

    # Initialize wandb
    run = wandb.init(project=args.project, entity=args.entity, config=vars(args), name=args.run_name)

    # state you would normally save
    model_state = {"example": True}
    optimizer_state = {"lr": 1e-3}

    best_metric = None
    best_epoch = -1
    best_ckpt_path = None

    start_epoch = 1
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming from checkpoint: {args.resume}")
            ckpt = torch.load(args.resume, map_location="cpu")
            start_epoch = ckpt.get("epoch", 1) + 1
            best_metric = ckpt.get("best_metric", None)
        else:
            print(f"Resume path provided but not found: {args.resume}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(epoch)
        val_loss = validate(epoch)

        metrics = {"train_loss": train_loss, "val_loss": val_loss}
        wandb.log({"epoch": epoch, **metrics})

        # Periodic checkpoint
        if args.save_freq > 0 and (epoch % args.save_freq == 0):
            ckpt_path = str(save_dir / f"checkpoint_epoch_{epoch}.pth")
            state = {
                "epoch": epoch,
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer_state,
                "metrics": metrics,
                "best_metric": best_metric,
            }
            save_local_checkpoint(state, ckpt_path)
            print(f"Saved periodic checkpoint: {ckpt_path}")
            upload_checkpoint_artifact(run, ckpt_path, "periodic", epoch, metrics, vars(args))

        # Best checkpoint
        current = val_loss
        is_better = False
        if best_metric is None:
            is_better = True
        else:
            if args.monitor_mode == "min":
                is_better = current < best_metric
            else:
                is_better = current > best_metric

        if is_better:
            best_metric = current
            best_epoch = epoch
            best_ckpt_path = str(save_dir / "best_checkpoint.pth")
            state = {
                "epoch": epoch,
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer_state,
                "metrics": metrics,
                "best_metric": best_metric,
            }
            save_local_checkpoint(state, best_ckpt_path)
            print(f"Saved new best checkpoint (epoch {epoch}): {best_ckpt_path}")
            upload_checkpoint_artifact(run, best_ckpt_path, "best", epoch, metrics, vars(args))

    # Final checkpoint
    final_ckpt = str(save_dir / "final_checkpoint.pth")
    final_state = {
        "epoch": args.epochs,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
        "metrics": {"train_loss": train_loss, "val_loss": val_loss},
        "best_metric": best_metric,
    }
    save_local_checkpoint(final_state, final_ckpt)
    print(f"Saved final checkpoint: {final_ckpt}")
    upload_checkpoint_artifact(run, final_ckpt, "final", args.epochs, final_state.get("metrics"), vars(args))

    # Record best info in run summary
    if best_metric is not None:
        run.summary["best_epoch"] = best_epoch
        run.summary["best_metric"] = best_metric
    run.finish()


if __name__ == "__main__":
    main()
