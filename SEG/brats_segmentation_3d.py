#!/usr/bin/env python3
"""
Brain tumor 3D segmentation with MONAI

This script shows how to construct a training workflow of multi-labels segmentation task.

Copyright (c) MONAI Consortium
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0

Target: Gliomas segmentation necrotic/active tumour and oedema
Modality: Multimodal multisite MRI data (FLAIR, T1w, T1gd,T2w)
Size: 750 4D volumes (484 Training + 266 Testing)
Source: BRATS 2016 and 2017 datasets.
"""

import os
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
)
from monai.utils import set_determinism
import onnxruntime
from tqdm import tqdm
import torch


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(
                torch.logical_or(
                    torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                )
            )
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d


def setup_environment():
    """Setup the environment and data directory"""
    print_config()
    
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    if directory is not None:
        os.makedirs(directory, exist_ok=True)
    root_dir = tempfile.mkdtemp() if directory is None else directory
    print(f"Data directory: {root_dir}")
    
    # Set deterministic training for reproducibility
    set_determinism(seed=0)
    
    return root_dir, directory


def get_transforms():
    """Define training and validation transforms"""
    train_transform = Compose(
        [
            # load 4 Nifti images and stack them together
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            RandSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )
    
    val_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )
    
    return train_transform, val_transform


def get_post_transforms():
    """Define post-processing transforms"""
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    
    # define post processing transforms for testset; at this time, we turn on the
    # data loader iterator mode
    post_transforms = Compose([
        EnsureTyped(keys="pred"),
        Invertd(
            keys="pred",
            transform=val_org_transform,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        Activationsd(keys="pred", sigmoid=True),
        AsDiscreted(keys="pred", threshold=0.5),
    ])
    
    return post_trans, post_transforms


def create_model(device):
    """Create and return the SegResNet model"""
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=4,
        out_channels=3,
        dropout_prob=0.2,
    ).to(device)
    return model


def train_model(model, train_loader, val_loader, device, root_dir, max_epochs=5):
    """Train the model"""
    VAL_AMP = True
    
    loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # use amp to accelerate training
    scaler = torch.cuda.amp.GradScaler()
    # enable cuDNN benchmark
    torch.backends.cudnn.benchmark = True

    best_metric = -1
    best_metric_epoch = -1
    best_metrics_epochs_and_time = [[], [], []]
    epoch_loss_values = []
    metric_values = []
    metric_values_tc = []
    metric_values_wt = []
    metric_values_et = []

    total_start = time.time()
    for epoch in range(max_epochs):
        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step_start = time.time()
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            print(
                f"{step}/{len(train_loader)}"
                f", train_loss: {loss.item():.4f}"
                f", step time: {(time.time() - step_start):.4f}"
            )
        lr_scheduler.step()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():

                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    if VAL_AMP:
                        with torch.cuda.amp.autocast():
                            val_outputs = sliding_window_inference(
                                val_inputs, (240, 240, 160), 4, model, overlap=0.8
                            )
                    else:
                        val_outputs = sliding_window_inference(
                            val_inputs, (240, 240, 160), 4, model, overlap=0.8
                        )
                    val_outputs = post_trans(val_outputs)
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    dice_metric_batch(y_pred=val_outputs, y=val_labels)

                metric = dice_metric.aggregate().item()
                metric_batch = dice_metric_batch.aggregate()
                metric_tc = metric_batch[0].item()
                metric_wt = metric_batch[1].item()
                metric_et = metric_batch[2].item()
                dice_metric.reset()
                dice_metric_batch.reset()

                metric_values.append(metric)
                metric_values_tc.append(metric_tc)
                metric_values_wt.append(metric_wt)
                metric_values_et.append(metric_et)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    best_metrics_epochs_and_time[0].append(best_metric)
                    best_metrics_epochs_and_time[1].append(best_metric_epoch)
                    best_metrics_epochs_and_time[2].append(time.time() - total_start)
                    torch.save(
                        model.state_dict(),
                        os.path.join(root_dir, "best_metric_model.pth"),
                    )
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                    f"\nbest mean dice: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )
        print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
    total_time = time.time() - total_start

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")


def evaluate_model(model, val_org_loader, post_transforms, device, root_dir):
    """Evaluate the trained model"""
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
    
    def inference(input):
        def _compute(input):
            return sliding_window_inference(
                inputs=input,
                roi_size=(240, 240, 160),
                sw_batch_size=1,
                predictor=model,
                overlap=0.5,
            )

        if VAL_AMP:
            with torch.autocast("cuda"):
                return _compute(input)
        else:
            return _compute(input)

    model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth"), weights_only=True))
    model.eval()

    with torch.no_grad():
        for val_data in val_org_loader:
            val_inputs = val_data["image"].to(device)
            val_data["pred"] = inference(val_inputs)
            val_data = [post_transforms(i) for i in decollate_batch(val_data)]
            val_outputs, val_labels = from_engine(["pred", "label"])(val_data)
            dice_metric(y_pred=val_outputs, y=val_labels)
            dice_metric_batch(y_pred=val_outputs, y=val_labels)

        metric_org = dice_metric.aggregate().item()
        metric_batch_org = dice_metric_batch.aggregate()

        dice_metric.reset()
        dice_metric_batch.reset()

    metric_tc, metric_wt, metric_et = metric_batch_org[0].item(), metric_batch_org[1].item(), metric_batch_org[2].item()

    print("Metric on original image spacing: ", metric_org)
    print(f"metric_tc: {metric_tc:.4f}")
    print(f"metric_wt: {metric_wt:.4f}")
    print(f"metric_et: {metric_et:.4f}")


def convert_to_onnx(model, device, root_dir):
    """Convert PyTorch model to ONNX format"""
    dummy_input = torch.randn(1, 4, 240, 240, 160).to(device)
    onnx_path = os.path.join(root_dir, "best_metric_model.onnx")
    torch.onnx.export(model, dummy_input, onnx_path, verbose=False)
    print(f"Model saved to ONNX format: {onnx_path}")


def main():
    """Main execution function"""
    # Setup environment
    root_dir, directory = setup_environment()
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    train_ds = DecathlonDataset(
        root_dir=root_dir,
        task="Task01_BrainTumour",
        transform=train_transform,
        section="training",
        download=True,
        cache_rate=0.0,
        num_workers=4,
    )
    
    val_ds = DecathlonDataset(
        root_dir=root_dir,
        task="Task01_BrainTumour",
        transform=val_transform,
        section="validation",
        download=False,
        cache_rate=0.0,
        num_workers=4,
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model
    model = create_model(device)
    
    # Train model
    val_interval = 2
    max_epochs = 300
    
    print("Starting training...")
    train_model(model, train_loader, val_loader, device, root_dir, max_epochs)
    
    # Setup validation on original image spacing
    val_org_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )
    
    val_org_ds = DecathlonDataset(
        root_dir=root_dir, task="Task01_BrainTumour", transform=val_org_transform, section="validation", download=False, cache_rate=0.0, num_workers=4
    )
    val_org_loader = DataLoader(val_org_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    # Get post transforms
    post_trans, post_transforms = get_post_transforms()
    
    # Evaluate model
    print("Evaluating model...")
    evaluate_model(model, val_org_loader, post_transforms, device, root_dir)
    
    # Convert to ONNX
    print("Converting to ONNX...")
    convert_to_onnx(model, device, root_dir)
    
    # Cleanup
    if directory is None:
        shutil.rmtree(root_dir)
        print("Cleaned up temporary directory")


if __name__ == "__main__":
    main()
