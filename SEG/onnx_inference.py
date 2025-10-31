#!/usr/bin/env python3
"""
ONNX Inference for Brain Tumor Segmentation

This script demonstrates how to use the ONNX model for inference
and compares results with the original PyTorch model.
"""

import os
import matplotlib.pyplot as plt
import torch
import onnxruntime
from monai.apps import DecathlonDataset
from monai.data import DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
)
from monai.networks.nets import SegResNet
from tqdm import tqdm


class ConvertToMultiChannelBasedOnBratsClassesd:
    """
    Convert labels to multi channels based on brats classes
    """
    def __init__(self, keys):
        self.keys = keys

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


def setup_onnx_inference(root_dir, device):
    """Setup ONNX inference session and validation data"""
    VAL_AMP = True
    
    # Load ONNX model
    onnx_model_path = os.path.join(root_dir, "best_metric_model.onnx")
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    
    # Setup validation transforms
    val_transform = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])
    
    # Create validation dataset
    val_ds = DecathlonDataset(
        root_dir=root_dir,
        task="Task01_BrainTumour",
        transform=val_transform,
        section="validation",
        download=False,
        cache_rate=0.0,
        num_workers=4,
    )
    
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    # Define ONNX inference function
    def onnx_infer(inputs):
        ort_inputs = {ort_session.get_inputs()[0].name: inputs.cpu().numpy()}
        ort_outs = ort_session.run(None, ort_inputs)
        return torch.Tensor(ort_outs[0]).to(inputs.device)

    def predict(input):
        def _compute(input):
            return sliding_window_inference(
                inputs=input,
                roi_size=(240, 240, 160),
                sw_batch_size=1,
                predictor=onnx_infer,
                overlap=0.5,
            )

        if VAL_AMP:
            with torch.autocast("cuda"):
                return _compute(input)
        else:
            return _compute(input)
    
    return ort_session, val_ds, val_loader, predict


def run_onnx_inference(root_dir, device):
    """Run ONNX inference and compute metrics"""
    ort_session, val_ds, val_loader, predict = setup_onnx_inference(root_dir, device)
    
    # Setup metrics and post-processing
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    
    print("Running ONNX inference...")
    for val_data in tqdm(val_loader, desc="ONNX Inference Progress"):
        val_inputs, val_labels = (
            val_data["image"].to(device),
            val_data["label"].to(device),
        )

        ort_outs = predict(val_inputs)
        val_outputs = post_trans(torch.Tensor(ort_outs[0]).to(device)).unsqueeze(0)

        dice_metric(y_pred=val_outputs, y=val_labels)
        dice_metric_batch(y_pred=val_outputs, y=val_labels)
    
    onnx_metric = dice_metric.aggregate().item()
    onnx_metric_batch = dice_metric_batch.aggregate()
    onnx_metric_tc = onnx_metric_batch[0].item()
    onnx_metric_wt = onnx_metric_batch[1].item()
    onnx_metric_et = onnx_metric_batch[2].item()

    print(f"ONNX metric: {onnx_metric:.4f}")
    print(f"ONNX metric_tc: {onnx_metric_tc:.4f}")
    print(f"ONNX metric_wt: {onnx_metric_wt:.4f}")
    print(f"ONNX metric_et: {onnx_metric_et:.4f}")
    
    return onnx_metric, onnx_metric_tc, onnx_metric_wt, onnx_metric_et


def compare_pytorch_onnx_outputs(root_dir, device):
    """Compare PyTorch and ONNX model outputs visually"""
    VAL_AMP = True
    
    # Setup models
    ort_session, val_ds, val_loader, onnx_predict = setup_onnx_inference(root_dir, device)
    
    # Load PyTorch model
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=4,
        out_channels=3,
        dropout_prob=0.2,
    ).to(device)
    
    model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth"), weights_only=True))
    model.eval()
    
    # Define PyTorch inference function
    def pytorch_inference(input):
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
    
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    
    with torch.no_grad():
        # Select one image to evaluate and visualize the model output
        val_input = val_ds[6]["image"].unsqueeze(0).to(device)
        val_output = pytorch_inference(val_input)
        val_output = post_trans(val_output[0])
        ort_output = onnx_predict(val_input)
        ort_output = post_trans(torch.Tensor(ort_output[0]).to(device)).unsqueeze(0)
        
        # Visualize the 4 input image channels
        plt.figure("image", (24, 6))
        for i in range(4):
            plt.subplot(1, 4, i + 1)
            plt.title(f"image channel {i}")
            plt.imshow(val_ds[6]["image"][i, :, :, 70].detach().cpu(), cmap="gray")
        plt.show()
        
        # Visualize the 3 channels label corresponding to this image
        plt.figure("label", (18, 6))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(f"label channel {i}")
            plt.imshow(val_ds[6]["label"][i, :, :, 70].detach().cpu())
        plt.show()
        
        # Visualize the 3 channels PyTorch model output
        plt.figure("pytorch_output", (18, 6))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(f"PyTorch output channel {i}")
            plt.imshow(val_output[i, :, :, 70].detach().cpu())
        plt.show()
        
        # Visualize the 3 channels ONNX model output
        plt.figure("onnx_output", (18, 6))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(f"ONNX output channel {i}")
            plt.imshow(ort_output[0, i, :, :, 70].detach().cpu())
        plt.show()


def main():
    """Main function for ONNX inference"""
    # Setup
    root_dir = os.environ.get("MONAI_DATA_DIRECTORY", "/tmp/monai_data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Data directory: {root_dir}")
    
    # Check if ONNX model exists
    onnx_model_path = os.path.join(root_dir, "best_metric_model.onnx")
    if not os.path.exists(onnx_model_path):
        print(f"ONNX model not found at {onnx_model_path}")
        print("Please run the training script first to generate the ONNX model.")
        return
    
    # Run ONNX inference
    run_onnx_inference(root_dir, device)
    
    # Compare outputs (optional - requires matplotlib display capability)
    try:
        compare_pytorch_onnx_outputs(root_dir, device)
    except Exception as e:
        print(f"Could not display comparison plots: {e}")
        print("This is normal if running in a headless environment.")


if __name__ == "__main__":
    main()
