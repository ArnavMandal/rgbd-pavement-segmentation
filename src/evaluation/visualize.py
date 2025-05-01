#!/usr/bin/env python3
"""
Visualization script to compare RGB and RGBD model predictions.

Usage:
    # Run from project root directory
    python src/evaluation/visualize.py --rgb_model models/best_unet_rgb.pth --rgbd_model models/best_unet_rgbd.pth --n_samples 3
"""

import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path

# Add project root to path for imports when running from src/evaluation
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import PavementDataset
from src.models.unet import UNet


# Default paths (relative to project root)
RGB_MODEL_PATH = "models/best_unet_rgb.pth"
RGBD_MODEL_PATH = "models/best_unet_rgbd.pth"
TEST_CSV = "data/splits/test.csv"
RGB_DIR = "data/processed/rgb"
MASK_DIR = "data/processed/mask"
DEPTH_DIR = "data/processed/depth"
OUT_DIR = "visualizations"


def load_model(model_path, n_channels):
    """Load a trained model with error handling."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepend project root if path is not absolute
    if not os.path.isabs(model_path):
        model_path = os.path.join(project_root, model_path)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None, device
    
    try:
        model = UNet(n_channels=n_channels, n_classes=1).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Successfully loaded {n_channels}-channel model from {model_path}")
        return model, device
    except Exception as e:
        print(f"Error loading model from {model_path}: {str(e)}")
        return None, device


def process_image(model, image, device):
    """Process an image through the model and return the prediction."""
    if model is None:
        return None
        
    with torch.no_grad():
        # Make sure image is the right shape [batch_size, channels, height, width]
        # If already has batch dimension, don't add another
        if image.dim() == 3:  # [C, H, W]
            image = image.unsqueeze(0)  # Add batch dimension [1, C, H, W]
        elif image.dim() == 5:  # Extra dimension somewhere
            image = image.squeeze(1)  # Remove extra dimension
            
        # Move to device
        image = image.to(device)
        
        # Process through model
        logits = model(image)
        preds = (torch.sigmoid(logits) > 0.5).cpu().float()
    
    # Ensure we return a 2D mask (H, W)
    return preds.squeeze().numpy()


def main(args):
    # Create output directory
    out_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output will be saved to {out_dir}")

    # Load models
    rgb_model, device = load_model(args.rgb_model, 3)
    rgbd_model, device = load_model(args.rgbd_model, 4)
    
    if rgb_model is None and rgbd_model is None:
        print("Error: Both models failed to load. Exiting.")
        return

    # Create dataset
    dataset_args = {
        'split_csv': os.path.join(project_root, args.split_csv),
        'rgb_dir': os.path.join(project_root, args.rgb_dir),
        'mask_dir': os.path.join(project_root, args.mask_dir),
        'depth_dir': os.path.join(project_root, args.depth_dir),
        'split_prefix': args.split_prefix
    }
    
    try:
        dataset = PavementDataset(**dataset_args)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        print(f"Dataset loaded with {len(dataset)} images")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return

    # Process images
    print(f"Processing {min(args.n_samples, len(dataset))} images...")
    
    for i, batch in enumerate(dataloader):
        if i >= args.n_samples:
            break
            
        img = batch["image"]  # Shape: [batch_size, channels, height, width]
        mask = batch["mask"]
        img_id = batch["id"][0]  # Get the ID
        
        print(f"Processing image {img_id}, tensor shape: {img.shape}")
        
        # Process with RGB model (use first 3 channels if image has 4)
        rgb_pred = None
        if rgb_model is not None:
            if img.shape[1] == 4:  # If RGBD image (4 channels)
                # Extract only RGB channels
                rgb_img = img[0, :3, :, :]  # Shape: [3, H, W]
                rgb_pred = process_image(rgb_model, rgb_img, device)
            else:  # Already RGB
                rgb_pred = process_image(rgb_model, img[0], device)
                
        # Process with RGBD model
        rgbd_pred = None
        if rgbd_model is not None:
            if img.shape[1] == 4:  # If RGBD image
                rgbd_pred = process_image(rgbd_model, img[0], device)
            else:  # Add a fake depth channel for RGB-only images
                # Create a zero-filled depth channel
                rgb = img[0]  # Shape: [3, H, W]
                zeros = torch.zeros((1, rgb.shape[1], rgb.shape[2]))  # Shape: [1, H, W]
                rgbd_img = torch.cat([rgb, zeros], dim=0)  # Shape: [4, H, W]
                rgbd_pred = process_image(rgbd_model, rgbd_img, device)
        
        # Create visualization
        # Create a grid of subplots based on which models are available
        n_plots = 2  # Always show original and ground truth
        if rgb_pred is not None:
            n_plots += 1
        if rgbd_pred is not None:
            n_plots += 1
            
        fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 4, 4))
        
        # Prepare the RGB image for display (denormalize)
        # Get the first 3 channels (RGB) of the first image in batch
        if img.shape[1] == 4:  # RGBD image
            rgb_img_display = img[0, :3, :, :]  # Just RGB channels
        else:
            rgb_img_display = img[0]  # Already RGB
            
        # Convert to numpy for display
        rgb_img_np = rgb_img_display.permute(1, 2, 0).numpy()
        if rgb_img_np.shape[2] == 4:  # If RGBD, only use RGB channels
            mean = [0.485, 0.456, 0.406, 0.5]
            std = [0.229, 0.224, 0.225, 0.5]
            rgb_img_np = (rgb_img_np * std + mean)[:, :, :3]  # Show only RGB channels
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            rgb_img_np = rgb_img_np * std + mean
        
        rgb_img_np = rgb_img_np.clip(0, 1)  # Ensure values are in valid range
        
        # Plot original image
        plot_idx = 0
        axes[plot_idx].imshow(rgb_img_np)
        axes[plot_idx].set_title("Original Image")
        axes[plot_idx].axis("off")
        
        # Plot ground truth
        plot_idx += 1
        axes[plot_idx].imshow(rgb_img_np)  # Background
        mask_np = mask[0].squeeze().numpy()
        mask_overlay = np.ma.masked_where(mask_np < 0.5, mask_np)
        axes[plot_idx].imshow(mask_overlay, cmap='spring', alpha=0.7)
        axes[plot_idx].set_title("Ground Truth")
        axes[plot_idx].axis("off")
        
        # Plot RGB model prediction if available
        if rgb_pred is not None:
            plot_idx += 1
            axes[plot_idx].imshow(rgb_img_np)  # Background
            rgb_overlay = np.ma.masked_where(rgb_pred < 0.5, rgb_pred)
            axes[plot_idx].imshow(rgb_overlay, cmap='winter', alpha=0.7)
            axes[plot_idx].set_title("RGB Model")
            axes[plot_idx].axis("off")
        
        # Plot RGBD model prediction if available
        if rgbd_pred is not None:
            plot_idx += 1
            axes[plot_idx].imshow(rgb_img_np)  # Background
            rgbd_overlay = np.ma.masked_where(rgbd_pred < 0.5, rgbd_pred)
            axes[plot_idx].imshow(rgbd_overlay, cmap='autumn', alpha=0.7)
            axes[plot_idx].set_title("RGBD Model")
            axes[plot_idx].axis("off")
        
        # Add overall title and save
        plt.suptitle(f"Image: {img_id}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"comparison_{img_id}.png"), dpi=150)
        plt.close(fig)
        
        print(f"Saved visualization for image {img_id}")
    
    print(f"Completed {min(args.n_samples, len(dataset))} visualizations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize and compare RGB and RGBD model predictions")
    
    # Model paths
    parser.add_argument("--rgb_model", type=str, default=RGB_MODEL_PATH,
                      help="Path to the RGB model checkpoint")
    parser.add_argument("--rgbd_model", type=str, default=RGBD_MODEL_PATH,
                      help="Path to the RGBD model checkpoint")
    
    # Dataset paths
    parser.add_argument("--split_csv", type=str, default=TEST_CSV,
                      help="Path to the split CSV file")
    parser.add_argument("--rgb_dir", type=str, default=RGB_DIR,
                      help="Directory with RGB images")
    parser.add_argument("--mask_dir", type=str, default=MASK_DIR,
                      help="Directory with mask images")
    parser.add_argument("--depth_dir", type=str, default=DEPTH_DIR,
                      help="Directory with depth images")
                      
    # Output settings
    parser.add_argument("--output_dir", type=str, default=OUT_DIR,
                      help="Directory to save visualizations")
    parser.add_argument("--n_samples", type=int, default=5,
                      help="Number of images to visualize")
    parser.add_argument("--split_prefix", type=str, default="test_",
                      help="Prefix for the split files (e.g., 'test_', 'val_')")
    
    args = parser.parse_args()
    main(args)
