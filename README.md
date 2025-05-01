# RGB-D Pavement Defect Segmentation

> Fuse RGB images with monocular depth estimates to segment pavement defects  
> (cracks, potholes) using a U-Net in PyTorch.

---

## Overview

This project implements a complete pipeline for multimodal pavement-defect segmentation:

1. **Data acquisition**  
   • Download KITTI-Road RGB frames and corresponding binary road/ego-lane masks.  
2. **Depth generation**  
   • Run MiDaS to synthesize monocular depth maps from RGB.  
3. **Preprocessing**  
   • Clean & rename all images/masks.  
   • Resize, normalize, and split into train/val/test sets.  
4. **Modeling**  
   • Define a U-Net that fuses RGB + depth inputs.  
   • Train with BCE+Logits loss and Adam optimizer.  
5. **Evaluation & Visualization**  
   • Compute IoU, precision, recall on held-out test set.  
   • Save mask overlays and TensorBoard logs.

---
## Project Structure
```
.
├── data/
│   ├── processed/          # Processed data ready for training
│   │   ├── depth/          # Generated depth maps
│   │   ├── mask/           # Binary segmentation masks
│   │   └── rgb/            # RGB images
│   ├── raw/                # Original raw data
│   ├── raw_clean/          # Cleaned raw data
│   ├── raw_masks/          # Original masks
│   ├── raw_masks_clean/    # Cleaned masks
│   └── splits/             # Train/val/test splits
├── models/                 # Saved model checkpoints
│   ├── best_unet_rgb.pth   # RGB-only model
│   └── best_unet_rgbd.pth  # RGBD model
├── notebooks/              # Jupyter notebooks
├── scripts/                # Utility scripts
├── src/
│   ├── data/               # Dataset implementation
│   ├── data_preprocessing/ # Data processing scripts
│   ├── evaluation/         # Evaluation & visualization
│   ├── models/             # Model architectures
│   └── training/           # Training scripts
├── visualizations/         # Model prediction visualizations
└── requirements.txt        # Project dependencies
```

## Setup
1. Clone this repo
2. Run `pip install -r requirements.txt`

## 📝 Summary of Results

We compared two variants of our U-Net segmentation model on the test split:

1. **RGB-only (3-channel)**
   - Mean Intersection-over-Union (IoU): **0.9907 ± 0.0093**  
   - Mean Dice score: **0.9952 ± 0.0048**

2. **RGB-D (4-channel fusion)**
   - Mean Intersection-over-Union (IoU): **0.9888 ± 0.0112**  
   - Mean Dice score: **0.9942 ± 0.0058**

> **Key takeaways:**  
> - Both models achieve excellent performance with IoU and Dice scores above 0.98
> - The RGB-only model shows slightly better performance by the metrics
> - Generated depth maps using MiDaS provide an additional modality for analysis
> - Visual comparisons of both models are available in the visualizations directory
> - Future work could explore different depth estimation methods or fusion strategies

These findings suggest that while monocular depth estimation adds an interesting dimension to the analysis, our current RGB-only model already achieves highly accurate segmentation results for this task.

