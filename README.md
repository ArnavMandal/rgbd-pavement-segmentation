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
├── assets
├── data
│   ├── processed
│   ├── raw
│   └── splits
├── models
├── notebooks
├── results
│   ├── metrics
│   └── predictions
└── src
    ├── data_preprocessing
    ├── evaluation
    ├── models
    └── training

16 directories
```

## Setup
1. Clone this repo
2. Run `pip install -r requirements.txt`

