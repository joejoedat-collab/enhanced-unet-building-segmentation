# Enhanced U-Net with ASPP and Attention Gates for Building Segmentation

This repository provides the official implementation of an **Enhanced U-Net**
architecture integrating **Batch Normalization (BN)**, **Attention Gates (AG)**,
and **Atrous Spatial Pyramid Pooling (ASPP)** for **building footprint segmentation**
in high-resolution aerial imagery.

The proposed approach is evaluated on the **Massachusetts Buildings Dataset**
using a **patch-based training strategy** and demonstrates improved performance
in terms of IoU, F1 score, precision, and accuracy, while maintaining
computational efficiency.

---

## Key Contributions
- Enhanced U-Net architecture with:
  - Batch Normalization for stable and faster convergence
  - Attention Gates for selective feature refinement
  - ASPP with dilation rates **6, 12, and 18** for multi-scale context modeling
- Patch-based training and inference (256×256, stride 128)
- Precision-oriented loss design (**BCE + Dice**)
- Threshold optimization at inference time
- FLOPs-optimized design (≤ 0.05 GFLOPs) >>>>>To confirm after traning last model

---

##  Model Architecture
The final model consists of:
- A deep encoder–decoder U-Net backbone
- Attention Gates applied to skip connections
- An ASPP module placed at the bottleneck layer
- Boundary refinement prior to final prediction

The final implementation is available in: models/enhanced_unet_final.py
