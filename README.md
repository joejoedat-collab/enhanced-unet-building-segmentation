
# Enhanced U-Net with ASPP and Attention Gates for Building Segmentation

This repository provides the official implementation of an **Enhanced U-Net**
architecture integrating **Batch Normalization (BN)**, **Attention Gates (AG)**,
and **Atrous Spatial Pyramid Pooling (ASPP)** for **building footprint segmentation**
in high-resolution aerial imagery.

The proposed approach is evaluated on the **Massachusetts Buildings and Inria Datasets**
using a **patch-based training strategy** and demonstrates improved performance
in terms of IoU, F1 score, precision, and accuracy, while maintaining
computational efficiency.

---

## Key Contributions
- Enhanced U-Net architecture with:
  - Batch Normalization for stable and faster convergence
  - Attention Gates for selective feature refinement
  - ASPP with dilation rates **3, 6, and 9** for multi-scale context modeling
- Patch-based training and inference (256×256, stride 128) on the Massachusetts Building Dataset
- Patch-based training and inference (512×512, stride 256) on the Inria Building Dataset
- Precision-oriented loss design (**BCE + Dice**)
- Threshold optimization at inference time


---

##  Model Architecture
The final model consists of:
- A deep encoder–decoder U-Net backbone
- Attention Gates applied to skip connections
- An ASPP module placed at the bottleneck layer
- Boundary refinement prior to final prediction

The final implementation is available in: models/enhanced_unet_final.py

---
##  The directories are structured as seen below
enhanced-unet-building-segmentation/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── models/
│   └── enhanced_unet.py         
│
├── losses/
│   └── losses.py    Dice, BCE+Dice, Tversky
│
├── data/
│   └── README.md           dataset instructions only
│
├── training/
│   ├── train_patch_based.py   training loop
│   └── callbacks.py
│
├── evaluation/
│   ├── metrics.py
│   ├── threshold_search.py
│   └── flops.py
│
├── inference/
│   └── predict_full_image.py
│
├── experiments/
│   └── ablation_table.md
│
└── figures/
    └── architecture.png



