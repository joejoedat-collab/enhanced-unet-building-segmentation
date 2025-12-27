
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

---
Enhanced-UNet-Building-Segmentation/
│
├── README.md                    # Main documentation
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT or Apache-2.0
│
├── data/
│   └── README.md                # Dataset download instructions (NO data files)
│
├── models/
│   ├── unet_baseline.py
│   ├── unet_bn.py
│   ├── deep_unet_bn.py
│   ├── deep_unet_bn_ag.py
│   ├── deep_unet_bn_aspp.py
│   ├── deep_unet_bn_ag_aspp.py
│   └── enhanced_unet_final.py   # ⭐ YOUR FINAL MODEL (the one you attached)
│
├── training/
│   ├── train_baseline.py
│   ├── train_enhanced.py
│   └── callbacks.py
│
├── evaluation/
│   ├── metrics.py
│   ├── threshold_search.py
│   └── flops_counter.py
│
├── experiments/
│   └── ablation_results.csv     # Table used in your paper
│
├── figures/
│   ├── architecture.png
│   ├── qualitative_results.png
│   └── training_curves.png
│
└── scripts/
    ├── predict_full_image.py
    └── visualize_results.py



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



