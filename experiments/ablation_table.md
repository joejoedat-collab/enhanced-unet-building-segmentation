# Ablation Study Results

This table summarizes the quantitative performance of the baseline U-Net model
and its successive enhancements evaluated on the **Massachusetts Buildings Dataset**.
The ablation study investigates the impact of **Batch Normalization (BN)**,
**network depth**, **Attention Gates (AG)**, **Atrous Spatial Pyramid Pooling (ASPP)**,
and **loss function selection**.

---

## Quantitative Results

| Model Variant | IoU | F1 Score | Precision | Recall | Accuracy | Loss |
|--------------|-----|----------|-----------|--------|----------|------|
| U-Net (baseline) | 0.670 | 0.802 | 0.792 | 0.813 | 0.926 | BCE |
| U-Net (baseline) + BN | 0.717 | 0.822 | 0.825 | 0.860 | 0.940 | BCE |
| Deep U-Net + BN | 0.725 | 0.840 | 0.826 | 0.856 | 0.940 | BCE |
| Deep U-Net + BN + AG | 0.727 | 0.842 | 0.824 | 0.861 | 0.940 | BCE |
| Deep U-Net + BN + ASPP | 0.724 | 0.840 | 0.829 | 0.851 | 0.940 | BCE |
| Deep U-Net + BN + AG + ASPP | 0.730 | 0.844 | 0.832 | 0.856 | 0.941 | BCE |
| Deep U-Net + BN + AG + ASPP | 0.734 | 0.847 | 0.837 | 0.857 | 0.942 | Dice |
| Deep U-Net + BN + AG + ASPP | 0.712 | 0.832 | **0.891** | 0.781 | 0.942 | Tversky |
| Deep U-Net + BN + AG + ASPP | 0.708 | 0.829 | 0.883 | 0.782 | 0.940 | Focal Tversky |
| **Deep U-Net + BN + AG + ASPP (Enhanced U-Net)** | **0.744** | **0.853** | 0.843 | **0.863** | **0.954** | Dice + BCE |

---

## Key Observations

- Batch Normalization significantly improves convergence stability and accuracy.
- Increasing network depth yields consistent gains in IoU and F1 score.
- Attention Gates enhance recall by improving feature selectivity.
- ASPP contributes to better multi-scale context modeling, improving IoU.
- Loss function choice strongly affects the precision–recall trade-off.
- The **Enhanced U-Net with Dice + BCE loss** achieves the best overall performance.

---

## Notes

- All models were trained using the same patch-based protocol (256×256, stride 128).
- The reported results correspond to the optimal decision threshold selected on the validation set.

