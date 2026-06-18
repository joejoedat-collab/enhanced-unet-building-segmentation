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

## Table 1. Test Results from Model Enhancement Experiments

| Model Variant               | IoU       | F1 Score  | Precision | Recall    | Accuracy  | Loss Function  | Dilation Rates |
| --------------------------- | --------- | --------- | --------- | --------- | --------- | -------------- | -------------- |
| U-Net (Baseline)            | 0.667     | 0.801     | 0.796     | 0.805     | 0.926     | BCE            | -              |
| U-Net + Batch Normalization | 0.729     | 0.843     | 0.829     | 0.857     | 0.941     | BCE            | -              |
| Deep U-Net + BN             | 0.734     | 0.847     | 0.836     | 0.858     | 0.943     | BCE            | -              |
| Deep U-Net + BN + AG        | 0.728     | 0.842     | 0.832     | 0.861     | 0.941     | BCE            | -              |
| Deep U-Net + BN + ASPP      | 0.729     | 0.843     | 0.832     | 0.854     | 0.941     | BCE            | 6,12,18        |
| Deep U-Net + BN + ASPP      | 0.731     | 0.845     | 0.829     | 0.861     | 0.941     | BCE            | 3,6,9          |
| Deep U-Net + BN + ASPP      | 0.727     | 0.842     | 0.828     | 0.856     | 0.940     | BCE            | 4,8,12         |
| Deep U-Net + BN + ASPP      | 0.720     | 0.837     | 0.821     | 0.853     | 0.938     | BCE            | 2,4,6          |
| Deep U-Net + BN + AG + ASPP | 0.734     | 0.847     | 0.829     | 0.865     | 0.942     | BCE            | 3,6,9          |
| Deep U-Net + BN + AG + ASPP | 0.736     | 0.848     | 0.837     | 0.859     | 0.943     | Dice           | 3,6,9          |
| Deep U-Net + BN + AG + ASPP | 0.709     | 0.830     | 0.888     | 0.779     | 0.941     | Tversky        | 3,6,9          |
| Deep U-Net + BN + AG + ASPP | 0.735     | 0.847     | 0.848     | 0.846     | 0.943     | Dice + Tversky | 3,6,9          |
| **Enhanced U-Net**          | **0.737** | **0.848** | **0.848** | **0.849** | **0.944** | **BCE + Dice** | **3,6,9**      |
| HRNet                       | 0.684     | 0.813     | 0.776     | 0.852     | 0.927     | BCE + Dice     | -              |
| DeepLabv3+                  | 0.691     | 0.817     | 0.811     | 0.824     | 0.932     | BCE + Dice     | -              |

**Best Performing Model:** Enhanced U-Net with BCE + Dice loss and ASPP dilation rates of 3, 6, and 9, achieving an IoU of 0.737 and F1-score of 0.848 on the Massachusetts Buildings Dataset.

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

