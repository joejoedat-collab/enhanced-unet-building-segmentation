
## Table 1. ASPP Dilation Rate Evaluation on the Massachusetts Buildings Test Set

| Model Variant              | IoU       | F1 Score  | Precision | Recall    | Accuracy  | Loss Function | Dilation Rates |
| -------------------------- | --------- | --------- | --------- | --------- | --------- | ------------- | -------------- |
| Deep U-Net + BN + ASPP     | 0.729     | 0.843     | 0.832     | 0.854     | 0.941     | BCE           | 6,12,18        |
| **Deep U-Net + BN + ASPP** | **0.731** | **0.845** | **0.829** | **0.861** | **0.941** | **BCE**       | **3,6,9**      |
| Deep U-Net + BN + ASPP     | 0.727     | 0.842     | 0.828     | 0.856     | 0.940     | BCE           | 4,8,12         |
| Deep U-Net + BN + ASPP     | 0.720     | 0.837     | 0.821     | 0.853     | 0.938     | BCE           | 2,4,6          |

**Best Dilation Configuration:** ASPP dilation rates of **3, 6, and 9** achieved the highest IoU (**0.731**) and F1-score (**0.845**), indicating a better balance between local feature preservation and multi-scale contextual information extraction. Smaller dilation rates (2,4,6) produced the lowest performance, while larger dilation rates (6,12,18) provided competitive results but slightly lower generalization on the test set.

