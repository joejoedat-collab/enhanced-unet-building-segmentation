#Evaluation Matrix
EPS = 1e-7

def metrics_at(y_true, y_prob, t):
    y_pred = (y_prob > t).astype(np.uint8)

    tp = np.sum((y_true==1) & (y_pred==1))
    tn = np.sum((y_true==0) & (y_pred==0))
    fp = np.sum((y_true==0) & (y_pred==1))
    fn = np.sum((y_true==1) & (y_pred==0))

    iou = tp / (tp + fp + fn + EPS)
    prec = tp / (tp + fp + EPS)
    rec  = tp / (tp + fn + EPS)
    f1   = 2 * prec * rec / (prec + rec + EPS)
    acc  = (tp + tn) / (tp + tn + fp + fn + EPS)
    return iou, f1, prec, rec, acc

#Threshold
# wider, denser grid tends to yield a better threshold
THRESH_GRID = np.linspace(0.20, 0.60, 81)

y_val_prob = model.predict(x_val, verbose=0)
best = (0.5, -1)  # (t, IoU)
for t in THRESH_GRID:
    iou, *_ = metrics_at(y_val, y_val_prob, t)
    if iou > best[1]:
        best = (t, iou)
best_t = best[0]

y_test_prob = model.predict(x_test, verbose=0)
iou, f1, prec, rec, acc = metrics_at(y_test, y_test_prob, best_t)
print(f"[TEST] IoU={iou:.4f} | F1={f1:.4f} | Precision={prec:.4f} | Recall={rec:.4f} | Accuracy={acc:.4f} | thr={best_t:.3f}")
