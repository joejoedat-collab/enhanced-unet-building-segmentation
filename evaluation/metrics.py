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

