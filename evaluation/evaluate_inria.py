# =========================================================
# 9) FULL-TILE SLIDING INFERENCE (Gaussian blending)
# =========================================================
def gaussian_window(ps=PATCH_SIZE, sigma=0.125):
    ax = np.linspace(-1, 1, ps)
    w1d = np.exp(-0.5 * (ax/(sigma*2))**2)
    w = np.outer(w1d, w1d).astype(np.float32)
    w /= (w.max() + 1e-8)
    return w[..., None]

WIN = gaussian_window(PATCH_SIZE) if USE_GAUSSIAN_BLEND else None

def load_full_tile(fname):
    img_path = os.path.join(IMG_DIR, fname)
    msk_path = os.path.join(MASK_DIR, fname)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    msk = (msk > 127).astype(np.uint8)
    return img, msk

def predict_full_tile(model, img_rgb_01, patch_size=PATCH_SIZE, stride=SLIDE_STRIDE):
    h, w = img_rgb_01.shape[:2]

    pad_h = (patch_size - (h % patch_size)) % patch_size
    pad_w = (patch_size - (w % patch_size)) % patch_size
    img_pad = np.pad(img_rgb_01, ((0,pad_h),(0,pad_w),(0,0)), mode="reflect")

    H, W = img_pad.shape[:2]
    prob = np.zeros((H, W, 1), dtype=np.float32)
    wsum = np.zeros((H, W, 1), dtype=np.float32)

    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patch = img_pad[i:i+patch_size, j:j+patch_size, :]
            p = model.predict(patch[None, ...], verbose=0)[0].astype(np.float32)

            if WIN is not None:
                prob[i:i+patch_size, j:j+patch_size, :] += p * WIN
                wsum[i:i+patch_size, j:j+patch_size, :] += WIN
            else:
                prob[i:i+patch_size, j:j+patch_size, :] += p
                wsum[i:i+patch_size, j:j+patch_size, :] += 1.0

    prob /= (wsum + 1e-8)
    return prob[:h, :w, :]


# =========================================================
# 9.5) THRESHOLD SEARCH ON VALIDATION TILES
# =========================================================
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

# wider, denser grid tends to yield a better threshold
#THRESH_GRID = np.linspace(0.20, 0.60, 81)
THRESH_GRID = np.linspace(0.30, 0.55, 151)

# Collect all validation true masks and predicted probabilities for threshold search
all_val_y_true = []
all_val_y_prob = []
print("Collecting validation predictions for threshold search...")
for fname in sorted(val_files):
    img, gt = load_full_tile(fname)
    prob = predict_full_tile(model, img, PATCH_SIZE, SLIDE_STRIDE)
    all_val_y_true.append(gt.flatten())
    all_val_y_prob.append(prob.flatten())

y_val_combined = np.concatenate(all_val_y_true)
y_val_prob_combined = np.concatenate(all_val_y_prob)

best_iou = -1.0
best_t = 0.5  # Default initial value
print("Searching best threshold on combined validation pixels...")
for t in THRESH_GRID:
    iou, _, _, _, _ = metrics_at(y_val_combined, y_val_prob_combined, t)
    if iou > best_iou:
        best_iou = iou
        best_t = t

print(f"Best threshold found: {best_t:.3f} with IoU={best_iou:.4f}")

# Collect all test true masks and predicted probabilities for final evaluation
all_test_y_true = []
all_test_y_prob = []
print("Collecting test predictions for final evaluation...")
for fname in sorted(test_files):
    img, gt = load_full_tile(fname)
    prob = predict_full_tile(model, img, PATCH_SIZE, SLIDE_STRIDE)
    all_test_y_true.append(gt.flatten())
    all_test_y_prob.append(prob.flatten())

y_test_combined = np.concatenate(all_test_y_true)
y_test_prob_combined = np.concatenate(all_test_y_prob)

iou, f1, prec, rec, acc = metrics_at(y_test_combined, y_test_prob_combined, best_t)
print(f"[TEST] IoU={iou:.4f} | F1={f1:.4f} | Precision={prec:.4f} | Recall={rec:.4f} | Accuracy={acc:.4f} | thr={best_t:.3f}")


# =========================================================
# 10) FULL-TILE TEST EVALUATION (paper-grade)
# =========================================================
ious, f1s, accs, precs, recs = [], [], [], [], []

N_SHOW = 5
for idx, fname in enumerate(sorted(test_files)):
    img, gt = load_full_tile(fname)
    prob = predict_full_tile(model, img, PATCH_SIZE, SLIDE_STRIDE)
    pred = (prob[..., 0] > best_t).astype(np.uint8)

    tp = np.sum((gt==1) & (pred==1))
    tn = np.sum((gt==0) & (pred==0))
    fp = np.sum((gt==0) & (pred==1))
    fn = np.sum((gt==1) & (pred==0))

    iou = tp / (tp + fp + fn + EPS)
    prec = tp / (tp + fp + EPS)
    rec  = tp / (tp + fn + EPS)
    f1   = 2 * prec * rec / (prec + rec + EPS)
    acc  = (tp + tn) / (tp + tn + fp + fn + EPS)

    ious.append(iou); f1s.append(f1); accs.append(acc); precs.append(prec); recs.append(rec)

    if idx < N_SHOW:
        plt.figure(figsize=(15,5))
        plt.subplot(1,3,1); plt.imshow(img); plt.title(f"Image: {fname}"); plt.axis("off")
        plt.subplot(1,3,2); plt.imshow(gt, cmap="gray"); plt.title("GT"); plt.axis("off")
        plt.subplot(1,3,3); plt.imshow(pred, cmap="gray"); plt.title(f"Pred (IoU={iou:.3f}, thr={best_t:.2f})"); plt.axis("off")
        plt.tight_layout()
        plt.show()

print("\n[TEST - FULL TILE RESULTS]")
print(f"Mean IoU       : {np.mean(ious):.4f}")
print(f"Mean F1        : {np.mean(f1s):.4f}")
print(f"Mean Accuracy  : {np.mean(accs):.4f}")
print(f"Mean Precision : {np.mean(precs):.4f}")
print(f"Mean Recall    : {np.mean(recs):.4f}")
print(f"Patch={PATCH_SIZE}, Stride={SLIDE_STRIDE}, GaussianBlend={USE_GAUSSIAN_BLEND}")

