# ---------------- FULL-IMAGE PREDICTION (weighted tiles) ----------------
def _gaussian_window(ps=256, sigma=0.125):
    ax = np.linspace(-1, 1, ps)
    w1d = np.exp(-0.5 * (ax/(sigma*2))**2)
    w = np.outer(w1d, w1d).astype(np.float32)
    w /= w.max()
    return w[..., None]

def predict_full_image(model, image_path, patch_size=256, stride=128, threshold=0.5):
    # Load RGB in [0,1]
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0
    h, w = rgb.shape[:2]

    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    img_padded = np.pad(rgb, ((0,pad_h),(0,pad_w),(0,0)), mode='reflect')

    H, W = img_padded.shape[:2]
    prob = np.zeros((H, W, 1), dtype=np.float32)
    wsum = np.zeros((H, W, 1), dtype=np.float32)
    win = _gaussian_window(ps=patch_size)

    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patch = img_padded[i:i+patch_size, j:j+patch_size]
            p = model.predict(patch[None, ...], verbose=0)[0]
            prob[i:i+patch_size, j:j+patch_size] += p * win
            wsum[i:i+patch_size, j:j+patch_size] += win

    prob /= (wsum + 1e-8)
    prob = prob[:h, :w]
    return (prob > threshold).astype(np.uint8), prob


# ---------------- SAMPLE DISPLAY ----------------
def display_sample_test_image():
    sample_image_name = sorted(os.listdir(x_test_dir))[0]
    img_path = os.path.join(x_test_dir, sample_image_name)
    mask_path = os.path.join(y_test_dir, sample_image_name)

    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Sample Test Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Ground Truth Mask")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# ---------------- PATCH PREDICTION ----------------
def predict_full_image(model, image_path, patch_size=256, stride=128, threshold=0.5):
    original_img = cv2.imread(image_path)
    original_img = original_img / 255.0
    h, w = original_img.shape[:2]

    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    img_padded = np.pad(original_img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')

    new_h, new_w = img_padded.shape[:2]
    prediction_mask = np.zeros((new_h, new_w, 1))
    count_mask = np.zeros((new_h, new_w, 1))

    for i in range(0, new_h - patch_size + 1, stride):
        for j in range(0, new_w - patch_size + 1, stride):
            patch = img_padded[i:i+patch_size, j:j+patch_size]
            patch_input = np.expand_dims(patch, axis=0)
            pred = model.predict(patch_input, verbose=0)[0]
            prediction_mask[i:i+patch_size, j:j+patch_size] += pred
            count_mask[i:i+patch_size, j:j+patch_size] += 1

    prediction_mask /= count_mask
    prediction_mask = prediction_mask[:h, :w]
    prediction_mask = (prediction_mask > threshold).astype(np.uint8)
    return prediction_mask

# ---------------- MASK LOADER ----------------
def load_mask(mask_path, target_size=(1500, 1500)):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, target_size)
    return (mask > 127).astype(np.uint8)

# ---------------- EVALUATION + VISUALIZATION ----------------
def evaluate_and_visualize(model, img_dir, mask_dir, patch_size=256, stride=128, threshold=0.5, n_samples=10):
    ious, f1s, accs, precisions, recalls = [], [], [], [], []
    filenames = sorted(os.listdir(img_dir))

    print("Evaluating on full-size test images...")

    for idx, fname in enumerate(filenames):
        img_path = os.path.join(img_dir, fname)
        mask_path = os.path.join(mask_dir, fname)

        y_true_gt = load_mask(mask_path) # Renamed to avoid confusion with flattened y_true
        predicted_mask_full = predict_full_image(model, img_path, patch_size, stride, threshold) # Renamed to avoid confusion with flattened y_pred

        # Flatten the arrays
        y_true = y_true_gt.flatten()
        y_pred = predicted_mask_full.flatten()

        # Binarize prediction if not already
        y_pred = (y_pred > 0.5).astype(np.uint8)

        # Intersection and union
        intersection = np.sum(y_true * y_pred)
        union = np.sum(y_true) + np.sum(y_pred) - intersection

        # Core metrics
        iou = intersection / (union + 1e-7)
        f1 = 2 * intersection / (np.sum(y_true) + np.sum(y_pred) + 1e-7)
        acc = np.mean(y_true == y_pred)

        # Precision and Recall
        precision = np.sum((y_true == 1) & (y_pred == 1)) / (np.sum(y_pred) + 1e-7)
        recall = np.sum((y_true == 1) & (y_pred == 1)) / (np.sum(y_true) + 1e-7)

        ious.append(iou)
        f1s.append(f1)
        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)


        # Show a few examples
        if idx < n_samples:
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(img_rgb)
            plt.title("Original Image")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(y_true_gt, cmap='gray') # Use original 2D mask for display
            plt.title("Ground Truth Mask")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(predicted_mask_full, cmap='gray') # Use original 2D predicted mask for display
            plt.title("Predicted Mask")
            plt.axis('off')
            #plt.suptitle(f"{fname} - IoU:{iou:.3f} F1:{f1:.3f}", fontsize=12)
            plt.tight_layout()
            plt.show()

    print(f"\nEvaluation Results:")
    print(f"Average IoU: {np.mean(ious):.4f}")
    print(f"Average F1 Score: {np.mean(f1s):.4f}")
    print(f"Average Pixel Accuracy: {np.mean(accs):.4f}")
    print(f"Average Precision: {np.mean(precisions):.4f}")
    print(f"Average Recall: {np.mean(recalls):.4f}")

# ---------------- RUN ----------------

