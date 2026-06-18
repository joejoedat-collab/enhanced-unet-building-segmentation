# ---------------- CONFIG ----------------
PATCH_SIZE = 256
STRIDE = 128
EPOCHS = 50
BATCH_SIZE = 4
SEED = 42

 #---------------- PATCH EXTRACTOR ----------------
def extract_patches(images_dir, masks_dir, patch_size=PATCH_SIZE, stride=STRIDE):
    image_patches = []
    mask_patches = []

    filenames = sorted(os.listdir(images_dir))

    for fname in filenames:
        img = cv2.imread(os.path.join(images_dir, fname))
        mask = cv2.imread(os.path.join(masks_dir, fname), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            continue

        h, w = img.shape[:2]
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                img_patch = img[i:i+patch_size, j:j+patch_size]
                mask_patch = mask[i:i+patch_size, j:j+patch_size]

                image_patches.append(img_patch)
                mask_patches.append(np.expand_dims(mask_patch, axis=-1))

    X = np.array(image_patches) / 255.0
    Y = (np.array(mask_patches) > 127).astype(np.float32)  # binarize
    return shuffle(X, Y, random_state=SEED)

