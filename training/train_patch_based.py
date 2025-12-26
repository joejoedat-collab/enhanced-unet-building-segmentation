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

# --- AUGMENTATION ---
image_datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
                                   zoom_range=0.1, horizontal_flip=True)
mask_datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
                                  zoom_range=0.1, horizontal_flip=True,
                                  preprocessing_function=lambda x: (x > 0.5).astype(np.float32))

image_generator = image_datagen.flow(x_train, batch_size=BATCH_SIZE, seed=SEED)
mask_generator = mask_datagen.flow(y_train, batch_size=BATCH_SIZE, seed=SEED)
train_generator = zip(image_generator, mask_generator)

#Model compilation
precision_metric = tf.keras.metrics.Precision(name='precision', thresholds=0.5)
recall_metric    = tf.keras.metrics.Recall(name='recall', thresholds=0.5)
accuracy_metric  = tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.5)

model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[precision_metric, recall_metric, accuracy_metric])

