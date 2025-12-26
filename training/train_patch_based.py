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

#  --- Callback - Helpful: show the LR each epoch ---
class LrPrinter(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = (self.model.optimizer.learning_rate.numpy()
              if not hasattr(self.model.optimizer.learning_rate, '__call__')
              else tf.keras.backend.get_value(self.model.optimizer.learning_rate(self.model.optimizer.iterations)))
        print(f"\n[Epoch {epoch+1}] learning_rate={lr:.6g}")

log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

callbacks = [
    # 1) Reduce LR when val_loss stalls (stabilizes training)
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', patience=4, factor=0.5, min_lr=1e-6, verbose=1
    ),
    # 2) Save the best model by val_loss
    tf.keras.callbacks.ModelCheckpoint(
        filepath='best_by_valloss.keras', monitor='val_loss',
        save_best_only=True, save_weights_only=False, verbose=1
    ),
    # 3) TensorBoard logs (scalars, graphs)
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0),
    # 4) CSV log of metrics per epoch (handy for plots later)
    tf.keras.callbacks.CSVLogger('training_log.csv', append=True),
    # 5) Auto-resume training state if the session restarts
    tf.keras.callbacks.BackupAndRestore(backup_dir='./backup'),
    # 6) Print LR at the end of each epoch
    LrPrinter(),
]
print(f"[TensorBoard] run: {log_dir}")

# Image Generator
image_generator = image_datagen.flow(x_train, batch_size=BATCH_SIZE, seed=SEED)
mask_generator = mask_datagen.flow(y_train, batch_size=BATCH_SIZE, seed=SEED)
train_generator = zip(image_generator, mask_generator)

#Model compilation
precision_metric = tf.keras.metrics.Precision(name='precision', thresholds=0.5)
recall_metric    = tf.keras.metrics.Recall(name='recall', thresholds=0.5)
accuracy_metric  = tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.5)

model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[precision_metric, recall_metric, accuracy_metric])

 # Model Tranining
def combined_generator(img_gen, mask_gen):
    while True:
        yield next(img_gen), next(mask_gen)

train_generator = combined_generator(image_generator, mask_generator)

steps_per_epoch = max(1, len(x_train) // BATCH_SIZE)
history = model.fit(
    train_generator,
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    callbacks=callbacks,
    verbose=1
)


