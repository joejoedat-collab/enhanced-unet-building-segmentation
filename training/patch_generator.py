# --- AUGMENTATION ---
image_datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
                                   zoom_range=0.1, horizontal_flip=True)
mask_datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
                                  zoom_range=0.1, horizontal_flip=True,
                                  preprocessing_function=lambda x: (x > 0.5).astype(np.float32))
#----Patch Generator
image_generator = image_datagen.flow(x_train, batch_size=BATCH_SIZE, seed=SEED)
mask_generator = mask_datagen.flow(y_train, batch_size=BATCH_SIZE, seed=SEED)
train_generator = zip(image_generator, mask_generator)
