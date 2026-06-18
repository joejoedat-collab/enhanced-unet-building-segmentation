from tensorflow.keras.applications import MobileNetV2

def ASPP(x, filters):
    shape = x.shape
    y1 = layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
    y1 = layers.BatchNormalization()(y1)
    y1 = layers.Activation('relu')(y1)

    y2 = layers.Conv2D(filters, 3, padding='same', dilation_rate=6, use_bias=False)(x)
    y2 = layers.BatchNormalization()(y2)
    y2 = layers.Activation('relu')(y2)

    y3 = layers.Conv2D(filters, 3, padding='same', dilation_rate=12, use_bias=False)(x)
    y3 = layers.BatchNormalization()(y3)
    y3 = layers.Activation('relu')(y3)

    y4 = layers.Conv2D(filters, 3, padding='same', dilation_rate=18, use_bias=False)(x)
    y4 = layers.BatchNormalization()(y4)
    y4 = layers.Activation('relu')(y4)

    y5 = layers.GlobalAveragePooling2D()(x)
    y5 = layers.Reshape((1, 1, y5.shape[-1]))(y5)
    y5 = layers.Conv2D(filters, 1, padding='same', use_bias=False)(y5)
    y5 = layers.BatchNormalization()(y5)
    y5 = layers.Activation('relu')(y5)
    y5 = layers.Resizing(shape[1], shape[2])(y5)

    y = layers.Concatenate()([y1, y2, y3, y4, y5])
    y = layers.Conv2D(filters, 1, padding='same', use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)

    return y

def DeepLabV3Plus(input_shape=(256, 256, 3), num_classes=1):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    # Extract feature maps
    image_features = base_model.get_layer('block_13_expand_relu').output
    x = ASPP(image_features, 256)

    # Upsample
    x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)

    # Low-level features
    low_level_features = base_model.get_layer('block_3_expand_relu').output
    low_level_features = layers.Conv2D(48, 1, padding='same', use_bias=False)(low_level_features)
    low_level_features = layers.BatchNormalization()(low_level_features)
    low_level_features = layers.Activation('relu')(low_level_features)

    x = layers.Concatenate()([x, low_level_features])
    x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Final upsample
    x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    output = layers.Conv2D(num_classes, 1, activation='sigmoid')(x)

    return Model(inputs=base_model.input, outputs=output)
    model = DeepLabV3Plus()
    model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[precision_metric, recall_metric, accuracy_metric])
