#ASPP Block
def aspp_block(x, filters=2048):
    """
    Atrous Spatial Pyramid Pooling (ASPP) block.
    x: input feature map (e.g., bottleneck feature, 8x8 for 256x256 input with 5 pools)
    filters: number of filters in each ASPP branch and final projection.
    """
    # Branch 1: 1x1 conv
    b1 = layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
    b1 = layers.BatchNormalization()(b1)
    b1 = layers.Activation('relu')(b1)

    # Branch 2: 3x3 conv, dilation=6
    b2 = layers.Conv2D(filters, 3, padding='same', dilation_rate=6, use_bias=False)(x)
    b2 = layers.BatchNormalization()(b2)
    b2 = layers.Activation('relu')(b2)

    # Branch 3: 3x3 conv, dilation=12
    b3 = layers.Conv2D(filters, 3, padding='same', dilation_rate=12, use_bias=False)(x)
    b3 = layers.BatchNormalization()(b3)
    b3 = layers.Activation('relu')(b3)

    # Branch 4: 3x3 conv, dilation=18
    b4 = layers.Conv2D(filters, 3, padding='same', dilation_rate=18, use_bias=False)(x)
    b4 = layers.BatchNormalization()(b4)
    b4 = layers.Activation('relu')(b4)

    # Branch 5: Image-level features (Global Average Pooling)
    gap = layers.GlobalAveragePooling2D()(x)                  # (B, C)
    gap = layers.Reshape((1, 1, gap.shape[-1]))(gap)          # (B, 1, 1, C)
    gap = layers.Conv2D(filters, 1, padding='same', use_bias=False)(gap)
    gap = layers.BatchNormalization()(gap)
    gap = layers.Activation('relu')(gap)

    # Upsample GAP branch back to feature map size
    h, w = x.shape[1], x.shape[2]
    gap = layers.UpSampling2D(size=(h, w), interpolation='bilinear')(gap)

    # Concatenate all branches
    y = layers.Concatenate()([b1, b2, b3, b4, gap])

    # Project back to 'filters' channels
    y = layers.Conv2D(filters, 1, padding='same', use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)

    return y


# Deep U-Net ASPP
def deep_unet_aspp(input_size=(PATCH_SIZE, PATCH_SIZE, 3)):
    inputs = layers.Input(input_size)

    # =============== ENCODER ===============
    c1 = layers.Conv2D(64, 3, padding='same')(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation('relu')(c1)
    c1 = layers.Conv2D(64, 3, padding='same')(c1)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation('relu')(c1)
    p1 = layers.MaxPooling2D()(c1)  # 256 -> 128

    c2 = layers.Conv2D(128, 3, padding='same')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Activation('relu')(c2)
    c2 = layers.Conv2D(128, 3, padding='same')(c2)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Activation('relu')(c2)
    p2 = layers.MaxPooling2D()(c2)  # 128 -> 64

    c3 = layers.Conv2D(256, 3, padding='same')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Activation('relu')(c3)
    c3 = layers.Conv2D(256, 3, padding='same')(c3)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Activation('relu')(c3)
    p3 = layers.MaxPooling2D()(c3)  # 64 -> 32

    c4 = layers.Conv2D(512, 3, padding='same')(p3)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Activation('relu')(c4)
    c4 = layers.Conv2D(512, 3, padding='same')(c4)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Activation('relu')(c4)
    p4 = layers.MaxPooling2D()(c4)  # 32 -> 16

    c5 = layers.Conv2D(1024, 3, padding='same')(p4)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Activation('relu')(c5)
    c5 = layers.Conv2D(1024, 3, padding='same')(c5)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Activation('relu')(c5)
    p5 = layers.MaxPooling2D()(c5)  # 16 -> 8

    # =============== BOTTLENECK (before ASPP) ===============
    c6 = layers.Conv2D(2048, 3, padding='same')(p5)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Activation('relu')(c6)
    c6 = layers.Conv2D(2048, 3, padding='same')(c6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Activation('relu')(c6)   # 8×8, 2048

    # =============== ASPP on bottleneck ===============
    # Reduce to 1024 channels via a 1x1 conv first (optional but helps)
    c6_reduced = layers.Conv2D(1024, 1, padding='same', use_bias=False)(c6)
    c6_reduced = layers.BatchNormalization()(c6_reduced)
    c6_reduced = layers.Activation('relu')(c6_reduced)

    aspp_out = aspp_block(c6_reduced, filters=2048)  # 8×8, 2048

    # =============== DECODER (standard U-Net style) ===============
    u7 = layers.UpSampling2D()(aspp_out)     # 8 -> 16
    u7 = layers.Concatenate()([u7, c5])
    c7 = layers.Conv2D(1024, 3, padding='same')(u7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Activation('relu')(c7)
    c7 = layers.Conv2D(1024, 3, padding='same')(c7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Activation('relu')(c7)      # 16×16

    u8 = layers.UpSampling2D()(c7)          # 16 -> 32
    u8 = layers.Concatenate()([u8, c4])
    c8 = layers.Conv2D(512, 3, padding='same')(u8)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.Activation('relu')(c8)
    c8 = layers.Conv2D(512, 3, padding='same')(c8)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.Activation('relu')(c8)      # 32×32

    u9 = layers.UpSampling2D()(c8)          # 32 -> 64
    u9 = layers.Concatenate()([u9, c3])
    c9 = layers.Conv2D(256, 3, padding='same')(u9)
    c9 = layers.BatchNormalization()(c9)
    c9 = layers.Activation('relu')(c9)
    c9 = layers.Conv2D(256, 3, padding='same')(c9)
    c9 = layers.BatchNormalization()(c9)
    c9 = layers.Activation('relu')(c9)      # 64×64

    u10 = layers.UpSampling2D()(c9)         # 64 -> 128
    u10 = layers.Concatenate()([u10, c2])
    c10 = layers.Conv2D(128, 3, padding='same')(u10)
    c10 = layers.BatchNormalization()(c10)
    c10 = layers.Activation('relu')(c10)
    c10 = layers.Conv2D(128, 3, padding='same')(c10)
    c10 = layers.BatchNormalization()(c10)
    c10 = layers.Activation('relu')(c10)    # 128×128

    u11 = layers.UpSampling2D()(c10)        # 128 -> 256
    u11 = layers.Concatenate()([u11, c1])
    c11 = layers.Conv2D(64, 3, padding='same')(u11)
    c11 = layers.BatchNormalization()(c11)
    c11 = layers.Activation('relu')(c11)
    c11 = layers.Conv2D(64, 3, padding='same')(c11)
    c11 = layers.BatchNormalization()(c11)
    c11 = layers.Activation('relu')(c11)    # 256×256

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c11)

    return Model(inputs, outputs, name="DeepUNet_ASPP")
