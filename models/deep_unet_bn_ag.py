#Deep U-Net Batch normalization + Attention Gates
def deep_unet_att(input_size=(PATCH_SIZE, PATCH_SIZE, 3)):
    inputs = layers.Input(input_size)

    # =============== ENCODER ===============
    c1 = layers.Conv2D(64, 3, padding='same')(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation('relu')(c1)
    c1 = layers.Conv2D(64, 3, padding='same')(c1)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation('relu')(c1)
    p1 = layers.MaxPooling2D()(c1)          # 256 -> 128

    c2 = layers.Conv2D(128, 3, padding='same')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Activation('relu')(c2)
    c2 = layers.Conv2D(128, 3, padding='same')(c2)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Activation('relu')(c2)
    p2 = layers.MaxPooling2D()(c2)          # 128 -> 64

    c3 = layers.Conv2D(256, 3, padding='same')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Activation('relu')(c3)
    c3 = layers.Conv2D(256, 3, padding='same')(c3)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Activation('relu')(c3)
    p3 = layers.MaxPooling2D()(c3)          # 64 -> 32

    c4 = layers.Conv2D(512, 3, padding='same')(p3)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Activation('relu')(c4)
    c4 = layers.Conv2D(512, 3, padding='same')(c4)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Activation('relu')(c4)
    p4 = layers.MaxPooling2D()(c4)          # 32 -> 16

    c5 = layers.Conv2D(1024, 3, padding='same')(p4)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Activation('relu')(c5)
    c5 = layers.Conv2D(1024, 3, padding='same')(c5)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Activation('relu')(c5)
    p5 = layers.MaxPooling2D()(c5)          # 16 -> 8

    # =============== BOTTLENECK ===============
    c6 = layers.Conv2D(2048, 3, padding='same')(p5)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Activation('relu')(c6)
    c6 = layers.Conv2D(2048, 3, padding='same')(c6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Activation('relu')(c6)      # 8×8, 2048

    # =============== DECODER + ATTENTION ===============
    # Level 5 skip (c5, 16x16, 1024)
    u7  = layers.UpSampling2D()(c6)         # 8 -> 16
    att5 = attention_gate(c5, u7, inter_channels=512)
    u7  = layers.Concatenate()([u7, att5])
    c7  = layers.Conv2D(1024, 3, padding='same')(u7)
    c7  = layers.BatchNormalization()(c7)
    c7  = layers.Activation('relu')(c7)
    c7  = layers.Conv2D(1024, 3, padding='same')(c7)
    c7  = layers.BatchNormalization()(c7)
    c7  = layers.Activation('relu')(c7)     # 16×16

    # Level 4 skip (c4, 32x32, 512)
    u8  = layers.UpSampling2D()(c7)         # 16 -> 32
    att4 = attention_gate(c4, u8, inter_channels=256)
    u8  = layers.Concatenate()([u8, att4])
    c8  = layers.Conv2D(512, 3, padding='same')(u8)
    c8  = layers.BatchNormalization()(c8)
    c8  = layers.Activation('relu')(c8)
    c8  = layers.Conv2D(512, 3, padding='same')(c8)
    c8  = layers.BatchNormalization()(c8)
    c8  = layers.Activation('relu')(c8)     # 32×32

    # Level 3 skip (c3, 64x64, 256)
    u9  = layers.UpSampling2D()(c8)         # 32 -> 64
    att3 = attention_gate(c3, u9, inter_channels=128)
    u9  = layers.Concatenate()([u9, att3])
    c9  = layers.Conv2D(256, 3, padding='same')(u9)
    c9  = layers.BatchNormalization()(c9)
    c9  = layers.Activation('relu')(c9)
    c9  = layers.Conv2D(256, 3, padding='same')(c9)
    c9  = layers.BatchNormalization()(c9)
    c9  = layers.Activation('relu')(c9)     # 64×64

    # Level 2 skip (c2, 128x128, 128)
    u10  = layers.UpSampling2D()(c9)        # 64 -> 128
    att2 = attention_gate(c2, u10, inter_channels=64)
    u10  = layers.Concatenate()([u10, att2])
    c10  = layers.Conv2D(128, 3, padding='same')(u10)
    c10  = layers.BatchNormalization()(c10)
    c10  = layers.Activation('relu')(c10)
    c10  = layers.Conv2D(128, 3, padding='same')(c10)
    c10  = layers.BatchNormalization()(c10)
    c10  = layers.Activation('relu')(c10)   # 128×128

    # Level 1 skip (c1, 256x256, 64)
    u11  = layers.UpSampling2D()(c10)       # 128 -> 256
    att1 = attention_gate(c1, u11, inter_channels=32)
    u11  = layers.Concatenate()([u11, att1])
    c11  = layers.Conv2D(64, 3, padding='same')(u11)
    c11  = layers.BatchNormalization()(c11)
    c11  = layers.Activation('relu')(c11)
    c11  = layers.Conv2D(64, 3, padding='same')(c11)
    c11  = layers.BatchNormalization()(c11)
    c11  = layers.Activation('relu')(c11)   # 256×256

    # =============== OUTPUT ===============
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c11)

    return Model(inputs, outputs, name="DeepUNet_Att")
