#Attention Gates
def attention_gate(x, g, inter_channels):
    """
    x = skip connection (high-resolution features)
    g = gating signal (decoder features, lower resolution)
    inter_channels = number of intermediate channels
    """

    theta_x = layers.Conv2D(inter_channels, 1, padding='same', use_bias=False)(x)
    theta_x = layers.BatchNormalization()(theta_x)

    phi_g = layers.Conv2D(inter_channels, 1, padding='same', use_bias=False)(g)
    phi_g = layers.BatchNormalization()(phi_g)

    add = layers.Add()([theta_x, phi_g])
    act = layers.Activation('relu')(add)

    psi = layers.Conv2D(1, 1, padding='same')(act)
    psi = layers.Activation('sigmoid')(psi)

    # Multiply attention map with skip connection
    y = layers.Multiply()([x, psi])
    return y
  
# ASPP Block
def aspp_block(x, filters=1024):

    b1 = layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
    b1 = layers.BatchNormalization()(b1)
    b1 = layers.Activation('relu')(b1)

    b2 = layers.Conv2D(filters, 3, padding='same', dilation_rate=6, use_bias=False)(x)
    b2 = layers.BatchNormalization()(b2)
    b2 = layers.Activation('relu')(b2)

    b3 = layers.Conv2D(filters, 3, padding='same', dilation_rate=12, use_bias=False)(x)
    b3 = layers.BatchNormalization()(b3)
    b3 = layers.Activation('relu')(b3)

    b4 = layers.Conv2D(filters, 3, padding='same', dilation_rate=18, use_bias=False)(x)
    b4 = layers.BatchNormalization()(b4)
    b4 = layers.Activation('relu')(b4)

    # Global context branch
    gap = layers.GlobalAveragePooling2D()(x)
    gap = layers.Reshape((1, 1, gap.shape[-1]))(gap)
    gap = layers.Conv2D(filters, 1, padding='same', use_bias=False)(gap)
    gap = layers.BatchNormalization()(gap)
    gap = layers.Activation('relu')(gap)
    h, w = x.shape[1], x.shape[2]
    gap = layers.UpSampling2D(size=(h, w), interpolation='bilinear')(gap)

    y = layers.Concatenate()([b1, b2, b3, b4, gap])
    y = layers.Conv2D(filters, 1, padding='same', use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)

    return y

# Deep U-Net Attention ASPP
def deep_unet_aspp_att(input_size=(PATCH_SIZE, PATCH_SIZE, 3)):
    inputs = layers.Input(input_size)

    # ================== ENCODER ==================
    c1 = conv_block(inputs, 64)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 128)
    p2 = layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 256)
    p3 = layers.MaxPooling2D()(c3)

    c4 = conv_block(p3, 512)
    p4 = layers.MaxPooling2D()(c4)

    c5 = conv_block(p4, 1024)
    p5 = layers.MaxPooling2D()(c5)

    # ================== BOTTLENECK ==================
    c6 = conv_block(p5, 512)          # 8×8 resolution
    c6 = layers.Conv2D(512, 1, padding='same', use_bias=False)(c6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Activation('relu')(c6)

    # ASPP applied here
    aspp = aspp_block(c6, filters=1024)

    # ================== DECODER ==================

    # ---- Level 1: c5 skip ----
    u7 = layers.UpSampling2D()(aspp)
    att5 = attention_gate(c5, u7, 512)
    u7 = layers.Concatenate()([u7, att5])
    c7 = conv_block(u7, 1024)

    # ---- Level 2: c4 skip ----
    u8 = layers.UpSampling2D()(c7)
    att4 = attention_gate(c4, u8, 256)
    u8 = layers.Concatenate()([u8, att4])
    c8 = conv_block(u8, 512)

    # ---- Level 3: c3 skip ----
    u9 = layers.UpSampling2D()(c8)
    att3 = attention_gate(c3, u9, 128)
    u9 = layers.Concatenate()([u9, att3])
    c9 = conv_block(u9, 256)

    # ---- Level 4: c2 skip ----
    u10 = layers.UpSampling2D()(c9)
    att2 = attention_gate(c2, u10, 64)
    u10 = layers.Concatenate()([u10, att2])
    c10 = conv_block(u10, 128)

    # ---- Level 5: c1 skip ----
    u11 = layers.UpSampling2D()(c10)
    att1 = attention_gate(c1, u11, 32)
    u11 = layers.Concatenate()([u11, att1])
    c11 = conv_block(u11, 64)

    # ================== OUTPUT LAYER ==================
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c11)

    return Model(inputs, outputs, name="DeepUNet_ASPP_Attention")
