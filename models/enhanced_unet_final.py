# Attention Gates
# -------------------------------------------
#1. Attention Gate (optimized)
# -------------------------------------------
def attention_gate(x, g, inter_channels):
    inter_channels = max(8, inter_channels // 2)   # REDUCED CHANNELS

    theta_x = layers.Conv2D(inter_channels, 1, padding='same', use_bias=False)(x)
    theta_x = layers.BatchNormalization()(theta_x)

    phi_g = layers.Conv2D(inter_channels, 1, padding='same', use_bias=False)(g)
    phi_g = layers.BatchNormalization()(phi_g)

    add = layers.Add()([theta_x, phi_g])
    act = layers.Activation('relu')(add)

    psi = layers.Conv2D(1, 1, padding='same')(act)
    psi = layers.Activation('sigmoid')(psi)

    y = layers.Multiply()([x, psi])
    return y

# 2. ASPP (optimized: filters 256 instead of 2048)
# -------------------------------------------
def aspp_block(x, filters=256):

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

    # Global context
    gap = layers.GlobalAveragePooling2D()(x)
    gap = layers.Reshape((1, 1, gap.shape[-1]))(gap)
    gap = layers.Conv2D(filters, 1, padding='same', use_bias=False)(gap)
    gap = layers.BatchNormalization()(gap)
    gap = layers.Activation('relu')(gap)
    gap = layers.UpSampling2D(size=(x.shape[1], x.shape[2]), interpolation='bilinear')(gap)

    y = layers.Concatenate()([b1, b2, b3, b4, gap])
    y = layers.Conv2D(filters, 1, padding='same', use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)

    return y

# 3. Standard convolution block
# -------------------------------------------
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    return x
  
# -------------------------------------------
# 4. FINAL OPTIMIZED Enhanced U-Net
# -------------------------------------------
def deep_unet_aspp_att(input_size=(PATCH_SIZE, PATCH_SIZE, 3)):
    inputs = layers.Input(input_size)

    # ----- ENCODER -----
    c1 = conv_block(inputs, 64)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 128)
    p2 = layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 256)
    p3 = layers.MaxPooling2D()(c3)

    c4 = conv_block(p3, 512)
    p4 = layers.MaxPooling2D()(c4)

    c5 = conv_block(p4, 1024)  # reduced from 1024
    p5 = layers.MaxPooling2D()(c5)

    # ----- BOTTLENECK -----
    c6 = conv_block(p5, 512)    # reduced from 2048
    c6 = layers.Conv2D(512, 1, padding='same', use_bias=False)(c6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Activation('relu')(c6)

    # ASPP (filtered to 512)
    aspp = aspp_block(c6, filters=512)

    # ----- DECODER -----
    u7 = layers.UpSampling2D()(aspp)
    att5 = attention_gate(c5, u7, 1024)
    u7 = layers.Concatenate()([u7, att5])
    c7 = conv_block(u7, 512)

    u8 = layers.UpSampling2D()(c7)
    att4 = attention_gate(c4, u8, 512)
    u8 = layers.Concatenate()([u8, att4])
    c8 = conv_block(u8, 256)

    u9 = layers.UpSampling2D()(c8)
    att3 = attention_gate(c3, u9, 256)
    u9 = layers.Concatenate()([u9, att3])
    c9 = conv_block(u9, 128)

    u10 = layers.UpSampling2D()(c9)
    att2 = attention_gate(c2, u10, 128)
    u10 = layers.Concatenate()([u10, att2])
    c10 = conv_block(u10, 64)

    u11 = layers.UpSampling2D()(c10)
    att1 = attention_gate(c1, u11, 64)
    u11 = layers.Concatenate()([u11, att1])
    c11 = conv_block(u11, 32)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c11)

    return Model(inputs, outputs, name="deep_unet_aspp_att")
