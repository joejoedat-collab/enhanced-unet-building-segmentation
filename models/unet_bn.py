
# ============================================================
# 4. Final Optimized Model
# Deep U-Net + BN + Residual ASPP + Residual Soft AG
# ============================================================
def deep_unet_5levels_bn_aspp_softag_optimized(input_size=(256, 256, 3)):
    inputs = layers.Input(input_size)

    # ================= Encoder =================
    c1 = conv_block(inputs, 64)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 128)
    p2 = layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 256)
    p3 = layers.MaxPooling2D()(c3)

    c4 = conv_block(p3, 384)
    p4 = layers.MaxPooling2D()(c4)

    c5 = conv_block(p4, 512)
    p5 = layers.MaxPooling2D()(c5)

    # ================= Bottleneck + ASPP =================
    b = conv_block(p5, 768)
    b = light_aspp_residual(b, filters=256)

    # ================= Decoder + Soft Attention Gates =================
    u1 = layers.UpSampling2D()(b)
    att5 = soft_attention_gate(c5, u1, inter_channels=256)
    u1 = layers.Concatenate()([u1, att5])
    d1 = conv_block(u1, 384)

    u2 = layers.UpSampling2D()(d1)
    att4 = soft_attention_gate(c4, u2, inter_channels=192)
    u2 = layers.Concatenate()([u2, att4])
    d2 = conv_block(u2, 384)

    u3 = layers.UpSampling2D()(d2)
    att3 = soft_attention_gate(c3, u3, inter_channels=128)
    u3 = layers.Concatenate()([u3, att3])
    d3 = conv_block(u3, 256)

    u4 = layers.UpSampling2D()(d3)
    att2 = soft_attention_gate(c2, u4, inter_channels=64)
    u4 = layers.Concatenate()([u4, att2])
    d4 = conv_block(u4, 128)

    u5 = layers.UpSampling2D()(d4)
    att1 = soft_attention_gate(c1, u5, inter_channels=32)
    u5 = layers.Concatenate()([u5, att1])
    d5 = conv_block(u5, 64)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(d5)

    return Model(
        inputs,
        outputs,
        name="deep_unet_5levels_bn_aspp_softag_optimized"
    )

model = deep_unet_5levels_bn_aspp_softag_optimized(
    input_size=(PATCH_SIZE, PATCH_SIZE, 3)
)
model.compile(
    optimizer='adam',
    loss=bce_precision_dice_loss_v2,
    metrics=[precision_metric, recall_metric, accuracy_metric]
)

