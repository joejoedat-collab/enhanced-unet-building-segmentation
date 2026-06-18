# ---------------- HRNET ----------------
def conv_bn_relu(x, filters, kernel_size=3, strides=1, dilation=1):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same', dilation_rate=dilation, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def basic_block(x, filters):
    shortcut = x
    x = conv_bn_relu(x, filters)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.ReLU()(x)
    return x

def fuse_layers(branches, filters):
    fused = []
    for i in range(len(branches)):
        y = branches[i]
        for j in range(len(branches)):
            if j > i:
                yj = layers.Conv2D(filters[i], 1, use_bias=False)(branches[j])
                yj = layers.BatchNormalization()(yj)
                yj = layers.UpSampling2D(size=2 ** (j - i), interpolation='bilinear')(yj)
                y = layers.Add()([y, yj])
            elif j < i:
                yj = layers.Conv2D(filters[i], 3, strides=2 ** (i - j), padding='same', use_bias=False)(branches[j])
                yj = layers.BatchNormalization()(yj)
                y = layers.Add()([y, yj])
        y = layers.ReLU()(y)
        fused.append(y)
    return fused

def build_hrnet(input_shape=(256, 256, 3), num_classes=1):
    inputs = Input(shape=input_shape)
    x = conv_bn_relu(inputs, 64, 3, strides=2)
    x = conv_bn_relu(x, 64, 3, strides=2)
    x = basic_block(x, 64)
    x1 = conv_bn_relu(x, 32)
    x2 = conv_bn_relu(layers.AveragePooling2D(2)(x), 64)
    branches = [x1, x2]
    filters = [32, 64]
    for _ in range(3):
        branches = [basic_block(b, f) for b, f in zip(branches, filters)]
        branches = fuse_layers(branches, filters)
    upsampled = [branches[0], layers.UpSampling2D(size=2, interpolation='bilinear')(branches[1])]
    x = layers.Concatenate()(upsampled)
    x = conv_bn_relu(x, 128)
    x = conv_bn_relu(x, 128)
    x = layers.Conv2D(num_classes, 1, activation='sigmoid')(x)
    x = layers.UpSampling2D(size=4, interpolation='bilinear')(x)
    return Model(inputs, x)
model = build_hrnet()
model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[precision_metric, recall_metric, accuracy_metric])
