
#Tversky Loss
def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    TP = tf.reduce_sum(y_true_f * y_pred_f)
    FP = tf.reduce_sum((1 - y_true_f) * y_pred_f)
    FN = tf.reduce_sum(y_true_f * (1 - y_pred_f))
    return 1 - (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    return 1 - (2. * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )

def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-6):
    """
    alpha penalizes false positives.
    beta penalizes false negatives.

    Since your precision is low, alpha should be higher than beta.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fp = tf.reduce_sum((1 - y_true_f) * y_pred_f)
    fn = tf.reduce_sum(y_true_f * (1 - y_pred_f))

    return 1 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)


def dice_tversky_loss(y_true, y_pred):
    dice = dice_loss(y_true, y_pred)
    tv = tversky_loss(y_true, y_pred, alpha=0.6, beta=0.4)
    return 0.7 * dice + 0.3 * tv


def precision_dice_loss(y_true, y_pred, fp_weight=1.2, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fp = tf.reduce_sum((1 - y_true_f) * y_pred_f)
    fn = tf.reduce_sum(y_true_f * (1 - y_pred_f))

    return 1 - (2 * tp + smooth) / (2 * tp + fp_weight * fp + fn + smooth)


def bce_precision_dice_loss_v2(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    pdice = precision_dice_loss(y_true, y_pred, fp_weight=1.2)

    return 0.7 * bce + 1.3 * pdice



