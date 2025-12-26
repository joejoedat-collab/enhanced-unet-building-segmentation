#Tversky Loss function
def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    TP = tf.reduce_sum(y_true_f * y_pred_f)
    FP = tf.reduce_sum((1 - y_true_f) * y_pred_f)
    FN = tf.reduce_sum(y_true_f * (1 - y_pred_f))
    return 1 - (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)


#Focal Tvesky loss function
def focal_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
    tl = tversky_loss(y_true, y_pred, alpha, beta, smooth)
    return tf.pow(tl, gamma)

#Dice loss function
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    inter = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * inter + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    bce  = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    # heavier BCE reduces false positives -> precision ↑
    return 1.2 * bce + 0.8 * dice
