import keras.backend as K
from keras.losses import binary_crossentropy

ALPHA = 0.8
GAMMA = 2


def FocalLoss(targets, inputs, alpha=ALPHA, gamma=GAMMA):
    """
    Focal loss focuses on the examples that the model gets wrong rather than the ones that it can confidently predict,
     ensuring that predictions on hard examples improve over time rather than becoming overly confident with easy ones

    :param targets: target values
    :param inputs: model predictions
    :param alpha: weighting factor (default ALPHA)
    :param gamma: power factor (default GAMMA)
    :return: focal_loss -- Focal Loss value
    """
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    BCE = K.binary_crossentropy(targets, inputs)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(alpha * K.pow((1-BCE_EXP), gamma) * BCE)

    return focal_loss


def DiceBCELoss(targets, inputs, smooth=1e-6):
    """
    Calculates a loss function that combines Dice Loss and BCE Loss.

    :param targets: target values
    :param inputs: model predictions
    :param smooth: smoothing constant (default 1e-6)
    :return: Dice_BCE -- Dice-BCE Loss value
    """

    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    inputs = K.expand_dims(inputs)
    targets = K.expand_dims(targets)

    BCE =  binary_crossentropy(targets, inputs)
    intersection = K.dot(K.transpose(targets), inputs)
    dice_loss = 1 - (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    Dice_BCE = BCE + dice_loss

    return Dice_BCE

def dice_p_bce(in_gt, in_pred):
    """
    Calculates the combined loss function of Dice Loss and BCE Loss separately.

    :param in_gt: target values
    :param in_pred: model predictions
    """
    return 1e-3*binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)


def DiceLoss(targets, inputs, smooth=1e-6):
    """
    Dice loss considers the loss information both locally and globally, which is critical for high accuracy.

    :param targets: target values
    :param inputs: model predictions
    :param smooth: Parameter to prevent division by zero
    """
    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    inputs = K.expand_dims(inputs)
    targets = K.expand_dims(targets)

    intersection = K.dot(K.transpose(targets), inputs)
    dice = (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return 1 - dice


def dice_coef(y_true, y_pred, smooth=1):
    """
    It is used to calculate the similarity between two images
    """
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice
