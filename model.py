from tensorflow.python.keras import backend as K
from tensorflow.python.keras.optimizer_v2.adam import Adam
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input,Conv2D, concatenate, Conv2DTranspose, UpSampling2D
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.layers.core import Activation
from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
def dice_coef_loss_fun(y_true, y_pred):
    p0 = y_pred  # 预测为类别i的像素
    p1 = 1 - y_pred  # 预测不为类别i的像素
    g0 = y_true
    g1 = 1 - y_true
    # 求得每个sample的每个类的dice
    fn = K.sum(p1 * g0, axis=(1, 2))
    fp = K.sum(p0 * g1, axis=(1, 2))
    tp = K.sum(p0 * g0, axis=(1, 2))
    den = tp + 0.5 * fp + 0.5 * fn + 1e-6
    dices = tp / den  #[batchsize, class_num]
    dices = K.mean(dices, axis=0)  #所有类别dice求平均的dice
    return 1 - K.mean(dices)

def build_model(classes = 5,
                target_size = (512,512),
                img_channel = 3,
                learning_rate = 0.0001,
                loss_f = None):
    h, w = target_size
    activation = 'softmax'  #多分类任务使用softmax
    def conv2d_BN(inputs, channel_size, kernel_size, strides=1, active=True, norm=True):
        x = Conv2D(channel_size, kernel_size, padding='same', strides=strides, kernel_initializer='he_normal')(inputs)
        if norm:
            x = BatchNormalization(momentum=0.99)(x)
        if active == True:
            x = Activation('relu')(x)
        return x
    def conv2dT_BN(inputs, channel_size, kernel_size, strides=2, active=True):
        #x = UpSampling2D(size=(2,2))(inputs)
        x = Conv2DTranspose(channel_size, kernel_size, padding='same', strides=strides, kernel_initializer='he_normal')(inputs)
        x = BatchNormalization(momentum=0.99)(x)
        if active == True:
            x = Activation('relu')(x)
        return x
    inputs = Input((h, w, img_channel))
    conv1 = conv2d_BN(inputs, 32, 3)
    conv1 = conv2d_BN(conv1, 32, 3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv2d_BN(pool1, 64, 3)
    conv2 = conv2d_BN(conv2, 64, 3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv2d_BN(pool2, 128, 3)
    conv3 = conv2d_BN(conv3, 128, 3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv2d_BN(pool3, 256, 3)
    conv4 = conv2d_BN(conv4, 256, 3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = conv2d_BN(pool4, 512, 3)
    conv5 = conv2d_BN(conv5, 512, 3)

    up6 = concatenate([conv2dT_BN(conv5, 256, 2), conv4], axis=3)
    conv6 = conv2d_BN(up6, 256, 3)
    conv6 = conv2d_BN(conv6, 256, 3)

    up7 = concatenate([conv2dT_BN(conv6, 128, 2), conv3], axis=3)
    conv7 = conv2d_BN(up7, 128, 3)
    conv7 = conv2d_BN(conv7, 128, 3)

    up8 = concatenate([conv2dT_BN(conv7, 64, 2), conv2], axis=3)
    conv8 = conv2d_BN(up8, 64, 3)
    conv8 = conv2d_BN(conv8, 64, 3)

    up9 = concatenate([conv2dT_BN(conv8, 32, 2), conv1], axis=3)
    conv9 = conv2d_BN(up9, 32, 3)
    conv9 = conv2d_BN(conv9, 32, 3)

    conv10 = Conv2D(classes, (1, 1), activation=activation)(conv9)

    model = Model(inputs, conv10)
    print(model.summary())




    """
    Optimizer config：

    """
    if learning_rate != None:
        lr = learning_rate
    else:
        lr = 0.0001
    opt = Adam(learning_rate=lr)

    """
    Loss functions:

    """


    if loss_f == 'diceloss':
        loss = dice_coef_loss_fun
    elif loss_f == 'CE':
        loss = 'categorical_crossentropy'
    return model, opt, loss
