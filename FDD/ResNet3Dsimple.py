from keras import Model
from keras.layers import Input, Conv3D, BatchNormalization, Dropout, GlobalAveragePooling3D, Add
from keras.layers import Dense, Activation, MaxPooling3D, Concatenate, AveragePooling3D
from keras.regularizers import l2
from keras.utils import plot_model
import random

def ResNet50(input_shape=None, res_blocks=4, res_layers = [3, 4, 6, 3], nb_classes=None, weight_decay=1e-4):

    img_input = Input(shape=input_shape)

    # Stage 1
    nb_channels = 64
    x = Conv3D(nb_channels, (3, 7, 7), padding='same', strides=(1, 2, 2),
               use_bias=False, kernel_regularizer=l2(weight_decay))(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D((1,3,3), strides=(1,2,2), padding='same')(x)

    # Stage 2
    x = convolution_block(x, [64,64])
    x = identity_block(x, [64,64])
    x = identity_block(x, [64, 64])

    # Stage 3
    x = convolution_block(x, [128, 128], s=2)
    x = identity_block(x, [128, 128])
    x = identity_block(x, [128, 128])
    x = identity_block(x, [128, 128])

    # Stage 4
    x = convolution_block(x, [256, 256], s=2)
    x = identity_block(x, [256, 256])
    x = identity_block(x, [256, 256])
    x = identity_block(x, [256, 256])
    x = identity_block(x, [256, 256])
    x = identity_block(x, [256, 256])

    # Stage 5
    x = convolution_block(x, [512, 512], s=2)
    x = identity_block(x, [512, 512])
    x = identity_block(x, [512, 512])

    x = GlobalAveragePooling3D()(x)
    out = Dense(nb_classes)(x)

    return Model(img_input, out)

def identity_block(inputs, filters):
    F1, F2 = filters
    x = inputs
    x = Conv3D(F1, kernel_size=(3,3,3), strides=(1,1,1), padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv3D(F2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)

    x = Add()([x, inputs])
    x = Activation('relu')(x)
    return x

def convolution_block(inputs, filters, s=1):
    F1, F2 = filters
    x = inputs
    x = Conv3D(F1, kernel_size=(3, 3, 3), strides=(s, s, s), padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv3D(F2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)

    inputs = Conv3D(F1, kernel_size=(3, 3, 3), strides=(s, s, s), padding='same',
               kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(inputs)
    inputs = BatchNormalization()(inputs)
    x = Add()([x, inputs])
    x = Activation('relu')(x)
    return x

# model = ResNet50(input_shape=(8,224,224,3), nb_classes=51)
# model.summary()



