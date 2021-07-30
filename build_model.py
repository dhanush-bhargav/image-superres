import tensorflow as tf
import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dense
from tensorflow.keras.layers import Add, Input, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU, PReLU, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Constant

def build_generator_net(B):
    
    input_layer = Input(shape=(24, 24, 3), dtype=tf.float32)

    x = Conv2D(filters=64, kernel_size=(9,9), strides=(1,1), padding='same')(input_layer)
    x = PReLU(alpha_initializer=Constant(0.2))(x)

    prev_x = x

    for i in range(0,B):
        next_x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same")(prev_x)
        next_x = BatchNormalization()(next_x)
        next_x = PReLU(alpha_initializer=Constant(0.2))(next_x)
        next_x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same")(next_x)
        next_x = BatchNormalization()(next_x)
        prev_x = Add()([prev_x, next_x])

    prev_x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same")(prev_x)
    prev_x = BatchNormalization()(prev_x)
    x = Add()([prev_x, x])

    x = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = tf.nn.depth_to_space(x, block_size=2, data_format="NHWC")
    x = PReLU(alpha_initializer=Constant(0.2))(x)

    x = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = tf.nn.depth_to_space(x, block_size=2, data_format="NHWC")
    x = PReLU(alpha_initializer=Constant(0.2))(x)

    output_layer = Conv2D(filters=3, kernel_size=(9,9), strides=(1,1), padding='same')(x)

    return Model(input_layer, output_layer)

def build_discriminator_net(n, k, s):
  
    input_layer = Input(shape=(96,96, 3), dtype=tf.float32)

    x = Conv2D(filters=n, kernel_size=(k,k), strides=(s,s), padding='valid')(input_layer)
    x = LeakyReLU(alpha=0.2)(x)

    for i in range(0,7):

        s = 2

        if(i%2!=0):
            n = n*2
            s = 1

        x = Conv2D(filters=n, kernel_size=(k,k), strides=(s,s), padding='valid')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

    x = Flatten()(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)

    output_layer = Dense(1,activation='sigmoid')(x)

    return Model(input_layer, output_layer)