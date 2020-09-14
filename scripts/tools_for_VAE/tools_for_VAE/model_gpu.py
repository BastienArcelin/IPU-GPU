# Import necessary librairies

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
import sys
import os
import logging
#import galsim
import random
import cmath as cm
import math
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Add, Multiply, Reshape, Flatten, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Input, Dense, Dropout, MaxPool2D, Flatten,  Reshape, UpSampling2D, Cropping2D, Conv2DTranspose, PReLU, Concatenate, Lambda, BatchNormalization, concatenate, LeakyReLU

from tensorflow import keras

import tensorflow as tf
tfd = tfp.distributions


#@tf.function
def create_model_wo_ls(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
    input_layer = Input(shape=(input_shape)) 
    #conv_activation='sigmoid'

    # Encoding part
    h = BatchNormalization()(input_layer)
    for i in range(len(filters)):
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same')(h)
        h = PReLU()(h)
        h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same', strides=(2,2))(h)
        h = PReLU()(h)
    h = Flatten()(h)
    h = Dense(tfp.layers.MultivariateNormalTriL.params_size(final_dim),
                activation=None)(h)
    h = tfp.layers.MultivariateNormalTriL(final_dim)(h)

    model = Model(input_layer,h)

    return model

def create_model_wo_ls_3(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
    model = tf.keras.Sequential()
    model.add(BatchNormalization())
    for i in range(len(filters)):
        model.add(Conv2D(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same'))
        model.add(PReLU())
        model.add(Conv2D(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same', strides=(2,2)))
        model.add(PReLU())
    model.add(keras.layers.Flatten())
    model.add(Dense(tfp.layers.MultivariateNormalTriL.params_size(final_dim), activation=None))
    model.add(tfp.layers.MultivariateNormalTriL(final_dim))
    model.build((None, 64,64,6))
    #m = tf.keras.Sequential(h)

    return model
