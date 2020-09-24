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
# IPU 
from tensorflow.python import ipu

import tensorflow as tf
tfd = tfp.distributions



def create_model_det(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
    h = []
    h.append(BatchNormalization())
    for i in range(len(filters)):
        h.append(Conv2D(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same'))
        h.append(PReLU())
        h.append(Conv2D(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same', strides=(2,2)))
        h.append(PReLU())
    h.append(keras.layers.Flatten())
    h.append(Dense(tfp.layers.MultivariateNormalTriL.params_size(final_dim), activation=None))
    h.append(Dense(final_dim))#tfp.layers.MultivariateNormalTriL(final_dim))#
    m = ipu.keras.Sequential(h)

    return m
