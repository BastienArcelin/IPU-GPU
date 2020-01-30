# Import necessary librairies

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
import sys
import os
import logging
import galsim
import random
import cmath as cm
import math
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Add, Multiply, Reshape, Flatten, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Input, Dense, Dropout, MaxPool2D, Flatten,  Reshape, UpSampling2D, Cropping2D, Conv2DTranspose, PReLU, Concatenate, Lambda, BatchNormalization, concatenate, LeakyReLU

import tensorflow as tf
tfd = tfp.distributions


#### Create encooder
def build_encoder(latent_dim, hidden_dim, filters, kernels,nb_of_bands, conv_activation=None, dense_activation=None):#'sofplus'
    """
    Return encoder as model
    latent_dim : dimension of the latent variable
    hidden_dim : dimension of the dense hidden layer
    filters: list of the sizes of the filters used for this model
    list of the size of the kernels used for each filter of this model
    conv_activation: type of activation layer used after the convolutional layers
    dense_activation: type of activation layer used after the dense layers
    nb_of bands : nb of band-pass filters needed in the model
    """
    # input_layer = Input(shape=(64,64,nb_of_bands))

    # h = Reshape((64,64,nb_of_bands))(input_layer)
    # h = BatchNormalization()(h)
    # for i in range(len(filters)):
    #     h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same')(h)
    #     h = PReLU()(h)
    #     h = Conv2D(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same', strides=(2,2))(h)
    #     h = PReLU()(h)
    # h = Flatten()(h)
    # h = Dense(hidden_dim, activation=dense_activation)(h)
    # h = PReLU()(h)
    # mu = Dense(latent_dim)(h)
    # sigma = Dense(latent_dim, activation='softplus')(h)

    tfd = tfp.distributions

    prior = tfd.Independent(tfd.Normal(loc=tf.zeros(latent_dim), scale=1),
                            reinterpreted_batch_ndims=1)

    encoder = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(64,64,nb_of_bands)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters[0], (kernels[0], kernels[0]),
                    padding='same', activation=conv_activation),
        tf.keras.layers.PReLU(),
        tf.keras.layers.Conv2D(filters[0], (kernels[0], kernels[0]),
                    padding='same', activation=conv_activation, strides=(2,2)),
        tf.keras.layers.PReLU(),
        tf.keras.layers.Conv2D(filters[1], (kernels[1], kernels[1]),
                    padding='same', activation=conv_activation),
        tf.keras.layers.PReLU(),
        tf.keras.layers.Conv2D(filters[1], (kernels[1], kernels[1]),
                    padding='same', activation=conv_activation, strides=(2,2)),
        tf.keras.layers.PReLU(),
        tf.keras.layers.Conv2D(filters[2], (kernels[2], kernels[2]),
                    padding='same', activation=conv_activation),
        tf.keras.layers.PReLU(),
        tf.keras.layers.Conv2D(filters[2], (kernels[2], kernels[2]),
                    padding='same', activation=conv_activation, strides=(2,2)),
        tf.keras.layers.PReLU(),
        tf.keras.layers.Conv2D(filters[3], (kernels[3], kernels[3]),
                    padding='same', activation=conv_activation),
        tf.keras.layers.PReLU(),
        tf.keras.layers.Conv2D(filters[3], (kernels[3], kernels[3]),
                    padding='same', activation=conv_activation, strides=(2,2)),
        tf.keras.layers.PReLU(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(latent_dim),
                activation=None),
        tf.keras.layers.Flatten(),
        tfp.layers.MultivariateNormalTriL(
            latent_dim,
            activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=0.01)),
    ])


    return encoder



#### Create encooder
def build_decoder(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):#tf.keras.activations.softmax
    """
    Return decoder as model
    input_shape: shape of the input data
    latent_dim : dimension of the latent variable
    hidden_dim : dimension of the dense hidden layer
    filters: list of the sizes of the filters used for this model
    list of the size of the kernels used for each filter of this model
    conv_activation: type of activation layer used after the convolutional layers
    dense_activation: type of activation layer used after the dense layers
    """
    # input_layer = Input(shape=(latent_dim,))
    # h = Dense(hidden_dim, activation=dense_activation)(input_layer)
    # h = PReLU()(h)
    # h = Dense(hidden_dim*2, activation=dense_activation)(input_layer)
    # h = PReLU()(h)
    # h = tfp.distributions.MultivariateNormalDiag()

    prior_output = tfd.Independent(tfd.Normal(loc=tf.zeros(final_dim), scale=1),
                            reinterpreted_batch_ndims=1)

    decoder = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(latent_dim)),

        tf.keras.layers.Dense(512, activation=dense_activation),#512
        tf.keras.layers.PReLU(),
        tf.keras.layers.Dense(256, activation=dense_activation),#256
        tf.keras.layers.PReLU(),

        tf.keras.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(final_dim),
                activation=None),
        #tf.keras.layers.PReLU(),
        tfp.layers.MultivariateNormalTriL(final_dim)#,activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior_output, weight=0.000001))
    ])

    return decoder



# Function to define model

def vae_model(latent_dim, final_dim, nb_of_bands):
    """
    Function to create VAE model
    nb_of bands : nb of band-pass filters needed in the model
    """

    #### Parameters to fix
    # batch_size : size of the batch given to the network
    # input_shape: shape of the input data
    # latent_dim : dimension of the latent variable
    # hidden_dim : dimension of the dense hidden layer
    # filters: list of the sizes of the filters used for this model
    # kernels: list of the size of the kernels used for each filter of this model

    batch_size = 100 
    
    input_shape = (64, 64, nb_of_bands)
    hidden_dim = 256
    filters = [32, 64, 128, 256]
    kernels = [3,3,3,3]
    
    # Build the encoder
    encoder = build_encoder(latent_dim, hidden_dim, filters, kernels, nb_of_bands)
    # Build the decoder
    decoder = build_decoder(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)

    
    return encoder, decoder



# Function to define model

def net_model(latent_dim, nb_of_bands):
    """
    Function to create VAE model
    nb_of bands : nb of band-pass filters needed in the model
    """

    #### Parameters to fix
    # batch_size : size of the batch given to the network
    # input_shape: shape of the input data
    # latent_dim : dimension of the latent variable
    # hidden_dim : dimension of the dense hidden layer
    # filters: list of the sizes of the filters used for this model
    # kernels: list of the size of the kernels used for each filter of this model

    batch_size = 100 
    
    input_shape = (64, 64, nb_of_bands)
    hidden_dim = 256
    filters = [32, 64, 128, 256]
    kernels = [3,3,3,3]

    conv_activation=None
    dense_activation=None #tf.nn.softmax

    tfd = tfp.distributions
    prior = tfd.Independent(tfd.Normal(loc=tf.zeros(latent_dim), scale=1),
                        reinterpreted_batch_ndims=1)


    # Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.
    def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        c = np.log(np.expm1(1.))
        return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t[..., :n],
                        scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
            reinterpreted_batch_ndims=1)),
        ])

    # Specify the prior over `keras.layers.Dense` `kernel` and `bias`.
    def prior_trainable(kernel_size, bias_size=0, dtype=None):
        n = kernel_size + bias_size
        return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t, scale=1),
            reinterpreted_batch_ndims=1)),
        ])


    encoder = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(64,64,nb_of_bands)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters[0], (kernels[0], kernels[0]),
                    padding='same', activation=conv_activation),
        tf.keras.layers.PReLU(),
        tf.keras.layers.Conv2D(filters[0], (kernels[0], kernels[0]),
                    padding='same', activation=conv_activation, strides=(2,2)),
        tf.keras.layers.PReLU(),
        tf.keras.layers.Conv2D(filters[1], (kernels[1], kernels[1]),
                    padding='same', activation=conv_activation),
        tf.keras.layers.PReLU(),
        tf.keras.layers.Conv2D(filters[1], (kernels[1], kernels[1]),
                    padding='same', activation=conv_activation, strides=(2,2)),
        tf.keras.layers.PReLU(),
        tf.keras.layers.Conv2D(filters[2], (kernels[2], kernels[2]),
                    padding='same', activation=conv_activation),
        tf.keras.layers.PReLU(),
        tf.keras.layers.Conv2D(filters[2], (kernels[2], kernels[2]),
                    padding='same', activation=conv_activation, strides=(2,2)),
        tf.keras.layers.PReLU(),
        tf.keras.layers.Conv2D(filters[3], (kernels[3], kernels[3]),
                    padding='same', activation=conv_activation),
        tf.keras.layers.PReLU(),
        tf.keras.layers.Conv2D(filters[3], (kernels[3], kernels[3]),
                    padding='same', activation=conv_activation, strides=(2,2)),
        tf.keras.layers.PReLU(),
        tf.keras.layers.Flatten(),
        #tf.keras.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(latent_dim),
        #        activation=dense_activation),
        tfp.layers.DenseVariational(tfp.layers.MultivariateNormalTriL.params_size(latent_dim), 
        posterior_mean_field, prior_trainable, kl_weight=0.001),
        #tfp.layers.DenseFlipout(tfp.layers.MultivariateNormalTriL.params_size(latent_dim),
        #        activation=dense_activation),
        tf.keras.layers.PReLU(),
        #tf.keras.layers.Flatten(),
        tfp.layers.MultivariateNormalTriL(
            latent_dim, activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=0.01)),
    ])

    # Build the encoder
    #encoder = build_encoder(latent_dim, hidden_dim, filters, kernels, nb_of_bands)
    encoder.summary()
    return encoder