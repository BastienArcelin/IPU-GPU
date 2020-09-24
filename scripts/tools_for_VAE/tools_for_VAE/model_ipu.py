# Import necessary librairies
import tensorflow.keras
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, BatchNormalization, Conv2D, PReLU
import tensorflow as tf
tfd = tfp.distributions
from tensorflow import keras

# IPU 
from tensorflow.python import ipu


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
