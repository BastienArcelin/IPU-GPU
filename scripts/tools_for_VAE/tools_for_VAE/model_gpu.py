# Import necessary librairies
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, BatchNormalization, Conv2D, PReLU
import tensorflow as tf
tfd = tfp.distributions

def create_model_det(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
    model = tf.keras.Sequential()
    model.add(BatchNormalization())
    for i in range(len(filters)):
        model.add(Conv2D(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same'))
        model.add(PReLU())
        model.add(Conv2D(filters[i], (kernels[i],kernels[i]), activation=conv_activation, padding='same', strides=(2,2)))
        model.add(PReLU())
    model.add(tf.keras.layers.Flatten())
    model.add(Dense(tfp.layers.MultivariateNormalTriL.params_size(final_dim), activation=None))
    model.add(tfp.layers.MultivariateNormalTriL(final_dim))
    model.build((None, 64,64,6))
    return model
