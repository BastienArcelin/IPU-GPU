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

# Probabilistic models
import tensorflow.compat.v1 as tf1
from tensorflow_probability.python.layers import util as tfp_layers_util

# Weights initialization for posteriors
def get_posterior_fn():
  return tfp_layers_util.default_mean_field_normal_fn(
      loc_initializer=tf1.initializers.he_normal(), 
      untransformed_scale_initializer=tf1.initializers.random_normal(
          mean=-9.0, stddev=0.1)
      )
# kernel divergence weight in loss
kernel_divergence_fn=(lambda q, p, ignore: tfd.kl_divergence(q, p) / 8000)

def create_model_full_prob(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None):
    model = tf.keras.Sequential()
    model.add(BatchNormalization())
    for i in range(len(filters)):
        model.add(tfp.layers.Convolution2DReparameterization(filters[i], (kernels[i],kernels[i]), 
                                            kernel_posterior_fn=get_posterior_fn(),
                                            #kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                            kernel_divergence_fn=kernel_divergence_fn,
                                            activation=conv_activation, 
                                            padding='same'))
        model.add(PReLU())
        model.add(tfp.layers.Convolution2DReparameterization(filters[i], (kernels[i],kernels[i]), 
                                            kernel_posterior_fn=get_posterior_fn(),
                                            #kernel_posterior_fn=ktied_distribution.get_ktied_posterior_fn(),
                                            kernel_divergence_fn=kernel_divergence_fn,
                                            activation=conv_activation, 
                                            padding='same', 
                                            strides=(2,2)))
        model.add(PReLU())
    model.add(tf.keras.layers.Flatten())
    model.add(tfp.layers.DenseReparameterization(tfp.layers.MultivariateNormalTriL.params_size(final_dim),
                                    kernel_posterior_fn=get_posterior_fn(),#ktied_distribution.get_ktied_posterior_fn(),
                                    kernel_divergence_fn = kernel_divergence_fn,
                                    activation=dense_activation))
    #model.add(tfp.layers.MultivariateNormalTriL(final_dim))
    model.add(tfp.layers.MultivariateNormalTriL(final_dim))
    model.build((None, 64,64,6))
    return model