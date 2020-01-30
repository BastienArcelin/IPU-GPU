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
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.models import Model, Sequential
from scipy.stats import norm
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.layers import Conv2D, Input, Dense, Dropout, MaxPool2D, Flatten,  Reshape, UpSampling2D, Cropping2D, Conv2DTranspose, PReLU, Concatenate, Lambda, BatchNormalization, concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, TerminateOnNaN, ModelCheckpoint
import tensorflow as tf
import tensorflow_probability as tfp

sys.path.insert(0,'../tools_for_VAE/')
from tools_for_VAE import model, vae_functions, utils, generator
from tools_for_VAE.callbacks import changeAlpha, changelr

######## Set some parameters
batch_size = 100
latent_dim = 2
epochs = int(sys.argv[2])
# load = str(sys.argv[5]).lower() == 'true'
bands = [4,5,6,7,8,9]

######## Load VAE
#encoder, decoder = model.vae_model(latent_dim, len(bands))
encoder = model.net_model(latent_dim, len(bands))

#encoder, decoder = model.vae_model_2(latent_dim, len(bands))

######## Build the model

model = Model(inputs=encoder.inputs, outputs = encoder(encoder.inputs))

#model = Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs[0]))
#model, Dkl = vae_functions.build_vanilla_vae(encoder, decoder, full_cov=False, coeff_KL = 0)

print(model.summary())

######## Define the loss function
# alpha = K.variable(1e-2)

negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)

# def vae_loss(x, x_decoded_mean):
#     log_likelihood = negative_log_likelihood(x, x_decoded_mean)
#     kl_loss = K.get_value(alpha) * Dkl
#     return log_likelihood + K.mean(kl_loss)

######## Compile the VAE
model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3), loss=negative_log_likelihood)#negative_log_likelihood#vae_loss
# , amsgrad = True
######## Callabcks
path_weights = '/sps/lsst/users/barcelin/TFP/weights/'
checkpointer_loss = ModelCheckpoint(filepath=path_weights+'test/weights_loss_noisy_v4.{epoch:02d}-{val_loss:.2f}.ckpt', 
                                    monitor='val_loss', 
                                    verbose=1, 
                                    save_best_only=True, 
                                    save_weights_only=True, 
                                    mode='min', 
                                    period=1)

# Define all used callbacks
callbacks = [ changelr(model)]#checkpointer_loss,, ReduceLROnPlateau(), TerminateOnNaN()]# checkpointer_mse earlystop, checkpointer_loss,vae_hist,, alphaChanger

######## Create generators

images_dir = '/sps/lsst/users/barcelin/data/TFP/GalSim_COSMOS/isolated_galaxies/'+str(sys.argv[1])
list_of_samples = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'training')) if x.endswith('.npy')]
list_of_samples_val = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'validation')) if x.endswith('.npy')]


training_generator = generator.BatchGenerator_ellipticity(bands, list_of_samples, total_sample_size=None,
                                    batch_size=batch_size, 
                                    trainval_or_test='training',
                                    do_norm=False,
                                    denorm = False,
                                    list_of_weights_e=None)

validation_generator = generator.BatchGenerator_ellipticity(bands, list_of_samples_val, total_sample_size=None,
                                    batch_size=batch_size, 
                                    trainval_or_test='validation',
                                    do_norm=False,
                                    denorm = False,
                                    list_of_weights_e= None)

######## Train the network
hist = model.fit_generator(generator = training_generator, epochs=epochs,
                  steps_per_epoch=64,#128
                  verbose=2,
                  shuffle=True,
                  validation_data=validation_generator,
                  validation_steps=16,#16
                  #callbacks=callbacks,
                  #max_queue_size=4,
                  workers=0,#4 
                  use_multiprocessing = True)


test = training_generator.__getitem__(2)
out_mean_var = model(test[0])

fig = plt.figure()
plt.plot(test[1][:,0], out_mean_var.mean().numpy()[:,0], 'o')
x = np.linspace(-1,1)
plt.plot(x, x)
plt.xlim(-1,1)
plt.ylim(-1,1)
fig.savefig('test_e1.png')

fig = plt.figure()
plt.plot(test[1][:,1], out_mean_var.mean().numpy()[:,1], 'o')
x = np.linspace(-1,1)
plt.plot(x, x)
plt.xlim(-1,1)
plt.ylim(-1,1)
fig.savefig('test_e2.png')

# fig = plt.figure()
# plt.plot(test[1][:,2], out_mean_var.mean().numpy()[:,2], 'o')
# x = np.linspace(0,4)
# plt.plot(x, x)
# fig.savefig('test_z.png')