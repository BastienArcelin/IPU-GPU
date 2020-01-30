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
latent_dim = 32
final_dim = 1
epochs = int(sys.argv[2])
flipout_or_LS = str(sys.argv[3])
# load = str(sys.argv[5]).lower() == 'true'
bands = [4,5,6,7,8,9]

######## Build the model
if flipout_or_LS == 'flipout':
    encoder = model.net_model(final_dim, len(bands))
    model = Model(inputs=encoder.inputs, outputs = encoder(encoder.inputs))
    kl = sum(model.losses)
    bayes_folder = 'flipout/'
    lr = 1.e-3

elif flipout_or_LS == 'LS' : 
    encoder, decoder = model.vae_model(latent_dim, final_dim, len(bands))
    model = Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs[0]))
    kl=0
    bayes_folder = 'LS/'
    lr = 1.e-4


print(model.summary())

######## Define the loss function
negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)


######## Compile the VAE
model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr), loss=negative_log_likelihood)

######## Callabcks
path_save = '/sps/lsst/users/barcelin/TFP/Redshift/'

checkpointer_loss = ModelCheckpoint(filepath=path_save+'weights/'+bayes_folder+'weights_loss_noisy_v4.{epoch:02d}-{val_loss:.2f}.ckpt', 
                                    monitor='val_loss', 
                                    verbose=1, 
                                    save_best_only=True, 
                                    save_weights_only=True, 
                                    mode='min', 
                                    period=1)

# Define all used callbacks
callbacks = [ checkpointer_loss]#checkpointer_loss,changelr(model)

######## Create generators

images_dir = '/sps/lsst/users/barcelin/data/TFP/GalSim_COSMOS/isolated_galaxies/'+str(sys.argv[1])
list_of_samples = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'training')) if x.endswith('.npy')]
list_of_samples_val = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'validation')) if x.endswith('.npy')]


training_generator = generator.BatchGenerator_redshift(bands, list_of_samples, total_sample_size=None,
                                    batch_size=batch_size, 
                                    trainval_or_test='training',
                                    do_norm=False,
                                    denorm = False,
                                    list_of_weights_e=None)

validation_generator = generator.BatchGenerator_redshift(bands, list_of_samples_val, total_sample_size=None,
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
                  callbacks=callbacks,
                  #max_queue_size=4,
                  workers=0,#4 
                  use_multiprocessing = True)


test = training_generator.__getitem__(2)
out_mean_var = model(test[0])

fig = plt.figure()
plt.plot(test[1][:,0], out_mean_var.mean().numpy()[:,0], '.')
x = np.linspace(0,4)
plt.plot(x, x)
plt.xlim(0,4)
plt.ylim(0,4)
fig.savefig(path_save+'plots/'+bayes_folder+'test_z_mean.png')

fig = plt.figure()
plt.plot(test[1][:,0], out_mean_var.mean().numpy()[:,0], '.', label = 'mean')
plt.plot(test[1][:,0], out_mean_var.mean().numpy()[:,0]+ 2*out_mean_var.stddev().numpy()[:,0], '+', label = 'mean + 2stddev')
plt.plot(test[1][:,0], out_mean_var.mean().numpy()[:,0]- 2*out_mean_var.stddev().numpy()[:,0], '+', label = 'mean - 2stddev')
x = np.linspace(0,4)
plt.plot(x, x)
plt.xlim(0,4)
plt.ylim(0,4)
fig.savefig(path_save+'plots/'+bayes_folder+'test_z_mean_stddev.png')

# fig = plt.figure()
# plt.plot(test[1][:,0], out_mean_var.mean().numpy()[:,0], 'o', label = 'mean')
# plt.plot(test[1][:,0], out_mean_var.mean().numpy()[:,0]- 2*out_mean_var.stddev().numpy()[:,0], '+', label = 'mean + 2stddev')
# x = np.linspace(0,4)
# plt.plot(x, x)
# plt.xlim(0,4)
# plt.ylim(0,4)
# fig.savefig(path_save+'plots/test_z_mean_stddev.png')