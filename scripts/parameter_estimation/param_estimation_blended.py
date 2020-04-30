#### Import librairies
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import collections
from importlib import reload

import tensorflow
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Add, Multiply, Reshape, Flatten, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Input, Dense, Dropout, MaxPool2D, Flatten,  Reshape, UpSampling2D, Cropping2D, Conv2DTranspose, PReLU, Concatenate, Lambda, BatchNormalization, concatenate, LeakyReLU

tfd = tfp.distributions

sys.path.insert(0,'../../scripts/tools_for_VAE/')
import tools_for_VAE.layers as layers
from tools_for_VAE import utils, vae_functions, generator, model
from tensorflow.keras import backend as K


######## Parameters
nb_of_bands = 6
batch_size = 100 

input_shape = (64, 64, nb_of_bands)
hidden_dim = 256
latent_dim = 32
final_dim = 9
filters = [32, 64, 128, 256, 512]#, 1024]
kernels = [3,3,3,3,3]#,3]

conv_activation = None
dense_activation = None

steps_per_epoch = 32
validation_steps = 8

bands = [4,5,6,7,8,9]


#### Loading data
# Direct loading
# data = np.load('/sps/lsst/users/barcelin/data/TFP/GalSim_COSMOS/isolated_galaxies/centered/training/galaxies_isolated_20191024_0_images.npy', mmap_mode = 'c')
# labels = pd.read_csv('/sps/lsst/users/barcelin/data/TFP/GalSim_COSMOS/isolated_galaxies/centered/training/galaxies_isolated_20191024_0_data.csv')

# temp_labels = labels[(np.abs(labels['e1'])<=1.) & (np.abs(labels['e2'])<=1)]
# e1 = np.exp(np.array(temp_labels['e1']))*2
# e2 = np.exp(np.array(temp_labels['e2']))*2
# z = np.array(temp_labels['redshift'])

# new_labels = np.zeros((len(e1),final_dim))
# new_labels[:,0] = e1
# new_labels[:,1] = e2
# new_labels[:,2] = z
# print(new_labels.shape)
# #new_labels = np.array(new_labels['e1'])

# training_data = data[:2000,1,4:]
# training_data = np.transpose(training_data, axes = (0,2,3,1))
# validation_data = data[2000:2500,1,4:]
# validation_data = np.transpose(validation_data, axes = (0,2,3,1))

# training_labels = new_labels[:2000]
# validation_labels = new_labels[2000:2500]



# With generator
images_dir = '/sps/lsst/users/barcelin/data/TFP/GalSim_COSMOS/blended_galaxies/random/'

list_of_samples = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'training')) if x.endswith('.npy')]
list_of_samples_val = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'validation')) if x.endswith('.npy')]
list_of_samples_test = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'test')) if x.endswith('.npy')]
print(list_of_samples_test)

training_generator = generator.BatchGenerator(bands, list_of_samples, total_sample_size=None,
                                    batch_size=batch_size, 
                                    trainval_or_test='training',
                                    do_norm=False,
                                    denorm = False,
                                    list_of_weights_e=None)

validation_generator = generator.BatchGenerator(bands, list_of_samples_val, total_sample_size=None,
                                    batch_size=batch_size, 
                                    trainval_or_test='validation',
                                    do_norm=False,
                                    denorm = False,
                                    list_of_weights_e=None)

test_generator = generator.BatchGenerator_multi_galaxies(bands, list_of_samples_test, total_sample_size=None,
                                    batch_size=batch_size, 
                                    trainval_or_test='test',
                                    do_norm=False,
                                    denorm = False,
                                    list_of_weights_e=None)

#### Model definition
model_choice = 'wo_ls'
# With latent space
if model_choice == 'ls':
    net = model.create_model(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)
# Without latent space
if model_choice == 'wo_ls': # create_model_wo_ls
    net = model.create_model_wo_ls_multi(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)
# Full probabilistic model
if model_choice == 'full_prob':
    net = model.create_model_full_prob(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)
net.summary()

#### Loss definition
if model_choice == 'full_prob':
    kl = sum(net.losses)
    def loss(x, dists):
        nll = -dists.log_prob(x)
        print(nll)
        kl = sum(net.losses)
        print(kl)
        return nll + kl, collections.namedtuple('loss','nll,kl')(nll, kl)
    negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x) + kl *(10/(batch_size*steps_per_epoch)-1)

else:
    negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)




net.compile(optimizer=tf.optimizers.Adam(learning_rate=3e-1), 
              loss=negative_log_likelihood , metrics = ['mse', 'acc'], experimental_run_tf_function=False)


loading_path = '/sps/lsst/users/barcelin/TFP/weights/blended_multi_3/loss/'#blended_multi_3
latest = tf.train.latest_checkpoint(loading_path)
net.load_weights(latest)



# Callbacks
saving_path = '/sps/lsst/users/barcelin/TFP/weights/blended_multi_3/'
checkpointer_mse = tf.keras.callbacks.ModelCheckpoint(filepath=saving_path+'mse/weights_noisy_v4.{epoch:02d}-{val_mse:.2f}.ckpt', monitor='val_mse', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)
checkpointer_loss = tf.keras.callbacks.ModelCheckpoint(filepath=saving_path+'loss/weights_noisy_v4.{epoch:02d}-{val_loss:.2f}.ckpt', monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)
checkpointer_acc = tf.keras.callbacks.ModelCheckpoint(filepath=saving_path+'acc/weights_noisy_v4.{epoch:02d}-{val_acc:.2f}.ckpt', monitor='val_acc', verbose=1, save_best_only=True,save_weights_only=True, mode='max', period=1)

callbacks = [checkpointer_mse, checkpointer_loss, checkpointer_acc]


######## Train the network
hist = net.fit_generator(test_generator, epochs=5000, # training
          steps_per_epoch=steps_per_epoch,#128
          verbose=1,
          shuffle=True,
          validation_data=test_generator, # validation
          validation_steps=validation_steps,#16
          callbacks= callbacks,
          workers=0,#4 
          use_multiprocessing = True)

saving_path = '/sps/lsst/users/barcelin/TFP/weights/blended_multi_3/'#blended_multi_3
net.save_weights(saving_path+'cp-{epoch:04d}.ckpt')
# 2 galaxies dans /weights/blended_multi/


## Plots
loading_path = '/sps/lsst/users/barcelin/TFP/weights/blended_multi_3/loss/'#blended_multi_3
latest = tf.train.latest_checkpoint(loading_path)
net.load_weights(latest)


test = test_generator.__getitem__(2)

training_data = test[0]
training_labels = test[1]
out = net(training_data)

fig = plt.figure()
sns.distplot(out.mean().numpy()[:,0], bins = 20)
sns.distplot(training_labels[:,0], bins = 20)
fig.savefig('test_distrib_e1.png')


fig = plt.figure()
sns.distplot(out.mean().numpy()[:,1], bins = 20)
sns.distplot(training_labels[:,1], bins = 20)
fig.savefig('test_distrib_e2.png')


fig = plt.figure()
sns.distplot(out.mean().numpy()[:,2], bins = 20)
sns.distplot(training_labels[:,2], bins = 20)
fig.savefig('test_distrib_z.png')

fig = plt.figure()
sns.distplot(out.mean().numpy()[:,3], bins = 20)
sns.distplot(training_labels[:,3], bins = 20)
fig.savefig('test_distrib_e1_2.png')


fig = plt.figure()
sns.distplot(out.mean().numpy()[:,4], bins = 20)
sns.distplot(training_labels[:,4], bins = 20)
fig.savefig('test_distrib_e2_2.png')


fig = plt.figure()
sns.distplot(out.mean().numpy()[:,5], bins = 20)
sns.distplot(training_labels[:,5], bins = 20)
fig.savefig('test_distrib_z_2.png')

fig = plt.figure()
sns.distplot(out.mean().numpy()[:,6], bins = 20)
sns.distplot(training_labels[:,6], bins = 20)
fig.savefig('test_distrib_e1_3.png')


fig = plt.figure()
sns.distplot(out.mean().numpy()[:,7], bins = 20)
sns.distplot(training_labels[:,7], bins = 20)
fig.savefig('test_distrib_e2_3.png')


fig = plt.figure()
sns.distplot(out.mean().numpy()[:,8], bins = 20)
sns.distplot(training_labels[:,8], bins = 20)
fig.savefig('test_distrib_z_3.png')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(training_labels[:,0], out.mean().numpy()[:,0], '.', label = 'mean')
axes[0].plot(training_labels[:,0], out.mean().numpy()[:,0]+ 2*out.stddev().numpy()[:,0], '+', label = 'mean + 2stddev')
axes[0].plot(training_labels[:,0], out.mean().numpy()[:,0]- 2*out.stddev().numpy()[:,0], '+', label = 'mean - 2stddev')
x = np.linspace(-1,1)
axes[0].plot(x, x)
axes[0].legend()
axes[0].set_ylim(-1,1)
axes[0].set_title('$e1$')

axes[1].plot(training_labels[:,1], out.mean().numpy()[:,1], '.', label = 'mean')
axes[1].plot(training_labels[:,1], out.mean().numpy()[:,1]+ 2*out.stddev().numpy()[:,1], '+', label = 'mean + 2stddev')
axes[1].plot(training_labels[:,1], out.mean().numpy()[:,1]- 2*out.stddev().numpy()[:,1], '+', label = 'mean - 2stddev')
x = np.linspace(-1,1)
axes[1].plot(x, x)
axes[1].legend()
axes[1].set_ylim(-1,1)
axes[1].set_title('$e2$')

axes[2].plot(training_labels[:,2], out.mean().numpy()[:,2], '.', label = 'mean')
axes[2].plot(training_labels[:,2], out.mean().numpy()[:,2]+ 2*out.stddev().numpy()[:,2], '+', label = 'mean + 2stddev')
axes[2].plot(training_labels[:,2], out.mean().numpy()[:,2]- 2*out.stddev().numpy()[:,2], '+', label = 'mean - 2stddev')
x = np.linspace(0,4)
axes[2].plot(x, x)
axes[2].legend()
axes[2].set_ylim(-1,5.5)
axes[2].set_title('$z$')

fig.savefig('test_train.png')


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(training_labels[:,3], out.mean().numpy()[:,3], '.', label = 'mean')
axes[0].plot(training_labels[:,3], out.mean().numpy()[:,3]+ 2*out.stddev().numpy()[:,3], '+', label = 'mean + 2stddev')
axes[0].plot(training_labels[:,3], out.mean().numpy()[:,3]- 2*out.stddev().numpy()[:,3], '+', label = 'mean - 2stddev')
x = np.linspace(-1,1)
axes[0].plot(x, x)
axes[0].legend()
axes[0].set_ylim(-1,1)
axes[0].set_title('$e1$')

axes[1].plot(training_labels[:,4], out.mean().numpy()[:,4], '.', label = 'mean')
axes[1].plot(training_labels[:,4], out.mean().numpy()[:,4]+ 2*out.stddev().numpy()[:,4], '+', label = 'mean + 2stddev')
axes[1].plot(training_labels[:,4], out.mean().numpy()[:,4]- 2*out.stddev().numpy()[:,4], '+', label = 'mean - 2stddev')
x = np.linspace(-1,1)
axes[1].plot(x, x)
axes[1].legend()
axes[1].set_ylim(-1,1)
axes[1].set_title('$e2$')

axes[2].plot(training_labels[:,5], out.mean().numpy()[:,5], '.', label = 'mean')
axes[2].plot(training_labels[:,5], out.mean().numpy()[:,5]+ 2*out.stddev().numpy()[:,5], '+', label = 'mean + 2stddev')
axes[2].plot(training_labels[:,5], out.mean().numpy()[:,5]- 2*out.stddev().numpy()[:,5], '+', label = 'mean - 2stddev')
x = np.linspace(0,4)
axes[2].plot(x, x)
axes[2].legend()
axes[2].set_ylim(-1,5.5)
axes[2].set_title('$z$')
fig.savefig('test_train_2.png')


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(training_labels[:,6], out.mean().numpy()[:,6], '.', label = 'mean')
axes[0].plot(training_labels[:,6], out.mean().numpy()[:,6]+ 2*out.stddev().numpy()[:,6], '+', label = 'mean + 2stddev')
axes[0].plot(training_labels[:,6], out.mean().numpy()[:,6]- 2*out.stddev().numpy()[:,6], '+', label = 'mean - 2stddev')
x = np.linspace(-1,1)
axes[0].plot(x, x)
axes[0].legend()
axes[0].set_ylim(-1,1)
axes[0].set_title('$e1$')

axes[1].plot(training_labels[:,7], out.mean().numpy()[:,7], '.', label = 'mean')
axes[1].plot(training_labels[:,7], out.mean().numpy()[:,7]+ 2*out.stddev().numpy()[:,7], '+', label = 'mean + 2stddev')
axes[1].plot(training_labels[:,7], out.mean().numpy()[:,7]- 2*out.stddev().numpy()[:,7], '+', label = 'mean - 2stddev')
x = np.linspace(-1,1)
axes[1].plot(x, x)
axes[1].legend()
axes[1].set_ylim(-1,1)
axes[1].set_title('$e2$')

axes[2].plot(training_labels[:,8], out.mean().numpy()[:,8], '.', label = 'mean')
axes[2].plot(training_labels[:,8], out.mean().numpy()[:,8]+ 2*out.stddev().numpy()[:,8], '+', label = 'mean + 2stddev')
axes[2].plot(training_labels[:,8], out.mean().numpy()[:,8]- 2*out.stddev().numpy()[:,8], '+', label = 'mean - 2stddev')
x = np.linspace(0,4)
axes[2].plot(x, x)
axes[2].legend()
axes[2].set_ylim(-1,5.5)
axes[2].set_title('$z$')
fig.savefig('test_train_3.png')

