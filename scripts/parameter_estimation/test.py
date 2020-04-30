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
final_dim = 3
filters = [32, 64, 128, 256, 512]
kernels = [3,3,3,3, 3]

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
images_dir = '/sps/lsst/users/barcelin/data/TFP/GalSim_COSMOS/isolated_galaxies/centered/'
images_dir = '/sps/lsst/users/barcelin/data/TFP/GalSim_COSMOS/blended_galaxies/centered/'
#list_of_samples = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'training')) if x.endswith('.npy')]
#list_of_samples_val = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'validation')) if x.endswith('.npy')]
list_of_samples_test = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'test')) if x.endswith('.npy')]
print(list_of_samples_test)
# training_generator = generator.BatchGenerator(bands, list_of_samples, total_sample_size=None,
#                                     batch_size=batch_size, 
#                                     trainval_or_test='training',
#                                     do_norm=False,
#                                     denorm = False,
#                                     list_of_weights_e=None)

# validation_generator = generator.BatchGenerator(bands, list_of_samples_val, total_sample_size=None,
#                                     batch_size=batch_size, 
#                                     trainval_or_test='validation',
#                                     do_norm=False,
#                                     denorm = False,
#                                     list_of_weights_e=None)

test_generator = generator.BatchGenerator(bands, list_of_samples_test, total_sample_size=None,
                                    batch_size=batch_size, 
                                    trainval_or_test='test',
                                    do_norm=False,
                                    denorm = False,
                                    list_of_weights_e=None)

#### Model definition
model_choice = 'full_prob'
# With latent space
if model_choice == 'ls':
    net = model.create_model(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)
# Without latent space
if model_choice == 'wo_ls':
    net = model.create_model_wo_ls(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)
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
    #print("la")
    #print(kl.eval())
    negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x) + kl *(-1+1/(batch_size))

else:
    negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)




net.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3), 
              loss=negative_log_likelihood , metrics = ['mse', 'acc'], experimental_run_tf_function=False)


## /test_3/ : with beta = 0.01 , lr = 10-3
## /test_2/ : with beta = 1.
loading_path = '/sps/lsst/users/barcelin/TFP/weights/test_5/'#test_5
latest = tf.train.latest_checkpoint(loading_path)
net.load_weights(latest)

######## Train the network
hist = net.fit_generator(test_generator, epochs=500, # training
          steps_per_epoch=steps_per_epoch,#128
          verbose=1,
          shuffle=True,
          validation_data=test_generator, # validation
          validation_steps=validation_steps,#16
          workers=0,#4 
          use_multiprocessing = True)

saving_path = '/sps/lsst/users/barcelin/TFP/weights/test_5/'#test_5
net.save_weights(saving_path+'cp-{epoch:04d}.ckpt')


#### Plots
# training

## REGENERER AVEC NOUVELLES IMAGES ET RENORMALISATION CORRECTE
# n_batch = 2
# test = np.zeros((2, n_batch*100))

# testing_data = np.zeros((n_batch*100, 64, 64, 6))
# testing_labels = np.zeros((n_batch*100, final_dim))

# for i in range (n_batch):
#     print(i)
#     testing_data[i*batch_size:(i+1)*batch_size]=test_generator.__getitem__(2)[0]
#     testing_labels[i*batch_size:(i+1)*batch_size]=test_generator.__getitem__(2)[1]

#test = np.concatenate(test, axis = 1)
#print(test.shape)

test = test_generator.__getitem__(2)

training_data = test[0]
training_labels = test[1]
out = net(training_data)

fig = plt.figure()
sns.distplot(out.mean().numpy()[:,0], bins = 20)
sns.distplot(training_labels[:,0], bins = 20)
fig.savefig('full_prob/test_distrib_e1.png')


fig = plt.figure()
sns.distplot(out.mean().numpy()[:,1], bins = 20)
sns.distplot(training_labels[:,1], bins = 20)
fig.savefig('full_prob/test_distrib_e2.png')


fig = plt.figure()
sns.distplot(out.mean().numpy()[:,2], bins = 20)
sns.distplot(training_labels[:,2], bins = 20)
fig.savefig('full_prob/test_distrib_e3.png')

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

fig.savefig('full_prob/test_train.png')

# validation 
# out = net(validation_data)
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# axes[0].plot(validation_labels[:,0], out.mean().numpy()[:,0], '.', label = 'mean')
# axes[0].plot(validation_labels[:,0], out.mean().numpy()[:,0]+ 2*out.stddev().numpy()[:,0], '+', label = 'mean + 2stddev')
# axes[0].plot(validation_labels[:,0], out.mean().numpy()[:,0]- 2*out.stddev().numpy()[:,0], '+', label = 'mean - 2stddev')
# x = np.linspace(0,4)
# axes[0].plot(x, x)
# axes[0].legend()
# axes[0].set_ylim(-0.5,6.5)
# axes[0].set_title('$e1$')

# axes[1].plot(validation_labels[:,1], out.mean().numpy()[:,1], '.', label = 'mean')
# axes[1].plot(validation_labels[:,1], out.mean().numpy()[:,1]+ 2*out.stddev().numpy()[:,1], '+', label = 'mean + 2stddev')
# axes[1].plot(validation_labels[:,1], out.mean().numpy()[:,1]- 2*out.stddev().numpy()[:,1], '+', label = 'mean - 2stddev')
# x = np.linspace(0,4)
# axes[1].plot(x, x)
# axes[1].legend()
# axes[1].set_ylim(-0.5,6.5)
# axes[1].set_title('$e2$')

# axes[2].plot(validation_labels[:,2], out.mean().numpy()[:,2], '.', label = 'mean')
# axes[2].plot(validation_labels[:,2], out.mean().numpy()[:,2]+ 2*out.stddev().numpy()[:,2], '+', label = 'mean + 2stddev')
# axes[2].plot(validation_labels[:,2], out.mean().numpy()[:,2]- 2*out.stddev().numpy()[:,2], '+', label = 'mean - 2stddev')
# x = np.linspace(0,4)
# axes[2].plot(x, x)
# axes[2].legend()
# axes[2].set_ylim(-0.5,6.5)
# axes[2].set_title('$z$')

# fig.savefig('test_val.png')

