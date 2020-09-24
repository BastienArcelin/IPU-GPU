#### Import librairies
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import collections
from importlib import reload
import time

# Tensorflow
import tensorflow
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Add, Multiply, Reshape, Flatten, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, Input, Dense, Dropout, MaxPool2D, Flatten,  Reshape, UpSampling2D, Cropping2D, Conv2DTranspose, PReLU, Concatenate, Lambda, BatchNormalization, concatenate, LeakyReLU

tfd = tfp.distributions

sys.path.insert(0,'../../scripts/tools_for_VAE/')
import tools_for_VAE.layers as layers
from tools_for_VAE import utils, vae_functions, generator, model_gpu
from tensorflow.keras import backend as K

#tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


######## Parameters
nb_of_bands = 6
batch_size = 8
print('The batchsize is : '+str(batch_size))

input_shape = (64, 64, nb_of_bands)
hidden_dim = 256
latent_dim = 32
final_dim = 3
filters = [32, 64, 128, 256]
kernels = [3,3,3,3]

conv_activation = None
dense_activation = None

steps_per_epoch = 1125#1100#00
validation_steps = 125#150#50

bands = [4,5,6,7,8,9]

images_dir = '/sps/lsst/users/barcelin/data/TFP/GalSim_COSMOS/isolated_galaxies/centered'

# list_of_samples = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'test')) if x.endswith('.npy')]
# list_of_samples_val = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'test')) if x.endswith('.npy')]
# list_of_samples_test = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'test')) if x.endswith('.npy')]
# print(list_of_samples_test)

data = np.load('/sps/lsst/users/barcelin/data/TFP/GalSim_COSMOS/isolated_galaxies/centered/test/galaxies_isolated_20191024_0_images.npy', mmap_mode = 'c')
data_label = pd.read_csv('/sps/lsst/users/barcelin/data/TFP/GalSim_COSMOS/isolated_galaxies/centered/test/galaxies_isolated_20191024_0_data.csv')
#training_data = np.zeros((9000,2))
x_train = tf.transpose(data[:9000,1], perm= [0,2,3,1])[:,:,:,4:]
y_train = np.zeros((9000,3))
y_train[:,0] = data_label[:9000]['e1']
y_train[:,1] = data_label[:9000]['e2']
y_train[:,2] = data_label[:9000]['redshift']
y_train = tf.convert_to_tensor(y_train)
ds_train = tf.data.Dataset.from_tensor_slices((np.expand_dims(x_train, axis = 1), np.expand_dims(y_train,axis = 1)))

x_val = tf.transpose(data[9000:,1], perm= [0,2,3,1])[:,:,:,4:]
y_val = np.zeros((1000,3))
y_val[:,0] = data_label[9000:]['e1']
y_val[:,1] = data_label[9000:]['e2']
y_val[:,2] = data_label[9000:]['redshift']
y_val = tf.convert_to_tensor(y_val)
ds_val = tf.data.Dataset.from_tensor_slices((np.expand_dims(x_val, axis = 1), np.expand_dims(y_val, axis = 1)))

#### Model definition
model_choice = 'wo_ls'
# With latent space
if model_choice == 'ls':
    net = model_gpu.create_model(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)
# Without latent space
if model_choice == 'wo_ls':
    net = model_gpu.create_model_wo_ls(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)
# Full probabilistic model
if model_choice == 'full_prob':
    net = model_gpu.create_model_full_prob(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)


#### Loss definition
if model_choice == 'full_prob':
    kl = sum(net.losses)
    negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x) + kl *(-1+1/(batch_size))

else:
    negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)


net.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3), 
            loss="mean_squared_error")

net.summary()


loading_path = '/sps/lsst/users/barcelin/weights/gpu_benchmark/'
latest = tf.train.latest_checkpoint(loading_path)
net.load_weights(latest)

print('weights loaded')



# training_data, training_labels  = next(iter(test_ds))
# print(training_data.numpy().shape)

# ds = tf.data.Dataset.from_tensor_slices((tf.cast(training_data, tf.float32)))

t_1 = time.time()
out = []
for i in range (len(x_val)):
    out.append(net(x_val[i]))
out = np.array(out)
#out = net(x_val)
t_2 = time.time()
print('prediction done in: '+str(t_2-t_1)+' seconds')
print(out.shape)

fig = plt.figure()
sns.distplot(out.mean().numpy()[:,0], bins = 20)
sns.distplot(y_val[:,0], bins = 20)
fig.savefig('test_distrib_e1.png')


fig = plt.figure()
sns.distplot(out.mean().numpy()[:,1], bins = 20)
sns.distplot(y_val[:,1], bins = 20)
fig.savefig('test_distrib_e2.png')


fig = plt.figure()
sns.distplot(out.mean().numpy()[:,2], bins = 20)
sns.distplot(y_val[:,2], bins = 20)
fig.savefig('test_distrib_e3.png')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(y_val[:,0], out.mean().numpy()[:,0], '.', label = 'mean')
axes[0].plot(y_val[:,0], out.mean().numpy()[:,0]+ 2*out.stddev().numpy()[:,0], '+', label = 'mean + 2stddev')
axes[0].plot(y_val[:,0], out.mean().numpy()[:,0]- 2*out.stddev().numpy()[:,0], '+', label = 'mean - 2stddev')
x = np.linspace(-1,1)
axes[0].plot(x, x)
axes[0].legend()
axes[0].set_ylim(-1,1)
axes[0].set_title('$e1$')

axes[1].plot(y_val[:,1], out.mean().numpy()[:,1], '.', label = 'mean')
axes[1].plot(y_val[:,1], out.mean().numpy()[:,1]+ 2*out.stddev().numpy()[:,1], '+', label = 'mean + 2stddev')
axes[1].plot(y_val[:,1], out.mean().numpy()[:,1]- 2*out.stddev().numpy()[:,1], '+', label = 'mean - 2stddev')
x = np.linspace(-1,1)
axes[1].plot(x, x)
axes[1].legend()
axes[1].set_ylim(-1,1)
axes[1].set_title('$e2$')

axes[2].plot(y_val[:,2], out.mean().numpy()[:,2], '.', label = 'mean')
axes[2].plot(y_val[:,2], out.mean().numpy()[:,2]+ 2*out.stddev().numpy()[:,2], '+', label = 'mean + 2stddev')
axes[2].plot(y_val[:,2], out.mean().numpy()[:,2]- 2*out.stddev().numpy()[:,2], '+', label = 'mean - 2stddev')
x = np.linspace(0,4)
axes[2].plot(x, x)
axes[2].legend()
axes[2].set_ylim(-1,5.5)
axes[2].set_title('$z$')

fig.savefig('test_train.png')