#### Import librairies
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import collections
from importlib import reload
#from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops

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

##### What's written below does not help increasing speed 
#from tensorflow.keras.mixed_precision import experimental as mixed_precision
#policy = mixed_precision.Policy('mixed_float16')
#mixed_precision.set_policy(policy)
#print('Compute dtype: %s' % policy.compute_dtype)
#print('Variable dtype: %s' % policy.variable_dtype)


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

#With generator (unchanged)
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

# test_generator = generator.BatchGenerator(bands, list_of_samples_test, total_sample_size=None,
#                                     batch_size=batch_size, 
#                                     trainval_or_test='test',
#                                     do_norm=False,
#                                     denorm = False,
#                                     list_of_weights_e=None)


# NEW: Wrap the generator.BatchGenerator objects in a generator-style function
# which we can then pass to tf.data.Dataset.from_generator()
# # (One per train/val/test dataset at the moment, but could be refactored for neatness!)
# def training_batch_generator():
#     multi_enqueuer = keras.utils.OrderedEnqueuer(training_generator,
#                                                 use_multiprocessing=False)
#     multi_enqueuer.start(workers=10, max_queue_size=10)
#     while True:
#         batch_x, batch_y = next(multi_enqueuer.get())
#         yield batch_x, batch_y

# def validation_batch_generator():
#     multi_enqueuer = keras.utils.OrderedEnqueuer(validation_generator,
#                                                     use_multiprocessing=False)
#     multi_enqueuer.start(workers=10, max_queue_size=10)
#     while True:
#         batch_x, batch_y = next(multi_enqueuer.get())
#         yield batch_x, batch_y

# def test_batch_generator():
#     multi_enqueuer = keras.utils.OrderedEnqueuer(test_generator,
#                                                     use_multiprocessing=False)
#     multi_enqueuer.start(workers=10, max_queue_size=10)
#     while True:
#         batch_x, batch_y = next(multi_enqueuer.get())
#         yield batch_x, batch_y

# # Recommended to specify the expected output shapes and types here
# output_types = (tf.float32, tf.float32)
# output_shapes = (tf.TensorShape([batch_size, 64, 64, nb_of_bands]),
#                     tf.TensorShape([batch_size, 3]))

# training_ds = tf.data.Dataset.from_generator(training_batch_generator,
#                                                 output_types=output_types,
#                                                 output_shapes=output_shapes).repeat()

# validation_ds = tf.data.Dataset.from_generator(validation_batch_generator,
#                                                 output_types=output_types,
#                                                 output_shapes=output_shapes).repeat()

# test_ds = tf.data.Dataset.from_generator(test_batch_generator,
#                                             output_types=output_types,
#                                             output_shapes=output_shapes).repeat()

# strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
# with strategy.scope():


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


######## Train the network
hist = net.fit(x_train, y_train, batch_size = batch_size, epochs=5,#x_train, y_train#training_ds
                    steps_per_epoch=steps_per_epoch,
                    verbose=2,
                    shuffle=True,
                    validation_data=(x_val,y_val),#(x_val,y_val),#validation_ds,
                    validation_steps=validation_steps)#,
                    #workers=0, 
                    #use_multiprocessing = True)


net.summary()

# saving_path = '/home/astrodeep/bastien/weights/'#test_5
# net.save_weights('test')