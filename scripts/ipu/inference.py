#### Import librairies
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import collections
from importlib import reload
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops

# IPU 
from tensorflow.python import ipu
from tensorflow.python.ipu.scopes import ipu_scope
cfg = ipu.utils.create_ipu_config()#profiling=True,
                                  #profile_execution=True,
                                  #report_directory='fixed_fullModel'
cfg = ipu.utils.auto_select_ipus(cfg, 1)
ipu.utils.configure_ipu_system(cfg)

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
from tools_for_VAE import utils, vae_functions, generator, model_ipu
from tensorflow.keras import backend as K


######## Parameters
nb_of_bands = 6
batch_size = 1

input_shape = (64, 64, nb_of_bands)
hidden_dim = 256
latent_dim = 32
final_dim = 3
filters = [32, 64, 128, 256]
kernels = [3,3,3,3]

conv_activation = None
dense_activation = None

steps_per_epoch = 512
validation_steps = 64

bands = [4,5,6,7,8,9]

images_dir = '/home/astrodeep/bastien/data/'

list_of_samples = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'test')) if x.endswith('.npy')]
list_of_samples_val = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'test')) if x.endswith('.npy')]
list_of_samples_test = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'test')) if x.endswith('.npy')]
print(list_of_samples_test)


###### Generator
test_generator = generator.BatchGenerator(bands, list_of_samples_test, total_sample_size=None,
                                    batch_size=batch_size, 
                                    trainval_or_test='test',
                                    do_norm=False,
                                    denorm = False,
                                    list_of_weights_e=None)

def test_batch_generator():
    multi_enqueuer = keras.utils.OrderedEnqueuer(test_generator,
                                                    use_multiprocessing=False)
    multi_enqueuer.start(workers=10, max_queue_size=10)
    while True:
        batch_x, batch_y = next(multi_enqueuer.get())
        yield batch_x, batch_y

# Recommended to specify the expected output shapes and types here
output_types = (tf.float32, tf.float32)
output_shapes = (tf.TensorShape([batch_size, 64, 64, nb_of_bands]),
                    tf.TensorShape([batch_size, 3]))
test_ds = tf.data.Dataset.from_generator(test_batch_generator,
                                            output_types=output_types,
                                            output_shapes=output_shapes).repeat()

def get_dataset(only_features=False):
    x_train, y_train  = next(iter(test_ds))
    print('entrée fonction')
    if only_features:
        print('if = true')
        print(x_train.shape)
        print(x_train[:1].shape)
        ds = tf.data.Dataset.from_tensor_slices((np.expand_dims(x_train[:1], axis = 0).astype("float32")))
    else:
        ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds = ds.cache().prefetch(tf.data.experimental.AUTOTUNE)
    return ds

# IPU
# Create an IPU distribution strategy
strategy = ipu.ipu_strategy.IPUStrategy()
#with ipu_scope("/device:IPU:0"):
with strategy.scope():
    #### Model definition
    model_choice = 'wo_ls'
    # With latent space
    if model_choice == 'ls':
        net = model_ipu.create_model(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)
    # Without latent space
    if model_choice == 'wo_ls':
        net = model_ipu.create_model_wo_ls_2(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)
    # Full probabilistic model
    if model_choice == 'full_prob':
        net = model_ipu.create_model_full_prob(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)
    
    net.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3), 
                loss="mean_squared_error")



    net.load_weights('test')#loading_path)
    print('weights loaded')
    print('ici')
    noise_data = get_dataset(only_features=True)
    print('get dataset ok')
    out = net.predict(noise_data)
    print('prediction ok')
    print(out)