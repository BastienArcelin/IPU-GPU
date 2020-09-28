#### Import librairies
import sys
import os
import numpy as np
import pandas as pd
import time
import tensorflow as tf

# IPU 
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.python import ipu
from tensorflow.python.ipu.scopes import ipu_scope
cfg = ipu.utils.create_ipu_config()#profiling=True,
                                  #profile_execution=True,
                                  #report_directory='fixed_fullModel'
cfg = ipu.utils.auto_select_ipus(cfg, 1)
ipu.utils.configure_ipu_system(cfg)

# Needed files
#sys.path.insert(0,'../../scripts/tools_for_VAE/')
#from tools_for_VAE import model_ipu
import model_ipu
from callbacks import time_callback

######## Parameters
nb_of_bands = 6
batch_size = 1

input_shape = (64, 64, nb_of_bands)
hidden_dim = 256
latent_dim = 32
final_dim = 3
filters = [32, 64, 128, 256]
kernels = [3,3,3,3]
bands = [4,5,6,7,8,9]

conv_activation = None
dense_activation = None

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

images_dir = '/home/astrodeep/bastien/data/'

list_of_samples = [x for x in listdir_fullpath(os.path.join(images_dir,'test')) if x.endswith('.npy')]
list_of_samples_data = [x for x in listdir_fullpath(os.path.join(images_dir,'test')) if x.endswith('.csv')]
print(list_of_samples)


def create_dataset(batch_size):
    data = np.load(list_of_samples[0], mmap_mode = 'c')
    data_label = pd.read_csv(list_of_samples_data[0])
    x_train = tf.transpose(data[:8000,1], perm= [0,2,3,1])[:,:,:,4:]
    y_train = np.zeros((8000,3))
    y_train[:,0] = data_label[:8000]['e1']
    y_train[:,1] = data_label[:8000]['e2']
    y_train[:,2] = data_label[:8000]['redshift']
    y_train = tf.convert_to_tensor(y_train)

    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size, drop_remainder=True).repeat()
    ds_train = ds_train.map(lambda d, l:
                            (tf.cast(d, tf.float32), tf.cast(l, tf.float32)))
    steps_per_epoch = int(len(x_train)/batch_size)
    return ds_train, steps_per_epoch

# IPU
# Create an IPU distribution strategy
strategy = ipu.ipu_strategy.IPUStrategy()
with strategy.scope():
    #### Model definition
    model_choice = 'det'
    # Fully deterministic model
    if model_choice == 'det':
        net = model_ipu.create_model_det(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)
    # Full probabilistic model
    if model_choice == 'full_prob':
        net = model_ipu.create_model_full_prob(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)


    #### Loss definition
    if model_choice == 'full_prob':
        kl = sum(net.losses)
        negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x) + kl *(-1+1/(batch_size))

    else:
        negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)
    
    #### Compile network
    net.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3), 
                loss="mean_squared_error")

    ds_train, steps_per_epoch = create_dataset(batch_size)
    time_c = time_callback()
######## Train the network
    t_1 = time.time()
    hist = net.fit(ds_train, steps_per_epoch=steps_per_epoch, epochs=112, verbose = 1, callbacks = [time_c])#1125#9000
    t_2 = time.time()

    print('training took '+str(t_2-t_1)+' seconds')
    net.summary()

    net.save_weights('test')

