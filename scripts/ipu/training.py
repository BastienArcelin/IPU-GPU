#### Import librairies
import sys
import os
import numpy as np
import pandas as pd
import time
import tensorflow as tf
import tensorflow.keras.backend as K

# IPU 
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.python import ipu
from tensorflow.python.ipu.scopes import ipu_scope
cfg = ipu.utils.create_ipu_config()#profiling=True,
                                  #profile_execution=True,
                                  #report_directory='fixed_fullModel'
cfg = ipu.utils.auto_select_ipus(cfg, 1)


# Handle CMD arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--replication-factor', type=int, default=2,
                    help="Number of IPUs to replicate across")
opts = parser.parse_args()
# Auto select as many IPUs as we want to replicate across
# ...(must be a power of 2 - IPU driver MultiIPUs come only in powers of 2)
#cfg = ipu.utils.auto_select_ipus(cfg, opts.replication_factor)

ipu.utils.configure_ipu_system(cfg)

# Needed files
import model_ipu
from callbacks import time_callback

######## Parameters
nb_of_bands = 1
batch_size = 12

input_shape = (None,64, 64, nb_of_bands)
hidden_dim = 256
latent_dim = 32
final_dim = 3
filters = [32, 64, 128, 256]
kernels = [3,3,3,3]

conv_activation = None
dense_activation = None

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

# Load data_dir from environment variables
data_dir = str(os.environ.get('BENCHMARK_DATA'))

list_of_samples = [x for x in listdir_fullpath(data_dir) if x.endswith('.npy')]
list_of_samples_data = [x for x in listdir_fullpath(data_dir) if x.endswith('.csv')]
print(list_of_samples)


def create_dataset(batch_size):
    data = np.load(list_of_samples[0], mmap_mode = 'c')
    data_label = pd.read_csv(list_of_samples_data[0])
    x_train = tf.transpose(data[:8000,1], perm= [0,2,3,1])[:,:,:,4:4+nb_of_bands]
    y_train = np.zeros((8000,3))
    y_train[:,0] = data_label[:8000]['e1']
    y_train[:,1] = data_label[:8000]['e2']
    y_train[:,2] = data_label[:8000]['redshift']
    y_train = tf.convert_to_tensor(y_train)

    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size, drop_remainder=True).repeat()#.shuffle(10000)
    ds_train = ds_train.map(lambda d, l:
                            (tf.cast(d, tf.float32), tf.cast(l, tf.float32)))
    steps_per_epoch = int(len(x_train)/batch_size)
    return ds_train, steps_per_epoch

# IPU
# Create an IPU distribution strategy
strategy = ipu.ipu_strategy.IPUStrategy()
with strategy.scope():
    #### Model definition
    model_choice = 'full_prob'
    # Fully deterministic model
    if model_choice == 'det':
        net = model_ipu.create_model_det(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)
        negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)
    # Full probabilistic model
    if model_choice == 'full_prob':
        net = model_ipu.create_model_full_prob(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)
        kl = sum(net.losses)
        alpha = K.variable(0.5)
        negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)+ kl *(K.get_value(alpha)-1)

    net.summary()
    #### Compile network
    #net.load_weights('test')
    # Custom metrics
    def kl_metric(y_true, y_pred):
        return K.sum(net.losses)
    
    net.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-4), 
                loss=negative_log_likelihood,
                metrics=['acc',kl_metric])#negative_log_likelihood)#"mean_squared_error"

    ds_train, steps_per_epoch = create_dataset(batch_size)
    time_c = time_callback()
######## Train the network
    t_1 = time.time()
    hist = net.fit(ds_train, steps_per_epoch=steps_per_epoch, epochs=1200, verbose = 1, callbacks = [time_c])
    t_2 = time.time()

    print('training took '+str(t_2-t_1)+' seconds')
    net.summary()

    net.save_weights('test')