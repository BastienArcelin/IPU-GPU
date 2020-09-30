#### Import librairies
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import time

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
#from tools_for_VAE import generator, model_ipu
import generator
import model_ipu

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

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

# Load data_dir from environment variables
data_dir = str(os.environ.get('BENCHMARK_DATA'))

list_of_samples = [x for x in listdir_fullpath(data_dir) if x.endswith('.npy')]
list_of_samples_data = [x for x in listdir_fullpath(data_dir) if x.endswith('.csv')]
print(list_of_samples)

###### Generator
test_generator = generator.BatchGenerator(bands, list_of_samples, total_sample_size=None,
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

def get_dataset(only_features=False, size = 2000):
    #x_train, y_train  = next(iter(test_ds))
    data = np.load(list_of_samples[0], mmap_mode = 'c')
    data_label = pd.read_csv(list_of_samples_data[0])
    x_train = tf.transpose(data[8000:8000+size,1], perm= [0,2,3,1])[:,:,:,4:]
    y_train = np.zeros((size,3))
    y_train[:,0] = data_label[8000:8000+size]['e1']
    y_train[:,1] = data_label[8000:8000+size]['e2']
    y_train[:,2] = data_label[8000:8000+size]['redshift']
    y_train = tf.convert_to_tensor(y_train)

    if only_features:
        ds = tf.data.Dataset.from_tensor_slices((x_train)).batch(batch_size, drop_remainder=True)
        ds = ds.map(lambda d:
                    (tf.cast(d, tf.float32)))
    else:
        ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size, drop_remainder=True).repeat()
        ds = ds.map(lambda d, l:
                    (tf.cast(d, tf.float32), tf.cast(l, tf.float32)))
    ds = ds.cache().prefetch(tf.data.experimental.AUTOTUNE)
    return ds, y_train

# IPU
# Create an IPU distribution strategy
strategy = ipu.ipu_strategy.IPUStrategy()
#with ipu_scope("/device:IPU:0"):
with strategy.scope():
    #### Model definition
    model_choice = 'det'
    # Fully deterministic model
    if model_choice == 'det':
        net = model_ipu.create_model_det(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)
    # Full probabilistic model
    if model_choice == 'full_prob':
        net = model_ipu.create_model_full_prob(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)

    net.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3), 
                loss="mean_squared_error")

# Load model weights
    net.load_weights('test')

    # Do inference
    ## In once
    # Warm up stage
    noise_data, y = get_dataset(only_features=True, size = 8)
    print(noise_data)
    net.predict(noise_data, steps_per_run = 1)
    print('warm up OK')
    # Prediction
    noise_data, y = get_dataset(only_features=True, size = 2000)
    t0 = time.time()
    out = net.predict(noise_data)
    t1 = time.time()
    
    ## One by one
    # out_res = []
    # out_label = []
    # noise_data, y = get_dataset(only_features=True)
    # for i in range (11):
    #     print(i)
    #     if i ==1:
    #         t0 = time.time()
    #     out_label.append(y)
    #     out = net.predict(noise_data)
    #     out_res.append(out)
    #     print(out, y)
    # t1 = time.time()
    print('prediction ok, time: '+str(t1-t0))
    #print(out_res)


#### Plots
print(np.shape(out))
fig = plt.figure()
sns.distplot(out[0][:,0], bins = 20, label = 'output')
sns.distplot(y[:,0], bins = 20, label = 'input')
fig.savefig('test_distrib_e1.png')


fig = plt.figure()
sns.distplot(out[0][:,1], bins = 20)
sns.distplot(y[:,1], bins = 20)
fig.savefig('test_distrib_e2.png')


fig = plt.figure()
sns.distplot(out[0][:,2], bins = 20)
sns.distplot(y[:,2], bins = 20)
fig.savefig('test_distrib_e3.png')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(y[:,0], out[0][:,0], '.', label = 'mean')
x = np.linspace(-1,1)
axes[0].plot(x, x)
axes[0].legend()
axes[0].set_ylim(-1,1)
axes[0].set_title('$e1$')

axes[1].plot(y[:,1], out[0][:,1], '.', label = 'mean')
x = np.linspace(-1,1)
axes[1].plot(x, x)
axes[1].legend()
axes[1].set_ylim(-1,1)
axes[1].set_title('$e2$')

axes[2].plot(y[:,2], out[0][:,2], '.', label = 'mean')
x = np.linspace(0,4)
axes[2].plot(x, x)
axes[2].legend()
axes[2].set_ylim(-1,5.5)
axes[2].set_title('$z$')

fig.savefig('test_train.png')

