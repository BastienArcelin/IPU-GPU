import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time
import os
import model_ipu
import pandas as pd
print(tf.__version__)

from tensorflow.python.ipu import ipu_compiler,         \
                                  ipu_optimizer,        \
                                  ipu_estimator,        \
                                  scopes,               \
                                  loops,                \
                                  ipu_infeed_queue,     \
                                  ipu_outfeed_queue,    \
                                  utils,                \
                                  gradient_accumulation_optimizer as gao
from tensorflow.python.ipu.ops import normalization_ops


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

######## Parameters
nb_of_bands = 6
batch_size = 8

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

ds_train, steps_per_epoch = create_dataset(batch_size)

infeed_GAN = ipu_infeed_queue.IPUInfeedQueue(ds_train, feed_name='in_GAN')

outfeed_FULL = ipu_outfeed_queue.IPUOutfeedQueue(feed_name='out_FULL')

outfeed_test = ipu_outfeed_queue.IPUOutfeedQueue(feed_name='out_test')


with tf.device("cpu"):
        numPoints = tf.placeholder(np.int32, shape=(), name="numPoints")

from tensorflow.keras.layers import Input, Flatten, Dense, Reshape, Dropout, LeakyReLU, Conv2DTranspose, Conv2D, BatchNormalization, Activation
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


_EPSILON = K.epsilon()

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

def _loss_generator(y_true, y_pred):
    return tf.keras.losses.MeanSquaredError(y_true,y_pred)


# Build Generator model ...
with tf.variable_scope('gen'):
    #### Model definition
    model_choice = 'det'
    # Fully deterministic model
    if model_choice == 'det':
        net = model_ipu.create_model_det(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)
    # Full probabilistic model
    if model_choice == 'full_prob':
        net = model_ipu.create_model_full_prob(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)

net.summary()
optimizer_D = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08)


def train_model(X):
    g_vars = tf.trainable_variables('gen')  

    ds_train, steps_per_epoch = create_dataset(batch_size)
    fake_images = Generator(noise)

    real_images = X

    in_values, labels_D = next(ds_train)
    #labels_D = tf.concat([labels_D_0, labels_D_1],0)

    out_values = net(in_values)

    loss_D = _loss_generator(labels_D,out_values)
    training_op_D = optimizer_D.minimize(loss_D, var_list = g_vars)

    return outfeed_FULL.enqueue([loss_D]), training_op_D



def training_loop_FULL(numPoints):
    out = loops.repeat(numPoints, train_model, infeed_queue=infeed_GAN)
    return out



# Compile the graph with the IPU custom xla compiler
with scopes.ipu_scope("/device:IPU:0"):
    compiled_FULL = ipu_compiler.compile(training_loop_FULL, [numPoints])


# Ops to read the outfeed and initialize all variables
dequeue_outfeed_op_FULL = outfeed_FULL.dequeue()

init_op = tf.global_variables_initializer()


cfg = utils.create_ipu_config(profiling=True, use_poplar_text_report=True)
cfg = utils.auto_select_ipus(cfg, 1)
utils.configure_ipu_system(cfg)


t0 = time.time()

loss_list = np.empty((0,3))
# Run the model
with tf.Session() as sess:

    # Initialize
    sess.run(init_op)
    sess.run(infeed_GAN.initializer)
    # Run
    print('Running...')

    t0 = time.time()

    sess.run(compiled_FULL, feed_dict={numPoints: 1})

    t1 = time.time()

    losses = sess.run(dequeue_outfeed_op_FULL)

    print('time for warm up',t1-t0)


    t0 = time.time()

    sess.run(compiled_FULL, feed_dict={numPoints: 1000})

    t1 = time.time()

    losses = sess.run(dequeue_outfeed_op_FULL)

    print('time for 1000',t1-t0)



    t0 = time.time()

    sess.run(compiled_FULL, feed_dict={numPoints: 1000})

    t1 = time.time()

    losses = sess.run(dequeue_outfeed_op_FULL)

    print('time for 1000',t1-t0)


    t0 = time.time()

    sess.run(compiled_FULL, feed_dict={numPoints: 1000})

    t1 = time.time()

    losses = sess.run(dequeue_outfeed_op_FULL)

    print('time for 1000',t1-t0)







