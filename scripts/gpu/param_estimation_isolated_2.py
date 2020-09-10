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
from tools_for_VAE import utils, vae_functions, generator, model
from tensorflow.keras import backend as K


######## Parameters
nb_of_bands = 6
batch_size = 8

input_shape = (64, 64, nb_of_bands)
hidden_dim = 256
latent_dim = 32
final_dim = 3
filters = [32, 64, 128, 256]#, 512]
kernels = [3,3,3,3]#, 3]

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


#With generator (unchanged)
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

test_generator = generator.BatchGenerator(bands, list_of_samples_test, total_sample_size=None,
                                    batch_size=batch_size, 
                                    trainval_or_test='test',
                                    do_norm=False,
                                    denorm = False,
                                    list_of_weights_e=None)


# NEW: Wrap the generator.BatchGenerator objects in a generator-style function
# which we can then pass to tf.data.Dataset.from_generator()
# (One per train/val/test dataset at the moment, but could be refactored for neatness!)
def training_batch_generator():
    multi_enqueuer = keras.utils.OrderedEnqueuer(training_generator,
                                                use_multiprocessing=False)
    multi_enqueuer.start(workers=10, max_queue_size=10)
    while True:
        batch_x, batch_y = next(multi_enqueuer.get())
        yield batch_x, batch_y

def validation_batch_generator():
    multi_enqueuer = keras.utils.OrderedEnqueuer(validation_generator,
                                                    use_multiprocessing=False)
    multi_enqueuer.start(workers=10, max_queue_size=10)
    while True:
        batch_x, batch_y = next(multi_enqueuer.get())
        yield batch_x, batch_y

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

training_ds = tf.data.Dataset.from_generator(training_batch_generator,
                                                output_types=output_types,
                                                output_shapes=output_shapes).repeat()

validation_ds = tf.data.Dataset.from_generator(validation_batch_generator,
                                                output_types=output_types,
                                                output_shapes=output_shapes).repeat()

test_ds = tf.data.Dataset.from_generator(test_batch_generator,
                                            output_types=output_types,
                                            output_shapes=output_shapes).repeat()



#### Model definition
model_choice = 'wo_ls'
# With latent space
if model_choice == 'ls':
    net = model.create_model(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)
# Without latent space
if model_choice == 'wo_ls':
    net = model.create_model_wo_ls_2(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)
# Full probabilistic model
if model_choice == 'full_prob':
    net = model.create_model_full_prob(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)


#### Loss definition
if model_choice == 'full_prob':
    kl = sum(net.losses)
    negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x) + kl *(-1+1/(batch_size))

else:
    negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)


net.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3), 
            loss="mean_absolute_error")


#loading_path = '/home/astrodeep/bastien/weights/'
#latest = tf.train.latest_checkpoint(loading_path)
#net.load_weights(latest)


######## Train the network
# hist = net.fit(training_ds, steps_per_epoch=2, epochs=1, verbose = 1)#576
hist = net.fit(training_ds, epochs=10,
                    steps_per_epoch=steps_per_epoch,
                    verbose=1,
                    shuffle=True,
                    validation_data=validation_ds,
                    validation_steps=validation_steps)#,
                    #workers=0,#4 
                    #use_multiprocessing = True)


net.summary()

    # hist = net.fit(training_data, training_labels, epochs=50, # training
    #                     steps_per_epoch=steps_per_epoch,#128
    #                     verbose=1,
    #                     shuffle=True,
    #                     validation_data=(validation_data,validation_labels), # validation
    #                     validation_steps=validation_steps)#,#16
    #                     #workers=0,#4 
    #                     #use_multiprocessing = True)

saving_path = '/home/astrodeep/bastien/weights/'#test_5
net.save_weights('test')


#### Plots
# test_generator.__getitem__(2)

# training_data = test[0]
# training_labels = test[1]
# new_net = model.create_model_wo_ls_3(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)
# new_net.summary()
# loading_path = '/home/astrodeep/bastien/weights/cp.ckpt'
#latest = tf.train.latest_checkpoint(loading_path)
#new_net.load_weights(latest)


net.load_weights('test')#loading_path)
print('weights loaded')



training_data, training_labels  = next(iter(test_ds))
print(training_data.numpy().shape)


#ds = tf.data.Dataset.from_tensor_slices([training_data]).batch(8, drop_remainder=True)

# output_types = (tf.float32, tf.float32)
# output_shapes = (tf.TensorShape([batch_size, 64, 64, nb_of_bands]),
#                     tf.TensorShape([batch_size, 3]))
# ds = tf.data.Dataset.from_tensor_generator(test_batch_generator,
#                                              output_types=output_types,
#                                              output_shapes=output_shapes)
#out = net.evaluate(test_ds, steps = 1)#.batch(8, drop_remainder=True)
#print('evaluate ok: '+str(out))
ds = tf.data.Dataset.from_tensor_slices((tf.cast(training_data, tf.float32)))

out = net.predict(ds.batch(8, drop_remainder=True))#dataset # ,steps_per_epoch = 1)#training_data     , steps = 1,steps_per_run=8

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
fig.savefig('test_distrib_e3.png')

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

