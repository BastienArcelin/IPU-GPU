#### Import librairies
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

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
sys.path.insert(0,'../../scripts/tools_for_VAE/')
from tools_for_VAE import generator, model_ipu


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

images_dir = '/home/astrodeep/bastien/data/'


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
        ds = tf.data.Dataset.from_tensor_slices((x_train)).batch(batch_size, drop_remainder=True)
    else:
        ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
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


    net.load_weights('test')#loading_path)
    print('weights loaded')
    print('ici')
    #for i in range (100)
    
    print('get dataset ok')
    out_res = []
    out_label = []
    for i in range (10):
        print(i)
        noise_data, y = get_dataset(only_features=True)
        out_label.append(y)
        out = net.predict(noise_data)
        out_res.append(out)
    print('prediction ok')
    print(out)


#### Plots
    # net.load_weights('test')#loading_path)
    # print('weights loaded')
    # print('ici')
    # noise_data = get_dataset(only_features=True)
    # print('get dataset ok')
    # out = net.predict(noise_data)
    # print('prediction ok')
    # print(out)


# fig = plt.figure()
# sns.distplot(out.mean().numpy()[:,0], bins = 20)
# sns.distplot(training_labels[:,0], bins = 20)
# fig.savefig('test_distrib_e1.png')


# fig = plt.figure()
# sns.distplot(out.mean().numpy()[:,1], bins = 20)
# sns.distplot(training_labels[:,1], bins = 20)
# fig.savefig('test_distrib_e2.png')


# fig = plt.figure()
# sns.distplot(out.mean().numpy()[:,2], bins = 20)
# sns.distplot(training_labels[:,2], bins = 20)
# fig.savefig('test_distrib_e3.png')

# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# axes[0].plot(training_labels[:,0], out.mean().numpy()[:,0], '.', label = 'mean')
# axes[0].plot(training_labels[:,0], out.mean().numpy()[:,0]+ 2*out.stddev().numpy()[:,0], '+', label = 'mean + 2stddev')
# axes[0].plot(training_labels[:,0], out.mean().numpy()[:,0]- 2*out.stddev().numpy()[:,0], '+', label = 'mean - 2stddev')
# x = np.linspace(-1,1)
# axes[0].plot(x, x)
# axes[0].legend()
# axes[0].set_ylim(-1,1)
# axes[0].set_title('$e1$')

# axes[1].plot(training_labels[:,1], out.mean().numpy()[:,1], '.', label = 'mean')
# axes[1].plot(training_labels[:,1], out.mean().numpy()[:,1]+ 2*out.stddev().numpy()[:,1], '+', label = 'mean + 2stddev')
# axes[1].plot(training_labels[:,1], out.mean().numpy()[:,1]- 2*out.stddev().numpy()[:,1], '+', label = 'mean - 2stddev')
# x = np.linspace(-1,1)
# axes[1].plot(x, x)
# axes[1].legend()
# axes[1].set_ylim(-1,1)
# axes[1].set_title('$e2$')

# axes[2].plot(training_labels[:,2], out.mean().numpy()[:,2], '.', label = 'mean')
# axes[2].plot(training_labels[:,2], out.mean().numpy()[:,2]+ 2*out.stddev().numpy()[:,2], '+', label = 'mean + 2stddev')
# axes[2].plot(training_labels[:,2], out.mean().numpy()[:,2]- 2*out.stddev().numpy()[:,2], '+', label = 'mean - 2stddev')
# x = np.linspace(0,4)
# axes[2].plot(x, x)
# axes[2].legend()
# axes[2].set_ylim(-1,5.5)
# axes[2].set_title('$z$')

# fig.savefig('test_train.png')

