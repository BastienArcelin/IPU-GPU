#### Import librairies
import sys
import os
import numpy as np
import pandas as pd
import time
import tensorflow as tf

# Needed files
sys.path.insert(0,'../../scripts/tools_for_VAE/')
from tools_for_VAE import utils, model_gpu

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

######## Parameters
nb_of_bands = 6
batch_size = 14
print('The batchsize is : '+str(batch_size))

input_shape = (64, 64, nb_of_bands)
hidden_dim = 256
latent_dim = 32
final_dim = 3
filters = [32, 64, 128, 256]
kernels = [3,3,3,3]

conv_activation = None
dense_activation = None

bands = [4,5,6,7,8,9]

### Loading data
images_dir = '/sps/lsst/users/barcelin/data/TFP/GalSim_COSMOS/isolated_galaxies/centered'

list_of_samples = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'test')) if x.endswith('.npy')]
list_of_samples_labels = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'test')) if x.endswith('.csv')]

data = np.load(list_of_samples[0], mmap_mode = 'c')
data_label = pd.read_csv(list_of_samples_labels[0])

### Create samples
x_train = tf.transpose(data[:8000,1], perm= [0,2,3,1])[:,:,:,4:]
y_train = np.zeros((8000,3))
y_train[:,0] = data_label[:8000]['e1']
y_train[:,1] = data_label[:8000]['e2']
y_train[:,2] = data_label[:8000]['redshift']
y_train = tf.convert_to_tensor(y_train)

x_val = tf.transpose(data[8000:,1], perm= [0,2,3,1])[:,:,:,4:]
y_val = np.zeros((10000-len(x_train),3))
y_val[:,0] = data_label[8000:]['e1']
y_val[:,1] = data_label[8000:]['e2']
y_val[:,2] = data_label[8000:]['redshift']
y_val = tf.convert_to_tensor(y_val)

### Define number of steps
steps_per_epoch = int(len(x_train)/batch_size)
validation_steps = int((10000-len(x_train))/batch_size)


#### Model definition
model_choice = 'det'
# Fully deterministic model
if model_choice == 'det':
    net = model_gpu.create_model_det(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)
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
t_1 = time.time()
hist = net.fit(x_train, y_train, 
                batch_size = batch_size, 
                epochs=100,
                steps_per_epoch=steps_per_epoch,
                verbose=1,
                shuffle=True,
                validation_data=(x_val,y_val),
                validation_steps=0)
t_2 = time.time()

print('training in '+str(t_2-t_1)+' seconds')
net.summary()

saving_path = '/sps/lsst/users/barcelin/weights/gpu_benchmark/'
net.save_weights(saving_path+'test')