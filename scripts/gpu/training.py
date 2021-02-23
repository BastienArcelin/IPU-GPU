#### Import librairies
import sys
import os
import numpy as np
import pandas as pd
import time
import tensorflow as tf

# Needed files
#sys.path.insert(0,'../../scripts/tools_for_VAE/')
#from tools_for_VAE import model_gpu
import model_gpu, callbacks
from callbacks import time_callback

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

######## Parameters
nb_of_bands = 6
batch_size = 4
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

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]
# Load data_dir from environment variables
data_dir = str(os.environ.get('BENCHMARK_DATA'))

list_of_samples = [x for x in listdir_fullpath(data_dir) if x.endswith('.npy')]
list_of_samples_labels = [x for x in listdir_fullpath(data_dir) if x.endswith('.csv')]

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
    negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)
# Full probabilistic model
if model_choice == 'full_prob':
    net = model_gpu.create_model_full_prob(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)
    kl = sum(net.losses)
    alpha = K.variable(0.)
    negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)+ kl *(K.get_value(alpha)-1)


net.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3), 
            loss=negative_log_likelihood)#"mean_squared_error"

## Callbacks 
time_c = time_callback()
######## Train the network
t_1 = time.time()
hist = net.fit(x_train, y_train, 
                batch_size = batch_size, 
                epochs=620,
                steps_per_epoch=steps_per_epoch,
                verbose=1,
                shuffle=True,
                validation_data=(x_val,y_val),
                validation_steps=0,
                callbacks = [time_c])
t_2 = time.time()

print('training in '+str(t_2-t_1)+' seconds')
net.summary()

#saving_path = '/sps/lsst/users/barcelin/weights/gpu_benchmark/'
#net.save_weights(saving_path+'test')
