#### Import librairies
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
import tensorflow as tf

### Needed files
#sys.path.insert(0,'../../scripts/tools_for_VAE/')
#from tools_for_VAE import model_gpu
import model_gpu

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


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
bands = [4,5,6,7,8,9]

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]
# Load data_dir from environment variables
data_dir = str(os.environ.get('BENCHMARK_DATA'))

list_of_samples_test = [x for x in listdir_fullpath(data_dir) if x.endswith('.npy')]
list_of_samples_test_labels = [x for x in listdir_fullpath(data_dir) if x.endswith('.csv')]

data = np.load(list_of_samples_test[0], mmap_mode = 'c')
data_label = pd.read_csv(list_of_samples_test_labels[0])

### Define samples
x_train = tf.transpose(data[:8000,1], perm= [0,2,3,1])[:,:,:,4:]
y_train = np.zeros((8000,3))
y_train[:,0] = data_label[:8000]['e1']
y_train[:,1] = data_label[:8000]['e2']
y_train[:,2] = data_label[:8000]['redshift']
y_train = tf.convert_to_tensor(y_train)

x_val = tf.transpose(data[8000:,1], perm= [0,2,3,1])[:,:,:,4:]
y_val = np.zeros((2000,3))
y_val[:,0] = data_label[8000:]['e1']
y_val[:,1] = data_label[8000:]['e2']
y_val[:,2] = data_label[8000:]['redshift']
y_val = tf.convert_to_tensor(y_val)

#### Model definition
model_choice = 'det'
# Fully deterministic model
if model_choice == 'det':
    net = model_gpu.create_model_det(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)
# Full probabilistic model
if model_choice == 'full_prob':
    net = model_gpu.create_model_full_prob(input_shape, latent_dim, hidden_dim, filters, kernels, final_dim, conv_activation=None, dense_activation=None)


net.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-4), 
            loss="mean_squared_error")

net.summary()

### Load model
loading_path = '/sps/lsst/users/barcelin/weights/gpu_benchmark/'
latest = tf.train.latest_checkpoint(loading_path)
net.load_weights(latest)

print('weights loaded')

### Do inference
## In once
t0 = time.time()
out = net(x_val)
t1 = time.time()

## One by one
# out = []
# for i in range (1001):
#     if i == 1:
#         t0 = time.time()
#     out.append(net(np.expand_dims(x_val[i], axis = 1)))
# t1 = time.time()
# out = np.array(out)

print('prediction done in: '+str(t1-t0)+' seconds')

### Plot results
fig = plt.figure()
sns.distplot(out.mean().numpy()[:,0], bins = 20)
sns.distplot(y_val[:,0], bins = 20)
fig.savefig('test_distrib_e1.png')


fig = plt.figure()
sns.distplot(out.mean().numpy()[:,1], bins = 20)
sns.distplot(y_val[:,1], bins = 20)
fig.savefig('test_distrib_e2.png')


fig = plt.figure()
sns.distplot(out.mean().numpy()[:,2], bins = 20)
sns.distplot(y_val[:,2], bins = 20)
fig.savefig('test_distrib_e3.png')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(y_val[:,0], out.mean().numpy()[:,0], '.', label = 'mean')
axes[0].plot(y_val[:,0], out.mean().numpy()[:,0]+ 2*out.stddev().numpy()[:,0], '+', label = 'mean + 2stddev')
axes[0].plot(y_val[:,0], out.mean().numpy()[:,0]- 2*out.stddev().numpy()[:,0], '+', label = 'mean - 2stddev')
x = np.linspace(-1,1)
axes[0].plot(x, x)
axes[0].legend()
axes[0].set_ylim(-1,1)
axes[0].set_title('$e1$')

axes[1].plot(y_val[:,1], out.mean().numpy()[:,1], '.', label = 'mean')
axes[1].plot(y_val[:,1], out.mean().numpy()[:,1]+ 2*out.stddev().numpy()[:,1], '+', label = 'mean + 2stddev')
axes[1].plot(y_val[:,1], out.mean().numpy()[:,1]- 2*out.stddev().numpy()[:,1], '+', label = 'mean - 2stddev')
x = np.linspace(-1,1)
axes[1].plot(x, x)
axes[1].legend()
axes[1].set_ylim(-1,1)
axes[1].set_title('$e2$')

axes[2].plot(y_val[:,2], out.mean().numpy()[:,2], '.', label = 'mean')
axes[2].plot(y_val[:,2], out.mean().numpy()[:,2]+ 2*out.stddev().numpy()[:,2], '+', label = 'mean + 2stddev')
axes[2].plot(y_val[:,2], out.mean().numpy()[:,2]- 2*out.stddev().numpy()[:,2], '+', label = 'mean - 2stddev')
x = np.linspace(0,4)
axes[2].plot(x, x)
axes[2].legend()
axes[2].set_ylim(-1,5.5)
axes[2].set_title('$z$')

fig.savefig('test_train.png')