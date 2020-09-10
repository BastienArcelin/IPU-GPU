import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization
from tensorflow.python import ipu
import tensorflow_probability as tfp
import numpy as np
import os
import pandas as pd
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops

import sys
sys.path.insert(0,'../../scripts/tools_for_VAE/')
import tools_for_VAE.layers as layers
from tools_for_VAE import utils, vae_functions, generator, model


cfg = ipu.utils.create_ipu_config(profiling=True,
                                  profile_execution=True,
                                  report_directory='single_compilation')
cfg = ipu.utils.auto_select_ipus(cfg, 1)
ipu.utils.configure_ipu_system(cfg)

batch_size = 4

class BatchGenerator(keras.utils.Sequence):
    """
    Class to create batch generator for the LSST VAE.
    """
    def __init__(self, list_of_samples, batch_size):
        """
        Initialization function
        """
        self.bands = [4,5,6,7,8,9]
        self.nbands = len(self.bands)
        self.batch_size = batch_size
        self.list_of_samples = list_of_samples

        self.epoch = 0

        # Weights computed from the lengths of lists
        self.p = []
        for sample in self.list_of_samples:
            temp = np.load(sample, mmap_mode = 'c')
            self.p.append(float(len(temp)))
        self.p = np.array(self.p)
        self.total_sample_size = int(np.sum(self.p))
        print("[BatchGenerator] total_sample_size = ", self.total_sample_size)
        print("[BatchGenerator] len(list_of_samples) = ", len(self.list_of_samples))

        self.p /= np.sum(self.p)

        self.produced_samples = 0


    def __len__(self):
        """
        Function to define the length of an epoch
        """
        return int(float(self.total_sample_size) / float(self.batch_size))      


    def on_epoch_end(self):
        """
        Function executed at the end of each epoch
        """
        self.produced_samples = 0


    def __getitem__(self, idx):
        """
        Function which returns the input and target batches for the network
        """
        index = np.random.choice(list(range(len(self.p))), p=self.p)
        sample_filename = self.list_of_samples[index]
        filename = 'galaxies_blended_20191024_0_images.npy'
        sample = np.load(sample_filename, mmap_mode = 'c')
        data = pd.read_csv(sample_filename.replace('images.npy','data_classified.csv'))

        new_data = data[(np.abs(data['e1_0'])<=1.) &
                        (np.abs(data['e2_0'])<=1) ]

        indices = np.random.choice(new_data.index, size=self.batch_size, replace=False)
        self.produced_samples += len(indices)

        x = sample[indices,1][:,self.bands]
        
        y = np.zeros((self.batch_size, 3))

        y[:,0] = np.array(new_data['e1_0'][indices])
        y[:,1] = np.array(new_data['e2_0'][indices])
        y[:,2] = np.array(new_data['redshift_0'][indices])
        
        x = tf.transpose(x, perm= [0,2,3,1])

        return x, y


# Most layers taken out as unrelated to issue and caused longer compile time
def create_model():
  h = []
  h.append(BatchNormalization())
  h.append(keras.layers.Flatten())
  h.append(Dense(tfp.layers.MultivariateNormalTriL.params_size(3), activation=None))
  h.append(tfp.layers.MultivariateNormalTriL(3))
  m = ipu.keras.Sequential(h)

  return m


# Point to data files
images_dir = '/home/astrodeep/bastien/data/'
list_of_samples = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'test')) if x.endswith('.npy')]

# keras.utils.Sequence class to read from files
train_generator = BatchGenerator(list_of_samples,
                                 batch_size=batch_size)

# Wrapper function for the keras.utils.Sequence class to generate batches
def generator():
    multi_enqueuer = keras.utils.OrderedEnqueuer(train_generator,
                                                 use_multiprocessing=False) # Doesn't like multiprocessing...
    multi_enqueuer.start(workers=1, max_queue_size=1)
    while True:
        batch_x, batch_y = next(multi_enqueuer.get())
        yield batch_x, batch_y

# Create an IPU distribution strategy
strategy = ipu.ipu_strategy.IPUStrategy()

with strategy.scope():
  # Create an instance of the model
  model = create_model()

  # Create TF dataset from our generator() wrapper function
  output_types = (tf.float32, tf.float32)
  output_shapes = (tf.TensorShape([batch_size, 64, 64, 6]), tf.TensorShape([batch_size, 3]))
  training_ds = tf.data.Dataset.from_generator(generator,
                                               output_types=output_types,
                                               output_shapes=output_shapes).repeat().prefetch(tf.data.experimental.AUTOTUNE)

  # Train the model
#   negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)
#   model.compile(loss=negative_log_likelihood,
#                 optimizer=tf.optimizers.Adam(learning_rate=1e-3))

  model.compile(loss = "categorical_crossentropy", optimizer=tf.optimizers.Adam(learning_rate=1e-3))

  model.fit(training_ds, steps_per_epoch=4, epochs=4)

  model.summary()
