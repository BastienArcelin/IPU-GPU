## Load necessary librairies
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
tfk = tf.keras
tfkl = tfk.layers
tfd = tfp.distributions
tfb = tfp.bijectors

import time
import sys
sys.path.insert(0,'')
from flow import *

## Define the normalizing flow
hidden_dim = [256,256]
layers =8
bijectors = []
for i in range(0, layers):
    made = make_network(32, hidden_dim,2)
    bijectors.append(MAF(made))
    bijectors.append(tfb.Permute(permutation=[31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]))
    
bijectors = tfb.Chain(bijectors=list(reversed(bijectors[:-1])))

distribution = tfd.TransformedDistribution(
    distribution=tfd.Normal(loc=0., scale=1.),
    bijector=bijectors,
    event_shape=[32]
)

x_ = tfkl.Input(shape=(32,), dtype=tf.float32)
log_prob_ = distribution.log_prob(x_)
model = tfk.Model(x_, log_prob_)

model.compile(optimizer=tf.optimizers.Adam(), loss=lambda _, log_prob: -log_prob)
print('flow defined')

## Load weights
loading_path = '../../../nflow_weights/'
latest = tf.train.latest_checkpoint(loading_path)
model.load_weights(latest)

### Do inference
## Warm-up 
samples = distribution.sample(1000)
print('warm-up over')

## Actual inference
t0 = time.time()
samples = distribution.sample(100000)
t1 = time.time()

print('time for inference:' + str(t1-t0))