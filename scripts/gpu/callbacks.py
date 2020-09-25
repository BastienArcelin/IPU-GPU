import sys, os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, TerminateOnNaN, ModelCheckpoint
import tensorflow.keras.backend as K
import tensorflow as tf
import time

###### Callbacks
# Create a callback to compute time spent between 10th and 110th epoch
class time_callback(Callback):
    def __init__(self):
        '''
        Compute time spent between 10th and 110th epoch
        '''
        self.epoch = 1
        self.t1 =0
        self.t2 = 0
    
    def on_epoch_end(self, epoch, t1):
        if (self.epoch == 10):
            self.t1 =time.time()
            print('t1: '+str(self.t1))
        elif (self.epoch == 110):
            self.t2 = time.time()
            print('t2: '+str(self.t2))
            print('for 100 epochs from 10 to 110: '+str(self.t2 - self.t1))
        self.epoch +=1