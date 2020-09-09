import sys, os
import numpy as np
import matplotlib.pyplot as plt


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, Dense, Dropout, MaxPool2D, Flatten, BatchNormalization, Reshape, UpSampling2D, Cropping2D, Conv2DTranspose, PReLU, Concatenate, Lambda
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, TerminateOnNaN, ModelCheckpoint
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras import metrics


###### Callbacks
# Create a callback for changing KL coefficient in the loss
class changeAlpha(Callback):
    def __init__(self, alpha, vae, vae_loss, path):
        self.epoch = 1
        self.alpha = alpha
        self.vae = vae
        self.vae_loss = vae_loss
        self.path = path
        #self.epochs = epochs
    
    def on_epoch_end(self, alpha, vae):
        stable = 10
        new_alpha = 0.0001
        if self.epoch > stable and K.get_value(self.alpha)>1e-8 :
            #coef = self.epoch - (self.epochs + stable)/2
            #print(coef)
            new_alpha = K.get_value(self.alpha)/2#1/(1+np.exp(-(coef)))*0.01
            print(new_alpha, self.epoch)
            K.set_value(self.alpha, new_alpha)
            self.vae.compile('adam', loss=self.vae_loss, metrics=['mse'])
            K.set_value(self.vae.optimizer.lr, 0.0001)
            print('loss modified')
            self.epoch = 1
            np.save(self.path+'alpha', K.get_value(self.alpha))
        
        self.epoch +=1

# Create a callback to change learning rate of the optimizer during training
class changelr(Callback):
    def __init__(self,vae):
        self.epoch = 0
        self.vae = vae
    
    def on_epoch_end(self, alpha, vae):
        if (self.epoch == 100):
            self.epoch =0
            # actual_value = K.get_value(self.vae.optimizer.lr)
            # if (actual_value > 0.000009):
            #     new_value = actual_value/2
            #     K.set_value(self.vae.optimizer.lr, new_value)
            #     print(K.get_value(self.vae.optimizer.lr))
            if self.epoch == 200:
                K.set_value(self.vae.optimizer.lr, 10-5)
        self.epoch +=1