# Import necessary librairies

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import random
import tensorflow.keras
import pandas as pd
import scipy
from scipy.stats import norm

from random import choice

sys.path.insert(0,'../tools_for_VAE/')
from tools_for_VAE import utils


class BatchGenerator(tensorflow.keras.utils.Sequence):
    """
    Class to create batch generator for the LSST VAE.
    """
    def __init__(self, bands, list_of_samples,total_sample_size, batch_size, trainval_or_test, do_norm,denorm, list_of_weights_e):
        """
        Initialization function
        total_sample_size: size of the whole training (or validation) sample
        batch_size: size of the batches to provide
        list_of_samples: list of the numpy arrays which correspond to the whole training (or validation) sample
#        path: path to the first numpy array taken in which the batch will be taken
        training_or_validation: choice between training of validation generator
        x: input of the neural network
        y: target of the neural network
        r: random value to sample into the validation sample
        """
        self.bands = bands
        self.nbands = len(bands)
        self.total_sample_size = total_sample_size
        self.batch_size = batch_size
        self.list_of_samples = list_of_samples
        self.trainval_or_test = trainval_or_test
        
        self.epoch = 0
        self.do_norm = do_norm
        self.denorm = denorm

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
        self.list_of_weights_e = list_of_weights_e
        #self.shifts = shifts

    def __len__(self):
        """
        Function to define the length of an epoch
        """
        return int(float(self.total_sample_size) / float(self.batch_size))      

    def on_epoch_end(self):
        """
        Function executed at the end of each epoch
        """
        # indices = 0
        #print("Produced samples", self.produced_samples)
        self.produced_samples = 0
        
    def __getitem__(self, idx):
        """
        Function which returns the input and target batches for the network
        """
        # If the generator is a training generator, the whole sample is displayed
        #sample_filename = np.random.choice(self.list_of_samples, p=self.p)
        #index = np.random.choice(1)
        index = np.random.choice(list(range(len(self.p))), p=self.p)
        sample_filename = self.list_of_samples[index]
        sample = np.load(sample_filename, mmap_mode = 'c')
        data = pd.read_csv(sample_filename.replace('images.npy','data.csv'))

        new_data = data[(np.abs(data['e1'])<=1.) &
                        (np.abs(data['e2'])<=1) ]

        if self.list_of_weights_e == None:
            indices = np.random.choice(new_data.index, size=self.batch_size, replace=False)
        else:
            self.weights_e = np.load(self.list_of_weights_e[index])
            indices = np.random.choice(new_data.index, size=self.batch_size, replace=False, p = self.weights_e/np.sum(self.weights_e))
            #print(indices)
        self.produced_samples += len(indices)

        x = sample[indices,1][:,self.bands]
        #print(x.shape)
        
        y = np.zeros((self.batch_size, 3))
        y[:,0] = np.array(new_data['e1'][indices])#np.exp(np.array(new_data['e1'][indices]))*2/(np.max(np.exp(np.array(new_data['e1'][indices]))*2))#np.exp(np.array(new_data['e1'][indices]))*2 # np.array(new_data['e1'][indices])
        y[:,1] = np.array(new_data['e2'][indices])#np.exp(np.array(new_data['e2'][indices]))*2/(np.max(np.exp(np.array(new_data['e2'][indices]))*2))#np.exp(np.array(new_data['e2'][indices]))*2 # np.array(new_data['e2'][indices]) # 
        y[:,2] = np.array(new_data['redshift'][indices])#/(np.max(np.array(new_data['redshift'][indices])))
        
        # Preprocessing of the data to be easier for the network to learn
        if self.do_norm:
            x = utils.norm(x, self.bands, n_years = 5)
        if self.denorm:
            x = utils.denorm(x, self.bands, n_years = 5)

        #  flip : flipping the image array
        # rand = np.random.randint(4)
        # if rand == 1: 
        #     x = np.flip(x, axis=-1)
        # elif rand == 2 : 
        #     x = np.swapaxes(x, -1, -2)
        # elif rand == 3:
        #     x = np.swapaxes(np.flip(x, axis=-1), -1, -2)
        
        x = np.transpose(x, axes = (0,2,3,1))
        
        if self.trainval_or_test == 'training' or self.trainval_or_test == 'validation':
            return x, y
        elif self.trainval_or_test == 'test':
            return x, y#, data.loc[indices], indices




class BatchGenerator_multi_galaxies(tensorflow.keras.utils.Sequence):
    """
    Class to create batch generator for the LSST VAE.
    """
    def __init__(self, bands, list_of_samples,total_sample_size, batch_size, trainval_or_test, do_norm,denorm, list_of_weights_e):
        """
        Initialization function
        total_sample_size: size of the whole training (or validation) sample
        batch_size: size of the batches to provide
        list_of_samples: list of the numpy arrays which correspond to the whole training (or validation) sample
#        path: path to the first numpy array taken in which the batch will be taken
        training_or_validation: choice between training of validation generator
        x: input of the neural network
        y: target of the neural network
        r: random value to sample into the validation sample
        """
        self.bands = bands
        self.nbands = len(bands)
        self.total_sample_size = total_sample_size
        self.batch_size = batch_size
        self.list_of_samples = list_of_samples
        self.trainval_or_test = trainval_or_test
        
        self.epoch = 0
        self.do_norm = do_norm
        self.denorm = denorm

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
        self.list_of_weights_e = list_of_weights_e
        #self.shifts = shifts

    def __len__(self):
        """
        Function to define the length of an epoch
        """
        return int(float(self.total_sample_size) / float(self.batch_size))      

    def on_epoch_end(self):
        """
        Function executed at the end of each epoch
        """
        # indices = 0
        #print("Produced samples", self.produced_samples)
        self.produced_samples = 0
        
    def __getitem__(self, idx):
        """
        Function which returns the input and target batches for the network
        """
        # If the generator is a training generator, the whole sample is displayed
        #sample_filename = np.random.choice(self.list_of_samples, p=self.p)
        #index = np.random.choice(1)
        index = np.random.choice(list(range(len(self.p))), p=self.p)
        sample_filename = self.list_of_samples[index]
        sample = np.load(sample_filename, mmap_mode = 'c')
        data = pd.read_csv(sample_filename.replace('images.npy','data_classified.csv'))

        new_data = data[(np.abs(data['e1_0'])<=1.) &
                        (np.abs(data['e2_0'])<=1)]

        if self.list_of_weights_e == None:
            indices = np.random.choice(new_data.index, size=self.batch_size, replace=False)
        else:
            self.weights_e = np.load(self.list_of_weights_e[index])
            indices = np.random.choice(new_data.index, size=self.batch_size, replace=False, p = self.weights_e/np.sum(self.weights_e))
            #print(indices)
        self.produced_samples += len(indices)

        x = sample[indices,1][:,self.bands]
        #print(x.shape)
        
        y = np.zeros((self.batch_size, 9))
        y_1 = np.zeros((self.batch_size, 3))
        y_2 = np.zeros((self.batch_size, 3))
        y_3 = np.zeros((self.batch_size, 3))
        #for k in range (4):
            #e1 = 'e1_'+str(k)
            #e2 = 'e2_'+str(k)
            #redshift = 'redshift_'+str(k)
        # Classify galaxies from lowest magnitude to the highest
        # for j, i in enumerate(indices):
        #     if np.array(new_data['nb_blended_gal'][i]>1):
        #         new_data[new_data==10]=30
        #         _idx = np.argmin((new_data['mag_1'][i], new_data['mag_2'][i], new_data['mag_3'][i]))
        #         if _idx == 0:
        #             y[j,0] = np.array(new_data['e1_1'][i])
        #             y[j,1] = np.array(new_data['e2_1'][i])
        #             y[j,2] = np.array(new_data['redshift_1'][i])
        #             y[j,3] = np.array(new_data['mag_1'][i])
        #         elif _idx == 1:
        #             y[j,0] = np.array(new_data['e1_2'][i])
        #             y[j,1] = np.array(new_data['e2_2'][i])
        #             y[j,2] = np.array(new_data['redshift_2'][i])
        #             y[j,3] = np.array(new_data['mag_2'][i])
        #         elif _idx == 2:
        #             y[j,0] = np.array(new_data['e1_3'][i])
        #             y[j,1] = np.array(new_data['e2_3'][i])
        #             y[j,2] = np.array(new_data['redshift_3'][i])
        #             y[j,3] = np.array(new_data['mag_3'][i])
        #     else:
        #         y[j,0] = 0
        #         y[j,1] = 0
        #         y[j,2] = 0
        #         y[j,3] = 0

        y[:,0] = np.array(new_data['e1_0'][indices])
        y[:,1] = np.array(new_data['e2_0'][indices])
        y[:,2] = np.array(new_data['redshift_0'][indices])
        y[:,3] = np.array(new_data['e1_1'][indices])
        y[:,4] = np.array(new_data['e2_1'][indices])
        y[:,5] = np.array(new_data['redshift_1'][indices])
        y[:,6] = np.array(new_data['e2_2'][indices])
        y[:,7] = np.array(new_data['e2_2'][indices])
        y[:,8] = np.array(new_data['redshift_2'][indices])

        # y[:,9] = np.array(new_data['e1_3'][indices])
        # y[:,10] = np.array(new_data['e2_3'][indices])
        # y[:,11] = np.array(new_data['redshift_3'][indices])

        # y_1[:,0] = np.array(new_data['e1_0'][indices])
        # y_1[:,1] = np.array(new_data['e2_0'][indices])
        # y_1[:,2] = np.array(new_data['redshift_0'][indices])
        # y_2[:,0] = np.array(new_data['e1_1'][indices])
        # y_2[:,1] = np.array(new_data['e2_1'][indices])
        # y_2[:,2] = np.array(new_data['redshift_1'][indices])
        # y_3[:,0] = np.array(new_data['e1_2'][indices])
        # y_3[:,1] = np.array(new_data['e2_2'][indices])
        # y_3[:,2] = np.array(new_data['redshift_2'][indices])

        y[np.isnan(y)]=0
        y[y==30]=0
        y[y==10]=0

        # # Preprocessing of the data to be easier for the network to learn
        # if self.do_norm:
        #     x = utils.norm(x, self.bands, n_years = 5)
        # if self.denorm:
        #     x = utils.denorm(x, self.bands, n_years = 5)

        #  flip : flipping the image array
        # rand = np.random.randint(4)
        # if rand == 1: 
        #     x = np.flip(x, axis=-1)
        # elif rand == 2 : 
        #     x = np.swapaxes(x, -1, -2)
        # elif rand == 3:
        #     x = np.swapaxes(np.flip(x, axis=-1), -1, -2)
        
        x = np.transpose(x, axes = (0,2,3,1))
        
        if self.trainval_or_test == 'training' or self.trainval_or_test == 'validation':
            return x, y#(y_1,y_2,y_3)#y
        elif self.trainval_or_test == 'test':
            return x, y#(y_1,y_2,y_3)#y, data.loc[indices], indices





class BatchGenerator_peak(tensorflow.keras.utils.Sequence):
    """
    Class to create batch generator for the LSST VAE.
    """
    def __init__(self, bands, list_of_samples, path, total_sample_size, batch_size, trainval_or_test, do_norm,denorm, list_of_weights_e):
        """
        Initialization function
        total_sample_size: size of the whole training (or validation) sample
        batch_size: size of the batches to provide
        list_of_samples: list of the numpy arrays which correspond to the whole training (or validation) sample
#        path: path to the first numpy array taken in which the batch will be taken
        training_or_validation: choice between training of validation generator
        x: input of the neural network
        y: target of the neural network
        r: random value to sample into the validation sample
        """
        self.bands = bands
        self.nbands = len(bands)
        self.total_sample_size = total_sample_size
        self.batch_size = batch_size
        self.list_of_samples = list_of_samples
        self.trainval_or_test = trainval_or_test
        
        self.epoch = 0
        self.do_norm = do_norm
        self.denorm = denorm

        self.path = path
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
        self.list_of_weights_e = list_of_weights_e
        #self.shifts = shifts

    def __len__(self):
        """
        Function to define the length of an epoch
        """
        return int(float(self.total_sample_size) / float(self.batch_size))      

    def on_epoch_end(self):
        """
        Function executed at the end of each epoch
        """
        # indices = 0
        #print("Produced samples", self.produced_samples)
        self.produced_samples = 0
        
    def __getitem__(self, idx):
        """
        Function which returns the input and target batches for the network
        """
        # If the generator is a training generator, the whole sample is displayed
        #sample_filename = np.random.choice(self.list_of_samples, p=self.p)
        #index = np.random.choice(1)
        index = np.random.choice(list(range(len(self.p))), p=self.p)
        sample_filename = self.list_of_samples[index]
        sample = np.load(sample_filename, mmap_mode = 'c')
        data = pd.read_csv(sample_filename.replace('images.npy','data.csv'))
        shifts = np.load(self.path+'shifts/'+sample_filename[-38:].replace('images.npy','shifts.npy'))

        if self.list_of_weights_e == None:
            indices = np.random.choice(data.index, size=self.batch_size, replace=False)
        else:
            self.weights_e = np.load(self.list_of_weights_e[index])
            indices = np.random.choice(data.index, size=self.batch_size, replace=False, p = self.weights_e/np.sum(self.weights_e))
            #print(indices)
        self.produced_samples += len(indices)

        x = sample[indices,1][:,self.bands]
        #print(x.shape)
        
        y = np.zeros((self.batch_size, 2))
        y = shifts[indices,0]#np.exp(np.array(new_data['e1'][indices]))*2/(np.max(np.exp(np.array(new_data['e1'][indices]))*2))#np.exp(np.array(new_data['e1'][indices]))*2 # np.array(new_data['e1'][indices])
        #y[:,1] = np.array(new_data['e2'][indices])#np.exp(np.array(new_data['e2'][indices]))*2/(np.max(np.exp(np.array(new_data['e2'][indices]))*2))#np.exp(np.array(new_data['e2'][indices]))*2 # np.array(new_data['e2'][indices]) # 
        
        # Preprocessing of the data to be easier for the network to learn
        if self.do_norm:
            x = utils.norm(x, self.bands, n_years = 5)
        if self.denorm:
            x = utils.denorm(x, self.bands, n_years = 5)

        #  flip : flipping the image array
        # rand = np.random.randint(4)
        # if rand == 1: 
        #     x = np.flip(x, axis=-1)
        # elif rand == 2 : 
        #     x = np.swapaxes(x, -1, -2)
        # elif rand == 3:
        #     x = np.swapaxes(np.flip(x, axis=-1), -1, -2)
        
        x = np.transpose(x, axes = (0,2,3,1))
        
        if self.trainval_or_test == 'training' or self.trainval_or_test == 'validation':
            return x, y
        elif self.trainval_or_test == 'test':
            return x, y#, data.loc[indices], indices

