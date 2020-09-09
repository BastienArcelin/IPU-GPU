from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
import tensorflow as tf

from . import layers

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy
from IPython.display import clear_output, set_matplotlib_formats
from mpl_toolkits.axes_grid1 import make_axes_locatable

def build_vanilla_vae(encoder, decoder, coeff_KL,full_cov=False):
    """
    Returns the model to train and parameters to plot relevant information during training using the VAEHistory callback
    """
    input_vae = Input(shape=encoder.input.shape[1:])
    output_encoder = encoder(input_vae)

    z, Dkl = layers.SampleMultivariateGaussian(full_cov=full_cov, add_KL=False, return_KL=True, coeff_KL=coeff_KL)(output_encoder)
    
    model = Model(input_vae, decoder(z))
    #vae_utils = Model(input_vae, [*encoder(input_vae), z, Dkl, decoder(z)])

    return model, Dkl, z



class VAEHistory(Callback):
    def __init__(self, xval_sub, vae_utils, latent_dim, alpha, plot_bands=0, figroot=None, period=1):
        self.xval_sub = xval_sub
        self.latent_dim = latent_dim
        self.plot_bands = plot_bands
        self.vae_utils = vae_utils
        self.figroot = figroot
        
        self.counter = 0
        self.period = period
        
        self.epoch = 0
        self.loss = []
        self.val_loss = []
        self.D_KL = []
        
        self.zz = np.linspace(-5,5,1000)
        self.gauss = scipy.stats.norm.pdf(self.zz)
                
        self.colors = mpl.cm.jet(np.linspace(0,1,latent_dim))
        self.alpha = alpha

        self.counter = 0
        
    @tf.function
    def mask_outliers(points_plop, thresh=3.5):
        """
        Returns a boolean array with True if points are outliers and False 
        otherwise.

        Parameters:
        -----------
            points : An numobservations by numdimensions array of observations
            thresh : The modified z-score to use as a threshold. Observations with
                a modified z-score (based on the median absolute deviation) greater
                than this value will be classified as outliers.

        Returns:
        --------
            mask : A numobservations-length boolean array.

        References:
        ----------
            Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
            Handle Outliers", The ASQC Basic References in Quality Control:
            Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
        """
        points = tf.Tensor(points_plop)

        if len(points.shape) == 1:
            points = points[:,None]
        median = tf.math.median(points, axis=0)
        diff = tf.math.sum((points - median)**2, axis=-1)
        diff = tf.math.sqrt(diff)
        med_abs_deviation = tf.math.median(diff)

        modified_z_score = 0.6745 * diff / med_abs_deviation

        #return modified_z_score > thresh

        return np.ma.masked_array(data=points, mask=modified_z_score > thresh)

    def on_epoch_end(self, epoch, logs={}):
        self.epoch += 1
        self.counter +=1

        if self.counter % self.period == 0:
            # try: 
            self.loss.append(logs.get('loss'))
            self.val_loss.append(logs.get('val_loss'))
                    
            mu, sigma, z, dkl, out = self.vae_utils.predict(self.xval_sub[:,1])

            self.D_KL.append(np.mean(dkl))

            clear_output(wait=True)
            
            fig, axes = plt.subplots(2, 4, figsize=(4*4,2*4))
            axes = axes.ravel()
                    
            ax = axes[0]
            ax.plot(self.loss, c='b', label='train')
            ax.plot(self.val_loss, c='r', label='val')
            ax.legend()
            ax.set_xlabel('epoch')
            ax.set_title('Training/validation losses')
            ax.set_yscale('log')
            
            ax = axes[1]
            ax.plot(self.D_KL)
            ax.set_xlabel('epoch')
            ax.set_title('D_KL')
            ax.set_yscale('log')

            for dim in range(self.latent_dim):
                c = self.colors[dim]
                _ = axes[2].hist(mu[:,dim], bins=50, histtype='step', color=c, label=str(dim), density=True)
                _ = axes[3].hist(sigma[:,dim], bins=50, log=False, histtype='step', color=c, label=str(dim), density=True)
                _ = axes[4].hist(z[:,dim], bins=50, histtype='step', color=c, label=str(dim), density=True)
            
            axes[4].plot(self.zz, self.gauss, c='k')
            axes[4].set_xlim(-5,5)
            
            axes[2].set_title('mu_y(X)')
            axes[3].set_title('sigma_y(X)')
            axes[4].set_title('z~N(0,1)')
            
            idx = np.random.randint(len(self.xval_sub))
            # weird thing when indexing numpy arrays with list...
            if isinstance(self.plot_bands, int):
                img_in = self.xval_sub[idx,1,:,:,self.plot_bands]
                print('first'+str(img_in.shape))
                img_target = self.xval_sub[idx,0,:,:,self.plot_bands]
                img_out = out[idx,:,:,self.plot_bands]
            else:
                img_in = self.xval_sub[idx,1,:,:,self.plot_bands].transpose([1,2,0])
                print('second'+str(img_in.shape))
                img_target = self.xval_sub[idx,0,:,:,self.plot_bands].transpose([1,2,0])
                img_out = out[idx,:,:,self.plot_bands].transpose([1,2,0])
        
            ax = axes[5]
            #if self.plot_bands == 0:
            #    self.xval_sub = self.xval_sub.reshape((500,64,64,1))
            #print(self.xval_sub.reshape((500,64,64,1))[idx][:,:][self.plot_bands].shape)
            # print(self.xval_sub.shape)
            # print(self.xval_sub[idx,:,:,self.plot_bands].shape)
            # print(out.shape)
            # print(out[idx,:,:,self.plot_bands].shape)
            im = ax.imshow(np.clip(img_in[:,:,0], 0., 1.))#.reshape((64,64)))
            #ax.imshow(self.xval_sub[idx])
            if isinstance(self.plot_bands, int):
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                plt.colorbar(im, cax=cax)
            
            ax.set_title('input')
            ax.axis('off')

            ax = axes[6]
            #if self.plot_bands == 0:
            #    self.xval_sub = self.xval_sub.reshape((500,64,64,1))
            #print(self.xval_sub.reshape((500,64,64,1))[idx][:,:][self.plot_bands].shape)
            # print(self.xval_sub.shape)
            # print(self.xval_sub[idx,:,:,self.plot_bands].shape)
            # print(out.shape)
            # print(out[idx,:,:,self.plot_bands].shape)
            im = ax.imshow(np.clip(img_target[:,:,0], 0., 1.))#.reshape((64,64)))
            #ax.imshow(self.xval_sub[idx])
            if isinstance(self.plot_bands, int):
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                plt.colorbar(im, cax=cax)
            
            ax.set_title('target')
            ax.axis('off')
            
            ax = axes[7]
            #if self.plot_bands == 0:
            #    out = out.reshape((500,64,64,1))

            im = ax.imshow(np.clip(img_out[:,:,0], 0., 1.))#.reshape((64,64)))
            #ax.imshow(out[idx])
            if isinstance(self.plot_bands, int):
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                plt.colorbar(im, cax=cax)
            ax.set_title('output')
            ax.axis('off')
            
            plt.tight_layout()
            
            if self.figroot is not None:
                fig.savefig(self.figroot+"_epoch"+str(self.epoch)+".png")#, dpi=300)
                
            # if self.plot_bands == 0:
            #     self.xval_sub = self.xval_sub.reshape((500,64,64))

            self.counter = 0

            # except :
            #     print('erreur')
            #     pass

            #plt.show()
        

class VAEHistory_2(Callback):
    def __init__(self, xval_sub, vae_utils, latent_dim, alpha, plot_bands=0, figroot=None, period=1):
        self.xval_sub = xval_sub
        self.latent_dim = latent_dim
        self.plot_bands = plot_bands
        self.vae_utils = vae_utils
        self.figroot = figroot
        
        self.counter = 0
        self.period = period
        
        self.epoch = 0
        self.loss = []
        self.val_loss = []
        self.D_KL = []
        
        self.zz = np.linspace(-5,5,1000)
        self.gauss = scipy.stats.norm.pdf(self.zz)
                
        self.colors = mpl.cm.jet(np.linspace(0,1,latent_dim))
        self.alpha = alpha

        self.counter = 0
        
    def mask_outliers(points_plop, thresh=3.5):
        """
        Returns a boolean array with True if points are outliers and False 
        otherwise.

        Parameters:
        -----------
            points : An numobservations by numdimensions array of observations
            thresh : The modified z-score to use as a threshold. Observations with
                a modified z-score (based on the median absolute deviation) greater
                than this value will be classified as outliers.

        Returns:
        --------
            mask : A numobservations-length boolean array.

        References:
        ----------
            Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
            Handle Outliers", The ASQC Basic References in Quality Control:
            Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
        """
        points = np.array(points_plop)

        if len(points.shape) == 1:
            points = points[:,None]
        median = np.median(points, axis=0)
        diff = np.sum((points - median)**2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)

        modified_z_score = 0.6745 * diff / med_abs_deviation

        #return modified_z_score > thresh

        return np.ma.masked_array(data=points, mask=modified_z_score > thresh)

    def on_epoch_end(self, epoch, logs={}):
        self.epoch += 1
        self.counter +=1

        if self.counter % self.period == 0:
            # try: 
            self.loss.append(logs.get('loss'))
            self.val_loss.append(logs.get('val_loss'))
                    
            mu, sigma, z, dkl, out = self.vae_utils.predict(self.xval_sub[:,1])
            print("mu",mu[:25])
            print("sigma",sigma[:25])
            print("z",z[:25])
            print("dkl",dkl[:25])


            self.D_KL.append(np.mean(dkl))

            clear_output(wait=True)
            
            fig, axes = plt.subplots(2, 4, figsize=(4*4,2*4))
            axes = axes.ravel()
                    
            ax = axes[0]
            ax.plot(self.loss, c='b', label='train')
            ax.plot(self.val_loss, c='r', label='val')
            ax.legend()
            ax.set_xlabel('epoch')
            ax.set_title('Training/validation losses')
            ax.set_yscale('log')
            
            ax = axes[1]
            ax.plot(self.D_KL)
            ax.set_xlabel('epoch')
            ax.set_title('D_KL')
            ax.set_yscale('log')

            for dim in range(self.latent_dim):
                c = self.colors[dim]
                try:
                    _ = axes[2].hist(mu[:,dim], bins=50, histtype='step', color=c, label=str(dim), density=True)
                    _ = axes[3].hist(sigma[:,dim], bins=50, log=False, histtype='step', color=c, label=str(dim), density=True)
                    _ = axes[4].hist(z[:,dim], bins=50, histtype='step', color=c, label=str(dim), density=True) 
                except ValueError:
                    continue
            
            axes[4].plot(self.zz, self.gauss, c='k')
            axes[4].set_xlim(-5,5)
            
            axes[2].set_title('mu_y(X)')
            axes[3].set_title('sigma_y(X)')
            axes[4].set_title('z~N(0,1)')
            
            idx = np.random.randint(len(self.xval_sub))
            # weird thing when indexing numpy arrays with list...
            if isinstance(self.plot_bands, int):
                img_in = self.xval_sub[idx,1,:,:,self.plot_bands]
                img_target = self.xval_sub[idx,0,:,:,self.plot_bands]
                img_out = out[idx,:,:,self.plot_bands]
            else:
                img_in = self.xval_sub[idx,1,:,:,self.plot_bands].transpose([1,2,0])
                img_target = self.xval_sub[idx,0,:,:,self.plot_bands].transpose([1,2,0])
                img_out = out[idx,:,:,self.plot_bands].transpose([1,2,0])
        
            ax = axes[5]
            #if self.plot_bands == 0:
            #    self.xval_sub = self.xval_sub.reshape((500,64,64,1))
            #print(self.xval_sub.reshape((500,64,64,1))[idx][:,:][self.plot_bands].shape)
            # print(self.xval_sub.shape)
            # print(self.xval_sub[idx,:,:,self.plot_bands].shape)
            # print(out.shape)
            # print(out[idx,:,:,self.plot_bands].shape)
            #ax.imshow(self.xval_sub[idx])
            if isinstance(self.plot_bands, int):
                im = ax.imshow(img_in)#.reshape((64,64)))
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                plt.colorbar(im, cax=cax)
            else:
                im = ax.imshow(np.clip(img_in, 0., 1.))#.reshape((64,64)))

            ax.set_title('input')
            ax.axis('off')

            ax = axes[6]
            #if self.plot_bands == 0:
            #    self.xval_sub = self.xval_sub.reshape((500,64,64,1))
            #print(self.xval_sub.reshape((500,64,64,1))[idx][:,:][self.plot_bands].shape)
            # print(self.xval_sub.shape)
            # print(self.xval_sub[idx,:,:,self.plot_bands].shape)
            # print(out.shape)
            # print(out[idx,:,:,self.plot_bands].shape)
            #im = ax.imshow(np.clip(img_target, 0., 1.))#.reshape((64,64)))
            #ax.imshow(self.xval_sub[idx])
            if isinstance(self.plot_bands, int):
                im = ax.imshow(img_target)#.reshape((64,64)))
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                plt.colorbar(im, cax=cax)
            else:
                im = ax.imshow(np.clip(img_target, 0., 1.))#.reshape((64,64)))
            
            ax.set_title('target')
            ax.axis('off')
            
            ax = axes[7]
            #if self.plot_bands == 0:
            #    out = out.reshape((500,64,64,1))

            #im = ax.imshow(np.clip(img_out, 0., 1.))#.reshape((64,64)))
            #ax.imshow(out[idx])
            if isinstance(self.plot_bands, int):
                im = ax.imshow(img_out)#.reshape((64,64)))
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                plt.colorbar(im, cax=cax)
            else:
                im = ax.imshow(np.clip(img_out, 0., 1.))#.reshape((64,64)))

            ax.set_title('output')
            ax.axis('off')
            
            plt.tight_layout()
            
            if self.figroot is not None:
                fig.savefig(self.figroot+"_epoch"+str(self.epoch)+".png")#, dpi=300)
                
            # if self.plot_bands == 0:
            #     self.xval_sub = self.xval_sub.reshape((500,64,64))

            self.counter = 0

            # except :
            #     print('erreur')
            #     pass

            #plt.show()
        

