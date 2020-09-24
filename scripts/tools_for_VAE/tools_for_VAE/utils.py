import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
import sys
import os
import logging
#import galsim
import random
import cmath as cm
import math
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Add, Multiply, BatchNormalization, Reshape, Flatten, Conv2D,  PReLU,Conv2DTranspose
from tensorflow.keras.models import Model, Sequential
from scipy.stats import norm
import tensorflow as tf

from . import  vae_functions, plot, layers #model,

I_lsst = np.array([255.2383, 2048.9297, 3616.1757, 4441.0576, 4432.7823, 2864.145])
I_euclid = np.array([5925.8097, 3883.7892, 1974.2465,  413.3895])
beta = 2.5


