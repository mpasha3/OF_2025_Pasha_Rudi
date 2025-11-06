import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power
from numpy import array, diag, dot, maximum, empty, repeat, ones, sum
from imports import *
from funcs import *

    
## a helper function for creating the blurring operator
def get_column_sum(spread):
    length = 40
    raw = np.array([np.exp(-(((i-length/2)/spread[0])**2 + ((j-length/2)/spread[1])**2)/2) 
                    for i in range(length) for j in range(length)])
    return np.sum(raw[raw > 0.0001])

## blurs a single pixel at center with a specified Gaussian spread
def P(spread, center, shape):
    image = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            v = np.exp(-(((i-center[0])/spread[0])**2 + ((j-center[1])/spread[1])**2)/2)
            if v < 0.0001:
                continue
            image[i,j] = v
    return image

## matrix multiplication where A operates on a 2-d image producing a new 2-d image
def image_mult(A, image, shape):
    return im( A @ vec(image), shape)


## construct our blurring matrix with a Gaussian spread and zero boundary conditions
def build_A(spread, shape):
    #normalize = get_column_sum(spread)
    m = shape[0]
    n = shape[1]
    A = np.zeros((m*n, m*n))
    count = 0
    for i in range(m):
        for j in range(n):
            column = vec(P(spread, [i, j],  shape))
            A[:, count] = column
            count += 1
    normalize = np.sum(A[:, int(m*n/2 + n/2)])
    A = 1/normalize * A
    return A

def add_noise(b_true, opt, noise_level):

    if (opt == 'Gaussian'):
        noise = np.random.randn(b_true.shape[0]).reshape((-1,1))
        e = noise_level * np.linalg.norm(b_true) / np.linalg.norm(noise) * noise
        e = e.reshape((-1,1))
        b_true = b_true.reshape((-1,1))
        delta = la.norm(e)
        b = b_true + e # add noise
        b_meas = b_true + e

    elif (opt == 'Poisson'):
        # Add Poisson Noise 
        gamma = 1 # background counts assumed known
        b_meas = np.random.poisson(lam=b_true+gamma) 

        delta = 0
    else:
        mu_obs = np.zeros(self.p*self.q)      # mean of noise
        e = np.random.laplace(self.p*self.q)
        sig_obs = noise_level * np.linalg.norm(b_true)/np.linalg.norm(e)
        b_meas = b_true + sig_obs*e
        delta = la.norm(sig_obs*e)
    return (b_meas , delta)

