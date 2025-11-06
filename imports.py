import matplotlib
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power
from numpy import array, diag, dot, maximum, empty, repeat, ones, sum
from numpy.linalg import inv
from multiprocessing import Pool
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power
from numpy import array, diag, dot, maximum, empty, repeat, ones, sum
from numpy.linalg import inv
import time
import scipy as sp
import scipy.stats as sps
import scipy.io as spio
import numpy as np
import operators
import astra

from venv import create
import pylops
from scipy.ndimage import convolve
from scipy import sparse
import scipy.special as spe
from scipy import linalg as la

import os
import time


from decompositions import golub_kahan_2, arnoldi
from gcv import *
from trips.utilities.reg_param.discrepancy_principle import *
from funcs import *
from operators import *
from MMGKS_mod import * 
from tomo_class import *
from MMGKS_dyn import *
from optical_flow_solver import *
from data_generator import *
#from discrepancy_principle import discrepancy_principle

from scipy import sparse
import numpy as np
from scipy import linalg as la

from tqdm import tqdm
from collections.abc import Iterable
import matplotlib
import matplotlib.pyplot as plt

from blur_funcs import *
from error_funcs import *

