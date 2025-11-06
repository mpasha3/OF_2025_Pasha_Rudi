#!/usr/bin/env python
"""
Builds functions for generalized cross validation
--------------------------------------------------------------------------
Created in 2023 for TRIPs-Py library
"""
__authors__ = "Mirjeta Pasha, Silvia Gazzola, Connor Sanderford, and Ugochukwu Obinna Ugwu"
__affiliations__ = 'Tufts University, University of Bath, Arizona State University, and Tufts University'
__copyright__ = "Copyright 2023, TRIPs-Py library"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "mirjeta.pasha@tufts.edu; mirjeta.pasha1@gmail.com; sg968@bath.ac.uk; csanderf@asu.edu; connorsanderford@gmail.com; Ugochukwu.Ugwu@tufts.edu"

import numpy as np 
from scipy.optimize import newton, minimize
import scipy.linalg as la
import scipy.optimize as op
from pylops import Identity, LinearOperator
from utils import operator_qr, operator_svd, is_identity
import scipy

"""
Generalized crossvalidation
"""

def gcv_numerator_2(reg_param, Q_A, R_A, R_L, b,**kwargs):
    variant = kwargs['variant'] if ('variant' in kwargs) else 'standard'

    # the observation term:

    R_A_2 = R_A.T @ R_A

    R_A_2 = R_A_2.todense() if isinstance(R_A_2, LinearOperator) else R_A_2

    # The regularizer term:

    R_L_2 = (R_L.T @ R_L)
    
    R_L_2 = R_L_2.todense() if isinstance(R_L_2, LinearOperator) else R_L_2

    # the inverse term:

    inverted = np.linalg.lstsq(( R_A_2 + reg_param * R_L_2), (R_A.T @ Q_A.T @ b) ,rcond=None)[0]  # la.solve( ( R_A_2 + reg_param * R_L_2), (R_A.T @ Q_A.T @ b) )

    if variant == 'modified':
        return ((np.linalg.norm( R_A @ inverted - Q_A.T @ b ))**2 + np.linalg.norm(b - Q_A@(Q_A.T@b))**2)
    else:
        return (np.linalg.norm( R_A @ inverted - Q_A.T @ b ))**2

        # return np.sqrt((np.linalg.norm( R_A @ inverted - Q_A.T @ b ))**2 + np.linalg.norm(b - Q_A@(Q_A.T@b))**2)

def gcv_denominator_2(reg_param, R_A, R_L, b, **kwargs):

    variant = kwargs['variant'] if ('variant' in kwargs) else 'standard'
    # print(variant)
    # the observation term:

    R_A_2 = R_A.T @ R_A

    R_A_2 = R_A_2.todense() if isinstance(R_A_2, LinearOperator) else R_A_2

    # The regularizer term:

    R_L_2 = (R_L.T @ R_L)

    R_L_2 = R_L_2.todense() if isinstance(R_L_2, LinearOperator) else R_L_2

    inverted = np.linalg.lstsq(( R_A_2 + reg_param * R_L_2), R_A.T,rcond=None)[0]  #la.solve( ( R_A_2 + reg_param * R_L_2), R_A.T )

    if variant == 'modified':
       m = kwargs['fullsize']
       trace_term = (m - R_A.shape[1]) - np.trace(R_A @ inverted) # b.size - np.trace(R_A @ inverted) # this is defined with respect to the projected quantities 
    else:
        # in this way works even if we revert to the fully projected pb (call with Q_A.T@b)
        # trace_term = b.size - np.trace(R_A @ inverted) # this is defined with respect to the projected quantities
        trace_term = R_A.shape[0]- np.trace(R_A @ inverted)
    return trace_term**2

def generalized_crossvalidation_2(Q_A, R_A, R_L, b, **kwargs):

    if 'tol' in kwargs:
        tol = kwargs['tol']
    else:
        tol = 10**(-12)

    # function to minimize
    gcv_func = lambda reg_param: gcv_numerator_2(reg_param, Q_A, R_A, R_L, b) / gcv_denominator_2(reg_param, R_A, R_L, b, **kwargs)
    lambdah = op.fminbound(func = gcv_func, x1 = 1e-9, x2 = 100, args=(), xtol=1e-12, maxfun=1000, full_output=0, disp=0)
    
    return lambdah

def gcv_ext_numerator(reg_param, Q_A, R_A, Q_L, R_L, b, b_,**kwargs):

    # the observation term:

    variant = kwargs['variant'] if ('variant' in kwargs) else 'standard'

    R_A_2 = R_A.T @ R_A

    R_A_2 = R_A_2.todense() if isinstance(R_A_2, LinearOperator) else R_A_2

    # The regularizer term:

    R_L_2 = (R_L.T @ R_L)
    
    R_L_2 = R_L_2.todense() if isinstance(R_L_2, LinearOperator) else R_L_2

    # the inverse term:

    inverted = la.solve( ( R_A_2 + reg_param * R_L_2), (R_A.T @ Q_A.T @ b + R_L.T @ Q_L.T @ ((reg_param)*b_)) )


    if variant == 'modified':
        return ((np.linalg.norm( R_A @ inverted - Q_A.T @ b ))**2 + reg_param *(np.linalg.norm( R_L @ inverted - Q_L.T @ b_ ))**2 
 + np.linalg.norm(b - Q_A@(Q_A.T@b))**2 + np.linalg.norm(b_ - Q_L@(Q_L.T@b_)))
    # else: 
    #     return (np.linalg.norm( R_A @ inverted - Q_A.T @ b ))**2 + reg_param *(np.linalg.norm( R_L @ inverted - Q_L.T @ b_ ))**2 
    else: 
        return (np.linalg.norm( R_A @ inverted - Q_A.T @ b ))**2
def gcv_ext_denominator(reg_param, R_A, R_L, b, **kwargs):

    variant = kwargs['variant'] if ('variant' in kwargs) else 'standard'
    # print(variant)
    # the observation term:

    R_A_2 = R_A.T @ R_A

    R_A_2 = R_A_2.todense() if isinstance(R_A_2, LinearOperator) else R_A_2

    # The regularizer term:

    R_L_2 = (R_L.T @ R_L)

    R_L_2 = R_L_2.todense() if isinstance(R_L_2, LinearOperator) else R_L_2

    inverted = la.solve( ( R_A_2 + reg_param * R_L_2), R_A.T )

    if variant == 'modified':
       m = kwargs['fullsize']
       trace_term = (m - R_A.shape[1]) - np.trace(R_A @ inverted) # b.size - np.trace(R_A @ inverted) # this is defined with respect to the projected quantities 
    else:
        # in this way works even if we revert to the fully projected pb (call with Q_A.T@b)
        #trace_term = b.size - np.trace(R_A @ inverted) # this is defined with respect to the projected quantities
        trace_term = R_A.shape[0]- np.trace(R_A @ inverted)    
        #trace_term = b.size - np.trace(R_A @ inverted)
    return trace_term**2

def generalized_crossvalidation_ext(Q_A, R_A, Q_L, R_L, b,b_, **kwargs):

    if 'tol' in kwargs:
        tol = kwargs['tol']
    else:
        tol = 10**(-12)

    # function to minimize
    gcv_func = lambda reg_param: gcv_ext_numerator(reg_param, Q_A, R_A,Q_L, R_L, b,b_) / gcv_ext_denominator(reg_param, R_A, R_L, b, **kwargs)
    #lambdah = op.minimize(fun = gcv_func, x0=[1e-2],bounds=[[1e-9,1]], method='Powell',tol = 10**(-20))['x'] #op.fminbound(func = gcv_func, x1 = 1e-13, x2 = 1e8, args=(), xtol=1e-12, maxfun=3000, full_output=0, disp=0)
    lambdah = op.fminbound(func = gcv_func, x1 = 1e-9, x2 = 100, args=(), xtol=1e-12, maxfun=1000, full_output=0, disp=0)
    return lambdah


def tolu(l,A,Q_A,R_A,B,C,Q_C,R_C,D,opt=2):
    AC = np.concatenate(((1-l)*A,l*C))
    BD = np.concatenate(((1-l)*B,l*D))
    Q_AC,R_AC = la.qr(AC,mode='economic')
    BD = np.concatenate(((1-l)*B,l*D))
    def x_star(l):
      return np.linalg.lstsq(R_AC,Q_AC.T@BD,rcond= None)[0]
    def O1(l):
      rhs = la.norm(Q_A)**2*la.norm(R_A@x_star(l) - Q_A.T@B)**2 /(la.norm(Q_A)**2*la.norm(R_A@x_star(l) - Q_A.T@B)**2 + la.norm(Q_C)**2*la.norm(R_C@x_star(l) - Q_C.T@D)**2)
      return l - rhs   
    def O2(l):
      rhs = l* (la.norm(Q_A)**2*la.norm(R_A@x_star(l) - Q_A.T@B)**2 + la.norm(Q_C)**2*la.norm(R_C@x_star(l) - Q_C.T@D)**2) - la.norm(Q_A)**2*la.norm(R_A@x_star(l) - Q_A.T@B)**2 
      return rhs
    if (opt==1): 
      O = O1
    else:
      O = O2
    O_ = lambda l: -O(l)
    return O_(l) 
def gg(A,Q_A,R_A,B,C,Q_C,R_C,D):
    return (scipy.optimize.fminbound(tolu,0,1,[A,Q_A,R_A,B,C,Q_C,R_C,D]))
