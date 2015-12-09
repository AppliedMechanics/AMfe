# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 10:57:03 2015

@author: johannesr
"""
import sys
import scipy as sp
import numpy as np

sys.path.insert(0, '../amfe')
# %cd /Users/johannesr/Documents/004_AMfe/amfe

import material 


A10 = 3
A01 = 4
kappa = 90

F = sp.rand(3,3)
F = sp.array([[ 0.59615078,  0.60223582,  0.41063672],
       [ 0.38285746,  0.96897483,  0.81334762],
       [ 0.93432345,  0.98928816,  0.70385361]])

# 2D-Analysis
F[:,-1] = 0
F[-1,:] = 0
F[-1, -1] = 1

E = 1/2*(F.T.dot(F) - sp.eye(3))

my_rivlin = material.MooneyRivlin(A10, A01, kappa)
S, S_v, C_SE = my_rivlin.S_Sv_and_C(E)

Ev = sp.array([E[0,0], E[1,1], E[2,2], 2*E[1,2], 2*E[0,2], 2*E[0,1]])

def calc_Sv(Ev):
    E = sp.array([[Ev[0]  , Ev[5]/2, Ev[4]/2],
                  [Ev[5]/2, Ev[1]  , Ev[3]/2],
                  [Ev[4]/2, Ev[3]/2, Ev[2]  ]])
    S, S_v, C_SE = my_rivlin.S_Sv_and_C(E)
    return S_v


def jacobian(func, X):
    '''
    Compute the jacobian of func with respect to u using a finite differences scheme. 
    
    '''
    ndof = X.shape[0]
    jac = np.zeros((ndof, ndof))
    h = np.sqrt(np.finfo(float).eps)*1E-3
    f = func(X).copy()
    for i in range(ndof):
        X_tmp = X.copy()
        X_tmp[i] += h
        f_tmp = func(X_tmp)
        jac[:,i] = (f_tmp - f) / h
    return jac

C_SE_exp = jacobian(calc_Sv, Ev)
print((C_SE - C_SE_exp)/(C_SE_exp))

print('2D-Analysis:')

#E[:,-1] = 0
#E[-1,:] = 0
S, S_v, C_SE = my_rivlin.S_Sv_and_C(E)
S2d, S_v2d, C_SE2d = my_rivlin.S_Sv_and_C_2d(E[:2,:2])


#%%