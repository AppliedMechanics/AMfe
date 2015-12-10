# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 17:25:07 2015



Compile the module:
f2py3 -c  --fcompiler=gnu95 -m element element.f90
f2py3 -c  --fcompiler=gnu95 -m f90_assembly assembly.f90

"""

import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0,'..')

import amfe

#%%
cd f2py
#%%
# ! f2py3 -c  --fcompiler=gnu95 -m callback test_callback.f90
# Generate the signature file
# ! f2py3 -m callback test_callback.f90  -h test_callback.pyf --overwrite-signature
# run with signature file
# ! f2py3 -c test_callback.pyf test_callback.f90

#%%

import callback

#callback.test(3,4)
#
#def my_func(a, b):
#    return np.sqrt(a**2 + b**2)
#
#callback.test2(3, 4, my_func)


def vec_func(a, b):
    c = a
    d = b
    return c, d
    


a = sp.rand(6)
b = sp.rand(6)
c = sp.zeros(6)

c_new = callback.test3(a, b, vec_func)

#%%
%%timeit
for i in range(1000000):
    c_new = callback.test3(a, b, vec_func)


#%% 
import f90_element
X = sp.rand(12)
u = sp.rand(12)
C_SE = sp.rand(3,3)
#%%
%%timeit
for i in range(1000000):
    K, f = f90_element.tri6_k_and_f(X, u, C_SE, 0)