# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 17:25:07 2015



Compile the module:
f2py3 -c  --fcompiler=gnu95 -m element element.f90

"""

import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0,'..')

import amfe


import element
import time

A = sp.rand(5,8)

B = element.scatter_matrix(A, 2)

#plt.matshow(B)

# use tri3
x = sp.array([0,0,3,1,2,2.])
u = sp.array([0,0,-0.5,0,0,0.])

# use tri6
x = np.array([0,0, 3,1, 2,2, 1.5,0.5, 2.5,1.5, 1,1])
u = np.array([0,0, -0.5,0, 0,0, -0.25,0, -0.25,0, 0,0])



t = 1.

C_SE = sp.array([[ 64.,  16.,   0.],
       [ 16.,  64.,   0.],
       [  0.,   0.,  24.]])


my_amfe_element = amfe.Tri6()
my_amfe_element.C_SE = C_SE

N = int(1E1)

t1 = time.time()
for i in range(N):
    K_f, f_f = element.tri6_k_and_f(x, u, C_SE, t)

t2 = time.time()
a = 0
for i in range(N):
    # K, f = my_amfe_element.k_and_f_int(x, u)
    a += 1
t3 = time.time()

#np.max(abs(K - K_f))

print('Kurzes Profiling:\nZeit für Fortran:', t2-t1, 'Zeit für Python:', t3-t2)
print('Performance-Gewinn: Faktor', (t3-t2)/(t2-t1) )

