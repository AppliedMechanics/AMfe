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


x = sp.array([0,0,3,1,2,2.])
u = sp.array([0,0,-0.5,0,0,0.])
t = 1.



C_SE = sp.array([[ 64.,  16.,   0.],
       [ 16.,  64.,   0.],
       [  0.,   0.,  24.]])

my_amfe_element = amfe.Tri3()
my_amfe_element.C_SE = C_SE

N = int(1E6)

t1 = time.time()
for i in range(N):
    K, f = element.tri3_k_and_f(x, u, C_SE, t)
t2 = time.time()

#for i in range(N):
#    K_2, f2 = my_amfe_element.k_and_f_int(x, u)
t3 = time.time()

print('Kurzes Profiling:\nZeit für Fortran:', t2-t1, 'Zeit für Python:', t3-t2)
print('Performance-Gewinn: Faktor', (t3-t2)/(t2-t1) )



