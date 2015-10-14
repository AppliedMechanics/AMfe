# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 11:23:01 2015

@author: johannesr
"""

import numpy as np
import scipy as sp
import test_element as te
import sys
sys.path.insert(0,'..')

import amfe

import time


my_fast_element = te.Tri3(E_modul=60, poisson_ratio=1/4)
my_amfe_element = amfe.Tri3(E_modul=60, poisson_ratio=1/4)

x = np.array([0,0,3,1,2,2.])
u = np.array([0,0,-0.5,0,0,0.])


a = 10000
t1 = time.time()
for i in range(a):
    K, f = my_fast_element.k_and_f_int(x, u)
t2 = time.time()
for i in range(a):
    K_amfe, f_amfe = my_amfe_element.k_and_f_int(x, u)
t3 = time.time()
print('amfe:', t3-t2, 'Cython:', t2-t1)

