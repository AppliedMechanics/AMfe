# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 12:06:30 2015

@author: johannesr
"""

import numpy as np
import scipy as sp

import test_element as te

A = sp.rand(10,10)

for i in range(100000):
    B = te.scatter_geometric_matrix(A, 6)

print(np.asarray(B))

#my_ele = te.Element()