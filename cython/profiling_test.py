# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 12:06:30 2015

@author: johannesr
"""

import numpy as np
import scipy as sp

from test_element import scatter_geometric_matrix

A = sp.rand(10,10)

for i in range(100000):
    B = scatter_geometric_matrix(A, 6)

print(np.asarray(B))