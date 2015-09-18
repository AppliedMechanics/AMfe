# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import scipy as sp

import io




nodes = sp.array([0,0,1,0,1,1,0,1.]).reshape((-1,2))
elements = sp.array([0, 1, 2, 3]).reshape((1,-1))
apdl_quad4 = produce_apdl('quad4', nodes,  elements, '182', rho=1, E=60, nu=1/4)



x_quad4
apdl_quad4 = produce_apdl('quad4', x_quad4.reshape((-1,2)), 
                          np.array([[0,1,2,3],]), ansys_dict['Quad4'], rho=1., E=60., nu=1/4)


#matrix_data = read_hbmat('m_mat.matrix')
#matrix_data.A


#import matplotlib.pyplot as plt
#plt.matshow(matrix_data.A)











