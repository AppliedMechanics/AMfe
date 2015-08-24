# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 08:44:57 2015

@author: johannesr
f2py3 -c  --fcompiler=gnu95 -m f90_assembly assembly.f90 && cp f90_assembly.so ../amfe

This is a test routine for the assembly process performed in python or in FORTRAN.

"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time


# make amfe running
import sys
sys.path.insert(0,'..')
import amfe


gmsh_input_file = '../meshes/test_meshes/bar_Quad4.msh'
paraview_output_file = '../results/gmsh_test/gmsh_import'

my_mesh = amfe.Mesh()
my_mesh.import_msh(gmsh_input_file, mesh_3d=False)

element_class_dict = {'Quad4' : amfe.Quad4(),
                      'Tet4' : amfe.Tet4()}

my_assembly = amfe.Assembly(my_mesh, element_class_dict)

K_sparse = my_assembly.assemble_k()
K_sparse = K_sparse.tocsr()

my_assembly.preallocate_csr()


#%%
# show the matrix structure:
if True:
    C = my_assembly.C_csr.copy()
    C.data += 1
#    plt.matshow(C.A)
    plt.spy(C, marker='.')


residual_1 = K_sparse.indptr - C.indptr
residual_2 = K_sparse.indices - C.indices

#%%
# Testing the new stuff:

my_element = element_class_dict['Quad4']

u = np.zeros(my_mesh.no_of_dofs)
K_new, f_new = my_assembly.assemble_matrix_and_vector(u, my_element.k_and_f_int)

K_res = K_sparse - K_new
print('The maximum difference between the two assembly routines are',
      K_res.max()/K_new.max())

#%%


B = my_assembly.C_csr.copy()
C = my_assembly.C_csr.copy()
i = 3
idxs = my_assembly.global_element_indices[i]
X = my_assembly.node_coords[idxs]
K_local = np.arange(0, 8*8.).reshape((8,8))
amfe.fill_csr_matrix(C.indptr, C.indices, C.data, K_local, idxs)
amfe.f90_assembly.fill_csr_matrix(B.indptr, B.indices, B.data, K_local, idxs)

#plt.plot(C.data, 'g')
#plt.plot(B.data, 'r')

#%%

#
# Test the get_index_of_csr_data-routine
#
N = int(1E4)
row = sp.random.randint(0, N, N)
col = sp.random.randint(0, N, N)
val = sp.rand(N)
A = sp.sparse.csr_matrix((val, (row, col)))
print('row:', row, '\ncol:', col)
for i in range(N):
    a = amfe.get_index_of_csr_data(row[i], col[i], A.indptr, A.indices)
    b = amfe.f90_assembly.get_index_of_csr_data(row[i], col[i], A.indptr, A.indices)
#    print(val[i] - A.data[b])
    if (a != b):
        print('ACHTUNG!!! Hier ist etwas falsch')


#%%

if False:
    X = sp.rand(2E7)
    a = sp.array([5,2,9,8,56,4,234,8])
    t1 = time.time()
    a = sp.random.randint(0,2E5, 6)
    for i in range(int(3E6)):
        x = X[a]
        b = X[a]
    t2 = time.time()
    print('Time for picking entries:', t2-t1)