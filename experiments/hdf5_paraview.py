# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 09:24:50 2016

@author: rutzmoser
"""

import h5py
import numpy as np
import scipy as sp
import amfe
from experiments.quadratic_manifold.benchmark_example import benchmark_system, paraview_output_file


#%%

mesh_class = benchmark_system.mesh_class
mesh_class.save_mesh_xdmf('my_xdmf_test/test')

#%%

# Try to integrate the system to get some displacements
ndof = benchmark_system.dirichlet_class.no_of_constrained_dofs
my_newmark = amfe.NewmarkIntegrator(benchmark_system)
my_newmark.integrate(np.zeros(ndof), 
                                      np.zeros(ndof), np.arange(0, 0.4, 1E-3))

q_array = np.array(benchmark_system.u_output)

#%%

# Export of data to HDF5


#%%



f = h5py.File('hdf5/my_file.hdf5', 'w')
sepp = f.create_dataset('/Sepp', data=np.zeros((10,10)))
sepp.attrs['shape'] = (10, 10)
sepp.attrs['info'] = 'my shape is lovely!'
sepp.attrs['paraview_export'] = True
# sepp = f['/Sepp']
f.close()


#%%
f = h5py.File('hdf5/my_file.hdf5', 'r')
sepp = f['Sepp']
sepp.attrs['paraview_export']
f.close()




#%%
with h5py.File('hdf5/mesh_file.hdf5', 'w') as f:
    f.create_dataset('/nodes', data=nodes)
    f.create_dataset('/topology', data=topology, dtype=np.int)
    f.create_dataset('/displacements', data=q_array.T)










