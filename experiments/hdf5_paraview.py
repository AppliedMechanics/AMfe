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
mesh_class.save_mesh_xdmf('my_xdmf_test')

#%%

# Building the XDMF-File

from xml.etree.ElementTree import Element, SubElement
from xml.etree import ElementTree
from xml.dom import minidom








filename = 'testfile'

topology_type = 'Triangle'
dims_connectivity = '546 3'

a, b = 710, 399
timesteps = [1, ]


my_str = prettify_xml(xml_root)
with open('myfile.xdmf', 'w') as f:
    f.write(my_str)

#%%
#
#x = sp.rand(100,100)
#f = h5py.File('mesh_file.hdf5', 'w')
#f.create_dataset('/name/data', data=x)
#f.close()
#
##%%
#
#f = h5py.File('my_file.hdf5', 'r')
## list the keys
#list(f.keys())
#list(f.items())
#list(f.values())
#
#dataset = f['name/data']
#vals = dataset.value
#
#f.close()
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
nodes = mesh_class.nodes

topology = np.zeros((546,3))
topology[:,:3] = np.array(mesh_class.ele_nodes[:-4])


with h5py.File('hdf5/mesh_file.hdf5', 'w') as f:
    f.create_dataset('/nodes', data=nodes)
    f.create_dataset('/topology', data=topology, dtype=np.int)
    f.create_dataset('/displacements', data=q_array.T)










