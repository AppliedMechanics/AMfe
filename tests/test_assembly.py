# -*- coding: utf-8 -*-
"""Test Routine for assembly"""


import unittest
import numpy as np
import scipy as sp
import copy
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal

import amfe


class AssemblyTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_index_getter(self):
        N = int(1E3)
        row = sp.random.randint(0, N, N)
        col = sp.random.randint(0, N, N)
        val = sp.rand(N)
        A = sp.sparse.csr_matrix((val, (row, col)))
        # print('row:', row, '\ncol:', col)
        for i in range(N):
            a = amfe.assembly.get_index_of_csr_data(row[i], col[i], A.indptr, A.indices)
            b = A[row[i], col[i]]
            #    print(val[i] - A.data[b])
            assert_array_equal(A.data[a], b)

    def test_nodes_observer(self):
        # This test tests if the nodes observer updates the nodes properties right if nodes_voigt have changed
        my_material = amfe.KirchhoffMaterial()
        my_system_ref = amfe.MechanicalSystem()
        my_system_ref.mesh_class.nodes = np.array([[0., 0.],
                                               [2., 0.],
                                               [4., 0.],
                                               [0., 1.5],
                                               [2., 1.5],
                                               [4., 1.5]])
        my_system_ref.mesh_class.no_of_dofs_per_node = 2
        my_system_ref.mesh_class.connectivity = [np.array([0, 1, 4, 3], dtype='int'), np.array([1,2,5,4], dtype='int')]
        quad_class = copy.deepcopy(my_system_ref.mesh_class.element_class_dict['Quad4'])
        quad_class.material = my_material
        object_series = [quad_class, quad_class]
        my_system_ref.mesh_class.ele_obj.extend(object_series)
        my_system_ref.mesh_class._update_mesh_props()
        my_system_ref.assembly_class = amfe.Assembly(my_system_ref.mesh_class)
        my_system_ref.dirichlet_class.no_of_unconstrained_dofs = 12
        my_system_ref.dirichlet_class.no_of_constrained_dofs = 12
        my_system_ref.assembly_class.preallocate_csr()
        M_ref = my_system_ref.M()
        K_ref, f_ref = my_system_ref.K_and_f()

        my_system_mod = amfe.MechanicalSystem()
        my_system_mod.mesh_class.nodes = np.array([[0.,0.],
                                                   [2., 0.],
                                                   [4., 0.],
                                                   [0., 1.5],
                                                   [2., 2.],
                                                   [4., 2.4]])
        my_system_mod.mesh_class.no_of_dofs_per_node = 2
        my_system_mod.mesh_class.connectivity = [np.array([0, 1, 4, 3], dtype='int'),
                                                 np.array([1, 2, 5, 4], dtype='int')]

        quad_class = copy.deepcopy(my_system_mod.mesh_class.element_class_dict['Quad4'])
        quad_class.material = my_material
        object_series = [quad_class, quad_class]
        my_system_mod.mesh_class.ele_obj.extend(object_series)
        my_system_mod.mesh_class._update_mesh_props()
        my_system_mod.assembly_class = amfe.Assembly(my_system_mod.mesh_class)
        my_system_mod.dirichlet_class.no_of_unconstrained_dofs = 12
        my_system_mod.dirichlet_class.no_of_constrained_dofs = 12
        my_system_mod.assembly_class.preallocate_csr()
        M_mod = my_system_mod.M()
        K_mod, f_mod = my_system_mod.K_and_f()

        my_system_mod.assembly_class.add_observer(amfe.NodesObserver(mechanical_system=my_system_mod))
        my_system_mod.assembly_class.nodes_voigt = my_system_ref.assembly_class.nodes_voigt
        M_mod = my_system_mod.M()
        K_mod, f_mod = my_system_mod.K_and_f()


        assert_array_equal(M_mod.indices, M_ref.indices)
        assert_array_equal(M_mod.indptr, M_ref.indptr)
        assert_almost_equal(M_mod.data, M_ref.data)
        assert_almost_equal(f_mod, f_ref)
        assert_array_equal(K_mod.indices, K_ref.indices)
        assert_array_equal(K_mod.indptr, K_ref.indptr)
        assert_almost_equal(K_mod.data, K_ref.data)
