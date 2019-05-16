# Copyright (c) 2018, Lehrstuhl für Angewandte Mechanik, Technische Universität München.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
#


from unittest import TestCase

import numpy as np
from numpy.testing import assert_, assert_allclose, assert_array_equal
from numpy.linalg import norm

from amfe.mor.hyper_red.ecsw import sparse_nnls, ecsw_assemble_G_and_b, ecsw_get_weights_by_component
from amfe.mor.ui import create_ecsw_hyperreduced_component_from_weights
from amfe.io.tools import amfe_dir
from amfe.io.mesh.reader import GidJsonMeshReader
from amfe.io.mesh.writer import AmfeMeshConverter
from amfe.mor.hyper_red.ecsw_assembly import EcswAssembly
from amfe.assembly import StructuralAssembly
from amfe.component import StructuralComponent
from amfe.material import KirchhoffMaterial


class TestNnls(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_nnls(self):
        # Copyright Notice:
        #   The Nnls testcase is a modified version of the nnls test case of the Scipy-package.
        #   This was distributed under BSD-3 License
        # Copyright(c) 2001, 2002 Enthought, Inc.
        # All rights reserved.
        #
        # Copyright (c) 2003-2019 SciPy Developers.
        # All rights reserved.
        #
        # Author: Uwe Schmitt
        # Sep 2008
        #

        # Build a matrix a
        a = np.arange(25.0).reshape(-1, 5)
        # Build a vector x
        x = np.arange(5.0)
        # Calculate the correct right hand side
        y = np.dot(a, x)
        # Calculate tau from residual tolerance tol = tau * norm(y)
        tol = 1e-7
        tau = tol/norm(y)

        # run algorithm
        x, stats = sparse_nnls(a, y, tau)
        # get last residual before return
        res = stats[-1][1]
        # test if residual is smaller than desired tolerance
        assert_(res <= tol)
        assert_(norm(np.dot(a, x.toarray()).reshape(-1)-y) <= 1e-7)
        # test if all x are greater equal zero
        np.all(np.greater_equal(x.toarray(), 0.0))

        # make a second test that does not converge
        a = np.array([[0.21235441, 0.32701625, 0.67680346, 0.72724123, 0.51983536],
                      [0.82603172, 0.76654767, 0.69746447, 0.58220156, 0.2564705 ],
                      [0.04594648, 0.78409449, 0.85036132, 0.4888821 , 0.92390904],
                      [0.10404788, 0.37767343, 0.30689839, 0.77633873, 0.42464905],
                      [0.66897911, 0.59824198, 0.60212744, 0.02402656, 0.75641132]])
        y = np.array([0., 0., 0., 0., 0.19731525])
        tau = 1e-1/norm(y)

        # this should not converge test if RuntimeError is raised
        with self.assertRaises(RuntimeError):
            sparse_nnls(a, y, tau)


class TestEcsw(TestCase):
    def setUp(self):
        # Define input file path
        file = amfe_dir('tests/meshes/gid_json_4_tets.json')
        # Define Reader Object, initialized with AmfeMeshConverter
        reader = GidJsonMeshReader(file)

        # Initialize component
        converter = AmfeMeshConverter()
        reader.parse(converter)
        self.my_mesh = converter.return_mesh()
        self.my_component = StructuralComponent(self.my_mesh)
        my_material = KirchhoffMaterial()
        self.my_component.assign_material(my_material, ['left', 'right'], 'S')

        # Get number of dofs for snapshot generation
        self.no_of_dofs = self.my_component.mapping.no_of_dofs
        # create 2 random snapshots
        self.no_of_snapshots = 2
        self.S = np.random.rand(self.no_of_dofs, self.no_of_snapshots) * 0.05
        self.W = np.eye(self.no_of_dofs)
        self.timesteps = np.zeros(self.no_of_snapshots)

    def tearDown(self):
        pass

    def test_assemble_g_b(self):
        # store an example for f_int for later comparison to check if the old assembly is recovered
        # after ecsw_assemble_G_and_b has finished
        no_of_dofs = self.S.shape[0]
        dq = ddq = np.zeros(no_of_dofs)
        t = 0.0
        f_old = self.my_component.f_int(self.S[:, 0], dq, t)

        # run ecsw_assemble_G_and_b
        G, b = ecsw_assemble_G_and_b(self.my_component, self.S, self.W, self.timesteps)

        # test shape of G and b
        no_of_elements = self.my_component.no_of_elements
        self.assertEqual(G.shape, (self.no_of_dofs*self.no_of_snapshots, no_of_elements))

        # ----------------------------------
        # Check if G is correct

        # Test first entry of G
        g11_actual = G[0:self.no_of_dofs, 0]
        connectivity = self.my_mesh.get_connectivity_by_elementids([1])[0]
        X_local = self.my_mesh.nodes_df.loc[connectivity].values.reshape(-1)
        u_local_indices = self.my_component.mapping.nodal2global.loc[connectivity].values.reshape(-1)
        u_local = self.S[u_local_indices, 0]
        fe_local = self.my_component.ele_obj[0].f_int(X_local, u_local)
        global_dofs = self.my_component.mapping.elements2global[0]
        g11_desired = np.zeros(self.no_of_dofs)
        g11_desired[global_dofs] = fe_local
        assert_allclose(g11_actual, g11_desired)
        # Test second entry of G
        g21_actual = G[self.no_of_dofs:, 0]
        connectivity = self.my_mesh.get_connectivity_by_elementids([1])[0]
        X_local = self.my_mesh.nodes_df.loc[connectivity].values.reshape(-1)
        u_local_indices = self.my_component.mapping.nodal2global.loc[connectivity].values.reshape(-1)
        u_local = self.S[u_local_indices, 1]
        fe_local = self.my_component.ele_obj[0].f_int(X_local, u_local)
        global_dofs = self.my_component.mapping.elements2global[0]
        g21_desired = np.zeros(self.no_of_dofs)
        g21_desired[global_dofs] = fe_local
        assert_allclose(g21_actual, g21_desired)
        # Test third entry of G
        g12_actual = G[0:self.no_of_dofs, 1]
        connectivity = self.my_mesh.get_connectivity_by_elementids([2])[0]
        X_local = self.my_mesh.nodes_df.loc[connectivity].values.reshape(-1)
        u_local_indices = self.my_component.mapping.nodal2global.loc[connectivity].values.reshape(-1)
        u_local = self.S[u_local_indices, 0]
        fe_local = self.my_component.ele_obj[1].f_int(X_local, u_local)
        global_dofs = self.my_component.mapping.elements2global[1]
        g12_desired = np.zeros(self.no_of_dofs)
        g12_desired[global_dofs] = fe_local
        assert_allclose(g12_actual, g12_desired)

        # --------------------------------------
        # check if b is correct:
        b_desired = np.sum(G, 1)
        assert_allclose(b, b_desired)

        # --------------------------------------
        # Check if old assembly is recovered

        # get f_new for comparison to f_old
        f_new = self.my_component.f_int(self.S[:, 0], dq, t)
        # test if old assembly is recovered in the component
        assert_allclose(f_new, f_old)

    def test_reduce_with_ecsw(self):
        # store old ids:
        comp_id_old = id(self.my_component)
        mesh_id_old = id(self.my_component.mesh)

        # first mode: deepcopy
        weights, indices, stats = ecsw_get_weights_by_component(self.my_component, self.S, self.W,
                                                                self.timesteps, tau=0.01)
        ecsw_component = create_ecsw_hyperreduced_component_from_weights(self.my_component, weights, indices,
                                                                         copymode='deep')
        self.assertNotEqual(id(ecsw_component), comp_id_old)
        self.assertNotEqual(id(ecsw_component.mesh), mesh_id_old)
        self.assertIsInstance(ecsw_component.assembly, EcswAssembly)

        # second mode: shallow
        weights, indices, stats = ecsw_get_weights_by_component(self.my_component, self.S, self.W,
                                                                self.timesteps, tau=0.01)
        ecsw_component = create_ecsw_hyperreduced_component_from_weights(self.my_component, weights, indices,
                                                                         copymode='shallow')
        self.assertNotEqual(id(ecsw_component), comp_id_old)
        self.assertEqual(id(ecsw_component.mesh), mesh_id_old)
        self.assertIsInstance(ecsw_component.assembly, EcswAssembly)

        # third mode: overwrite
        weights, indices, stats = ecsw_get_weights_by_component(self.my_component, self.S, self.W,
                                                                self.timesteps, tau=0.01)
        ecsw_component = create_ecsw_hyperreduced_component_from_weights(self.my_component, weights, indices,
                                                                         copymode='overwrite')
        self.assertEqual(id(ecsw_component), comp_id_old)
        self.assertEqual(id(ecsw_component.mesh), mesh_id_old)
        self.assertIsInstance(ecsw_component.assembly, EcswAssembly)

        # test wrong copymode
        with self.assertRaises(ValueError):
            weights, indices, stats = ecsw_get_weights_by_component(self.my_component, self.S, self.W,
                                                                    self.timesteps, tau=0.01, conv_stats=False)
            ecsw_component = create_ecsw_hyperreduced_component_from_weights(self.my_component, weights, indices,
                                                                             copymode='foo')
        # test if function with option stats = false is executable
        weights, indices, stats = ecsw_get_weights_by_component(self.my_component, self.S, self.W,
                                                                self.timesteps, tau=0.01, conv_stats=False)
        ecsw_component = create_ecsw_hyperreduced_component_from_weights(self.my_component, weights, indices)
        self.assertIsInstance(ecsw_component.assembly, EcswAssembly)


class EcswTest(TestCase):
    def setUp(self):
        self.nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float)
        self.iconnectivity = [np.array([0, 1, 2], dtype=np.int), np.array([0, 2, 3], dtype=np.int),
                              np.array([1, 2], dtype=np.int), np.array([2, 3], dtype=np.int)]

        self.asm = StructuralAssembly()

        class DummyTri3Element:
            def __init__(self):
                pass

            def m_int(self, X, u, t=0.):
                M = np.array([[2, 0, -0.5, 0, -0.5, 0],
                              [0, 2, 0, -0.5, 0, -0.5],
                              [-0.5, 0, 2, 0, -0.5, 0],
                              [0, -0.5, 0, 2, 0, -0.5],
                              [-0.5, 0, -0.5, 0, 2, 0],
                              [0, -0.5, 0, -0.5, 0, 2]], dtype=float)
                return M

            def k_and_f_int(self, X, u, t=0.):
                K = np.array([[4, -0.5, -0.5, -0.2, -0.5, -0.2],
                              [-0.2, 4, -0.2, -0.5, -0.2, -0.5],
                              [-0.5, -0.2, 4, -0.2, -0.5, -0.2],
                              [-0.2, -0.5, -0.2, 4, -0.2, -0.5],
                              [-0.5, -0.2, -0.5, -0.2, 4, -0.2],
                              [-0.2, -0.5, -0.2, -0.5, -0.2, 4]], dtype=float)
                f = np.array([3, 1, 3, 1, 3, 1], dtype=float)
                return K, f

            def k_f_S_E_int(self, X, u, t=0):
                K, f = self.k_and_f_int(X, u, t)
                S = np.ones((3, 6), dtype=float)
                E = 2*np.ones((3, 6), dtype=float)
                return K, f, S, E

        self.ele = DummyTri3Element()

    def tearDown(self):
        self.asm = None

    def test_assemble_k_and_f_ecsw_test1(self):

        weights = [5]
        indices = np.array([1], dtype=int)
        asm = EcswAssembly(weights, indices)
        ele_obj = np.array([self.ele, self.ele], dtype=object)
        element2dofs = np.array([np.array([0, 1, 2, 3, 4, 5], dtype=int), np.array([0, 1, 4, 5, 6, 7], dtype=int)])

        K_global = asm.preallocate(8, element2dofs[indices])
        f_global = np.zeros(K_global.shape[0])

        memory_K_global_before = id(K_global)
        memory_K_global_data_before = id(K_global.data)
        memory_f_global_before = id(f_global)

        dofvalues = np.array([0.0, 0.1, 0.2, 0.05, 0.0, 0.05, 0.02, 0.04])
        asm.assemble_k_and_f(self.nodes, ele_obj, self.iconnectivity[0:2], element2dofs, dofvalues, K_csr=K_global, f_glob=f_global)
        K_global_desired = np.zeros((8, 8), dtype=float)
        f_global_desired = np.zeros(8, dtype=float)
        # element 1
        # Not assembled:
        # K_global_desired[0:6, 0:6], f_global_desired[0:6] = self.ele.k_and_f_int(None, None)

        # element 2
        K_local, f_local = self.ele.k_and_f_int(None, None)
        # diagonals
        K_global_desired[0:2, 0:2] += weights[0]*K_local[0:2, 0:2]
        K_global_desired[4:, 4:] += weights[0]*K_local[2:, 2:]
        # off-diagonals
        K_global_desired[0:2, 4:] += weights[0]*K_local[0:2, 2:]
        K_global_desired[4:, 0:2] += weights[0]*K_local[2:, 0:2]
        # f_int:
        f_global_desired[0:2] += weights[0]*f_local[0:2]
        f_global_desired[4:] += weights[0]*f_local[2:]

        assert_array_equal(K_global.todense(), K_global_desired)
        assert_array_equal(f_global, f_global_desired)


        # Test if preallocation is working
        memory_K_global_after = id(K_global)
        memory_K_global_data_after = id(K_global.data)
        memory_f_global_after = id(f_global)

        self.assertTrue(memory_K_global_after == memory_K_global_before)
        self.assertTrue(memory_K_global_data_after == memory_K_global_data_before)
        self.assertTrue(memory_f_global_after == memory_f_global_before)

    def test_assemble_k_and_f_ecsw_test2(self):

        weights = np.array([5.0, 4.0])
        indices = np.array([1, 0], dtype=int)
        asm = EcswAssembly(weights, indices)
        ele_obj = np.array([self.ele, self.ele], dtype=object)
        element2dofs = np.array([np.array([0, 1, 2, 3, 4, 5], dtype=int), np.array([0, 1, 4, 5, 6, 7], dtype=int)])

        K_global, f_global = asm.assemble_k_and_f(self.nodes, ele_obj, self.iconnectivity[0:2], element2dofs, K_csr=None,
                                                 f_glob=None)
        K_global_desired = np.zeros((8, 8), dtype=float)
        f_global_desired = np.zeros(8, dtype=float)
        # element 0
        K_global_desired[0:6, 0:6], f_global_desired[0:6] = self.ele.k_and_f_int(None, None)
        K_global_desired[0:6, 0:6] = weights[1]*K_global_desired[0:6, 0:6]
        f_global_desired[0:6] = weights[1]*f_global_desired[0:6]

        # element 1
        K_local, f_local = self.ele.k_and_f_int(None, None)
        # diagonals
        K_global_desired[0:2, 0:2] += weights[0]*K_local[0:2, 0:2]
        K_global_desired[4:, 4:] += weights[0]*K_local[2:, 2:]
        # off-diagonals
        K_global_desired[0:2, 4:] += weights[0]*K_local[0:2, 2:]
        K_global_desired[4:, 0:2] += weights[0]*K_local[2:, 0:2]
        # f_int:
        f_global_desired[0:2] += weights[0]*f_local[0:2]
        f_global_desired[4:] += weights[0]*f_local[2:]

        assert_array_equal(K_global.todense(), K_global_desired)
        assert_array_equal(f_global, f_global_desired)

    def test_assemble_k_f_S_E_ecsw(self):

        weights = [5]
        indices = np.array([1], dtype=int)
        asm = EcswAssembly(weights, indices)

        ele_obj = np.array([self.ele, self.ele], dtype=object)
        element2dofs = [np.array([0, 1, 2, 3, 4, 5], dtype=int), np.array([0, 1, 4, 5, 6, 7], dtype=int)]
        elements_on_node = np.array([weights[0], np.Inf, weights[0], weights[0]])

        K_global = asm.preallocate(8, element2dofs)
        f_global = np.zeros(K_global.shape[0])

        memory_K_global_before = id(K_global)
        memory_K_global_data_before = id(K_global.data)
        memory_f_global_before = id(f_global)

        K_global, f_global, S_global, E_global= asm.assemble_k_f_S_E(self.nodes, ele_obj, self.iconnectivity[0:2],
                                                                     element2dofs, elements_on_node, K_csr=K_global, f_glob=f_global)

        memory_K_global_after = id(K_global)
        memory_K_global_data_after = id(K_global.data)
        memory_f_global_after = id(f_global)

        # test fully preallocated version
        self.assertTrue(memory_K_global_after == memory_K_global_before)
        self.assertTrue(memory_K_global_data_after == memory_K_global_data_before)
        self.assertTrue(memory_f_global_after == memory_f_global_before)

        K_global_desired = np.zeros((8, 8), dtype=float)
        f_global_desired = np.zeros(8, dtype=float)
        # element 1
        # Not assembled:
        # K_global_desired[0:6, 0:6], f_global_desired[0:6] = self.ele.k_and_f_int(None, None)

        # element 2
        K_local, f_local = self.ele.k_and_f_int(None, None)
        # diagonals
        K_global_desired[0:2, 0:2] += weights[0] * K_local[0:2, 0:2]
        K_global_desired[4:, 4:] += weights[0] * K_local[2:, 2:]
        # off-diagonals
        K_global_desired[0:2, 4:] += weights[0] * K_local[0:2, 2:]
        K_global_desired[4:, 0:2] += weights[0] * K_local[2:, 0:2]
        # f_int:
        f_global_desired[0:2] += weights[0] * f_local[0:2]
        f_global_desired[4:] += weights[0] * f_local[2:]

        assert_array_equal(K_global.todense(), K_global_desired)
        assert_array_equal(f_global, f_global_desired)


        S_global_desired = np.ones((4, 6))

        E_global_desired = np.ones((4, 6))*2


        # Set 2nd rows to zero as this node has no element in ecsw assembly:
        S_global_desired[1, :] = 0.0
        E_global_desired[1, :] = 0.0

        assert_array_equal(K_global.todense(), K_global_desired)
        assert_array_equal(f_global, f_global_desired)
        assert_array_equal(S_global, S_global_desired)
        assert_array_equal(E_global, E_global_desired)
