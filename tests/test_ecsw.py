# Copyright (c) 2018, Lehrstuhl für Angewandte Mechanik, Technische Universität München.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
#


from unittest import TestCase

import numpy as np
from numpy.testing import assert_, assert_allclose
from numpy.linalg import norm

from amfe.hyper_red.ecsw import sparse_nnls, assemble_g_and_b
from amfe.tools import amfe_dir
from amfe.io import GidJsonMeshReader, AmfeMeshConverter
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

    def test_assemble_g_b(self):
        # Define input file path
        file = amfe_dir('tests/meshes/gid_json_4_tets.json')
        # Define Reader Object, initialized with AmfeMeshConverter
        reader = GidJsonMeshReader(file, AmfeMeshConverter())

        # Initialize component
        my_mesh = reader.parse()
        my_component = StructuralComponent(my_mesh)
        my_material = KirchhoffMaterial()
        my_component.assign_material(my_material, ['left', 'right'], 'S')

        # Get number of dofs for snapshot generation
        no_of_dofs = my_component._constraints.no_of_constrained_dofs
        # create 2 random snapshots
        no_of_snapshots = 2
        S = np.random.rand(no_of_dofs, no_of_snapshots)*0.05
        timesteps = np.zeros(no_of_snapshots)

        # store an example for f_int for later comparison to check if the old assembly is recovered
        # after assemble_g_and_b has finished
        f_old = my_component.f_int(S[:, 0])

        # run assemble_g_and_b
        G, b = assemble_g_and_b(my_component, S, timesteps)

        # test shape of G and b
        no_of_elements = my_component.no_of_elements
        self.assertEqual(G.shape, (no_of_dofs*no_of_snapshots, no_of_elements))

        # ----------------------------------
        # Check if G is correct

        # Test first entry of G
        g11_actual = G[0:no_of_dofs, 0]
        connectivity = my_mesh.get_connectivity_by_elementids([1])[0]
        X_local = my_mesh.nodes_df.loc[connectivity].values.reshape(-1)
        u_local_indices = my_component._mapping.nodal2global.loc[connectivity].values.reshape(-1)
        u_local = S[u_local_indices, 0]
        fe_local = my_component.ele_obj[0].f_int(X_local, u_local)
        global_dofs = my_component._mapping.elements2global[0]
        g11_desired = np.zeros(no_of_dofs)
        g11_desired[global_dofs] = fe_local
        assert_allclose(g11_actual, g11_desired)
        # Test second entry of G
        g21_actual = G[no_of_dofs:, 0]
        connectivity = my_mesh.get_connectivity_by_elementids([1])[0]
        X_local = my_mesh.nodes_df.loc[connectivity].values.reshape(-1)
        u_local_indices = my_component._mapping.nodal2global.loc[connectivity].values.reshape(-1)
        u_local = S[u_local_indices, 1]
        fe_local = my_component.ele_obj[0].f_int(X_local, u_local)
        global_dofs = my_component._mapping.elements2global[0]
        g21_desired = np.zeros(no_of_dofs)
        g21_desired[global_dofs] = fe_local
        assert_allclose(g21_actual, g21_desired)
        # Test third entry of G
        g12_actual = G[0:no_of_dofs, 1]
        connectivity = my_mesh.get_connectivity_by_elementids([2])[0]
        X_local = my_mesh.nodes_df.loc[connectivity].values.reshape(-1)
        u_local_indices = my_component._mapping.nodal2global.loc[connectivity].values.reshape(-1)
        u_local = S[u_local_indices, 0]
        fe_local = my_component.ele_obj[1].f_int(X_local, u_local)
        global_dofs = my_component._mapping.elements2global[1]
        g12_desired = np.zeros(no_of_dofs)
        g12_desired[global_dofs] = fe_local
        assert_allclose(g12_actual, g12_desired)

        # --------------------------------------
        # check if b is correct:
        b_desired = np.sum(G, 1)
        assert_allclose(b, b_desired)

        # --------------------------------------
        # Check if old assembly is recovered

        # get f_new for comparison to f_old
        f_new = my_component.f_int(S[:, 0])
        # test if old assembly is recovered in the component
        assert_allclose(f_new, f_old)
