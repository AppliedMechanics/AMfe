# -*- coding: utf-8 -*-
'''
Test for testing io module
'''

import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal

import amfe


class IOTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_gidascii_to_amfemesh(self):
        # Reference for nodes
        nodes_reference = np.array([[1.34560000e-02, 3.56167570e-02, 0.00000000e+00],
                                    [1.02791191e+00, 3.91996620e-02, 1.23863900e-03],
                                    [6.35836584e-02, 1.05638658e+00, 8.97892300e-03],
                                    [1.05296566e+00, 1.04992142e+00, 5.77563650e-03],
                                    [2.04236782e+00, 4.27825670e-02, 2.47727800e-03],
                                    [2.04234766e+00, 1.04345626e+00, 2.57235000e-03]])
        el_df_reference = {'active': {1: False, 2: False, 3: False, 4: False}, 'el_type': {1: 'Tri3', 2: 'Tri3', 3: 'Tri3', 4: 'Tri3'},
         'idx': {1: 1, 2: 2, 3: 3, 4: 4},
         'nodes': {1: np.array([5, 6, 4]), 2: np.array([4, 3, 2]), 3: np.array([4, 2, 5]), 4: np.array([1, 2, 3])},
         'phys_group': {1: 0, 2: 0, 3: 0, 4: 0}}

        # Define input file path
        file = amfe.amfe_dir('tests/meshes/gid_4_tets_v2.msh')
        # Define Reader Object, initialized with AmfeMeshConverter
        reader = amfe.io.GidAsciiMeshReader(file, amfe.io.AmfeMeshConverter())
        reader.parse(verbose=True)

        mesh = reader.builder.return_mesh()
        assert_allclose(mesh.nodes, nodes_reference, rtol=1e-12, atol=0.00)
        assert_equal(mesh.el_df.to_dict(), el_df_reference)
        assert_equal(mesh.no_of_dofs_per_node, 3)

if __name__ == '__main__':
    st = IOTest()
    st.setUp()
    st.test_gidascii_to_amfemesh()
