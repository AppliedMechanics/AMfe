# Copyright (c) 2018, Lehrstuhl für Angewandte Mechanik, Technische Universität München.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
#


from unittest import TestCase
import numpy as np
from numpy.testing import assert_array_equal

from amfe.material import BeamMaterial


class TestBeamMaterial(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_beam_material(self):
        E_desired = 21.0
        G_desired = 32.0
        rho_desired = 54.0
        A_desired = 12.49
        I_y_desired = 40.0
        I_z_desired = 14.67
        I_p_desired = I_y_desired + I_z_desired
        J_x_desired = 510.1
        X3_desired = (1.0, 500.0, -30.0)

        mat = BeamMaterial(E_desired, G_desired, rho_desired, A_desired, I_y_desired, I_z_desired,
                           J_x_desired, X3_desired)

        self.assertEqual(mat.E_modulus, E_desired)
        self.assertEqual(mat.G_modulus, G_desired)
        self.assertEqual(mat.rho, rho_desired)
        self.assertEqual(mat.crosssec, A_desired)
        self.assertEqual(mat.I_y, I_y_desired)
        self.assertEqual(mat.I_z, I_z_desired)
        self.assertEqual(mat.I_p, I_p_desired)
        self.assertEqual(mat.J_x, J_x_desired)
        assert_array_equal(mat.X3, np.array(X3_desired, dtype=float))
