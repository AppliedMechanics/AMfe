# Copyright (c) 2018, Lehrstuhl für Angewandte Mechanik, Technische Universität München.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
#


from unittest import TestCase

import numpy as np
from scipy.sparse import csr_matrix
from numpy.testing import assert_allclose

from amfe.reduced_basis import vibration_modes


class TestVibrationModes(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_vibration_modes(self):
        K = np.array([[5.0, -2.0, 0.0], [-2.0, 4.0, -2.0], [0.0, -2.0, 3.0]], dtype=float)
        M = np.diag([2.0, 2.0, 2.0])

        omega, Phi = vibration_modes(K, M, 2)
        for i, om in enumerate(omega):
            res = (K - om**2*M).dot(Phi[:, i])
            assert_allclose(res, np.zeros(3, dtype=float), atol=1e-12)

        # Test shift
        om2 = omega[1]
        omega, Phi = vibration_modes(K, M, 1, shift=om2)
        assert_allclose(omega[0], om2)

        # Test with csr_matrices instead of numpy arrays
        K = csr_matrix(K)
        M = csr_matrix(M)

        omega, Phi = vibration_modes(K, M, 2)
        for i, om in enumerate(omega):
            res = (K - om**2*M).dot(Phi[:, i])
            assert_allclose(res, np.zeros(3, dtype=float), atol=1e-12)
