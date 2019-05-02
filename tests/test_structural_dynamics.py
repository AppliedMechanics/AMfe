# Copyright (c) 2018, Lehrstuhl für Angewandte Mechanik, Technische Universität München.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
#


from unittest import TestCase

import numpy as np
from scipy.linalg import qr
from numpy.testing import assert_allclose, assert_array_almost_equal

from amfe.structural_dynamics import *


class TestStructuralDynamicsToolsMAC(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_mac_diag_ones(self):
        """
        Test mac criterion for getting ones on the diagonal, if the same matrix is
        given.
        """
        N = 100
        n = 10
        A = np.random.rand(N, n)
        macvals = modal_assurance(A, A)
        assert_allclose(np.diag(macvals), np.ones(n))

    def test_mac_symmetric(self):
        """
        Test if MAC returns symmetric result
        """
        N = 100
        n = 10
        A = np.random.rand(N, n)
        macvals = modal_assurance(A, A)
        result = macvals - macvals.T
        assert_allclose(result, np.zeros((n, n)))

    def test_mac_identity(self):
        N = 100
        n = 10
        A = np.random.rand(N, n)
        Q, __ = qr(A, mode='economic')
        macvals = modal_assurance(Q, Q)
        assert_allclose(macvals, np.eye(n), atol=1E-14)


class TestStructuralDynamicsToolsForceNorm(TestCase):
    def setUp(self):
        return

    def tearDown(self):
        return

    def test_force_norm(self):
        F = np.array([0.0, 3.0, 4.0])
        u = np.array([1.0, 1.3, 2.5])
        K = np.array([[4.0, -1.0, 0.0], [-1.0, 3.0, -1.5], [0.0, -1.5, 2.5]])
        M = np.diag([2.0, 6.0, 5.0])

        F_actual = force_norm(F, K, M, norm='euclidean')
        F_desired = np.linalg.norm(F)
        self.assertAlmostEqual(F_actual, F_desired)

        # Impedance norm: F^T K^{-1} F
        F_imp = K.dot(u)
        F_actual = force_norm(F_imp, K, M, norm='impedance')
        F_desired = np.sqrt(F_imp.dot(u))
        self.assertAlmostEqual(F_actual, F_desired)

        # Kinetic norm: F.T K^{-T} M K^{-1} F
        F_kin = F_imp
        F_actual = force_norm(F_kin, K, M, norm='kinetic')
        F_desired = np.sqrt(u.dot(M).dot(u))
        self.assertAlmostEqual(F_actual, F_desired)

        # Test for matrix F:
        F_mat = np.array([F, F]).reshape(3, 2)
        F_actual = force_norm(F_mat, K, M, norm='euclidean')
        F_desired = np.array([np.linalg.norm(F), np.linalg.norm(F)])
        assert_array_almost_equal(F_actual, F_desired)


class TestStructuralDynamicsToolsRayleigh(TestCase):
    def setUp(self):
        return

    def tearDown(self):
        return

    def test_rayleigh_coefficients(self):
        K = np.array([[4.0, -1.0, 0.0], [-1.0, 3.0, -1.5], [0.0, -1.5, 2.5]])
        M = np.diag([2.0, 6.0, 5.0])
        omegas, V = modal_analysis(K, M, 3, mass_norm=True)
        omega1 = omegas[0]
        omega2 = omegas[1]
        zeta = 0.01

        a, b = rayleigh_coefficients(zeta, omega1, omega2)

        D = a*M + b*K

        Ddiag = V.T.dot(D).dot(V)
        self.assertAlmostEqual(Ddiag[0, 0]/2/omega1, zeta)
        self.assertAlmostEqual(Ddiag[1, 1]/2/omega2, zeta)


class TestStructuralDynamicsToolsModalAnalysis(TestCase):
    def setUp(self):
        return

    def tearDown(self):
        return

    def test_modal_analysis(self):
        K = np.array([[4.0, -1.0, 0.0], [-1.0, 3.0, -1.5], [0.0, -1.5, 2.5]])
        M = np.diag([2.0, 6.0, 5.0])

        # Test mass norm true
        omegas, V = modal_analysis(K, M, 3, mass_norm=True)
        vmv_desired = np.diag([1.0, 1.0, 1.0])
        assert_array_almost_equal(vmv_desired, V.T.dot(M).dot(V))

        for i, omega in enumerate(omegas):
            residual = (K - omega**2 * M).dot(V[:, i])
            desired = np.array([0.0, 0.0, 0.0])
            assert_array_almost_equal(residual, desired)

        # Test mass norm false
        omegas, V = modal_analysis(K, M, 3, mass_norm=False)

        for i, omega in enumerate(omegas):
            residual = (K - omega ** 2 * M).dot(V[:, i])
            desired = np.array([0.0, 0.0, 0.0])
            assert_array_almost_equal(residual, desired)
            norm_of_eigenvector = np.linalg.norm(V[:, i])
            self.assertAlmostEqual(norm_of_eigenvector, 1.0)
